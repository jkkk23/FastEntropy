"""Train the fixed expert-guided zero-bit geometry-refinement module."""

import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    DATASETS,
    dataset_config,
    postprocess_training_output_dir,
)
import kit.codec as codec
import kit.data_preset as data
from kit.offset import offset_bounds
from kit.postprocess_training import (
    TreeDataset,
    child_bases,
    collate_one,
    deferred_tree_items,
    spherical_grid_to_xyz,
    tree_and_offset_unified_costs,
)
from kit.runtime import configure_torchsparse, select_device, set_seed
from kit.tree_refiner import (
    DecoupledBoETreeOffsetRefiner,
    occupancy_bytes_to_bits,
)


SEED = 1
NUM_WORKERS = 2
MAX_DEFERRED = 100000
LOG_EVERY = 5000
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
GATE_THRESHOLD = 0.5

FALLBACK_STEPS = 50000
TREE_STEPS = 80000
GATE_STEPS = 30000
JOINT_STEPS = 40000
FALLBACK_LEARNING_RATE = 1e-3
TREE_LEARNING_RATE = 3e-4
GATE_LEARNING_RATE = 3e-4
JOINT_EXPERT_LEARNING_RATE = 1e-4
JOINT_GATE_LEARNING_RATE = 3e-4
JOINT_WARMUP_STEPS = 5000

MASK_WEIGHT = 1.0
BIT_WEIGHT = 0.2
POPCOUNT_WEIGHT = 0.2
OFFSET_WEIGHT = 0.5
CHILD_OFFSET_WEIGHT = 0.75
TREE_GEOMETRY_WEIGHT = 0.5
FALLBACK_GEOMETRY_WEIGHT = 0.5
OFFSET_BETA = 0.1
CARTESIAN_WEIGHT = 0.25
CARTESIAN_BETA = 0.05

GATE_COST_MARGIN = 0.03
GATE_TEMPERATURE = 0.1
GATE_GEOMETRY_WEIGHT = 1.0
GATE_COUNT_WEIGHT = 0.5
GATE_COUNT_CLIP = 2.0
GATE_GEO_CLIP = 1.0
GATE_REGRET_WEIGHT = 1.0
GATE_UNCERTAIN_BAND = 0.25
GATE_MIN_SAMPLE_WEIGHT = 0.2

JOINT_EXPERT_WEIGHT = 0.35
ROUTED_ANCHOR_WEIGHT = 0.05
ROUTED_UNCERTAIN_WEIGHT = 0.1


def log(message):
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the fixed zero-bit geometry-refinement module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_folder", required=True)
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Entropy checkpoint or directory containing entropy_model.pt."
        ),
    )
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    return parser.parse_args()


def resolve_entropy_model(path):
    model_path = Path(path).expanduser().resolve()
    if model_path.is_dir():
        model_path = model_path / "entropy_model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Entropy model does not exist: {model_path}")
    return model_path


def create_postprocessor(net, device):
    return DecoupledBoETreeOffsetRefiner(
        channels=net.channels,
        hidden_channels=32,
        max_depth=16,
        boe_experts=12,
        target_mode="continuous",
    ).to(device)


def configure_stage(model, stage):
    if stage not in {"fallback", "tree", "gate", "joint"}:
        raise ValueError(f"Unsupported fixed training stage: {stage}")
    for name, parameter in model.named_parameters():
        is_fallback = name.startswith("fallback_offset.")
        is_gate = ".gate_head." in name
        if stage == "fallback":
            trainable = is_fallback
        elif stage == "tree":
            trainable = not is_fallback and not is_gate
        elif stage == "gate":
            trainable = is_gate
        else:
            trainable = not is_fallback
        parameter.requires_grad_(trainable)


def create_optimizer(model, stage):
    configure_stage(model, stage)
    if stage == "fallback":
        parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        return torch.optim.Adam(
            parameters,
            lr=FALLBACK_LEARNING_RATE,
        )
    if stage == "joint":
        expert_parameters = []
        gate_parameters = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".gate_head." in name:
                gate_parameters.append(parameter)
            else:
                expert_parameters.append(parameter)
        return torch.optim.AdamW(
            [
                {
                    "params": expert_parameters,
                    "lr": JOINT_EXPERT_LEARNING_RATE,
                },
                {
                    "params": gate_parameters,
                    "lr": JOINT_GATE_LEARNING_RATE,
                },
            ],
            weight_decay=WEIGHT_DECAY,
        )
    learning_rate = (
        TREE_LEARNING_RATE if stage == "tree" else GATE_LEARNING_RATE
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )


def _unified_costs(details, item, spherical_params):
    return tree_and_offset_unified_costs(
        details,
        details["offsets"],
        item,
        spherical_params,
        GATE_GEO_CLIP,
        GATE_GEOMETRY_WEIGHT,
        GATE_COUNT_WEIGHT,
        GATE_COUNT_CLIP,
    )


def _mask_losses(model, details, item):
    probabilities = details["probabilities"].float()
    target_bits = occupancy_bytes_to_bits(item["occupancy"]).to(
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    mask_bits = model.mask_bits.to(
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    predicted_bits = torch.matmul(probabilities, mask_bits).clamp(
        1e-6,
        1.0 - 1e-6,
    )
    bit_loss = F.binary_cross_entropy(
        predicted_bits,
        target_bits,
        reduction="none",
    ).mean(dim=1)
    target_count = target_bits.sum(dim=1)
    popcount_loss = F.smooth_l1_loss(
        details["expected_popcount"].float() / 8.0,
        target_count / 8.0,
        reduction="none",
    )
    return bit_loss, popcount_loss


def tree_loss_by_parent(model, details, item, spherical_params, costs=None):
    occupancy_target = item["occupancy"].long().view(-1) - 1
    mask_loss = F.cross_entropy(
        details["logits"].float(),
        occupancy_target,
        reduction="none",
    )
    bit_loss, popcount_loss = _mask_losses(model, details, item)

    factor = int(item["factor"])
    child_scale = max(float(factor) * 0.5, 1e-6)
    child_mask = item["child_offset_mask"].bool()
    predicted = details["child_offsets"]
    target = item["child_offset_targets"]
    local = F.smooth_l1_loss(
        predicted / child_scale,
        target / child_scale,
        beta=0.1,
        reduction="none",
    ).mean(dim=2)
    bases = child_bases(item["coords"], factor, torch.float32)
    predicted_xyz = spherical_grid_to_xyz(
        bases + predicted,
        spherical_params,
        item["dd"],
    )
    target_xyz = spherical_grid_to_xyz(
        bases + target,
        spherical_params,
        item["dd"],
    )
    cartesian = F.smooth_l1_loss(
        predicted_xyz,
        target_xyz,
        beta=CARTESIAN_BETA,
        reduction="none",
    ).mean(dim=2)
    weights = child_mask.to(local.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    child_loss = (
        ((local + CARTESIAN_WEIGHT * cartesian) * weights).sum(dim=1)
        / denominator
    )
    if costs is None:
        costs = _unified_costs(details, item, spherical_params)
    return (
        MASK_WEIGHT * mask_loss
        + BIT_WEIGHT * bit_loss
        + POPCOUNT_WEIGHT * popcount_loss
        + CHILD_OFFSET_WEIGHT * child_loss
        + TREE_GEOMETRY_WEIGHT * costs["tree_geometry"]
    )


def _mean_offset_targets(item):
    mask = item["gt_point_mask"]
    points = item["gt_points"].float()
    counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
    mean_points = (
        points * mask.unsqueeze(-1).to(points.dtype)
    ).sum(dim=1) / counts.to(points.dtype)
    factor = int(item["factor"])
    parent_base = item["coords"][:, 1:].float() * float(factor)
    lower, upper = offset_bounds(
        factor,
        item["stride"],
        "continuous",
    )
    return (mean_points - parent_base).clamp(float(lower), float(upper))


def offset_loss_by_parent(details, item, spherical_params, costs=None):
    factor = int(item["factor"])
    lower, upper = offset_bounds(factor, item["stride"], "continuous")
    scale = max(float(upper - lower), 1.0)
    offset_loss = F.smooth_l1_loss(
        details["offsets"].float() / scale,
        _mean_offset_targets(item).float() / scale,
        beta=OFFSET_BETA,
        reduction="none",
    ).mean(dim=1)
    if costs is None:
        costs = _unified_costs(details, item, spherical_params)
    return (
        OFFSET_WEIGHT * offset_loss
        + FALLBACK_GEOMETRY_WEIGHT * costs["offset_geometry"]
    )


def gate_targets(details, item, spherical_params):
    with torch.no_grad():
        costs = _unified_costs(details, item, spherical_params)
        gain = costs["offset_cost"] - costs["tree_cost"]
        decision = gain - GATE_COST_MARGIN
        target = torch.sigmoid(decision / GATE_TEMPERATURE)
        confidence = (
            decision.abs() / GATE_UNCERTAIN_BAND
        ).clamp(0.0, 1.0)
        weight = (
            GATE_MIN_SAMPLE_WEIGHT
            + (1.0 - GATE_MIN_SAMPLE_WEIGHT) * confidence
        )
    return costs, gain, target, weight


def gate_loss(details, costs, target, weight):
    logits = details["gate_logits"].float()
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    bce = (bce * weight).sum() / weight.sum().clamp_min(1.0)
    probability = torch.sigmoid(logits)
    expected_cost = (
        probability * costs["tree_cost"]
        + (1.0 - probability) * costs["offset_cost"]
    )
    best_cost = torch.minimum(costs["tree_cost"], costs["offset_cost"])
    regret = ((expected_cost - best_cost).clamp_min(0.0) * weight).sum()
    regret = regret / weight.sum().clamp_min(1.0)
    return bce + GATE_REGRET_WEIGHT * regret


def _weighted_mean(values, weights):
    weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def fixed_stage_loss(
    model,
    details,
    item,
    spherical_params,
    stage,
    stage_step,
):
    if stage == "fallback":
        factor = int(item["factor"])
        lower, upper = offset_bounds(
            factor,
            item["stride"],
            "continuous",
        )
        scale = max(float(upper - lower), 1.0)
        return F.smooth_l1_loss(
            details["offsets"].float() / scale,
            item["offset_targets"].float() / scale,
        )
    if stage == "tree":
        return tree_loss_by_parent(
            model,
            details,
            item,
            spherical_params,
        ).mean()

    costs, gain, target, weight = gate_targets(
        details,
        item,
        spherical_params,
    )
    calibration_loss = gate_loss(details, costs, target, weight)
    if stage == "gate":
        return calibration_loss

    tree_per_parent = tree_loss_by_parent(
        model,
        details,
        item,
        spherical_params,
        costs=costs,
    )
    if stage_step < JOINT_WARMUP_STEPS:
        expert_loss = JOINT_EXPERT_WEIGHT * tree_per_parent.mean()
    else:
        offset_per_parent = offset_loss_by_parent(
            details,
            item,
            spherical_params,
            costs=costs,
        )
        tree_mask = (gain > GATE_COST_MARGIN).float()
        offset_mask = ((-gain) > GATE_COST_MARGIN).float()
        uncertain_mask = (
            1.0 - tree_mask - offset_mask
        ).clamp_min(0.0)
        routed_loss = (
            _weighted_mean(tree_per_parent, tree_mask)
            + _weighted_mean(offset_per_parent, offset_mask)
            + ROUTED_UNCERTAIN_WEIGHT
            * _weighted_mean(
                torch.minimum(tree_per_parent, offset_per_parent),
                uncertain_mask,
            )
        )
        expert_loss = (
            JOINT_EXPERT_WEIGHT * routed_loss
            + ROUTED_ANCHOR_WEIGHT * tree_per_parent.mean()
        )
    return expert_loss + calibration_loss


def training_geometry(config):
    return {
        "sph_alpha": float(config.sph_alpha),
        "scale_rho": float(config.scale_rho),
        "scale_theta": float(config.scale_theta),
        "scale_phi": float(config.scale_phi),
        "radius_transform": "linear",
        "near_distance": float(config.near_distance),
        "far_distance": float(config.far_distance),
        "objective": "next_layer_8bit_occupancy",
        "loss": "boe_tree_offset_conservative_routed_joint_unified_set_gate",
        "tree_model_version": "v2",
        "fallback_offset_ckpt": "",
        "training_stage": "joint",
        "joint_expert_mode": "routed",
        "mask_weight": MASK_WEIGHT,
        "bit_weight": BIT_WEIGHT,
        "popcount_weight": POPCOUNT_WEIGHT,
        "offset_weight": OFFSET_WEIGHT,
        "child_offset_weight": CHILD_OFFSET_WEIGHT,
        "tree_geometry_weight": TREE_GEOMETRY_WEIGHT,
        "fallback_geometry_weight": FALLBACK_GEOMETRY_WEIGHT,
        "expert_geo_clip": GATE_GEO_CLIP,
        "routed_margin": -1.0,
        "routed_anchor_weight": ROUTED_ANCHOR_WEIGHT,
        "routed_uncertain_weight": ROUTED_UNCERTAIN_WEIGHT,
        "gate_regret_weight": GATE_REGRET_WEIGHT,
        "boe_route": "reuse_main_bitstream_expert_idx",
        "gate_target": "unified_normalized",
        "gate_fp_penalty": 1.5,
        "gate_fn_penalty": 1.25,
        "gate_offset_collapse_penalty": 1.0,
        "gate_geometry_weight": GATE_GEOMETRY_WEIGHT,
        "gate_count_weight": GATE_COUNT_WEIGHT,
        "gate_count_clip": GATE_COUNT_CLIP,
        "gate_geo_clip": GATE_GEO_CLIP,
        "target_mode": "continuous",
        "virtual_far_tree": True,
        "fallback_pretraining_steps": FALLBACK_STEPS,
        "tree_pretraining_steps": TREE_STEPS,
        "gate_calibration_steps": GATE_STEPS,
        "joint_training_steps": JOINT_STEPS,
        "routed_warmup_steps": JOINT_WARMUP_STEPS,
        "joint_fallback_lr": 0.0,
    }


def save_checkpoint(path, model, step, config):
    torch.save(
        {
            "deployment_gate_threshold": GATE_THRESHOLD,
            "stage": "boe_treeoffset_joint",
            "step": int(step),
            "training_geometry": training_geometry(config),
            "tree_config": model.get_config(),
            "tree_mode": "occupancy_8bit_fallback_ready",
            "tree_state_dict": model.state_dict(),
            "base_ckpt": "entropy_model.pt",
        },
        path,
    )


def train_stage(
    stage,
    max_steps,
    cumulative_start,
    model,
    net,
    loader,
    config,
    device,
):
    optimizer = create_optimizer(model, stage)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    model.train()
    stage_step = 0
    while stage_step < max_steps:
        for batch in loader:
            x = batch["x"].to(device)
            target_coords = batch["target_coords"].to(device)
            continuous_coords = batch["continuous_coords"].to(device)
            dd = int(np.random.choice(config.train_dd))
            near_radius, far_radius = codec.search_radii_in_bins(
                config.near_distance,
                config.far_distance,
                config.scale_rho,
                dd,
            )
            with torch.no_grad():
                items = deferred_tree_items(
                    net,
                    x,
                    target_coords,
                    continuous_coords,
                    dd,
                    near_radius,
                    far_radius,
                    MAX_DEFERRED,
                    offset_only=stage == "fallback",
                )
            if not items:
                continue

            weighted_loss = x.feats.new_zeros(())
            item_count = 0
            for item in items:
                if stage == "fallback":
                    details = {
                        "offsets": model.fallback_offset(
                            item["coords"],
                            item["features"].float(),
                            item["stride"],
                            item["factor"],
                            net.target_embedding,
                            depth=item["depth"],
                            expert_idx=item["boe_idx"],
                        )
                    }
                else:
                    details = model.forward_details(
                        item["coords"],
                        item["features"].float(),
                        item["stride"],
                        item["factor"],
                        net.target_embedding,
                        depth=item["depth"],
                        expert_idx=item["boe_idx"],
                    )
                loss = fixed_stage_loss(
                    model,
                    details,
                    item,
                    net.target_embedding,
                    stage,
                    stage_step,
                )
                weight = int(item["coords"].shape[0])
                weighted_loss = weighted_loss + loss * weight
                item_count += weight
            if item_count == 0:
                continue

            loss = weighted_loss / item_count
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if stage != "fallback":
                torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
            optimizer.step()
            stage_step += 1
            cumulative_step = cumulative_start + stage_step

            if stage_step == 1 or stage_step % LOG_EVERY == 0:
                log(
                    f"Stage {stage} | {stage_step}/{max_steps} | "
                    f"global {cumulative_step} | loss {loss.item():.6f} | "
                    f"items {item_count} | dd {dd}"
                )
            if stage_step >= max_steps:
                break
    return cumulative_start + stage_step


def main():
    args = parse_args()
    config = dataset_config(args.dataset)
    entropy_model = resolve_entropy_model(args.model)
    files = data.find_point_clouds(args.input_folder, args.dataset)
    output_dir = postprocess_training_output_dir(args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED, deterministic=True)
    configure_torchsparse()
    device = select_device()
    net, _ = codec.load_network(
        entropy_model,
        device,
        spherical_params=config,
    )
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    model = create_postprocessor(net, device)
    loader = torch.utils.data.DataLoader(
        TreeDataset(files, args.dataset, config),
        batch_size=1,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=str(device).startswith("cuda"),
        collate_fn=collate_one,
        prefetch_factor=1,
    )

    log(f"Training files: {len(files)}")
    log(f"Output: {output_dir}")
    train_stage(
        "fallback",
        FALLBACK_STEPS,
        0,
        model,
        net,
        loader,
        config,
        device,
    )
    cumulative_step = train_stage(
        "tree",
        TREE_STEPS,
        0,
        model,
        net,
        loader,
        config,
        device,
    )
    cumulative_step = train_stage(
        "gate",
        GATE_STEPS,
        cumulative_step,
        model,
        net,
        loader,
        config,
        device,
    )
    cumulative_step = train_stage(
        "joint",
        JOINT_STEPS,
        cumulative_step,
        model,
        net,
        loader,
        config,
        device,
    )
    final_path = output_dir / "postprocess.pt"
    save_checkpoint(
        final_path,
        model,
        cumulative_step,
        config,
    )
    log(f"Training complete: {final_path}")


if __name__ == "__main__":
    main()
