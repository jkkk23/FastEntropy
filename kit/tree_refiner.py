import math

import torch
import torch.nn as nn

from kit.offset import OffsetPredictor


_CONTINUOUS_FACTORS = (4, 2, 1)


def _supported_factors(target_mode):
    if str(target_mode) != "continuous":
        raise ValueError("Only continuous TreeOffset V2 is supported")
    return _CONTINUOUS_FACTORS


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _factor_key(factor, target_mode="continuous"):
    factor = int(factor)
    if factor not in _supported_factors(target_mode):
        raise ValueError(
            f"Unsupported TreeOffset factor {factor} for target_mode={target_mode}"
        )
    return str(factor)


def _factor_id(factor, target_mode):
    factor = int(factor)
    factors = _supported_factors(target_mode)
    if factor not in factors:
        raise ValueError(
            f"Unsupported TreeOffset factor {factor} for target_mode={target_mode}"
        )
    return factors.index(factor)


def _expanded_ids(value, count, device, upper_bound):
    if value is None:
        ids = torch.zeros(count, dtype=torch.long, device=device)
    elif torch.is_tensor(value):
        ids = value.to(device=device, dtype=torch.long).reshape(-1)
        if ids.numel() == 1:
            ids = ids.expand(count)
        elif ids.numel() != count:
            raise ValueError(
                f"Condition id count {ids.numel()} does not match {count}"
            )
    else:
        ids = torch.full((count,), int(value), dtype=torch.long, device=device)
    return ids.clamp(0, max(int(upper_bound) - 1, 0))


def occupancy_bytes_to_bits(occupancy):
    occupancy = occupancy.long().view(-1)
    bit_ids = torch.arange(8, dtype=torch.long, device=occupancy.device)
    return ((occupancy[:, None] >> bit_ids[None, :]) & 1).float()


def _geometry_features(coords, stride, factor, spherical_params, dtype):
    factor = int(factor)
    if factor not in {1, 2, 4}:
        raise ValueError(f"Unsupported TreeOffset geometry factor: {factor}")
    stride = float(stride)
    dd_scale = stride / float(factor)
    center_dd = (float(factor) - 1.0) * 0.5

    rho_grid = (coords[:, 1].float() * factor + center_dd) * dd_scale
    theta_grid = (coords[:, 2].float() * factor + center_dd) * dd_scale
    phi_grid = (coords[:, 3].float() * factor + center_dd) * dd_scale

    rho_value = rho_grid * float(spherical_params.scale_rho)
    if bool(getattr(spherical_params, "use_log_radius", False)):
        sph_alpha = float(getattr(spherical_params, "sph_alpha", 1.0))
        rho_metric = torch.expm1(rho_value.clamp_max(30.0)) / sph_alpha
    else:
        rho_metric = rho_value

    theta_deg = (
        theta_grid - float(spherical_params.theta_offset)
    ) * float(spherical_params.scale_theta)
    phi_deg = (
        phi_grid - float(spherical_params.phi_offset)
    ) * float(spherical_params.scale_phi)
    theta_rad = torch.deg2rad(theta_deg)
    phi_rad = torch.deg2rad(phi_deg)
    log_stride = math.log2(max(stride, 1.0)) / 16.0

    return torch.stack(
        (
            rho_metric / 100.0,
            torch.sin(theta_rad),
            torch.cos(theta_rad),
            torch.sin(phi_rad),
            torch.cos(phi_rad),
            torch.full_like(rho_metric, log_stride),
        ),
        dim=1,
    ).to(dtype=dtype)


def tree_child_coords(
    net,
    parent_coords,
    occupancy,
    factor,
    child_offsets=None,
):
    factor = int(factor)
    child_factor = float(factor) * 0.5
    child_center = (child_factor - 1.0) * 0.5
    child_coords = net.fcg(parent_coords, occupancy)
    positions = child_coords[:, 1:].float() * float(child_factor)
    if child_offsets is None:
        return positions + child_center

    active_bits = occupancy_bytes_to_bits(occupancy).bool()
    parent_idx, child_id = torch.nonzero(active_bits, as_tuple=True)
    if parent_idx.shape[0] != child_coords.shape[0]:
        raise RuntimeError("Tree child-offset order does not match FCG output")
    selected_offsets = child_offsets[parent_idx, child_id].to(positions.dtype)
    return positions + selected_offsets


def _child_offset_center_and_half(factor, target_mode="continuous"):
    factor = int(factor)
    _supported_factors(target_mode)
    child_factor = float(factor) * 0.5
    lower = -0.5
    upper = child_factor - 0.5
    return (lower + upper) * 0.5, (upper - lower) * 0.5


class BoETreeBranchExpert(nn.Module):
    """Tree-structured expansion expert with child-specific offsets."""

    def __init__(self, channels, hidden_channels, max_depth, target_mode="continuous"):
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.max_depth = max(int(max_depth), 1)
        self.target_mode = str(target_mode)
        self.supported_factors = _supported_factors(self.target_mode)

        self.feature_norm = nn.LayerNorm(self.channels)
        self.factor_embedding = nn.Embedding(len(self.supported_factors), 4)
        self.depth_embedding = nn.Embedding(self.max_depth, 4)
        input_channels = self.channels + 6 + 4 + 4
        self.trunk = nn.Sequential(
            nn.Linear(input_channels, self.hidden_channels),
            nn.ReLU(True),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
        )
        self.mask_head = nn.Linear(self.hidden_channels, 255)
        self.child_offset_head = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
            nn.Linear(self.hidden_channels, 8 * 3),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(self.hidden_channels + 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
        )

        nn.init.zeros_(self.child_offset_head[-1].weight)
        nn.init.zeros_(self.child_offset_head[-1].bias)
        nn.init.constant_(self.gate_head[-1].bias, -2.0)

    def encode_hidden(
        self,
        coords,
        features,
        stride,
        factor,
        spherical_params,
        depth=None,
    ):
        if features.ndim != 2 or features.shape[1] != self.channels:
            raise ValueError(
                f"Expected [N, {self.channels}] features, got "
                f"{tuple(features.shape)}"
            )
        factor_key = int(_factor_key(factor, self.target_mode))
        normalized = self.feature_norm(features.float()).to(features.dtype)
        geometry = _geometry_features(
            coords,
            stride,
            int(factor_key),
            spherical_params,
            normalized.dtype,
        )
        factor_id = _factor_id(factor, self.target_mode)
        factor_ids = torch.full(
            (coords.shape[0],),
            factor_id,
            dtype=torch.long,
            device=coords.device,
        )
        depth_ids = _expanded_ids(
            depth,
            coords.shape[0],
            coords.device,
            self.max_depth,
        )
        inputs = torch.cat(
            (
                normalized,
                geometry,
                self.factor_embedding(factor_ids).to(normalized.dtype),
                self.depth_embedding(depth_ids).to(normalized.dtype),
            ),
            dim=1,
        )
        return self.trunk(inputs)

    def child_offsets(self, hidden, factor):
        center, half = _child_offset_center_and_half(factor, self.target_mode)
        raw = self.child_offset_head(hidden).float().view(-1, 8, 3)
        return center + half * torch.tanh(raw)

    def gate_logits(self, hidden, stats):
        summary = torch.stack(
            (
                stats["top_probability"],
                stats["probability_margin"],
                1.0 - stats["entropy"],
                stats["expected_popcount"] / 8.0,
            ),
            dim=1,
        ).detach().to(hidden.dtype)
        return self.gate_head(torch.cat((hidden, summary), dim=1)).squeeze(1)


def _default_fallback_config(
    channels,
    hidden_channels,
    max_depth,
    boe_experts,
    target_mode,
):
    if int(boe_experts) != 12:
        raise ValueError("TreeOffset V2 requires 12 hierarchical BoE experts")
    return {
        "channels": int(channels),
        "hidden_channels": int(hidden_channels),
        "leaf_hidden_channels": 8,
        "max_depth": int(max_depth),
        "target_mode": str(target_mode),
        "boe_mode": "hierarchical_group_leaf",
        "boe_count_groups": 3,
        "boe_leaves_per_group": 4,
    }


class DecoupledBoETreeOffsetRefiner(nn.Module):
    """Expert-guided zero-bit geometry refinement with two reconstruction branches."""

    def __init__(
        self,
        channels,
        hidden_channels=32,
        max_depth=16,
        boe_experts=12,
        target_mode="continuous",
        fallback_config=None,
    ):
        super().__init__()
        self.model_type = "boe_tree_offset_v2"
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.max_depth = max(int(max_depth), 1)
        self.boe_experts = int(boe_experts)
        self.target_mode = str(target_mode)
        self.supported_factors = _supported_factors(self.target_mode)
        if self.boe_experts != 12:
            raise ValueError("TreeOffset V2 requires 12 hierarchical BoE experts")

        if fallback_config is None:
            fallback_config = _default_fallback_config(
                self.channels,
                self.hidden_channels,
                self.max_depth,
                self.boe_experts,
                self.target_mode,
            )
        fallback_config = dict(fallback_config)
        if int(fallback_config["channels"]) != self.channels:
            raise ValueError("fallback channels must match TreeOffset channels")
        if str(fallback_config.get("target_mode", self.target_mode)) != self.target_mode:
            raise ValueError("fallback target_mode must match TreeOffset target_mode")
        self.fallback_offset = OffsetPredictor(**fallback_config)

        self.tree_experts = nn.ModuleList(
            [
                BoETreeBranchExpert(
                    self.channels,
                    self.hidden_channels,
                    self.max_depth,
                    self.target_mode,
                )
                for _ in range(self.boe_experts)
            ]
        )

        mask_values = torch.arange(1, 256, dtype=torch.long)
        bit_ids = torch.arange(8, dtype=torch.long)
        mask_bits = ((mask_values[:, None] >> bit_ids[None, :]) & 1).float()
        self.register_buffer("mask_bits", mask_bits, persistent=False)
        self.register_buffer(
            "mask_popcount",
            mask_bits.sum(dim=1),
            persistent=False,
        )

    def get_config(self):
        return {
            "model_type": self.model_type,
            "channels": self.channels,
            "hidden_channels": self.hidden_channels,
            "max_depth": self.max_depth,
            "boe_experts": self.boe_experts,
            "target_mode": self.target_mode,
            "fallback_config": self.fallback_offset.get_config(),
        }

    def supports_factor(self, factor):
        return (
            int(factor) in self.supported_factors
            and self.fallback_offset.supports_factor(factor)
        )

    def _expert_id(self, expert_idx):
        if torch.is_tensor(expert_idx):
            flat = expert_idx.detach().reshape(-1)
            if flat.numel() == 0:
                return 0
            value = int(flat[0].item())
            if flat.numel() > 1 and bool((flat != value).any().item()):
                raise ValueError(
                    "DecoupledBoETreeOffsetRefiner expects one BoE id per batch item"
                )
        elif expert_idx is None:
            value = 0
        else:
            value = int(expert_idx)
        return max(0, min(value, self.boe_experts - 1))

    def mask_statistics(self, logits):
        probabilities = torch.softmax(logits.float(), dim=1)
        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=2,
            dim=1,
        )
        occupancy = top_indices[:, 0].long() + 1
        entropy = -(
            probabilities * torch.log(probabilities.clamp_min(1e-10))
        ).sum(dim=1) / math.log(probabilities.shape[1])
        expected_popcount = torch.matmul(
            probabilities,
            self.mask_popcount.to(probabilities.dtype),
        )
        return {
            "probabilities": probabilities,
            "occupancy": occupancy.view(-1, 1),
            "top_probability": top_probabilities[:, 0],
            "probability_margin": (
                top_probabilities[:, 0] - top_probabilities[:, 1]
            ),
            "entropy": entropy,
            "expected_popcount": expected_popcount,
            "predicted_popcount": self.mask_popcount[top_indices[:, 0]].long(),
        }

    def forward_details(
        self,
        coords,
        features,
        stride,
        factor,
        spherical_params,
        depth=None,
        expert_idx=None,
    ):
        expert_id = self._expert_id(expert_idx)
        expert = self.tree_experts[expert_id]
        hidden = expert.encode_hidden(
            coords,
            features,
            stride,
            factor,
            spherical_params,
            depth=depth,
        )
        logits = expert.mask_head(hidden)
        stats = self.mask_statistics(logits)
        offsets = self.fallback_offset(
            coords,
            features,
            stride,
            factor,
            spherical_params,
            depth=depth,
            expert_idx=expert_idx,
            confidence_threshold=0.0,
        )
        details = {
            "expert_id": expert_id,
            "hidden": hidden,
            "logits": logits,
            "offsets": offsets,
            "child_offsets": expert.child_offsets(hidden, factor),
        }
        details.update(stats)
        details["gate_logits"] = expert.gate_logits(hidden, stats)
        details["gate_probability"] = torch.sigmoid(
            details["gate_logits"].float()
        )
        return details

    def forward(
        self,
        coords,
        features,
        stride,
        factor,
        spherical_params,
        depth=None,
        expert_idx=None,
    ):
        return self.forward_details(
            coords,
            features,
            stride,
            factor,
            spherical_params,
            depth=depth,
            expert_idx=expert_idx,
        )["logits"]


def load_tree_refiner(path, device, expected_channels=None):
    checkpoint = _torch_load(path, map_location="cpu")
    if "tree_config" not in checkpoint or "tree_state_dict" not in checkpoint:
        raise ValueError(f"Invalid BoE TreeOffset checkpoint: {path}")

    config = dict(checkpoint["tree_config"])
    model_type = config.pop("model_type")
    if model_type != "boe_tree_offset_v2":
        raise ValueError(
            "Only the submitted decoupled TreeOffset V2 is supported; "
            f"got model_type={model_type!r}"
        )
    if expected_channels is not None and int(config["channels"]) != int(
        expected_channels
    ):
        raise ValueError(
            "TreeOffset feature channels do not match the base model: "
            f"{config['channels']} != {expected_channels}"
        )
    model = DecoupledBoETreeOffsetRefiner(**config)
    model.load_state_dict(checkpoint["tree_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def warmup_tree_refiner(model, spherical_params, device, num_points=4096):
    if model is None or not str(device).startswith("cuda"):
        return
    coords = torch.randint(
        0,
        4096,
        (int(num_points), 4),
        dtype=torch.int32,
        device=device,
    )
    coords[:, 0].zero_()
    features = torch.randn(
        int(num_points),
        model.channels,
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        for factor in model.supported_factors:
            model(
                coords,
                features,
                stride=factor,
                factor=factor,
                spherical_params=spherical_params,
                depth=8,
                expert_idx=0,
            )
    torch.cuda.synchronize()
