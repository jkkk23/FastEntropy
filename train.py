"""Train the fixed FastEntropy entropy model."""

import argparse
import datetime
import gc

import numpy as np
import torch
from torch.cuda import amp
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from tqdm import tqdm

from config import (
    DATASETS,
    OPTIMAL_ARCHITECTURE,
    dataset_config,
    training_output_dir,
)
from network import Network
import kit.data_preset as data
import kit.spherical as spherical
from kit.runtime import configure_torchsparse, select_device, set_seed


SEED = 12
BOE_CLUSTER_SAMPLES = 5000
BOE_CLUSTER_ITERATIONS = 30


def log(message):
    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {message}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the submitted FastEntropy architecture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        nargs="+",
        default=(50000, 170000),
    )
    parser.add_argument("--max_steps", type=int, default=220000)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    args = parser.parse_args()

    if args.channels <= 0:
        parser.error("--channels must be positive")
    if args.kernel_size <= 0 or args.kernel_size % 2 == 0:
        parser.error("--kernel_size must be a positive odd integer")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be positive")
    if not 0 < args.lr_decay <= 1:
        parser.error("--lr_decay must be in (0, 1]")
    if any(step <= 0 for step in args.lr_decay_steps):
        parser.error("--lr_decay_steps must contain positive integers")
    if args.max_steps <= 0:
        parser.error("--max_steps must be positive")
    return args


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, files, dataset, config):
        self.files = list(files)
        self.dataset = dataset
        self.config = config

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        points = data.load_points(self.files[index], self.dataset).points
        coords, _ = spherical.quantize_points(
            points,
            sph_alpha=self.config.sph_alpha,
            scale_rho=self.config.scale_rho,
            scale_theta=self.config.scale_theta,
            scale_phi=self.config.scale_phi,
            use_log_radius=False,
        )
        coords = torch.from_numpy(coords).int()
        features = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        return {"input": SparseTensor(coords=coords, feats=features)}


def create_network(channels, kernel_size, config):
    return Network(
        channels=channels,
        kernel_size=kernel_size,
        sph_alpha=config.sph_alpha,
        scale_rho=config.scale_rho,
        scale_theta=config.scale_theta,
        scale_phi=config.scale_phi,
        use_log_radius=False,
        **OPTIMAL_ARCHITECTURE,
    )


def _cluster_input(path, dataset, config):
    points = data.load_points(path, dataset).points
    coords, _ = spherical.quantize_points(
        points,
        sph_alpha=config.sph_alpha,
        scale_rho=config.scale_rho,
        scale_theta=config.scale_theta,
        scale_phi=config.scale_phi,
        use_log_radius=False,
    )
    coords = torch.from_numpy(coords).int()
    batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32)
    features = torch.ones((coords.shape[0], 1), dtype=torch.float32)
    return SparseTensor(coords=torch.cat((batch, coords), dim=1), feats=features)


def _kmeans(values, clusters):
    if values.shape[0] < clusters:
        repeats = int(np.ceil(clusters / values.shape[0]))
        values = values.repeat((repeats, 1))[:clusters]
    centers = values[torch.randperm(values.shape[0])[:clusters]].clone()
    labels = torch.zeros(values.shape[0], dtype=torch.long)
    for _ in range(BOE_CLUSTER_ITERATIONS):
        labels = torch.argmin(torch.cdist(values, centers), dim=1)
        next_centers = centers.clone()
        for index in range(clusters):
            mask = labels == index
            if mask.any():
                next_centers[index] = values[mask].mean(dim=0)
        if torch.allclose(next_centers, centers, atol=1e-6):
            break
        centers = next_centers
    labels = torch.argmin(torch.cdist(values, centers), dim=1)
    return centers, labels


def fit_boe_centers(files, dataset, config, channels, kernel_size):
    if len(files) > BOE_CLUSTER_SAMPLES:
        indices = np.random.choice(
            len(files),
            BOE_CLUSTER_SAMPLES,
            replace=False,
        )
        cluster_files = [files[index] for index in sorted(indices)]
    else:
        cluster_files = files

    net = create_network(channels, kernel_size, config).cpu().eval()
    count_descriptors = []
    symbol_descriptors = []
    with torch.inference_mode():
        for path in tqdm(cluster_files, desc="Fit BoE routing"):
            skipped_depths = int(np.random.choice(config.train_dd))
            pyramid = net.build_occupancy_pyramid(
                _cluster_input(path, dataset, config),
                skipped_depths=skipped_depths,
            )
            encoded_depths = max(len(pyramid) - 1, 0)
            scale = 2**skipped_depths
            near_radius = config.near_distance / config.scale_rho / scale
            far_radius = config.far_distance / config.scale_rho / scale
            for depth in range(1, encoded_depths):
                coords, occupancy = pyramid[depth + 1]
                coords, occupancy = Network.separate_coding_depth(
                    coords,
                    occupancy,
                    depth=depth,
                    encoded_depths=encoded_depths,
                    near_radius=near_radius,
                    far_radius=far_radius,
                )
                s0, s1 = net.split_occupancy_two_stage(occupancy)
                count_descriptors.append(
                    net._count_descriptor(occupancy[:, 0]).cpu()
                )
                symbol_descriptors.append(
                    net._symbol_descriptor(
                        s0[:, 0],
                        s1[:, 0],
                        sqrt=True,
                    ).cpu()
                )
    if not count_descriptors:
        raise RuntimeError("The training set produced no BoE routing descriptors")

    count_values = torch.stack(count_descriptors)
    symbol_values = torch.stack(symbol_descriptors)
    count_centers, group_labels = _kmeans(
        count_values,
        OPTIMAL_ARCHITECTURE["boe_count_groups"],
    )
    order = torch.argsort(count_centers[:, 8])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel())
    count_centers = count_centers[order]
    group_labels = inverse[group_labels]

    group_count = OPTIMAL_ARCHITECTURE["boe_count_groups"]
    leaves = OPTIMAL_ARCHITECTURE["boe_leaves_per_group"]
    symbol_centers = torch.zeros(group_count, leaves, symbol_values.shape[1])
    for group in range(group_count):
        mask = group_labels == group
        group_values = symbol_values[mask] if mask.any() else symbol_values
        symbol_centers[group], _ = _kmeans(group_values, leaves)
    return count_centers.numpy(), symbol_centers.numpy()


def save_checkpoint(path, net, step):
    torch.save(
        {
            "final_step": step,
            "model_config": net.get_model_config(),
            "model_state_dict": net.state_dict(),
        },
        path,
    )


def main():
    args = parse_args()
    config = dataset_config(args.dataset)
    files = data.find_point_clouds(args.input_folder, args.dataset)
    output_dir = training_output_dir(args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED, deterministic=True)
    configure_torchsparse()
    log(f"Training files: {len(files)}")
    log(f"Checkpoints: {output_dir}")
    count_centers, symbol_centers = fit_boe_centers(
        files,
        args.dataset,
        config,
        args.channels,
        args.kernel_size,
    )
    np.savez(
        output_dir / "boe_centers_hierarchical.npz",
        count_centers=count_centers.astype(np.float32),
        symbol_centers=symbol_centers.astype(np.float32),
    )

    device = select_device()
    net = create_network(args.channels, args.kernel_size, config)
    net.load_boe_centers(
        np.load(output_dir / "boe_centers_hierarchical.npz")
    )
    net.to(device).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    scaler = amp.GradScaler(enabled=str(device).startswith("cuda"))

    chunks = [files[index:index + 100] for index in range(0, len(files), 100)]
    global_step = 0
    epoch = 0
    try:
        while global_step < args.max_steps:
            np.random.shuffle(chunks)
            for chunk in chunks:
                dataset = PointCloudDataset(chunk, args.dataset, config)
                loader = torch.utils.data.DataLoader(
                    dataset,
                    shuffle=True,
                    batch_size=1,
                    collate_fn=sparse_collate_fn,
                    num_workers=1,
                    pin_memory=str(device).startswith("cuda"),
                    drop_last=True,
                    prefetch_factor=1,
                )
                for batch in loader:
                    sparse_input = batch["input"].to(
                        device=device,
                        non_blocking=True,
                    )
                    dd = int(np.random.choice(config.train_dd))
                    scale = 2**dd
                    near_radius = config.near_distance / config.scale_rho / scale
                    far_radius = config.far_distance / config.scale_rho / scale

                    optimizer.zero_grad(set_to_none=True)
                    with amp.autocast(enabled=str(device).startswith("cuda")):
                        result = net.forward_rate(
                            sparse_input,
                            skipped_depths=dd,
                            near_radius=near_radius,
                            far_radius=far_radius,
                            return_details=True,
                        )
                    scaler.scale(result["loss"]).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    global_step += 1

                    if global_step in args.lr_decay_steps:
                        for group in optimizer.param_groups:
                            group["lr"] *= args.lr_decay
                    if global_step % 100 == 0:
                        log(
                            f"Step {global_step}/{args.max_steps} | "
                            f"Rate {result['rate_bpp'].item():.5f} bpp | dd={dd}"
                        )
                    if global_step % 500 == 0:
                        save_checkpoint(
                            output_dir / "checkpoint.pt",
                            net,
                            global_step,
                        )
                    if global_step >= args.max_steps:
                        break
                del loader
                gc.collect()
                torch.cuda.empty_cache()
                if global_step >= args.max_steps:
                    break
            epoch += 1
            log(f"Completed epoch {epoch}")
    except KeyboardInterrupt:
        save_checkpoint(
            output_dir / "checkpoint_interrupted.pt",
            net,
            global_step,
        )
        raise SystemExit(130)

    save_checkpoint(
        output_dir / "entropy_model.pt",
        net,
        global_step,
    )
    log(f"Training complete: {output_dir / 'entropy_model.pt'}")


if __name__ == "__main__":
    main()
