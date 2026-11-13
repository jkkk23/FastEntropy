"""Decompress FastEntropy bitstreams with zero-bit geometry refinement."""

import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from config import DATASETS, dataset_config, resolve_model_bundle
import kit.codec as codec
import kit.data_preset as data
import kit.io as io
import kit.spherical as spherical
from kit.runtime import configure_torchsparse, select_device, set_seed
from kit.tree_refiner import load_tree_refiner, warmup_tree_refiner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decompress a folder of FastEntropy bitstreams.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="Directory containing entropy_model.pt and postprocess.pt.",
    )
    parser.add_argument("--dd", required=True, type=int)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    args = parser.parse_args()

    config = dataset_config(args.dataset)
    if args.dd not in config.train_dd:
        parser.error(f"--dd must be one of {config.train_dd} for {args.dataset}")
    return args


def _validate_postprocess(checkpoint, config):
    if checkpoint.get("tree_mode") != "occupancy_8bit_fallback_ready":
        raise ValueError("postprocess.pt is not the submitted geometry-refinement model")
    geometry = checkpoint.get("training_geometry", {})
    if geometry.get("objective") != "next_layer_8bit_occupancy":
        raise ValueError("postprocess.pt has an unsupported training objective")
    threshold = float(checkpoint.get("deployment_gate_threshold", 0.5))
    if threshold != 0.5:
        raise ValueError("The submitted refinement routing threshold must be 0.5")
    for name in ("scale_rho", "scale_theta", "scale_phi"):
        if not np.isclose(
            float(geometry[name]),
            float(getattr(config, name)),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Model bundle is incompatible with the {config.name} dataset"
            )
    return threshold


def main():
    args = parse_args()
    config = dataset_config(args.dataset)
    bundle = resolve_model_bundle(args.model)
    input_files = data.find_bitstreams(args.input_folder)
    os.makedirs(args.output_folder, exist_ok=True)

    set_seed(1)
    configure_torchsparse()
    device = select_device()
    net, _ = codec.load_network(
        bundle.entropy_model,
        device,
        spherical_params=config,
    )
    postprocess, postprocess_checkpoint = load_tree_refiner(
        bundle.postprocess,
        device,
        expected_channels=net.channels,
    )
    gate_threshold = _validate_postprocess(postprocess_checkpoint, config)
    codec.warmup_network(net, device, num_points=4096, coord_max=4096)
    warmup_tree_refiner(
        postprocess,
        net.target_embedding,
        device,
        num_points=4096,
    )

    elapsed_times = []
    use_fp16 = str(device).startswith("cuda")
    with torch.inference_mode():
        for input_path in tqdm(input_files):
            metadata, base_coords, base_feats, byte_stream = (
                codec.read_compressed_file(input_path)
            )
            stream_dataset = metadata.get("data_preset", "semantickitti")
            if stream_dataset != args.dataset:
                raise ValueError(
                    f"{input_path} is a {stream_dataset} bitstream, "
                    f"not {args.dataset}"
                )
            codec.apply_spherical_params(net, metadata)
            near_radius, far_radius = codec.search_radii_in_bins(
                config.near_distance,
                config.far_distance,
                float(metadata["scale_rho"]),
                args.dd,
            )

            if use_fp16:
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                coords_int = codec.decode_streams(
                    net,
                    base_coords,
                    base_feats,
                    byte_stream,
                    near_radius=near_radius,
                    far_radius=far_radius,
                    device=device,
                    skipped_depths=args.dd,
                    tree_refiner_model=postprocess,
                    tree_gate_threshold=gate_threshold,
                )
            reconstructed = spherical.dequantize_points(
                coords_int,
                metadata,
                args.dd,
            )
            reconstructed = data.restore_points(reconstructed, metadata)
            if use_fp16:
                torch.cuda.synchronize()
            elapsed_times.append(time.perf_counter() - started)

            output_path = codec.decompressed_output_path(
                args.output_folder,
                input_path,
            )
            io.save_ply_ascii_geo(reconstructed, output_path)

    print(
        "Decompressed {count} files | Time: {elapsed:.3f} s/frame".format(
            count=len(input_files),
            elapsed=float(np.mean(elapsed_times)),
        )
    )


if __name__ == "__main__":
    main()
