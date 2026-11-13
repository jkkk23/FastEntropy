"""Compress SemanticKITTI or Ford point-cloud folders with the submitted model."""

import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from config import DATASETS, dataset_config, resolve_model_bundle
import kit.codec as codec
import kit.data_preset as data
import kit.op as op
from kit.runtime import configure_torchsparse, select_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress a folder of point clouds.",
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


def main():
    args = parse_args()
    config = dataset_config(args.dataset)
    bundle = resolve_model_bundle(args.model)
    input_files = data.find_point_clouds(
        args.input_folder,
        args.dataset,
        require_unique_stems=True,
    )
    os.makedirs(args.output_folder, exist_ok=True)

    set_seed(1)
    configure_torchsparse()
    device = select_device()
    net, _ = codec.load_network(
        bundle.entropy_model,
        device,
        spherical_params=config,
    )
    codec.warmup_network(net, device, num_points=2048, coord_max=2048)
    near_radius, far_radius = codec.search_radii_in_bins(
        config.near_distance,
        config.far_distance,
        config.scale_rho,
        args.dd,
    )

    elapsed_times = []
    bits_per_point = []
    use_fp16 = str(device).startswith("cuda")
    with torch.inference_mode():
        for input_path in tqdm(input_files):
            sample = data.load_points(input_path, args.dataset)
            if use_fp16:
                torch.cuda.synchronize()
            started = time.perf_counter()
            coords_int, metadata = codec.quantize_points(
                sample.points,
                config,
                deduplicate=False,
            )
            metadata = data.merge_metadata(metadata, sample)
            sparse_input = codec.make_sparse_input(coords_int, device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                pyramid = codec.build_occupancy_pyramid(
                    net,
                    sparse_input,
                    skipped_depths=args.dd,
                )
                byte_stream, _, _ = codec.encode_pyramid(
                    net,
                    pyramid,
                    near_radius=near_radius,
                    far_radius=far_radius,
                    skipped_depths=args.dd,
                )
            if use_fp16:
                torch.cuda.synchronize()
            elapsed_times.append(time.perf_counter() - started)

            output_path = codec.compressed_output_path(
                args.output_folder,
                input_path,
            )
            base_coords, base_feats = pyramid[0]
            codec.write_compressed_file(
                output_path,
                metadata,
                base_coords,
                base_feats,
                byte_stream,
            )
            if sample.points.shape[0]:
                bits_per_point.append(
                    op.get_file_size_in_bits(output_path) / sample.points.shape[0]
                )

    print(
        "Compressed {count} files | Bpp: {bpp:.3f} | Time: {elapsed:.3f} s/frame".format(
            count=len(input_files),
            bpp=float(np.mean(bits_per_point)),
            elapsed=float(np.mean(elapsed_times)),
        )
    )


if __name__ == "__main__":
    main()
