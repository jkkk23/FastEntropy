import numpy as np
import torch
from torchsparse import SparseTensor

from network import Network
import kit.data_preset as data
import kit.op as op
import kit.spherical as spherical
from kit.offset import offset_bounds
from kit.tree_refiner import occupancy_bytes_to_bits


class TreeDataset(torch.utils.data.Dataset):
    def __init__(self, files, dataset, config):
        self.files = list(files)
        self.dataset = dataset
        self.config = config

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        points = data.load_points(self.files[index], self.dataset).points
        target_coords, _ = spherical.quantize_points(
            points,
            sph_alpha=self.config.sph_alpha,
            scale_rho=self.config.scale_rho,
            scale_theta=self.config.scale_theta,
            scale_phi=self.config.scale_phi,
            use_log_radius=False,
            deduplicate=False,
        )
        theta_offset, phi_offset = spherical.get_angle_offsets(
            self.config.scale_theta,
            self.config.scale_phi,
        )
        x, y, z = points.T
        rho = np.linalg.norm(points, axis=1)
        theta = np.degrees(np.arctan2(y, x))
        phi = np.degrees(
            np.arcsin(np.clip(z / np.maximum(rho, 1e-8), -1.0, 1.0))
        )
        continuous_coords = np.stack(
            (
                rho / self.config.scale_rho,
                theta / self.config.scale_theta + theta_offset,
                phi / self.config.scale_phi + phi_offset,
            ),
            axis=1,
        ).astype(np.float32)
        unique_coords = spherical.unique_coords(
            target_coords,
        )
        return {
            "unique_coords": torch.from_numpy(unique_coords).int(),
            "target_coords": torch.from_numpy(target_coords).int(),
            "continuous_coords": torch.from_numpy(continuous_coords).float(),
        }


def collate_one(batch):
    sample = batch[0]
    unique_coords = sample["unique_coords"]
    target_coords = sample["target_coords"]
    continuous_coords = sample["continuous_coords"]
    unique_batch = torch.zeros((unique_coords.shape[0], 1), dtype=torch.int32)
    target_batch = torch.zeros((target_coords.shape[0], 1), dtype=torch.int32)
    x = SparseTensor(
        coords=torch.cat((unique_batch, unique_coords), dim=1),
        feats=torch.ones((unique_coords.shape[0], 1), dtype=torch.float32),
    )
    return {
        "x": x,
        "target_coords": torch.cat((target_batch, target_coords), dim=1),
        "continuous_coords": torch.cat(
            (target_batch.float(), continuous_coords),
            dim=1,
        ),
    }


def predict_child_candidates(
    net,
    x_C,
    x_O,
    depth,
    encoded_depths,
    propagated_feats,
    propagated_core_idx=None,
    lineage_feats=None,
    return_lineage=False,
):
    x_F = net.core().prior_embedding(x_O.int()).view(-1, net.channels)
    x_F = net.fuse_cross_depth_parent(
        x_F,
        propagated_feats,
        depth,
        encoded_depths,
        source_core_idx=propagated_core_idx,
        target_core_idx=getattr(net, "_active_core_idx", None),
    )
    x_F = net.fuse_lineage_parent(x_F, lineage_feats)
    x = net.run_prior_resnet(SparseTensor(coords=x_C, feats=x_F), depth)
    (
        x_up_C,
        x_up_F,
        x_up_parent_O,
        x_up_child_id,
        x_up_parent_idx,
    ) = net.fcg(
        x_C,
        x_O,
        x.feats,
        return_info=True,
        return_parent_idx=True,
    )
    x_up_lineage = net.expand_lineage_to_children(
        lineage_feats,
        x_up_parent_idx,
        x_up_child_id,
        x_up_F.dtype,
        x_up_F.device,
    )
    x_up_F = net.fuse_child_features(
        x_up_F,
        x_up_parent_O,
        x_up_child_id,
        x_up_lineage,
    )
    x_up_C, x_up_F, x_up_lineage = net.order_CFs(
        x_up_C,
        x_up_F,
        x_up_lineage,
    )
    if return_lineage:
        return x_up_C, x_up_F, x_up_lineage
    return x_up_C, x_up_F


def score_child_features(
    net,
    x_up_C,
    x_up_F,
    depth,
    encoded_depths,
    skipped_depths,
    target_stride,
):
    x_up_F = net.target_embedding(
        x_up_F,
        x_up_C,
        target_stride=target_stride,
    )
    x_up_F = net.apply_coarse_grid_prior(
        x_up_C,
        x_up_F,
        depth,
        target_stride,
    )
    return net.run_target_resnet(
        SparseTensor(coords=x_up_C, feats=x_up_F),
        depth=depth,
        encoded_depths=encoded_depths,
        skipped_depths=skipped_depths,
    )


def row_keys(rows, minimum, spans):
    shifted = rows.long() - minimum
    return (
        ((shifted[:, 0] * spans[1] + shifted[:, 1]) * spans[2]
         + shifted[:, 2]) * spans[3]
        + shifted[:, 3]
    )


def gather_occupancy_by_coords(query_coords, ref_coords, ref_occupancy):
    if query_coords.shape[0] == 0:
        return ref_occupancy.new_empty((0, 1))
    all_coords = torch.cat((query_coords, ref_coords), dim=0).long()
    minimum = all_coords.min(dim=0).values
    spans = (all_coords - minimum).max(dim=0).values + 1

    query_keys = row_keys(query_coords, minimum, spans)
    ref_keys = row_keys(ref_coords, minimum, spans)
    sorted_keys, order = torch.sort(ref_keys)
    pos = torch.searchsorted(sorted_keys, query_keys)
    valid = (
        (pos < sorted_keys.numel())
        & (sorted_keys[pos.clamp_max(sorted_keys.numel() - 1)] == query_keys)
    )
    if not bool(valid.all().item()):
        missing = int((~valid).sum().item())
        raise RuntimeError(f"{missing} deferred coordinates lack GT occupancy")
    return ref_occupancy[order[pos]].long()


def limit_candidates(coords, features, occupancy, limit):
    if limit <= 0 or coords.shape[0] <= limit:
        return coords, features, occupancy
    indices = torch.randperm(coords.shape[0], device=coords.device)[:limit]
    return coords[indices], features[indices], occupancy[indices]


def build_offset_targets(
    original_coords,
    candidate_coords,
    skipped_depths,
    factor,
    continuous_coords,
):
    """Continuous spherical-mean targets used by the fixed fallback stage."""
    factor = int(factor)
    if candidate_coords.shape[0] == 0:
        return candidate_coords.new_empty((0, 3), dtype=torch.float32)
    dd_scale = 2 ** int(skipped_depths)
    grid_coords = original_coords.clone().long()
    grid_coords[:, 1:] = torch.div(
        grid_coords[:, 1:],
        dd_scale,
        rounding_mode="floor",
    )
    cell_coords = grid_coords.clone()
    cell_coords[:, 1:] = torch.div(
        cell_coords[:, 1:],
        factor,
        rounding_mode="floor",
    )
    unique_cells, inverse = torch.unique(
        cell_coords,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    continuous_grid = continuous_coords[:, 1:].float() / float(dd_scale)
    local_offsets = (
        continuous_grid
        - cell_coords[:, 1:].float() * float(factor)
    )
    sums = torch.zeros(
        (unique_cells.shape[0], 3),
        dtype=torch.float32,
        device=original_coords.device,
    )
    counts = torch.zeros(
        (unique_cells.shape[0], 1),
        dtype=torch.float32,
        device=original_coords.device,
    )
    sums.index_add_(0, inverse, local_offsets)
    counts.index_add_(
        0,
        inverse,
        torch.ones(
            (inverse.shape[0], 1),
            dtype=torch.float32,
            device=original_coords.device,
        ),
    )
    representatives = sums / counts.clamp_min(1.0)

    all_rows = torch.cat((unique_cells, candidate_coords.long()), dim=0)
    minimum = all_rows.min(dim=0).values
    spans = all_rows.max(dim=0).values - minimum + 1
    unique_keys = row_keys(unique_cells, minimum, spans)
    candidate_keys = row_keys(candidate_coords.long(), minimum, spans)
    sorted_keys, order = torch.sort(unique_keys)
    positions = torch.searchsorted(sorted_keys, candidate_keys)
    valid = positions < sorted_keys.shape[0]
    safe_positions = positions.clamp_max(max(sorted_keys.shape[0] - 1, 0))
    valid &= sorted_keys[safe_positions] == candidate_keys
    if not bool(valid.all().item()):
        raise RuntimeError("Deferred candidate could not be matched to input points")
    targets = representatives[order[safe_positions]]
    lower, upper = offset_bounds(
        factor,
        factor * dd_scale,
        "continuous",
    )
    return targets.clamp(lower, upper)


def build_deferred_geometry_targets(
    original_coords,
    candidate_coords,
    occupancy,
    skipped_depths,
    factor,
    continuous_coords,
):
    factor = int(factor)
    count = int(candidate_coords.shape[0])
    max_points = 8 if factor == 1 else factor ** 3
    device = candidate_coords.device

    grid_coords = original_coords.clone().long()
    grid_coords[:, 1:] = torch.div(
        grid_coords[:, 1:],
        2 ** int(skipped_depths),
        rounding_mode="floor",
    )

    if continuous_coords.shape != original_coords.shape:
        raise ValueError(
            "continuous_coords must have the same shape as original_coords"
        )
    point_values = continuous_coords[:, 1:].float() / float(
        2 ** int(skipped_depths)
    )

    if factor == 1:
        point_parents = grid_coords.clone()
        residual = (point_values - grid_coords[:, 1:].float()).clamp(-0.5, 0.5)
        child_bits = (residual >= 0.0).long()
        child_id_per_point = (
            child_bits[:, 0]
            + 2 * child_bits[:, 1]
            + 4 * child_bits[:, 2]
        )
        representative_cells = grid_coords.clone()
        representative_cells[:, 1:] = (
            representative_cells[:, 1:] * 2
            + child_bits.to(representative_cells.dtype)
        )
    else:
        point_parents = grid_coords.clone()
        point_parents[:, 1:] = torch.div(
            point_parents[:, 1:],
            factor,
            rounding_mode="floor",
        )
        representative_cells = grid_coords.clone()

    unique_representatives, inverse = torch.unique(
        representative_cells,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    representative_sums = torch.zeros(
        (unique_representatives.shape[0], 3),
        dtype=torch.float32,
        device=device,
    )
    representative_counts = torch.zeros(
        (unique_representatives.shape[0], 1),
        dtype=torch.float32,
        device=device,
    )
    representative_sums.index_add_(0, inverse, point_values)
    representative_counts.index_add_(
        0,
        inverse,
        torch.ones(
            (inverse.shape[0], 1),
            dtype=torch.float32,
            device=device,
        ),
    )
    representative_points = representative_sums / representative_counts.clamp_min(1.0)

    representative_parents = unique_representatives.clone()
    if factor == 1:
        representative_parents[:, 1:] = torch.div(
            representative_parents[:, 1:],
            2,
            rounding_mode="floor",
        )
    else:
        representative_parents[:, 1:] = torch.div(
            representative_parents[:, 1:],
            factor,
            rounding_mode="floor",
        )

    if factor == 1:
        parent_rows = torch.cat((point_parents, candidate_coords.long()), dim=0)
        parent_minimum = parent_rows.min(dim=0).values
        parent_spans = parent_rows.max(dim=0).values - parent_minimum + 1
        point_parent_keys = row_keys(point_parents, parent_minimum, parent_spans)
        candidate_parent_keys = row_keys(
            candidate_coords.long(),
            parent_minimum,
            parent_spans,
        )
        sorted_candidate_keys, candidate_order = torch.sort(candidate_parent_keys)
        point_positions = torch.searchsorted(
            sorted_candidate_keys,
            point_parent_keys,
        )
        valid_points = point_positions < sorted_candidate_keys.numel()
        safe_point_positions = point_positions.clamp_max(
            max(sorted_candidate_keys.numel() - 1, 0)
        )
        valid_points &= (
            sorted_candidate_keys[safe_point_positions] == point_parent_keys
        )
        point_candidate_indices = candidate_order[safe_point_positions[valid_points]]
        point_child_ids = child_id_per_point[valid_points]
        occupancy = torch.zeros(
            (count,),
            dtype=torch.long,
            device=device,
        )
        for bit_id in range(8):
            bit_parents = point_candidate_indices[point_child_ids == bit_id]
            if bit_parents.numel() > 0:
                occupancy[bit_parents] |= 1 << bit_id
        occupancy = occupancy.view(-1, 1)

    all_rows = torch.cat((representative_parents, candidate_coords.long()), dim=0)
    minimum = all_rows.min(dim=0).values
    spans = all_rows.max(dim=0).values - minimum + 1
    representative_parent_keys = row_keys(representative_parents, minimum, spans)
    candidate_keys = row_keys(candidate_coords.long(), minimum, spans)
    sorted_keys, order = torch.sort(candidate_keys)
    positions = torch.searchsorted(sorted_keys, representative_parent_keys)
    valid = positions < sorted_keys.numel()
    safe_positions = positions.clamp_max(max(sorted_keys.numel() - 1, 0))
    valid &= sorted_keys[safe_positions] == representative_parent_keys

    parent_indices = order[safe_positions[valid]]
    points = representative_points[valid]
    sort_order = torch.argsort(parent_indices)
    parent_indices = parent_indices[sort_order]
    points = points[sort_order]
    counts = torch.bincount(parent_indices, minlength=count)
    offsets = torch.cumsum(counts, dim=0) - counts
    ranks = torch.arange(
        parent_indices.shape[0],
        device=device,
    ) - torch.repeat_interleave(offsets, counts)
    if ranks.numel() and int(ranks.max().item()) >= max_points:
        raise RuntimeError("Deferred parent contains too many target representatives")

    gt_points = torch.zeros(
        (count, max_points, 3),
        dtype=torch.float32,
        device=device,
    )
    gt_point_mask = torch.zeros(
        (count, max_points),
        dtype=torch.bool,
        device=device,
    )
    gt_points[parent_indices, ranks] = points
    gt_point_mask[parent_indices, ranks] = True

    if occupancy is None:
        raise ValueError("TreeOffset occupancy target is required")
    child_offset_mask = occupancy_bytes_to_bits(occupancy).bool()

    child_factor = float(factor) * 0.5
    child_center = (child_factor - 1.0) * 0.5
    child_offset_targets = torch.full(
        (count, 8, 3),
        child_center,
        dtype=torch.float32,
        device=device,
    )

    active_parent, child_id = torch.nonzero(
        child_offset_mask,
        as_tuple=True,
    )
    if active_parent.numel() > 0:
        child_bits = torch.stack(
            (
                child_id & 1,
                (child_id >> 1) & 1,
                (child_id >> 2) & 1,
            ),
            dim=1,
        ).to(torch.float32)
        child_bases = (
            candidate_coords[active_parent, 1:].float() * float(factor)
            + child_bits * float(child_factor)
        )

        if factor == 1:
            child_cells = candidate_coords[active_parent].clone()
            child_cells[:, 1:] = (
                child_cells[:, 1:] * 2
                + child_bits.to(child_cells.dtype)
            )
            representative_child_cells = unique_representatives
        else:
            child_cells = candidate_coords[active_parent].clone()
            child_cells[:, 1:] = (
                child_cells[:, 1:] * 2
                + child_bits.to(child_cells.dtype)
            )
            representative_child_cells = unique_representatives.clone()
            representative_child_cells[:, 1:] = torch.div(
                representative_child_cells[:, 1:],
                int(child_factor),
                rounding_mode="floor",
            )

        unique_child_cells, child_inverse = torch.unique(
            representative_child_cells,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        child_sums = torch.zeros(
            (unique_child_cells.shape[0], 3),
            dtype=torch.float32,
            device=device,
        )
        child_counts = torch.zeros(
            (unique_child_cells.shape[0], 1),
            dtype=torch.float32,
            device=device,
        )
        child_sums.index_add_(0, child_inverse, representative_points)
        child_counts.index_add_(
            0,
            child_inverse,
            torch.ones(
                (child_inverse.shape[0], 1),
                dtype=torch.float32,
                device=device,
            ),
        )
        child_representatives = child_sums / child_counts.clamp_min(1.0)

        all_child_rows = torch.cat(
            (unique_child_cells, child_cells.long()),
            dim=0,
        )
        child_minimum = all_child_rows.min(dim=0).values
        child_spans = all_child_rows.max(dim=0).values - child_minimum + 1
        unique_keys = row_keys(
            unique_child_cells,
            child_minimum,
            child_spans,
        )
        child_keys = row_keys(
            child_cells.long(),
            child_minimum,
            child_spans,
        )
        sorted_keys, order = torch.sort(unique_keys)
        positions = torch.searchsorted(sorted_keys, child_keys)
        valid = (
            (positions < sorted_keys.numel())
            & (
                sorted_keys[
                    positions.clamp_max(sorted_keys.numel() - 1)
                ]
                == child_keys
            )
        )
        if not bool(valid.all().item()):
            missing = int((~valid).sum().item())
            raise RuntimeError(
                f"{missing} active tree children lack GT offset targets"
            )
        child_offset_targets[active_parent, child_id] = (
            child_representatives[order[positions]] - child_bases
        )

    lower, upper = offset_bounds(
        factor,
        factor * (2 ** int(skipped_depths)),
        "continuous",
    )
    direct_offsets = (
        gt_points - candidate_coords[:, None, 1:].float() * float(factor)
    )
    if bool((gt_point_mask & ((direct_offsets < lower - 1e-4) | (direct_offsets > upper + 1e-4)).any(dim=2)).any().item()):
        gt_points = torch.where(
            gt_point_mask[..., None],
            (
                candidate_coords[:, None, 1:].float() * float(factor)
                + direct_offsets.clamp(lower, upper)
            ),
            gt_points,
        )

    return {
        "gt_points": gt_points,
        "gt_point_mask": gt_point_mask,
        "child_offset_targets": child_offset_targets,
        "child_offset_mask": child_offset_mask,
        "occupancy": occupancy.long(),
    }


def deferred_tree_items(
    net,
    x,
    target_coords,
    continuous_coords,
    skipped_depths,
    near_radius,
    far_radius,
    max_deferred,
    offset_only=False,
):
    data_ls = net.build_occupancy_pyramid(x, skipped_depths=skipped_depths)
    encoded_depths = max(len(data_ls) - 1, 0)
    if encoded_depths < 2:
        return []

    items = []
    propagated_feats = None
    propagated_core_idx = None
    lineage_feats = None
    x_C, x_O = data_ls[0]
    for depth in range(encoded_depths):
        raw_gt_C, raw_gt_O = data_ls[depth + 1]
        gt_deferred_C = None
        gt_deferred_O = None
        factor = None

        if depth == encoded_depths - 1:
            gt_C, gt_deferred_C, gt_O, gt_deferred_O = op.separation(
                raw_gt_C,
                raw_gt_O,
                R=far_radius,
            )
            factor = 2
        elif depth == encoded_depths - 2:
            gt_C, gt_deferred_C, gt_O, gt_deferred_O = op.separation(
                raw_gt_C,
                raw_gt_O,
                R=near_radius / 2,
            )
            factor = 4
        else:
            gt_C, gt_O = raw_gt_C, raw_gt_O

        gt_C, gt_O = net.order_CF(gt_C, gt_O)
        gt_s0, gt_s1 = net.split_occupancy_two_stage(gt_O)
        core_idx = net.select_encoder_index(
            depth,
            gt_s0[:, 0],
            gt_s1[:, 0],
            occupancy=gt_O[:, 0],
        )
        net.set_active_core(core_idx)

        if net.uses_uniform_prior(depth):
            x_up_C = net.fcg(x_C, x_O)
            x_up_C, _ = Network.separate_coding_depth(
                x_up_C,
                x_up_C,
                depth=depth,
                encoded_depths=encoded_depths,
                near_radius=near_radius,
                far_radius=far_radius,
            )
            propagated_feats = None
            propagated_core_idx = None
            lineage_feats = None
        else:
            x_up_C, x_up_F, x_up_lineage = predict_child_candidates(
                net,
                x_C,
                x_O,
                depth,
                encoded_depths,
                propagated_feats,
                propagated_core_idx=propagated_core_idx,
                lineage_feats=lineage_feats,
                return_lineage=True,
            )
            target_stride = 2 ** (len(data_ls) + skipped_depths - depth - 1)
            x_up_all = score_child_features(
                net,
                x_up_C,
                x_up_F,
                depth,
                encoded_depths,
                skipped_depths,
                target_stride,
            )
            x_scored_C = x_up_all.coords
            x_scored_F = x_up_all.feats
            x_scored_lineage = x_up_lineage
            deferred_C = None
            deferred_F = None
            if depth == encoded_depths - 1:
                x_up_C, deferred_C, x_up_F, deferred_F = op.separation(
                    x_scored_C,
                    x_scored_F,
                    R=far_radius,
                )
                if x_scored_lineage is not None:
                    _, _, x_up_lineage, _ = op.separation(
                        x_scored_C,
                        x_scored_lineage,
                        R=far_radius,
                    )
            elif depth == encoded_depths - 2:
                x_up_C, deferred_C, x_up_F, deferred_F = op.separation(
                    x_scored_C,
                    x_scored_F,
                    R=near_radius / 2,
                )
                if x_scored_lineage is not None:
                    _, _, x_up_lineage, _ = op.separation(
                        x_scored_C,
                        x_scored_lineage,
                        R=near_radius / 2,
                    )
            else:
                x_up_C, x_up_F = x_scored_C, x_scored_F

            if (
                deferred_C is not None
                and deferred_C.shape[0] > 0
                and gt_deferred_C is not None
            ):
                deferred_F = deferred_F.detach()
                stride = 2 ** (
                    len(data_ls) + skipped_depths - depth - 1
                )
                if not offset_only:
                    occupancy = gather_occupancy_by_coords(
                        deferred_C,
                        gt_deferred_C,
                        gt_deferred_O,
                    )
                    deferred_C, deferred_F, occupancy = limit_candidates(
                        deferred_C,
                        deferred_F,
                        occupancy,
                        int(max_deferred),
                    )
                offset_targets = build_offset_targets(
                    target_coords,
                    deferred_C,
                    skipped_depths,
                    factor,
                    continuous_coords,
                )
                item = {
                    "coords": deferred_C,
                    "features": deferred_F,
                    "stride": int(stride),
                    "factor": int(factor),
                    "depth": int(depth),
                    "boe_idx": int(core_idx),
                    "offset_targets": offset_targets,
                    "dd": int(skipped_depths),
                }
                if not offset_only:
                    geometry_targets = build_deferred_geometry_targets(
                        target_coords,
                        deferred_C,
                        occupancy,
                        skipped_depths,
                        factor,
                        continuous_coords=continuous_coords,
                    )
                    item["occupancy"] = geometry_targets.pop("occupancy")
                    item.update(geometry_targets)
                items.append(item)

            x_up_C, x_up_F, x_up_lineage = net.order_CFs(
                x_up_C,
                x_up_F,
                x_up_lineage,
            )
            if depth == encoded_depths - 1 and x_up_C.shape[0] > 0:
                final_parent_occupancy = gather_occupancy_by_coords(
                    x_up_C,
                    gt_C,
                    gt_O,
                )
                final_C, final_F = net.fcg(x_up_C, final_parent_occupancy, x_up_F)
                if final_C.shape[0] > 0:
                    final_F = final_F.detach()
                    if not offset_only:
                        final_C, final_F, _ = limit_candidates(
                            final_C,
                            final_F,
                            torch.ones(
                                (final_C.shape[0], 1),
                                dtype=torch.long,
                                device=final_C.device,
                            ),
                            int(max_deferred),
                        )
                    offset_targets = build_offset_targets(
                        target_coords,
                        final_C,
                        skipped_depths,
                        1,
                        continuous_coords,
                    )
                    item = {
                        "coords": final_C,
                        "features": final_F,
                        "stride": int(2 ** int(skipped_depths)),
                        "factor": 1,
                        "depth": int(depth),
                        "boe_idx": int(core_idx),
                        "offset_targets": offset_targets,
                        "dd": int(skipped_depths),
                    }
                    if not offset_only:
                        geometry_targets = build_deferred_geometry_targets(
                            target_coords,
                            final_C,
                            None,
                            skipped_depths,
                            1,
                            continuous_coords=continuous_coords,
                        )
                        item["occupancy"] = geometry_targets.pop("occupancy")
                        item.update(geometry_targets)
                    items.append(item)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            propagated_feats = net.make_cross_depth_features(
                x_up_C,
                x_up.feats,
                gt_s0[:, 0],
                gt_s1[:, 0],
            )
            propagated_core_idx = (
                int(core_idx) if propagated_feats is not None else None
            )
            lineage_feats = net.update_lineage_memory(
                x_up.feats,
                gt_s0[:, 0],
                gt_s1[:, 0],
                x_up_lineage,
                depth,
            )

        x_C, x_O = x_up_C, gt_O
    return items


def spherical_grid_to_xyz(coords, spherical_params, dd):
    coords = coords.float()
    dd_scale = float(2 ** int(dd))
    rho_value = (
        coords[..., 0]
        * float(spherical_params.scale_rho)
        * dd_scale
    )
    rho = rho_value
    theta = (
        coords[..., 1]
        - float(spherical_params.theta_offset) / dd_scale
    ) * float(spherical_params.scale_theta) * dd_scale
    phi = (
        coords[..., 2]
        - float(spherical_params.phi_offset) / dd_scale
    ) * float(spherical_params.scale_phi) * dd_scale
    theta = torch.deg2rad(theta)
    phi = torch.deg2rad(phi)
    cos_phi = torch.cos(phi)
    return torch.stack(
        (
            rho * cos_phi * torch.cos(theta),
            rho * cos_phi * torch.sin(theta),
            rho * torch.sin(phi),
        ),
        dim=-1,
    )


def child_bases(parent_coords, factor, dtype):
    child_factor = float(int(factor)) * 0.5
    child_ids = torch.arange(8, device=parent_coords.device)
    bits = torch.stack(
        (
            child_ids & 1,
            (child_ids >> 1) & 1,
            (child_ids >> 2) & 1,
        ),
        dim=1,
    ).to(dtype)
    return (
        parent_coords[:, None, 1:].to(dtype) * float(factor)
        + bits[None, :, :] * float(child_factor)
    )


def predicted_tree_points(details, item):
    factor = int(item["factor"])
    bases = child_bases(item["coords"], factor, torch.float32)
    return bases + details["child_offsets"].float()


def _masked_mean(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def local_voxel_diagonal_squares(coords, factor, spherical_params, dd):
    factor = float(int(factor))
    base = coords[:, 1:].float() * factor
    lower = base - 0.5
    corner_bits = torch.tensor(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ),
        dtype=torch.float32,
        device=coords.device,
    )
    corners = lower[:, None, :] + corner_bits[None, :, :] * factor
    corner_xyz = spherical_grid_to_xyz(corners, spherical_params, dd)
    extent = corner_xyz.max(dim=1).values - corner_xyz.min(dim=1).values
    return extent.square().sum(dim=1).clamp_min(1e-8)


def tree_and_offset_unified_costs(
    details,
    offset_prediction,
    item,
    spherical_params,
    geo_clip,
    geometry_weight,
    count_weight,
    count_clip,
):
    dd = int(item["dd"])
    factor = int(item["factor"])
    predicted_mask = occupancy_bytes_to_bits(
        details["occupancy"]
    ).bool()
    target_child_mask = occupancy_bytes_to_bits(
        item["occupancy"]
    ).bool()
    target_grid = item["gt_points"]
    target_mask = item["gt_point_mask"]

    scale = local_voxel_diagonal_squares(
        item["coords"],
        factor,
        spherical_params,
        dd,
    )
    target_xyz = spherical_grid_to_xyz(target_grid, spherical_params, dd)

    tree_grid = predicted_tree_points(details, item)
    tree_xyz = spherical_grid_to_xyz(tree_grid, spherical_params, dd)
    pairwise_tree = torch.cdist(target_xyz.float(), tree_xyz.float()).square()
    large = torch.finfo(pairwise_tree.dtype).max / 16.0

    target_to_tree = pairwise_tree.masked_fill(
        ~predicted_mask[:, None, :],
        large,
    ).min(dim=2).values
    target_to_tree = _masked_mean(target_to_tree, target_mask)
    tree_to_target = pairwise_tree.transpose(1, 2).masked_fill(
        ~target_mask[:, None, :],
        large,
    ).min(dim=2).values
    tree_to_target = _masked_mean(tree_to_target, predicted_mask)
    tree_geometry = 0.5 * (target_to_tree + tree_to_target)

    offset_grid = (
        item["coords"][:, 1:].float() * float(factor)
        + offset_prediction.float()
    )
    offset_xyz = spherical_grid_to_xyz(offset_grid, spherical_params, dd)
    offset_distances = (
        target_xyz - offset_xyz[:, None, :]
    ).square().sum(dim=2)
    target_to_offset = _masked_mean(offset_distances, target_mask)
    offset_to_target = offset_distances.masked_fill(
        ~target_mask,
        large,
    ).min(dim=1).values
    offset_geometry = 0.5 * (target_to_offset + offset_to_target)

    geo_limit = float(geo_clip)
    tree_geometry = torch.nan_to_num(
        tree_geometry / scale,
        nan=geo_limit,
        posinf=geo_limit,
        neginf=0.0,
    ).clamp(0.0, geo_limit)
    offset_geometry = torch.nan_to_num(
        offset_geometry / scale,
        nan=geo_limit,
        posinf=geo_limit,
        neginf=0.0,
    ).clamp(0.0, geo_limit)

    target_count = target_child_mask.sum(dim=1).float().clamp_min(1.0)
    predicted_count = predicted_mask.sum(dim=1).float()
    offset_count = torch.ones_like(target_count)
    count_limit = float(count_clip)
    tree_count_error = (
        (predicted_count - target_count).abs() / target_count
    ).clamp(0.0, count_limit)
    offset_count_error = (
        (offset_count - target_count).abs() / target_count
    ).clamp(0.0, count_limit)
    tree_structure = float(count_weight) * tree_count_error
    offset_structure = float(count_weight) * offset_count_error

    false_positive = (predicted_mask & ~target_child_mask).sum(dim=1).float()
    false_negative = (~predicted_mask & target_child_mask).sum(dim=1).float()

    tree_geometry_cost = float(geometry_weight) * tree_geometry
    offset_geometry_cost = float(geometry_weight) * offset_geometry
    return {
        "tree_cost": tree_geometry_cost + tree_structure,
        "offset_cost": offset_geometry_cost + offset_structure,
        "tree_geometry": tree_geometry,
        "offset_geometry": offset_geometry,
        "tree_structure": tree_structure,
        "offset_structure": offset_structure,
        "tree_count_error": tree_count_error,
        "offset_count_error": offset_count_error,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "target_count": target_count,
        "predicted_count": predicted_count,
        "scale": scale,
    }
