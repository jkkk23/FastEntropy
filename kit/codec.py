import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torchac
from torchsparse import SparseTensor
from network import Network
import kit.op as op
import kit.spherical as spherical
from kit.checkpoint import load_model_state_dict, load_state_dict, resolve_model_config
from kit.tree_refiner import tree_child_coords
_BOE_STREAM_MAGIC = b'BOE1'

def _read_param(source, name, default=None):
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)

def apply_spherical_params(net, params):
    if params is None:
        return
    radius_transform = _read_param(params, 'radius_transform')
    use_log_radius = _read_param(params, 'use_log_radius')
    if use_log_radius is None and radius_transform is not None:
        use_log_radius = radius_transform == 'log'
    net.set_spherical_params(sph_alpha=_read_param(params, 'sph_alpha'), scale_rho=_read_param(params, 'scale_rho'), scale_theta=_read_param(params, 'scale_theta'), scale_phi=_read_param(params, 'scale_phi'), theta_offset=_read_param(params, 'theta_offset'), phi_offset=_read_param(params, 'phi_offset'), use_log_radius=use_log_radius)

def load_network(ckpt_path, device, spherical_params=None):
    (state_dict, checkpoint) = load_state_dict(ckpt_path, map_location='cpu')
    model_config = resolve_model_config(state_dict, checkpoint)
    net = Network(**model_config)
    apply_spherical_params(net, spherical_params)
    load_model_state_dict(net, state_dict)
    net.to(device).eval()
    return (net, checkpoint)

def warmup_network(net, device, num_points=4096, coord_max=4096):
    if str(device).startswith('cuda'):
        random_coords = torch.randint(low=0, high=coord_max, size=(num_points, 3), dtype=torch.int32, device=device)
        warmup_coords = torch.cat((random_coords[:, 0:1] * 0, random_coords), dim=-1).contiguous()
        warmup_feats = torch.ones((num_points, 1), dtype=torch.float32, device=device)
        net(SparseTensor(coords=warmup_coords, feats=warmup_feats))
        torch.cuda.synchronize()

def quantize_points(points, args, deduplicate=None):
    if deduplicate is None:
        deduplicate = True
    return spherical.quantize_points(points, sph_alpha=args.sph_alpha, scale_rho=args.scale_rho, scale_theta=args.scale_theta, scale_phi=args.scale_phi, use_log_radius=args.radius_transform == 'log', deduplicate=deduplicate)

def make_sparse_input(coords_int, device):
    coords_int = np.asarray(coords_int, dtype=np.int32)
    xyz_tensor = torch.from_numpy(coords_int).to(device=device, dtype=torch.int32)
    batch_column = torch.zeros((xyz_tensor.shape[0], 1), dtype=torch.int32, device=device)
    coords = torch.cat((batch_column, xyz_tensor), dim=-1).contiguous()
    feats = torch.ones((coords.shape[0], 1), dtype=torch.float32, device=device)
    return SparseTensor(coords=coords, feats=feats)

def search_radii_in_bins(near_distance, far_distance, scale_rho, skipped_depths):
    scale = 2 ** skipped_depths
    return (near_distance / scale_rho / scale, far_distance / scale_rho / scale)

def build_occupancy_pyramid(net, x, min_points=64, skipped_depths=0):
    return net.build_occupancy_pyramid(x, min_points=min_points, skipped_depths=skipped_depths)

def _predict_child_candidates(net, x_C, x_O, depth, encoded_depths=None, propagated_feats=None, propagated_core_idx=None, lineage_feats=None, return_lineage=False):
    x_F = net.core().prior_embedding(x_O.int()).view(-1, net.channels)
    x_F = net.fuse_cross_depth_parent(x_F, propagated_feats, depth, encoded_depths if encoded_depths is not None else depth + 1, source_core_idx=propagated_core_idx, target_core_idx=getattr(net, '_active_core_idx', None))
    x_F = net.fuse_lineage_parent(x_F, lineage_feats)
    x = net.run_prior_resnet(SparseTensor(coords=x_C, feats=x_F), depth)
    (x_up_C, x_up_F, x_up_parent_O, x_up_child_id, x_up_parent_idx) = net.fcg(x_C, x_O, x.feats, return_info=True, return_parent_idx=True)
    x_up_lineage = net.expand_lineage_to_children(lineage_feats, x_up_parent_idx, x_up_child_id, x_up_F.dtype, x_up_F.device)
    x_up_F = net.fuse_child_features(x_up_F, x_up_parent_O, x_up_child_id, x_up_lineage)
    (x_up_C, x_up_F, x_up_lineage) = net.order_CFs(x_up_C, x_up_F, x_up_lineage)
    if return_lineage:
        return (x_up_C, x_up_F, x_up_lineage)
    return (x_up_C, x_up_F)

def _score_child_features(net, x_up_C, x_up_F, depth, encoded_depths, skipped_depths=0, target_stride=1):
    x_up_F = net.target_embedding(x_up_F, x_up_C, target_stride=target_stride)
    x_up_F = net.apply_coarse_grid_prior(x_up_C, x_up_F, depth, target_stride)
    x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
    return net.run_target_resnet(x_up, depth=depth, encoded_depths=encoded_depths, skipped_depths=skipped_depths)

def _prob_to_cdf_norm(prob):
    cdf = prob.new_empty((prob.shape[0], prob.shape[1] + 1))
    cdf[:, 0].zero_()
    torch.cumsum(prob, dim=-1, out=cdf[:, 1:])
    cdf.clamp_(min=0, max=1)
    return op._convert_to_int_and_normalize(cdf, True)

def _prob_pair_to_cdf_norm(prob0, prob1):
    n0 = prob0.shape[0]
    cdf_cat = _prob_to_cdf_norm(torch.cat((prob0, prob1), dim=0))
    return (cdf_cat[:n0], cdf_cat[n0:])

def _entropy_encode(cdf, symbols):
    return torchac.encode_int16_normalized_cdf(cdf.cpu(), symbols.cpu())

def _entropy_decode(cdf, byte_stream, device):
    return torchac.decode_int16_normalized_cdf(cdf.cpu(), byte_stream).to(device)

def _pack_boe_stream(byte_stream, enc_idx_ls):
    enc_idx = np.asarray(enc_idx_ls, dtype=np.uint8)
    return _BOE_STREAM_MAGIC + np.array(enc_idx.shape[0], dtype=np.uint16).tobytes() + enc_idx.tobytes() + byte_stream

def _unpack_boe_stream(byte_stream):
    if not byte_stream.startswith(_BOE_STREAM_MAGIC):
        raise ValueError("Invalid FastEntropy expert-routing stream")
    cursor = len(_BOE_STREAM_MAGIC)
    count = int(np.frombuffer(byte_stream[cursor:cursor + 2], dtype=np.uint16)[0])
    cursor += 2
    enc_idx = np.frombuffer(byte_stream[cursor:cursor + count], dtype=np.uint8).astype(np.int64).tolist()
    cursor += count
    return (byte_stream[cursor:], enc_idx)

def _decode_deferred_coords(net, deferred_coords, deferred_feats, deferred_stride, factor, depth, expert_idx, tree_refiner_model, tree_gate_threshold):
    factor = int(factor)
    parent_base = deferred_coords[:, 1:].float() * float(factor)
    details = None
    if deferred_feats is not None:
        with torch.cuda.amp.autocast(enabled=False):
            details = tree_refiner_model.forward_details(deferred_coords, deferred_feats.float(), deferred_stride, factor, net.target_embedding, depth=depth, expert_idx=expert_idx)
        fallback_offsets = details['offsets']
    else:
        fallback_offsets = torch.full_like(parent_base, (float(factor) - 1.0) * 0.5)
    fallback_coords = parent_base + fallback_offsets
    if deferred_feats is None:
        return fallback_coords.cpu().numpy().astype(np.float64)
    if details is None:
        raise ValueError('The geometry-refinement model produced no branch details')
    tree_occupancy_all = details['occupancy']
    use_tree = torch.sigmoid(details['gate_logits'].float()) >= float(tree_gate_threshold)
    if not bool(use_tree.any().item()):
        return fallback_coords.cpu().numpy().astype(np.float64)
    tree_occupancy = tree_occupancy_all[use_tree]
    child_offsets = details.get('child_offsets')
    tree_coords = tree_child_coords(net, deferred_coords[use_tree], tree_occupancy, factor, child_offsets=child_offsets[use_tree] if child_offsets is not None else None)
    if bool((~use_tree).any().item()):
        coords = torch.cat((fallback_coords[~use_tree], tree_coords), dim=0)
    else:
        coords = tree_coords
    return coords.cpu().numpy().astype(np.float64)

def _encode_pyramid_fast(net, data_ls, near_radius, far_radius, skipped_depths=0):
    encoded_depths = max(len(data_ls) - 1, 0)
    device = next(net.parameters()).device
    use_async_entropy = device.type == 'cuda' and encoded_depths > 1
    entropy_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='torchac-encode') if use_async_entropy else None
    entropy_results = []
    propagated_feats = None
    propagated_core_idx = None
    lineage_feats = None
    enc_idx_ls = []
    (x_C, x_O) = data_ls[0]
    try:
        for depth in range(encoded_depths):
            (gt_x_up_C, gt_x_up_O) = data_ls[depth + 1]
            (gt_x_up_C, gt_x_up_O) = Network.separate_coding_depth(gt_x_up_C, gt_x_up_O, depth=depth, encoded_depths=encoded_depths, near_radius=near_radius, far_radius=far_radius)
            (gt_x_up_C, gt_x_up_O) = net.order_CF(gt_x_up_C, gt_x_up_O)
            (gt_x_up_O_s0, gt_x_up_O_s1) = net.split_occupancy_two_stage(gt_x_up_O)
            core_idx = net.select_encoder_index(depth, gt_x_up_O_s0[:, 0], gt_x_up_O_s1[:, 0], occupancy=gt_x_up_O[:, 0])
            net.set_active_core(core_idx)
            enc_idx_ls.append(core_idx)
            if net.uses_uniform_prior(depth):
                x_up_C = net.fcg(x_C, x_O)
                (x_up_C, _) = Network.separate_coding_depth(x_up_C, x_up_C, depth=depth, encoded_depths=encoded_depths, near_radius=near_radius, far_radius=far_radius)
                x_up = None
                propagated_feats = None
                propagated_core_idx = None
                lineage_feats = None
            else:
                (x_up_C, x_up_F, x_up_lineage) = _predict_child_candidates(net, x_C, x_O, depth=depth, encoded_depths=encoded_depths, propagated_feats=propagated_feats, propagated_core_idx=propagated_core_idx, lineage_feats=lineage_feats, return_lineage=True)
                target_stride = 2 ** (len(data_ls) + skipped_depths - depth - 1)
                x_up_all = _score_child_features(net, x_up_C, x_up_F, depth=depth, encoded_depths=encoded_depths, skipped_depths=skipped_depths, target_stride=target_stride)
                x_scored_C = x_up_all.coords
                x_scored_F = x_up_all.feats
                x_scored_lineage = x_up_lineage
                if near_radius is not None and far_radius is not None:
                    if depth == encoded_depths - 1:
                        (x_up_C, _, x_up_F, _) = op.separation(x_scored_C, x_scored_F, R=far_radius)
                        if x_scored_lineage is not None:
                            (_, _, x_up_lineage, _) = op.separation(x_scored_C, x_scored_lineage, R=far_radius)
                    elif depth == encoded_depths - 2:
                        (x_up_C, _, x_up_F, _) = op.separation(x_scored_C, x_scored_F, R=near_radius / 2)
                        if x_scored_lineage is not None:
                            (_, _, x_up_lineage, _) = op.separation(x_scored_C, x_scored_lineage, R=near_radius / 2)
                    else:
                        (x_up_C, x_up_F) = (x_scored_C, x_scored_F)
                else:
                    (x_up_C, x_up_F) = (x_scored_C, x_scored_F)
                (x_up_C, x_up_F, x_up_lineage) = net.order_CFs(x_up_C, x_up_F, x_up_lineage)
                x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            if net.uses_uniform_prior(depth):
                (x_up_O_prob_s0, x_up_O_prob_s1) = net.coarse_prior_two_stage(depth, x_up_C.shape[0], gt_x_up_O_s0[:, 0], torch.float32)
            elif net.uses_joint_head(x_up.feats.shape[0]):
                (x_up_O_prob_s0, x_up_O_prob_s1) = net.predict_joint_two_stage(x_up.feats, gt_x_up_O_s0[:, 0])
            else:
                (x_up_feats, x_up_O_prob_s0) = net.predict_s0(x_up.feats)
                x_up_O_prob_s1 = net.predict_s1_from_s0(x_up_feats, x_up_O_prob_s0, gt_x_up_O_s0[:, 0])
            if not net.uses_uniform_prior(depth):
                propagated_feats = net.make_cross_depth_features(x_up_C, x_up.feats, gt_x_up_O_s0[:, 0], gt_x_up_O_s1[:, 0])
                propagated_core_idx = int(core_idx) if propagated_feats is not None else None
                lineage_feats = net.update_lineage_memory(x_up.feats, gt_x_up_O_s0[:, 0], gt_x_up_O_s1[:, 0], x_up_lineage, depth)
            (x_up_O_cdf_s0_norm, x_up_O_cdf_s1_norm) = _prob_pair_to_cdf_norm(x_up_O_prob_s0, x_up_O_prob_s1)
            symbols_s0 = gt_x_up_O_s0[:, 0].to(torch.int16)
            symbols_s1 = gt_x_up_O_s1[:, 0].to(torch.int16)
            if entropy_executor is None:
                entropy_results.append((_entropy_encode(x_up_O_cdf_s0_norm, symbols_s0), _entropy_encode(x_up_O_cdf_s1_norm, symbols_s1)))
            else:
                cdf_s0_cpu = x_up_O_cdf_s0_norm.cpu()
                cdf_s1_cpu = x_up_O_cdf_s1_norm.cpu()
                symbols_s0_cpu = symbols_s0.cpu()
                symbols_s1_cpu = symbols_s1.cpu()
                entropy_results.append((entropy_executor.submit(torchac.encode_int16_normalized_cdf, cdf_s0_cpu, symbols_s0_cpu), entropy_executor.submit(torchac.encode_int16_normalized_cdf, cdf_s1_cpu, symbols_s1_cpu)))
            (x_C, x_O) = (x_up_C, gt_x_up_O)
    finally:
        if entropy_executor is not None:
            entropy_executor.shutdown(wait=True)
    if use_async_entropy:
        entropy_results = [(future_s0.result(), future_s1.result()) for (future_s0, future_s1) in entropy_results]
    byte_stream_s0_ls = [s[0] for s in entropy_results]
    byte_stream_s1_ls = [s[1] for s in entropy_results]
    byte_stream_ls = [s for pair in entropy_results for s in pair]
    return (_pack_boe_stream(op.pack_byte_stream_ls(byte_stream_ls), enc_idx_ls), op.pack_byte_stream_ls(byte_stream_s0_ls), op.pack_byte_stream_ls(byte_stream_s1_ls))

def encode_pyramid(net, data_ls, near_radius, far_radius, skipped_depths=0):
    return _encode_pyramid_fast(net, data_ls, near_radius, far_radius, skipped_depths)

def write_compressed_file(file_path, metadata, base_coords, base_feats, byte_stream):
    base_len = base_coords.shape[0]
    base_coords = base_coords[:, 1:].cpu().numpy()
    base_feats = base_feats.cpu().numpy()
    with open(file_path, 'wb') as f:
        spherical.write_header(f, metadata)
        f.write(np.array(base_len, dtype=np.int32).tobytes())
        f.write(np.array(base_coords, dtype=np.int32).tobytes())
        f.write(np.array(base_feats, dtype=np.uint8).tobytes())
        f.write(byte_stream)

def read_compressed_file(file_path):
    with open(file_path, 'rb') as f:
        metadata = spherical.read_header(f)
        base_len = np.frombuffer(f.read(4), dtype=np.int32)[0]
        base_coords = np.frombuffer(f.read(base_len * 4 * 3), dtype=np.int32).reshape(-1, 3)
        base_feats = np.frombuffer(f.read(base_len), dtype=np.uint8).reshape(-1, 1)
        byte_stream = f.read()
    return (metadata, base_coords, base_feats, byte_stream)

def decode_streams(net, base_coords, base_feats, byte_stream, near_radius, far_radius, device, skipped_depths=0, tree_refiner_model=None, tree_gate_threshold=0.5):
    if tree_refiner_model is None:
        raise ValueError('The submitted decoder requires zero-bit geometry refinement')
    base_coords = torch.tensor(base_coords, dtype=torch.int32, device=device)
    base_feats = torch.tensor(base_feats, dtype=torch.uint8, device=device)
    batch_column = torch.zeros((base_coords.shape[0], 1), dtype=torch.int32, device=device)
    x = SparseTensor(coords=torch.cat((batch_column, base_coords), dim=-1).int(), feats=base_feats).to(device)
    (byte_stream, enc_idx_ls) = _unpack_boe_stream(byte_stream)
    byte_stream_ls = op.unpack_byte_stream(byte_stream)
    deferred_far_coords = None
    deferred_near_coords = None
    deferred_far_feats = None
    deferred_near_feats = None
    deferred_far_stride = None
    deferred_near_stride = None
    deferred_far_depth = None
    deferred_near_depth = None
    deferred_far_expert_idx = None
    deferred_near_expert_idx = None
    final_refine_coords = None
    final_refine_feats = None
    final_refine_stride = None
    final_refine_depth = None
    final_refine_expert_idx = None
    streams_per_depth = 2
    if len(byte_stream_ls) % streams_per_depth != 0:
        raise ValueError(
            f"Invalid two-stage entropy stream count: {len(byte_stream_ls)}"
        )
    total_depths = len(byte_stream_ls) // streams_per_depth
    if len(enc_idx_ls) != total_depths:
        raise ValueError(f'BoE encoder-index count {len(enc_idx_ls)} does not match stream depth count {total_depths}')
    full_pyramid_depths = total_depths + skipped_depths + 1
    propagated_feats = None
    propagated_core_idx = None
    lineage_feats = None
    for depth in range(total_depths):
        core_idx = enc_idx_ls[depth]
        net.set_active_core(core_idx)
        if net.uses_uniform_prior(depth):
            x_up_C = net.fcg(x.coords, x.feats.int())
            x_up_F = None
            propagated_feats = None
            propagated_core_idx = None
            lineage_feats = None
        else:
            (x_up_C, x_up_F, x_up_lineage) = _predict_child_candidates(net, x.coords, x.feats.int(), depth=depth, encoded_depths=total_depths, propagated_feats=propagated_feats, propagated_core_idx=propagated_core_idx, lineage_feats=lineage_feats, return_lineage=True)
        if net.uses_uniform_prior(depth):
            separation_feats = x_up_C
            if depth == total_depths - 1:
                (x_up_C, deferred_far_coords, separation_feats, _) = op.separation(x_up_C, separation_feats, R=far_radius)
                deferred_far_stride = 2 ** (full_pyramid_depths - depth - 1)
                deferred_far_depth = depth
                deferred_far_expert_idx = enc_idx_ls[depth]
                deferred_far_feats = None
            elif depth == total_depths - 2:
                (x_up_C, deferred_near_coords, separation_feats, _) = op.separation(x_up_C, separation_feats, R=near_radius / 2)
                deferred_near_stride = 2 ** (full_pyramid_depths - depth - 1)
                deferred_near_depth = depth
                deferred_near_expert_idx = enc_idx_ls[depth]
                deferred_near_feats = None
            x_up = None
        else:
            target_stride = 2 ** (full_pyramid_depths - depth - 1)
            x_up_all = _score_child_features(net, x_up_C, x_up_F, depth=depth, encoded_depths=total_depths, skipped_depths=skipped_depths, target_stride=target_stride)
            x_scored_C = x_up_all.coords
            x_scored_F = x_up_all.feats
            x_scored_lineage = x_up_lineage
            if depth == total_depths - 1:
                (x_up_C, deferred_far_coords, x_up_F, deferred_far_feats) = op.separation(x_scored_C, x_scored_F, R=far_radius)
                if x_scored_lineage is not None:
                    (_, _, x_up_lineage, _) = op.separation(x_scored_C, x_scored_lineage, R=far_radius)
                deferred_far_stride = 2 ** (full_pyramid_depths - depth - 1)
                deferred_far_depth = depth
                deferred_far_expert_idx = enc_idx_ls[depth]
            elif depth == total_depths - 2:
                (x_up_C, deferred_near_coords, x_up_F, deferred_near_feats) = op.separation(x_scored_C, x_scored_F, R=near_radius / 2)
                if x_scored_lineage is not None:
                    (_, _, x_up_lineage, _) = op.separation(x_scored_C, x_scored_lineage, R=near_radius / 2)
                deferred_near_stride = 2 ** (full_pyramid_depths - depth - 1)
                deferred_near_depth = depth
                deferred_near_expert_idx = enc_idx_ls[depth]
            else:
                (x_up_C, x_up_F) = (x_scored_C, x_scored_F)
            (x_up_C, x_up_F, x_up_lineage) = net.order_CFs(x_up_C, x_up_F, x_up_lineage)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
        byte_stream_s0 = byte_stream_ls[2 * depth]
        byte_stream_s1 = byte_stream_ls[2 * depth + 1]
        if net.uses_uniform_prior(depth):
            joint = net.coarse_prior_distribution(depth, x_up_C.shape[0], torch.float32).view(-1, 16, 16)
            x_up_O_prob_s0 = joint.sum(dim=1)
            x_up_feats = None
        elif net.uses_joint_head(x_up.feats.shape[0]):
            joint = net.predict_joint_distribution(x_up.feats)
            x_up_O_prob_s0 = joint.sum(dim=1)
            x_up_feats = None
        else:
            (x_up_feats, x_up_O_prob_s0) = net.predict_s0(x_up.feats)
            joint = None
        x_up_O_cdf_s0_norm = _prob_to_cdf_norm(x_up_O_prob_s0)
        x_up_O_s0 = _entropy_decode(x_up_O_cdf_s0_norm, byte_stream_s0, device)
        if joint is not None:
            x_up_O_prob_s1 = net.condition_joint_on_s0(joint, x_up_O_prob_s0, x_up_O_s0)
        elif not net.uses_uniform_prior(depth):
            x_up_O_prob_s1 = net.predict_s1_from_s0(x_up_feats, x_up_O_prob_s0, x_up_O_s0)
        x_up_O_cdf_s1_norm = _prob_to_cdf_norm(x_up_O_prob_s1)
        x_up_O_s1 = _entropy_decode(x_up_O_cdf_s1_norm, byte_stream_s1, device)
        x_up_O = net.combine_occupancy_two_stage(x_up_O_s0, x_up_O_s1)
        if not net.uses_uniform_prior(depth):
            propagated_feats = net.make_cross_depth_features(x_up_C, x_up.feats, x_up_O_s0, x_up_O_s1)
            propagated_core_idx = int(core_idx) if propagated_feats is not None else None
            lineage_feats = net.update_lineage_memory(x_up.feats, x_up_O_s0, x_up_O_s1, x_up_lineage, depth)
        if depth == total_depths - 1 and (not net.uses_uniform_prior(depth)) and tree_refiner_model.supports_factor(1) and (x_up_C.shape[0] > 0):
            (final_refine_coords, final_refine_feats) = net.fcg(x_up_C, x_up_O.unsqueeze(-1), x_up.feats)
            final_refine_stride = 2 ** int(skipped_depths)
            final_refine_depth = depth
            final_refine_expert_idx = int(core_idx)
        x = SparseTensor(coords=x_up_C, feats=x_up_O.unsqueeze(-1)).to(device)
    if final_refine_coords is not None and final_refine_coords.shape[0] > 0:
        decoded_coords = _decode_deferred_coords(net, final_refine_coords, final_refine_feats, final_refine_stride, 1, final_refine_depth, final_refine_expert_idx, tree_refiner_model, tree_gate_threshold)
    else:
        decoded_coords = net.fcg(x.coords, x.feats)[:, 1:].cpu().numpy().astype(np.float64)
    coord_parts = [decoded_coords]
    if deferred_far_coords is not None:
        coord_parts.append(_decode_deferred_coords(net, deferred_far_coords, deferred_far_feats, deferred_far_stride, 2, deferred_far_depth, deferred_far_expert_idx, tree_refiner_model, tree_gate_threshold))
    if deferred_near_coords is not None:
        coord_parts.append(_decode_deferred_coords(net, deferred_near_coords, deferred_near_feats, deferred_near_stride, 4, deferred_near_depth, deferred_near_expert_idx, tree_refiner_model, tree_gate_threshold))
    return np.concatenate(coord_parts, axis=0)

def compressed_output_path(output_folder, input_path):
    return os.path.join(output_folder, os.path.splitext(os.path.basename(input_path))[0] + '.bin')

def decompressed_output_path(output_folder, input_path):
    return os.path.join(output_folder, os.path.splitext(os.path.basename(input_path))[0] + '.ply')
