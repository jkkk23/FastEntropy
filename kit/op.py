import os
import torch
import numpy as np
_CDF_ARANGE_CACHE = {}

def morton_sort_indices(C):
    """Sort batched 3D integer coordinates by batch and Morton code."""
    coords = C.long()
    xyz = coords[:, 1:]
    if bool((xyz < 0).any().item()):
        raise ValueError("Morton hierarchy requires non-negative quantized coordinates")
    max_bits = max(int(xyz.max().item()).bit_length(), 1)
    code = torch.zeros(coords.shape[0], dtype=torch.int64, device=C.device)
    for bit in range(max_bits):
        code |= ((xyz[:, 0] >> bit) & 1) << (3 * bit)
        code |= ((xyz[:, 1] >> bit) & 1) << (3 * bit + 1)
        code |= ((xyz[:, 2] >> bit) & 1) << (3 * bit + 2)

    batch = coords[:, 0] - coords[:, 0].min()
    batch_span = code.max().clamp_min(0) + 1
    return torch.argsort(batch * batch_span + code)


def morton_sort_C(C):
    return C[morton_sort_indices(C)]


def morton_unique_sort_C(C):
    """Sort coordinates by Morton order and drop duplicate leaf voxels."""
    coords = morton_sort_C(C)
    if coords.shape[0] < 2:
        return coords
    keep = torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)
    keep[1:] = torch.any(coords[1:] != coords[:-1], dim=1)
    return coords[keep]


def morton_downsample(C):
    """Build one parent level and its 8-bit occupancies from Morton-ordered children."""
    child_id = (
        (C[:, 1].long() & 1)
        + ((C[:, 2].long() & 1) << 1)
        + ((C[:, 3].long() & 1) << 2)
    )
    parents = C.clone()
    parents[:, 1:] = torch.div(parents[:, 1:], 2, rounding_mode="floor")

    unique_parents, inverse = torch.unique_consecutive(
        parents,
        dim=0,
        return_inverse=True,
    )
    occupancy = torch.zeros(
        unique_parents.shape[0],
        dtype=torch.int64,
        device=C.device,
    )
    occupancy.index_add_(0, inverse, 1 << child_id)
    return unique_parents.int(), occupancy.view(-1, 1).float()


def pack_byte_stream_ls(byte_stream_ls):
    stream_parts = [np.array(len(byte_stream_ls), dtype=np.uint16).tobytes()]
    for byte_stream in byte_stream_ls:
        stream_parts.append(np.array(len(byte_stream), dtype=np.uint32).tobytes())
        stream_parts.append(byte_stream)
    return b"".join(stream_parts)

def unpack_byte_stream(stream):
    len_bytes_stream_ls = np.frombuffer(stream[:2], dtype=np.uint16)[0]
    byte_stream_ls = []
    cursor = 2
    for idx in range(len_bytes_stream_ls):
        len_bytes_stream = np.frombuffer(stream[cursor:cursor+4], dtype=np.uint32)[0]
        byte_stream = stream[cursor+4:cursor+4+len_bytes_stream]
        byte_stream_ls.append(byte_stream)
        cursor = cursor + 4 + len_bytes_stream

    return byte_stream_ls

def _cached_cdf_arange(Lp, device):
    key = (int(Lp), str(device))
    r = _CDF_ARANGE_CACHE.get(key, None)
    if r is None:
        r = torch.arange(Lp, dtype=torch.int16, device=device)
        _CDF_ARANGE_CACHE[key] = r
    return r


def _convert_to_int_and_normalize(cdf_float, needs_normalization):
    r"""
    Faster int16 CDF normalization.
    Mathematically equivalent to the original implementation while avoiding
    repeated construction of factor and arange tensors.
    """
    Lp = cdf_float.shape[-1]

    if cdf_float.dtype != torch.float32:
        cdf_float = cdf_float.float()

    if needs_normalization:
        cdf_float = cdf_float.mul(float((1 << 16) - (Lp - 1))).round_()
    else:
        cdf_float = cdf_float.mul(float(1 << 16)).round_()

    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)

    if needs_normalization:
        cdf.add_(_cached_cdf_arange(Lp, cdf.device))

    return cdf

def get_file_size_in_bits(file_path):
    
    return os.stat(file_path).st_size * 8

def separation(x_C, X_O, R):
    r"""
    Split x_C=(R, theta, phi) into the two radial regions x<R and x>=R.
    x_C is an (N, 3) tensor containing spherical coordinates.
    """
#    print("X_C.shape:{} X_O.shape:{}".format(x_C.shape, X_O.shape))
    mask = x_C[:, 1] > R
    x_C_part1 = x_C[mask]
    x_C_part2 = x_C[~mask]
    X_O_part1 = X_O[mask]
    X_O_part2 = X_O[~mask]
#    print("X_C_part1:=\n,R:", x_C_part1, R)
#    print("x_C_part1.shape: {} x_C_part2.shape: {} X_O_part1.shape: {} X_O_part2.shape: {}".format(x_C_part1.shape, x_C_part2.shape, X_O_part1.shape, X_O_part2.shape))
    return x_C_part1, x_C_part2, X_O_part1, X_O_part2
   
