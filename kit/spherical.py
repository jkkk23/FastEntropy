import math
import json
import struct

import numpy as np


HEADER_MAGIC = b"RNSP"
JSON_HEADER_MAGIC = b"RNSJ"
HEADER_VERSION = 1
FLAG_LOG_RADIUS = 1
HEADER_STRUCT = struct.Struct("<4sBB2x4f2i")
JSON_HEADER_STRUCT = struct.Struct("<4sBI")


def unique_coords(coords):
    """Deduplicate integer XYZ rows while preserving lexicographic order."""
    if coords.shape[0] < 2:
        return coords

    coords64 = coords.astype(np.int64, copy=False)
    minimum = coords64.min(axis=0)
    shifted = coords64 - minimum
    spans = shifted.max(axis=0) + 1

    max_int64 = np.iinfo(np.int64).max
    if spans[1] > max_int64 // spans[2]:
        return np.unique(coords, axis=0)
    tail_span = int(spans[1] * spans[2])
    if spans[0] > max_int64 // tail_span:
        return np.unique(coords, axis=0)

    keys = (shifted[:, 0] * spans[1] + shifted[:, 1]) * spans[2]
    keys += shifted[:, 2]
    _, first_indices = np.unique(keys, return_index=True)
    return coords[first_indices]


def get_angle_offsets(scale_theta, scale_phi, margin=1):
    theta_offset = int(math.ceil(180.0 / scale_theta)) + margin
    phi_offset = int(math.ceil(90.0 / scale_phi))   + margin
    return theta_offset, phi_offset


def quantize_points(
    points,
    sph_alpha,
    scale_rho,
    scale_theta,
    scale_phi,
    use_log_radius=True,
    theta_offset=None,
    phi_offset=None,
    deduplicate=True,
):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    if theta_offset is None or phi_offset is None:
        theta_offset, phi_offset = get_angle_offsets(scale_theta, scale_phi)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    rho = np.linalg.norm(points, axis=1)
    theta = np.degrees(np.arctan2(y, x))
    phi = np.degrees(np.arcsin(np.clip(z / np.maximum(rho, 1e-8), -1.0, 1.0)))

    if use_log_radius:
        if sph_alpha <= 0:
            raise ValueError("sph_alpha must be positive when use_log_radius=True")
        rho_value = np.log1p(sph_alpha * rho)
    else:
        rho_value = rho

    rho_int = np.rint(rho_value / scale_rho).astype(np.int32)
    theta_int = np.rint(theta / scale_theta).astype(np.int32) + theta_offset
    phi_int = np.rint(phi / scale_phi).astype(np.int32) + phi_offset
    coords = np.stack([rho_int, theta_int, phi_int], axis=1)

    if deduplicate:
        coords = unique_coords(coords)

    metadata = {
        "sph_alpha": float(sph_alpha),
        "scale_rho": float(scale_rho),
        "scale_theta": float(scale_theta),
        "scale_phi": float(scale_phi),
        "theta_offset": int(theta_offset),
        "phi_offset": int(phi_offset),
        "use_log_radius": bool(use_log_radius),
    }
    return coords, metadata

def dequantize_points(coords, metadata,dd=0):
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")

    sph_alpha = float(metadata["sph_alpha"])
    scale_rho = float(metadata["scale_rho"])*(2**dd)
    scale_theta = float(metadata["scale_theta"])*(2**dd)
    scale_phi = float(metadata["scale_phi"])*(2**dd)    
    theta_offset = int(metadata["theta_offset"])
    phi_offset = int(metadata["phi_offset"])
    use_log_radius = bool(metadata.get("use_log_radius", False))

    rho_value = coords[:, 0] * scale_rho
    if use_log_radius:
        if sph_alpha <= 0:
            raise ValueError("sph_alpha must be positive when use_log_radius=True")
        rho = np.expm1(rho_value) / sph_alpha
    else:
        rho = rho_value
    theta_deg = (coords[:, 1] - theta_offset/(2**dd)) * scale_theta
    phi_deg = (coords[:, 2] - phi_offset/(2**dd)) * scale_phi
    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi_deg)
    cos_phi = np.cos(phi_rad)

    x = rho * cos_phi * np.cos(theta_rad)
    y = rho * cos_phi * np.sin(theta_rad)
    z = rho * np.sin(phi_rad)

    return np.stack([x, y, z], axis=1).astype(np.float32)


def write_header(file_obj, metadata):
    if _needs_json_header(metadata):
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        file_obj.write(JSON_HEADER_STRUCT.pack(JSON_HEADER_MAGIC, HEADER_VERSION, len(payload)))
        file_obj.write(payload)
        return

    flags = FLAG_LOG_RADIUS if metadata.get("use_log_radius", False) else 0
    payload = HEADER_STRUCT.pack(
        HEADER_MAGIC,
        HEADER_VERSION,
        flags,
        float(metadata["sph_alpha"]),
        float(metadata["scale_rho"]),
        float(metadata["scale_theta"]),
        float(metadata["scale_phi"]),
        int(metadata["theta_offset"]),
        int(metadata["phi_offset"])
    )
    file_obj.write(payload)


def read_header(file_obj):
    start = file_obj.tell()
    prefix = file_obj.read(4)
    file_obj.seek(start)

    if prefix == HEADER_MAGIC:
        raw = file_obj.read(HEADER_STRUCT.size)
        magic, version, flags, sph_alpha, scale_rho, scale_theta, scale_phi, theta_offset, phi_offset = HEADER_STRUCT.unpack(raw)
        if magic != HEADER_MAGIC or version != HEADER_VERSION:
            raise ValueError("Unsupported spherical header version")
        return {
            "sph_alpha": float(sph_alpha),
            "scale_rho": float(scale_rho),
            "scale_theta": float(scale_theta),
            "scale_phi": float(scale_phi),
            "theta_offset": int(theta_offset),
            "phi_offset": int(phi_offset),
            "use_log_radius": bool(flags & FLAG_LOG_RADIUS),
        }

    if prefix == JSON_HEADER_MAGIC:
        raw = file_obj.read(JSON_HEADER_STRUCT.size)
        magic, version, payload_len = JSON_HEADER_STRUCT.unpack(raw)
        if magic != JSON_HEADER_MAGIC or version != HEADER_VERSION:
            raise ValueError("Unsupported JSON spherical header version")
        payload = file_obj.read(payload_len)
        return json.loads(payload.decode("utf-8"))

    raise ValueError("Unsupported spherical header magic")


def _needs_json_header(metadata):
    binary_header_keys = {
        "sph_alpha",
        "scale_rho",
        "scale_theta",
        "scale_phi",
        "theta_offset",
        "phi_offset",
        "use_log_radius",
    }
    return any(key not in binary_header_keys for key in metadata)
