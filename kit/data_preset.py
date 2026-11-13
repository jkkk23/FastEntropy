"""SemanticKITTI/Ford input handling shared by the release entry points."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import dataset_config


@dataclass
class PointCloudSample:
    points: np.ndarray
    metadata: dict


def find_point_clouds(input_folder, dataset, require_unique_stems=False):
    folder = Path(input_folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    extension = dataset_config(dataset).extension
    files = sorted(path for path in folder.rglob(f"*{extension}") if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No {extension} point clouds found under input folder: {folder}"
        )
    if require_unique_stems:
        _check_unique_stems(files)
    return files


def find_bitstreams(input_folder):
    folder = Path(input_folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    files = sorted(path for path in folder.rglob("*.bin") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .bin bitstreams found under input folder: {folder}")
    _check_unique_stems(files)
    return files


def _check_unique_stems(files):
    seen = {}
    for path in files:
        previous = seen.setdefault(path.stem, path)
        if previous != path:
            raise ValueError(
                "Input files would overwrite the same output name: "
                f"{previous} and {path}"
            )


def load_points(path, dataset):
    config = dataset_config(dataset)
    if dataset == "semantickitti":
        points_native = _read_semantickitti_bin(path)
        return PointCloudSample(
            points=points_native.astype(np.float32, copy=False),
            metadata={},
        )

    points_native = _read_ply_xyz(path)
    return PointCloudSample(
        points=(points_native * config.unit_scale_to_meters).astype(
            np.float32,
            copy=False,
        ),
        metadata={
            "data_preset": "ford",
            "native_unit": "millimeter",
            "unit_scale_to_meters": config.unit_scale_to_meters,
        },
    )


def merge_metadata(spherical_metadata, sample):
    if not sample.metadata:
        return spherical_metadata
    metadata = dict(spherical_metadata)
    metadata.update(sample.metadata)
    return metadata


def restore_points(points_model, metadata):
    if not isinstance(metadata, dict) or metadata.get("data_preset") != "ford":
        return points_model
    scale = float(metadata["unit_scale_to_meters"])
    return (np.asarray(points_model, dtype=np.float32) / scale).astype(np.float32)


def _read_semantickitti_bin(path):
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        raise ValueError(
            f"SemanticKITTI file must contain float32 Nx4 values: {path}"
        )
    return values.reshape(-1, 4)[:, :3]


def _read_ply_xyz(path):
    with open(path, "rb") as handle:
        fmt = None
        count = None
        properties = []
        reading_vertices = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY missing end_header: {path}")
            text = line.decode("ascii", "strict").strip()
            if text.startswith("format "):
                fmt = text.split()[1]
            elif text.startswith("element vertex "):
                count = int(text.split()[-1])
                reading_vertices = True
            elif text.startswith("element "):
                reading_vertices = False
            elif reading_vertices and text.startswith("property "):
                fields = text.split()
                if len(fields) != 3:
                    raise ValueError(f"List vertex properties are unsupported: {path}")
                properties.append((fields[1], fields[2]))
            elif text == "end_header":
                break

        if count is None:
            raise ValueError(f"PLY missing vertex count: {path}")
        names = [name for _, name in properties]
        try:
            xyz_columns = tuple(names.index(axis) for axis in ("x", "y", "z"))
        except ValueError as exc:
            raise ValueError(f"PLY must contain x/y/z vertex properties: {path}") from exc

        if fmt == "ascii":
            array = np.loadtxt(
                handle,
                dtype=np.float32,
                max_rows=count,
                usecols=xyz_columns,
            )
            if array.ndim == 1:
                array = array.reshape(1, 3)
            if array.shape[0] != count:
                raise ValueError(
                    f"PLY declares {count} vertices but contains {array.shape[0]}: {path}"
                )
            return array

        if fmt == "binary_little_endian":
            dtype = np.dtype(
                [
                    (name, _ply_numpy_dtype(type_name))
                    for type_name, name in properties
                ]
            )
            array = np.fromfile(handle, dtype=dtype, count=count)
            if array.shape[0] != count:
                raise ValueError(
                    f"PLY declares {count} vertices but contains {array.shape[0]}: {path}"
                )
            return np.stack(
                (array["x"], array["y"], array["z"]),
                axis=1,
            ).astype(np.float32)

    raise ValueError(f"Unsupported PLY format: {fmt}")


def _ply_numpy_dtype(type_name):
    mapping = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "<i2",
        "int16": "<i2",
        "ushort": "<u2",
        "uint16": "<u2",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    try:
        return mapping[type_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported PLY property type: {type_name}") from exc
