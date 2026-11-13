"""Single publication configuration for FastEntropy."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    extension: str
    scale_rho: float
    scale_theta: float
    scale_phi: float
    near_distance: float
    far_distance: float
    train_dd: tuple[int, ...]
    unit_scale_to_meters: float = 1.0

    sph_alpha: float = 1.0
    radius_transform: str = "linear"


DATASETS = {
    "semantickitti": DatasetConfig(
        name="semantickitti",
        extension=".bin",
        scale_rho=200 / (2**16 - 1),
        scale_theta=0.02747 / 2,
        scale_phi=0.01373 / 2,
        near_distance=10.0,
        far_distance=20.0,
        train_dd=(0, 1, 2, 3, 4, 5),
    ),
    "ford": DatasetConfig(
        name="ford",
        extension=".ply",
        scale_rho=0.001,
        scale_theta=360.0 / (120000 - 1),
        scale_phi=(360.0 / (120000 - 1)) / 2,
        near_distance=15.0,
        far_distance=30.0,
        train_dd=(1, 2, 3, 4, 5, 6),
        unit_scale_to_meters=0.001,
    ),
}


# Every architectural choice except C and K is fixed to the best submitted path.
OPTIMAL_ARCHITECTURE = {
    "prior_res_blocks": 1,
    "target_res_blocks": 1,
    "bit_permutation": "phi_first",
    "uniform_prior_depths": 1,
    "coarse_k1_depths": 3,
    "coarse_grid_depths": 3,
    "coarse_grid_theta_bins": 32,
    "coarse_grid_phi_bins": 16,
    "joint_distribution_max_points": 512,
    "boe_k": 12,
    "boe_count_groups": 3,
    "boe_leaves_per_group": 4,
}


@dataclass(frozen=True)
class ModelBundle:
    root: Path
    entropy_model: Path
    postprocess: Path


def dataset_config(name: str) -> DatasetConfig:
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"dataset must be one of {tuple(DATASETS)}") from exc


def resolve_model_bundle(path: str | Path) -> ModelBundle:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Model bundle directory does not exist: {root}")
    entropy_model = root / "entropy_model.pt"
    postprocess = root / "postprocess.pt"
    missing = [str(item) for item in (entropy_model, postprocess) if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            "Model bundle must contain entropy_model.pt and postprocess.pt; "
            f"missing: {missing}"
        )
    return ModelBundle(root, entropy_model, postprocess)


def training_output_dir(dataset: str) -> Path:
    return PROJECT_ROOT / "checkpoints" / dataset


def postprocess_training_output_dir(dataset: str) -> Path:
    return PROJECT_ROOT / "checkpoints" / "postprocess" / dataset
