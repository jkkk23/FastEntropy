import ast
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import OPTIMAL_ARCHITECTURE, dataset_config, resolve_model_bundle
from kit.data_preset import find_point_clouds


def public_options(script):
    tree = ast.parse((ROOT / script).read_text(encoding="utf-8"))
    options = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and str(argument.value).startswith("--"):
                options.add(argument.value)
    return options


class ReleaseTest(unittest.TestCase):
    def test_public_cli_is_minimal(self):
        codec_options = {
            "--input_folder",
            "--output_folder",
            "--model",
            "--dd",
            "--dataset",
        }
        self.assertEqual(public_options("compress.py"), codec_options)
        self.assertEqual(public_options("decompress.py"), codec_options)
        self.assertEqual(
            public_options("train_postprocess.py"),
            {"--input_folder", "--model", "--dataset"},
        )
        self.assertEqual(
            public_options("train.py"),
            {
                "--input_folder",
                "--channels",
                "--kernel_size",
                "--learning_rate",
                "--lr_decay",
                "--lr_decay_steps",
                "--max_steps",
                "--dataset",
            },
        )

    def test_postprocess_training_schedule_is_fixed(self):
        tree = ast.parse(
            (ROOT / "train_postprocess.py").read_text(encoding="utf-8")
        )
        constants = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            if isinstance(node.targets[0], ast.Name):
                try:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        self.assertEqual(constants["FALLBACK_STEPS"], 50000)
        self.assertEqual(constants["TREE_STEPS"], 80000)
        self.assertEqual(constants["GATE_STEPS"], 30000)
        self.assertEqual(constants["JOINT_STEPS"], 40000)
        self.assertEqual(constants["JOINT_WARMUP_STEPS"], 5000)
        self.assertEqual(constants["GATE_THRESHOLD"], 0.5)

    def test_dataset_quality_levels(self):
        self.assertEqual(
            dataset_config("semantickitti").train_dd,
            (0, 1, 2, 3, 4, 5),
        )
        self.assertEqual(dataset_config("ford").train_dd, (1, 2, 3, 4, 5, 6))

    def test_training_allows_repeated_frame_names(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for sequence in ("00", "01"):
                folder = root / sequence
                folder.mkdir()
                (folder / "000000.bin").touch()
            self.assertEqual(len(find_point_clouds(root, "semantickitti")), 2)
            with self.assertRaises(ValueError):
                find_point_clouds(
                    root,
                    "semantickitti",
                    require_unique_stems=True,
                )

    def test_model_bundles_are_the_submitted_branch(self):
        for dataset in ("semantickitti", "ford"):
            bundle = resolve_model_bundle(ROOT / "models" / dataset)
            entropy = torch.load(
                bundle.entropy_model,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                {
                    key: entropy["model_config"][key]
                    for key in OPTIMAL_ARCHITECTURE
                },
                OPTIMAL_ARCHITECTURE,
            )
            state_dict = entropy["model_state_dict"]
            self.assertFalse(
                any(key.startswith("boe_cores.12.") for key in state_dict)
            )
            self.assertTrue(
                any(
                    key.startswith("group_boundary_feature_alignment.")
                    for key in state_dict
                )
            )
            postprocess = torch.load(
                bundle.postprocess,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                postprocess["tree_config"]["model_type"],
                "boe_tree_offset_v2",
            )
            self.assertEqual(
                postprocess["tree_mode"],
                "occupancy_8bit_fallback_ready",
            )
            self.assertEqual(postprocess["deployment_gate_threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
