from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.dataset import write_labeled_extxyz
from cluster_mlip.models import Atom, LabeledFrame, Record
from cluster_mlip.training import TrainingConfig, scan_dataset, write_training_campaign


def _frame(
    record_id: str,
    charge: int = 0,
    multiplicity: int = 1,
    *,
    link1_route: str = "#p UBPW91/Gen Force",
) -> LabeledFrame:
    record = Record(
        record_id,
        "src.log",
        [Atom("Fe", 0.0, 0.0, 0.0), Atom("O", 1.5, 0.0, 0.0)],
        charge,
        multiplicity,
        "minimum",
        metadata={"parent_record_id": record_id, "link1_route": link1_route},
    )
    return LabeledFrame(record, -10.0, [(0.1, 0.0, 0.0), (-0.1, 0.0, 0.0)], Path("src.log"))


def _dataset(tmp: Path, frames: list[LabeledFrame]) -> Path:
    dataset = tmp / "dataset"
    dataset.mkdir()
    write_labeled_extxyz(frames, dataset / "all.extxyz")
    write_labeled_extxyz(frames, dataset / "train.extxyz")
    write_labeled_extxyz(frames[:1], dataset / "valid.extxyz")
    write_labeled_extxyz(frames[:1], dataset / "test.extxyz")
    return dataset


class ScanDatasetTests(unittest.TestCase):
    def test_collects_ranges_and_routes(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(
                tmp,
                [_frame("a", -1, 6), _frame("b", 0, 1), _frame("c", 2, 11)],
            )
            facts = scan_dataset(dataset)
            self.assertEqual(min(facts.charges), -1)
            self.assertEqual(max(facts.multiplicities), 11)
            self.assertEqual(facts.label_routes, {"#p UBPW91/Gen Force"})

    def test_missing_split_is_reported(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = tmp / "dataset"
            dataset.mkdir()
            write_labeled_extxyz([_frame("a")], dataset / "train.extxyz")
            with self.assertRaises(FileNotFoundError):
                scan_dataset(dataset)


class ScratchCampaignTests(unittest.TestCase):
    def test_scratch_run_has_locked_gaussian_args_and_embedding(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(tmp, [_frame("a", 0, 1), _frame("b", 0, 5)])
            output = tmp / "models" / "run"
            plan = write_training_campaign(
                TrainingConfig(dataset_dir=dataset, output_dir=output, seeds=(11, 23))
            )
            self.assertEqual(plan["model"], "ScaleShiftMACE")
            self.assertEqual(len(plan["seed_runs"]), 2)
            argv = plan["seed_runs"][0]["argv"]
            self.assertIn("--default_dtype=float64", argv)
            self.assertIn("--stress_weight=0", argv)
            self.assertIn("--energy_key=REF_energy", argv)
            self.assertIn("--use_embedding_readout", argv)
            self.assertTrue(any(a.startswith("--embedding_specs=") for a in argv))
            self.assertTrue(any(a.startswith("--seed=11") for a in argv))
            self.assertTrue((output / "seed_11" / "run.sh").is_file())
            self.assertTrue((output / "run_all_seeds.sh").is_file())
            manifest = json.loads((output / "train_manifest.json").read_text())
            self.assertEqual(manifest["seeds"], [11, 23])
            # The embedding_specs JSON contains {}" and must be single-quoted
            # in the rendered script so the shell does not mangle it.
            script_text = (output / "seed_11" / "run.sh").read_text()
            self.assertIn("--embedding_specs='{", script_text)
            self.assertIn("--hidden_irreps='128x0e + 128x1o + 128x2e'", script_text)

    def test_multiplicity_outside_embedding_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(tmp, [_frame("a", 0, 250)])
            with self.assertRaises(ValueError):
                write_training_campaign(
                    TrainingConfig(dataset_dir=dataset, output_dir=tmp / "out")
                )

    def test_mixed_label_routes_are_refused_without_override(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(
                tmp,
                [
                    _frame("a", link1_route="#p UBPW91/Gen Force"),
                    _frame("b", link1_route="#p wB97XD/Gen Force"),
                ],
            )
            with self.assertRaises(ValueError):
                write_training_campaign(
                    TrainingConfig(dataset_dir=dataset, output_dir=tmp / "out")
                )
            plan = write_training_campaign(
                TrainingConfig(
                    dataset_dir=dataset, output_dir=tmp / "out2", allow_mixed_method=True
                )
            )
            self.assertEqual(len(plan["label_routes"]), 2)

    def test_refuses_nonempty_output_without_force(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(tmp, [_frame("a")])
            output = tmp / "out"
            output.mkdir()
            (output / "stale").write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_training_campaign(
                    TrainingConfig(dataset_dir=dataset, output_dir=output)
                )


class FinetuneCampaignTests(unittest.TestCase):
    def test_polar_foundation_uses_polarmace_and_no_custom_embedding(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(tmp, [_frame("a", 0, 1), _frame("b", -1, 2)])
            output = tmp / "ft"
            plan = write_training_campaign(
                TrainingConfig(
                    dataset_dir=dataset,
                    output_dir=output,
                    mode="finetune",
                    foundation_model="polar-1-m",
                )
            )
            self.assertEqual(plan["model"], "PolarMACE")
            argv = plan["seed_runs"][0]["argv"]
            self.assertIn("--foundation_model=polar-1-m", argv)
            self.assertIn("--total_charge_key=charge", argv)
            self.assertFalse(any(a.startswith("--embedding_specs=") for a in argv))
            self.assertTrue((output / "PREFLIGHT.md").is_file())

    def test_generic_foundation_keeps_custom_embedding(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(tmp, [_frame("a", 0, 1)])
            plan = write_training_campaign(
                TrainingConfig(
                    dataset_dir=dataset,
                    output_dir=tmp / "ft",
                    mode="finetune",
                    foundation_model="medium",
                )
            )
            argv = plan["seed_runs"][0]["argv"]
            self.assertEqual(plan["model"], "ScaleShiftMACE")
            self.assertIn("--foundation_model=medium", argv)
            self.assertTrue(any(a.startswith("--embedding_specs=") for a in argv))
            self.assertIn("--amsgrad", argv)


if __name__ == "__main__":
    unittest.main()
