from pathlib import Path
import tempfile
import unittest

from cluster_mlip.active_learning import (
    committee_force_disagreement,
    rank_candidates_by_disagreement,
    write_next_batch,
)
from cluster_mlip.analysis import scan_source
from cluster_mlip.dataset import read_labeled_extxyz, write_labeled_extxyz
from cluster_mlip.doctor import MISSING_OPTIONAL, OK, run_checks
from cluster_mlip.evaluate import summarize_evaluation
from cluster_mlip.label_report import summarize_labels
from cluster_mlip.manifest import write_experiment_manifest
from cluster_mlip.models import Atom, LabeledFrame, Record
from cluster_mlip.spin import validate_fragment_specification_shape

FIXTURES = Path(__file__).parent / "fixtures"


def _frame(record_id: str, charge: int, multiplicity: int, energy: float, force_x: float) -> LabeledFrame:
    record = Record(
        record_id, "src.log", [Atom("Fe", 0, 0, 0), Atom("O", 1.5, 0, 0)], charge, multiplicity, "minimum"
    )
    return LabeledFrame(record, energy, [(force_x, 0.0, 0.0), (-force_x, 0.0, 0.0)], Path("src.log"))


def _uniform_frame(record_id: str, charge: int, multiplicity: int, energy: float, force: float) -> LabeledFrame:
    """All three Cartesian components equal, so a uniform prediction offset
    gives an exact, easy-to-hand-check force MAE (no zero components to
    dilute the average)."""
    record = Record(
        record_id, "src.log", [Atom("Fe", 0, 0, 0), Atom("O", 1.5, 0, 0)], charge, multiplicity, "minimum"
    )
    return LabeledFrame(record, energy, [(force, force, force), (-force, -force, -force)], Path("src.log"))


class DoctorTests(unittest.TestCase):
    def test_run_checks_reports_python_ok(self):
        checks = run_checks()
        by_name = {check.name: check for check in checks}
        self.assertEqual(by_name["python"].status, OK)
        # These external tools are not guaranteed to exist in the test
        # environment; the check must degrade to a warning, never crash.
        self.assertIn(by_name["mace-torch"].status, (OK, MISSING_OPTIONAL))


class ManifestTests(unittest.TestCase):
    def test_manifest_hashes_dataset_files_and_records_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "train.extxyz").write_text("hello", encoding="utf-8")
            output = Path(tmp) / "manifest.json"
            manifest = write_experiment_manifest(dataset_dir, output, notes="unit test")
            self.assertEqual(manifest["notes"], "unit test")
            self.assertIn("train.extxyz", manifest["dataset_files"])
            self.assertIsNotNone(manifest["dataset_files"]["train.extxyz"]["sha256"])
            self.assertNotIn("valid.extxyz", manifest["dataset_files"])
            self.assertTrue(output.exists())


class LabelReportTests(unittest.TestCase):
    def test_summarize_labels_groups_by_charge_multiplicity_and_flags_outliers(self):
        frames = [
            _frame("a", 0, 1, -10.0, 0.1),
            _frame("b", 0, 1, -11.0, 0.2),
            _frame("c", 0, 3, -9.0, 50.0),  # blown-up force -> outlier
        ]
        summary = summarize_labels(frames, force_outlier_threshold_ev_ang=5.0)
        self.assertEqual(summary["n_frames"], 3)
        self.assertEqual(summary["n_groups"], 2)
        self.assertEqual(len(summary["outliers"]), 1)
        self.assertEqual(summary["outliers"][0]["record_id"], "c")

    def test_write_label_report_roundtrips_through_dataset_io(self):
        frames = [_frame("a", 0, 1, -10.0, 0.1)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.extxyz"
            write_labeled_extxyz(frames, path)
            loaded = read_labeled_extxyz(path)
        self.assertEqual(len(loaded), 1)
        self.assertAlmostEqual(loaded[0].energy_ev, -10.0)
        self.assertEqual(loaded[0].record.charge, 0)
        self.assertEqual(loaded[0].record.multiplicity, 1)
        self.assertAlmostEqual(loaded[0].forces_ev_ang[0][0], 0.1, places=6)


class EvaluateTests(unittest.TestCase):
    def test_summarize_evaluation_computes_known_errors(self):
        frames = [_uniform_frame("a", 0, 1, -10.0, 1.0), _uniform_frame("b", 0, 2, -20.0, 1.0)]
        # Predictions off by exactly 1 eV total energy (0.5 eV/atom, 2 atoms)
        # and 0.5 eV/Angstrom on every force component.
        predictions = [
            (-9.0, [(1.5, 1.5, 1.5), (-1.5, -1.5, -1.5)]),
            (-21.0, [(1.5, 1.5, 1.5), (-1.5, -1.5, -1.5)]),
        ]
        summary = summarize_evaluation(frames, predictions)
        self.assertEqual(summary["n_frames"], 2)
        self.assertAlmostEqual(summary["overall"]["energy_mae_ev_per_atom"], 0.5, places=6)
        self.assertAlmostEqual(summary["overall"]["force_mae_ev_ang"], 0.5, places=6)
        self.assertEqual(len(summary["by_charge_multiplicity"]), 2)

    def test_summarize_evaluation_rejects_length_mismatch(self):
        frames = [_frame("a", 0, 1, -10.0, 1.0)]
        with self.assertRaises(ValueError):
            summarize_evaluation(frames, [])


class ActiveLearningTests(unittest.TestCase):
    def test_committee_force_disagreement_is_zero_for_agreeing_models(self):
        forces = [[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]]
        self.assertAlmostEqual(committee_force_disagreement(forces), 0.0)

    def test_committee_force_disagreement_flags_the_worst_atom(self):
        # Atom 0 agrees across the committee; atom 1 does not.
        forces = [[(1.0, 0.0, 0.0), (0.0, 0.0, 0.0)], [(1.0, 0.0, 0.0), (0.0, 10.0, 0.0)]]
        self.assertGreater(committee_force_disagreement(forces), 0.0)

    def test_rank_candidates_orders_by_descending_disagreement(self):
        low = Record("low", "s", [Atom("Fe", 0, 0, 0)], 0, 1, "minimum")
        high = Record("high", "s", [Atom("Fe", 0, 0, 0)], 0, 1, "minimum")
        committee_forces = [
            [[(1.0, 0.0, 0.0)], [(1.0, 0.0, 0.0)]],  # model 1: low, high
            [[(1.0, 0.0, 0.0)], [(9.0, 0.0, 0.0)]],  # model 2: low agrees, high disagrees
        ]
        ranked = rank_candidates_by_disagreement([low, high], committee_forces)
        self.assertEqual([record.record_id for record, _ in ranked], ["high", "low"])

    def test_write_next_batch_writes_extxyz_and_selection_csv(self):
        candidates = [Record(f"c{i}", "s", [Atom("Fe", 0, 0, 0)], 0, 1, "minimum") for i in range(3)]
        committee_forces = [
            [[(float(i), 0.0, 0.0)] for i in range(3)],
            [[(float(i) + 1.0, 0.0, 0.0)] for i in range(3)],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "next_batch"
            selected = write_next_batch(candidates, committee_forces, output, top_k=2)
            self.assertEqual(len(selected), 2)
            self.assertTrue((output / "next_batch.extxyz").exists())
            self.assertTrue((output / "selection.csv").exists())


class ParallelScanTests(unittest.TestCase):
    def test_scan_source_with_jobs_matches_sequential(self):
        sequential_files, sequential_records = scan_source(FIXTURES, jobs=1)
        parallel_files, parallel_records = scan_source(FIXTURES, jobs=2)
        self.assertEqual(len(sequential_files), len(parallel_files))
        self.assertEqual(
            sorted(f.source for f in sequential_files), sorted(f.source for f in parallel_files)
        )
        self.assertEqual(
            sorted(r.record_id for r in sequential_records), sorted(r.record_id for r in parallel_records)
        )


class FragmentSpecShapeTests(unittest.TestCase):
    def test_valid_example_spec_has_no_shape_errors(self):
        import json

        payload = json.loads((Path(__file__).parent.parent / "examples" / "spin_fragments.example.json").read_text())
        self.assertEqual(validate_fragment_specification_shape(payload["guesses"]), [])

    def test_missing_and_wrong_typed_fields_are_all_reported_together(self):
        bad = [
            {
                "name": "bad",
                "target_multiplicity": "not-an-int",
                "fragments": [{"atoms": [1], "charge": 0}],
            }
        ]
        errors = validate_fragment_specification_shape(bad)
        joined = "\n".join(errors)
        self.assertIn("missing required key 'record_id'", joined)
        self.assertIn("'target_multiplicity' must be a int", joined)
        self.assertIn("at least two fragments", joined)


if __name__ == "__main__":
    unittest.main()
