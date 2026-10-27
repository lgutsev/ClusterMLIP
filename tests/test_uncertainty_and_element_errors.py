import json
from pathlib import Path
import tempfile
import unittest

from cluster_mlip.active_learning import (
    committee_error_calibration,
    committee_mean_force_error,
    spearman_rank_correlation,
    write_committee_calibration,
)
from cluster_mlip.evaluate import element_force_errors, summarize_evaluation, write_evaluation_report
from cluster_mlip.models import Atom, LabeledFrame, Record


def _fe_n_frame(record_id: str, fe_force: float, n_force: float) -> LabeledFrame:
    record = Record(
        record_id, "src.log", [Atom("Fe", 0, 0, 0), Atom("N", 1.8, 0, 0)], 0, 3, "minimum"
    )
    return LabeledFrame(record, -10.0, [(fe_force, 0.0, 0.0), (n_force, 0.0, 0.0)], Path("src.log"))


def _single_atom_frame(record_id: str, force_x: float) -> LabeledFrame:
    record = Record(record_id, "src.log", [Atom("Fe", 0, 0, 0)], 0, 5, "minimum")
    return LabeledFrame(record, -1.0, [(force_x, 0.0, 0.0)], Path("src.log"))


class ElementForceErrorTests(unittest.TestCase):
    def test_minority_element_worse_than_zero_force_is_visible(self):
        # Fe is predicted exactly; N is off by 0.3 eV/Ang on x although its true
        # force is only 0.1 eV/Ang, so N is worse than predicting zero.
        frame = _fe_n_frame("a", 1.0, 0.1)
        prediction = (-10.0, [(1.0, 0.0, 0.0), (0.4, 0.0, 0.0)])
        rows = {row["element"]: row for row in element_force_errors([(frame, prediction)])}
        self.assertEqual(set(rows), {"Fe", "N"})
        self.assertEqual(rows["N"]["n_atoms"], 1)
        self.assertAlmostEqual(rows["Fe"]["force_rmse_ev_ang"], 0.0)
        self.assertAlmostEqual(rows["N"]["force_mae_ev_ang"], 0.1)
        self.assertAlmostEqual(rows["N"]["force_rmse_ev_ang"], (0.09 / 3) ** 0.5)
        self.assertAlmostEqual(rows["N"]["zero_force_rmse_ev_ang"], (0.01 / 3) ** 0.5)
        self.assertGreater(rows["N"]["force_rmse_ev_ang"], rows["N"]["zero_force_rmse_ev_ang"])

    def test_summary_and_report_carry_the_element_table(self):
        frame = _fe_n_frame("a", 1.0, 0.1)
        prediction = (-10.0, [(1.0, 0.0, 0.0), (0.4, 0.0, 0.0)])
        self.assertEqual(len(summarize_evaluation([frame], [prediction])["by_element"]), 2)
        with tempfile.TemporaryDirectory() as tmp:
            write_evaluation_report([frame], [prediction], Path(tmp))
            self.assertIn("zero_force_rmse_ev_ang", (Path(tmp) / "by_element.csv").read_text())
            self.assertIn("## Forces by element", (Path(tmp) / "report.md").read_text())

    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            element_force_errors([(_fe_n_frame("a", 1.0, 0.1), (-10.0, [(0.0, 0.0, 0.0)]))])


class CommitteeCalibrationTests(unittest.TestCase):
    def test_mean_force_error_uses_the_committee_mean(self):
        members = [[(1.0, 0.0, 0.0)], [(3.0, 0.0, 0.0)]]
        self.assertAlmostEqual(committee_mean_force_error(members, [(0.0, 0.0, 0.0)]), 2.0)

    def test_spearman_handles_ties_and_degenerate_input(self):
        self.assertAlmostEqual(spearman_rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman_rank_correlation([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(spearman_rank_correlation([1, 2], [1, 2]))
        self.assertIsNone(spearman_rank_correlation([1, 1, 1], [1, 2, 3]))
        self.assertAlmostEqual(spearman_rank_correlation([1, 1, 2], [1, 1, 2]), 1.0)

    def test_disagreement_that_tracks_error_is_calibrated(self):
        # Frame i: members predict +-i around the truth plus a mean offset of i.
        references = [[(0.0, 0.0, 0.0)] for _ in range(10)]
        member_a = [[(2.0 * i, 0.0, 0.0)] for i in range(10)]
        member_b = [[(0.0, 0.0, 0.0)] for i in range(10)]
        result = committee_error_calibration([member_a, member_b], references, top_fraction=0.2)
        self.assertAlmostEqual(result["spearman_rho"], 1.0)
        self.assertEqual(result["n_top"], 2)
        self.assertAlmostEqual(result["top_error_recall"], 1.0)
        self.assertEqual(result["confident_failures"], [])

    def test_shared_mistake_is_reported_as_confident_failure(self):
        # Frame 0: both members agree and are both 5 eV/Ang wrong.
        references = [[(0.0, 0.0, 0.0)]] + [[(0.0, 0.0, 0.0)] for _ in range(4)]
        member_a = [[(5.0, 0.0, 0.0)]] + [[(0.1 * i, 0.0, 0.0)] for i in range(1, 5)]
        member_b = [[(5.0, 0.0, 0.0)]] + [[(-0.1 * i, 0.0, 0.0)] for i in range(1, 5)]
        result = committee_error_calibration([member_a, member_b], references, top_fraction=0.2)
        self.assertEqual(result["confident_failures"], [0])
        self.assertAlmostEqual(result["top_error_recall"], 0.0)

    def test_rejects_bad_inputs(self):
        references = [[(0.0, 0.0, 0.0)]]
        with self.assertRaises(ValueError):
            committee_error_calibration([[[(0.0, 0.0, 0.0)]]], references)
        with self.assertRaises(ValueError):
            committee_error_calibration([[], [[(0.0, 0.0, 0.0)]]], references)
        with self.assertRaises(ValueError):
            committee_error_calibration(
                [[[(0.0, 0.0, 0.0)]], [[(0.0, 0.0, 0.0)]]], references, top_fraction=0.0
            )

    def test_writes_per_frame_table_and_summary(self):
        frames = [_single_atom_frame(f"f{i}", 0.0) for i in range(4)]
        member_a = [[(float(i), 0.0, 0.0)] for i in range(4)]
        member_b = [[(0.0, 0.0, 0.0)] for _ in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_committee_calibration(frames, [member_a, member_b], Path(tmp), 0.25)
            rows = (Path(tmp) / "calibration.csv").read_text().splitlines()
            self.assertEqual(len(rows), 5)
            self.assertTrue(rows[0].startswith("record_id,formula,charge,multiplicity,pes_region"))
            saved = json.loads((Path(tmp) / "calibration.json").read_text())
        self.assertEqual(saved["n_frames"], 4)
        self.assertEqual(summary["confident_failures"], [])
        self.assertNotIn("disagreement_ev_ang", saved)


if __name__ == "__main__":
    unittest.main()
