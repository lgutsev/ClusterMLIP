from pathlib import Path
import unittest

from cluster_mlip.models import Atom, LabeledFrame, Record
from cluster_mlip.physical_checks import (
    fragmenting_force_check,
    low_coordination_error_check,
    rattled_direction_check,
    run_all_checks,
    spin_ordering_check,
    stationary_point_check,
)


def _frame(record_id, config_type, charge, mult, atoms, energy, forces):
    record = Record(record_id, "s.log", atoms, charge, mult, config_type)
    return LabeledFrame(record, energy, forces, Path("s.log"))


class StationaryPointCheckTests(unittest.TestCase):
    def test_small_predicted_force_at_minimum_passes(self):
        atoms = [Atom("Fe", 0, 0, 0), Atom("O", 1.7, 0, 0)]
        frame = _frame("m1", "minimum", 0, 3, atoms, -10.0, [(0.0, 0, 0), (0.0, 0, 0)])
        prediction = (-10.01, [(0.02, 0, 0), (-0.02, 0, 0)])
        result = stationary_point_check([frame], [prediction])
        self.assertEqual(result["n_frames_considered"], 1)
        self.assertTrue(result["passed"])

    def test_no_stationary_frames_reports_not_applicable(self):
        atoms = [Atom("Fe", 0, 0, 0), Atom("O", 1.7, 0, 0)]
        frame = _frame("w1", "warehouse_structure", 0, 1, atoms, -10.0, [(0, 0, 0), (0, 0, 0)])
        result = stationary_point_check([frame], [(-10.0, [(0, 0, 0), (0, 0, 0)])])
        self.assertIsNone(result["passed"])
        self.assertIsNone(result["metric_value"])


class RattledDirectionCheckTests(unittest.TestCase):
    def test_matching_direction_passes(self):
        atoms = [Atom("Fe", 0, 0, 0), Atom("O", 1.75, 0, 0)]
        frame = _frame("m1-r01", "minimum_rattled", 0, 3, atoms, -9.8, [(0.5, 0.1, 0), (-0.5, -0.1, 0)])
        prediction = (-9.75, [(0.48, 0.09, 0), (-0.48, -0.09, 0)])
        result = rattled_direction_check([frame], [prediction])
        self.assertEqual(result["n_frames_considered"], 1)
        self.assertTrue(result["passed"])

    def test_opposite_direction_fails(self):
        atoms = [Atom("Fe", 0, 0, 0), Atom("O", 1.75, 0, 0)]
        frame = _frame("m1-r01", "minimum_rattled", 0, 3, atoms, -9.8, [(0.5, 0, 0), (-0.5, 0, 0)])
        prediction = (-9.75, [(-0.5, 0, 0), (0.5, 0, 0)])
        result = rattled_direction_check([frame], [prediction])
        self.assertFalse(result["passed"])


class LowCoordinationErrorCheckTests(unittest.TestCase):
    def test_error_concentrated_on_flagged_atom_fails(self):
        # Diatomic: both atoms are low-coordination (degree 1 each), so the
        # "flagged" set is both atoms and this reduces to the frame's own
        # mean error -> ratio 1.0, passes trivially. Use a 3-atom fragment
        # instead so the flagged (degree-0, isolated) atom's error can
        # meaningfully diverge from the frame mean.
        atoms = [Atom("Fe", 0, 0, 0), Atom("O", 1.6, 0, 0), Atom("Fe", 50, 0, 0)]
        frame = _frame(
            "f1", "minimum", 0, 1, atoms, -5.0,
            [(0.1, 0, 0), (0.1, 0, 0), (0.1, 0, 0)],
        )
        # Isolated Fe (index 2) badly mispredicted; the bonded pair is exact.
        prediction = (-5.0, [(0.1, 0, 0), (0.1, 0, 0), (5.0, 0, 0)])
        result = low_coordination_error_check([frame], [prediction])
        self.assertEqual(result["n_frames_considered"], 1)
        self.assertGreater(result["metric_value"], 1.5)
        self.assertFalse(result["passed"])


class FragmentingForceCheckTests(unittest.TestCase):
    def test_opposite_net_force_on_fragments_fails(self):
        atoms = [Atom("Fe", 0, 0, 0), Atom("O", 1.6, 0, 0), Atom("Fe", 50, 0, 0)]
        frame = _frame("frag1", "minimum", 0, 1, atoms, -5.0, [(0.1, 0, 0), (0, 0, 0), (-0.1, 0, 0)])
        prediction = (-5.0, [(-0.1, 0, 0), (0, 0, 0), (0.1, 0, 0)])
        result = fragmenting_force_check([frame], [prediction])
        self.assertEqual(result["n_frames_considered"], 1)
        self.assertAlmostEqual(result["metric_value"], -1.0, places=6)
        self.assertFalse(result["passed"])

    def test_compact_frame_is_not_considered(self):
        atoms = [Atom("Fe", 0, 0, 0), Atom("O", 1.6, 0, 0)]
        frame = _frame("m1", "minimum", 0, 1, atoms, -5.0, [(0, 0, 0), (0, 0, 0)])
        result = fragmenting_force_check([frame], [(-5.0, [(0, 0, 0), (0, 0, 0)])])
        self.assertIsNone(result["passed"])


class SpinOrderingCheckTests(unittest.TestCase):
    def test_model_disagreeing_on_ground_state_fails(self):
        atoms = [Atom("Fe", 0, 0, 0), Atom("Fe", 2.5, 0, 0)]
        low_spin = _frame("s1", "minimum", 0, 1, atoms, -20.0, [(0, 0, 0), (0, 0, 0)])
        high_spin = _frame("s2", "minimum", 0, 9, atoms, -19.0, [(0, 0, 0), (0, 0, 0)])
        # Reference: multiplicity 1 is the ground state (-20.0 < -19.0).
        # Prediction: model thinks multiplicity 9 is lower.
        predictions = [(-19.9, [(0, 0, 0), (0, 0, 0)]), (-19.95, [(0, 0, 0), (0, 0, 0)])]
        result = spin_ordering_check([low_spin, high_spin], predictions)
        self.assertEqual(result["n_frames_considered"], 1)
        self.assertAlmostEqual(result["metric_value"], 0.0)
        self.assertFalse(result["passed"])

    def test_model_agreeing_on_ground_state_passes(self):
        atoms = [Atom("Fe", 0, 0, 0), Atom("Fe", 2.5, 0, 0)]
        low_spin = _frame("s1", "minimum", 0, 1, atoms, -20.0, [(0, 0, 0), (0, 0, 0)])
        high_spin = _frame("s2", "minimum", 0, 9, atoms, -19.0, [(0, 0, 0), (0, 0, 0)])
        predictions = [(-20.05, [(0, 0, 0), (0, 0, 0)]), (-18.9, [(0, 0, 0), (0, 0, 0)])]
        result = spin_ordering_check([low_spin, high_spin], predictions)
        self.assertAlmostEqual(result["metric_value"], 1.0)
        self.assertTrue(result["passed"])

    def test_no_multi_multiplicity_groups_reports_not_applicable(self):
        atoms = [Atom("Fe", 0, 0, 0), Atom("Fe", 2.5, 0, 0)]
        frame = _frame("s1", "minimum", 0, 1, atoms, -20.0, [(0, 0, 0), (0, 0, 0)])
        result = spin_ordering_check([frame], [(-20.0, [(0, 0, 0), (0, 0, 0)])])
        self.assertIsNone(result["passed"])


class RunAllChecksTests(unittest.TestCase):
    def test_returns_five_named_checks(self):
        atoms = [Atom("Fe", 0, 0, 0), Atom("O", 1.7, 0, 0)]
        frame = _frame("m1", "minimum", 0, 1, atoms, -10.0, [(0, 0, 0), (0, 0, 0)])
        results = run_all_checks([frame], [(-10.0, [(0, 0, 0), (0, 0, 0)])])
        names = {r["name"] for r in results}
        self.assertEqual(
            names,
            {
                "stationary_point_force", "rattled_force_direction",
                "low_coordination_error_concentration", "fragmenting_force_direction",
                "spin_state_ordering",
            },
        )

    def test_rejects_length_mismatch(self):
        atoms = [Atom("Fe", 0, 0, 0), Atom("O", 1.7, 0, 0)]
        frame = _frame("m1", "minimum", 0, 1, atoms, -10.0, [(0, 0, 0), (0, 0, 0)])
        with self.assertRaises(ValueError):
            run_all_checks([frame], [])


if __name__ == "__main__":
    unittest.main()
