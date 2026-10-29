import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cluster_mlip.magpt_diagnostic import (
    SpinState,
    assign_ids,
    distinct_states,
    drop_twin_geometries,
    evaluate_state,
    force_metrics,
    group_states_by_geometry,
    hartree_per_bohr_to_ev_per_ang,
    hartree_to_ev,
    minimum_vacuum_gap,
    moment_vectors,
    pairwise_ordering,
    place_in_cubic_box,
    relative_energies,
    same_geometry,
    spin_pattern_distance,
)

RNG = np.random.default_rng(7)
POS = RNG.normal(scale=2.0, size=(4, 3))


def state(m, energy, spins, positions=POS, frame="f"):
    return SpinState(
        frame_id=frame,
        symbols=["Fe"] * len(spins),
        positions=np.array(positions, dtype=float),
        multiplicity=m,
        charge=0,
        energy_ev=energy,
        local_spins=np.array(spins, dtype=float),
        spin_method="mulliken",
        forces_ev_ang=np.zeros((len(spins), 3)),
    )


class MomentVectorTests(unittest.TestCase):
    def test_collinear_z_keeps_sign_and_magnitude(self):
        m = moment_vectors([3.8, -5.6, 0.2, 2.0])
        np.testing.assert_allclose(m[:, 2], [3.8, -5.6, 0.2, 2.0])
        np.testing.assert_allclose(m[:, :2], 0.0)

    def test_no_renormalization_between_states(self):
        a = moment_vectors([3.0, 3.0])
        b = moment_vectors([4.0, 2.0])
        # same total, different magnitudes: must not be mapped onto a common per-site moment
        self.assertAlmostEqual(a[:, 2].sum(), b[:, 2].sum())
        self.assertFalse(np.allclose(a, b))

    def test_scale_and_axis(self):
        m = moment_vectors([2.0, -1.0], scale=1.1, axis="x")
        np.testing.assert_allclose(m, [[2.2, 0, 0], [-1.1, 0, 0]])

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            moment_vectors([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            moment_vectors([1.0, math.nan])
        with self.assertRaises(ValueError):
            moment_vectors([1.0], axis="w")


class MatchingTests(unittest.TestCase):
    def test_states_at_identical_geometry_group_together(self):
        a = state(53, -10.0, [4, 4, 4, 4])
        b = state(51, -10.3, [4, 4, 4, 2], positions=POS + 1e-5)
        c = state(53, -9.0, [4, 4, 4, 4], positions=POS + 0.2)
        groups = group_states_by_geometry([a, b, c])
        self.assertEqual([len(g) for g in groups], [2, 1])
        self.assertTrue(same_geometry(a.positions, b.positions))
        self.assertFalse(same_geometry(a.positions, c.positions))

    def test_duplicate_scf_solution_is_not_a_new_state(self):
        a = state(51, -10.0, [4, 4, 4, 2])
        dup = state(51, -10.00005, [4, 4, 4, 2.01])
        other = state(51, -10.0, [4, 4, -4, 2])  # same M and energy, different pattern: a distinct solution
        self.assertEqual(len(distinct_states([a, dup, other])), 2)

    def test_permuted_rerun_is_a_twin_geometry(self):
        g1 = [state(53, -10.0, [4, 4, 4, 4]), state(51, -10.3, [4, 4, 4, 2])]
        perm = POS[[2, 0, 3, 1]] @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        g2 = [state(53, -10.0, [4] * 4, positions=perm), state(51, -10.3, [4, 4, 2, 4], positions=perm)]
        g3 = [state(53, -11.0, [4] * 4, positions=perm), state(51, -10.3, [4, 4, 2, 4], positions=perm)]
        self.assertEqual(len(drop_twin_geometries([g1, g2])), 1)
        self.assertEqual(len(drop_twin_geometries([g1, g3])), 2)

    def test_ids_are_stable_and_unique(self):
        g = [state(53, -10.0, [4] * 4), state(51, -10.3, [4, 4, 4, 2]), state(51, -10.1, [4, 4, -4, 2])]
        assign_ids([g])
        ids = [s.state_id for s in g]
        self.assertEqual(len(set(ids)), 3)
        self.assertTrue(all(i.startswith(g[0].geometry_id) for i in ids))
        moved = [state(53, -10.0, [4] * 4, positions=POS @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))]
        assign_ids([moved])
        self.assertEqual(moved[0].geometry_id, g[0].geometry_id)

    def test_spin_pattern_distance(self):
        d = spin_pattern_distance([4, 4, -4, 0.1], [4, 4, 4, -0.1])
        self.assertEqual(d["n_sign_changes_strong_sites"], 1)  # weak site sign change not counted
        self.assertAlmostEqual(d["max_local_moment_diff"], 8.0)


class UnitTests(unittest.TestCase):
    def test_hartree_and_force_conversion(self):
        self.assertAlmostEqual(hartree_to_ev(1.0), 27.211386245988)
        np.testing.assert_allclose(hartree_per_bohr_to_ev_per_ang([1.0]), [27.211386245988 / 0.529177210903])

    def test_box_placement(self):
        pos, cell = place_in_cubic_box(POS, 25.0)
        np.testing.assert_allclose(pos.mean(axis=0), [12.5] * 3)
        np.testing.assert_allclose(cell, np.eye(3) * 25.0)
        np.testing.assert_allclose(pos - pos[0], POS - POS[0])  # pure translation
        extent = (POS.max(0) - POS.min(0)).max()
        self.assertAlmostEqual(minimum_vacuum_gap(POS, 25.0), 25.0 - extent)


class RelativeEnergyTests(unittest.TestCase):
    def test_same_reference_state_on_both_sides(self):
        np.testing.assert_allclose(relative_energies([-5.0, -5.3, -4.9], 1), [0.3, 0.0, 0.4])
        # an offset in absolute energies (different energy scales) cancels exactly
        np.testing.assert_allclose(relative_energies([1e5 - 5.0, 1e5 - 5.3], 1), [0.3, 0.0], atol=1e-9)

    def test_pairwise_ordering(self):
        o = pairwise_ordering([0.0, 0.1, 0.3], [0.0, 0.2, 0.1])
        self.assertEqual((o["concordant_pairs"], o["discordant_pairs"]), (2, 1))
        self.assertAlmostEqual(o["kendall_tau_a"], 1 / 3)
        self.assertAlmostEqual(o["spearman_rho"], 0.5)
        self.assertTrue(math.isnan(pairwise_ordering([0, 1], [0, 1])["spearman_rho"]))

    def test_force_metrics(self):
        ref = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
        m = force_metrics(ref, 2 * ref)
        self.assertAlmostEqual(m["cosine"], 1.0)
        self.assertAlmostEqual(m["mae"], 1 / 3)


class FakeModel:
    """Deterministic stand-in: E depends on positions and moments, so plumbing errors show up."""

    def __init__(self):
        self.calls = []

    def __call__(self, positions, cell, moments, symbols):
        self.calls.append((None if cell is None else cell.copy(), moments.copy()))
        com = positions - positions.mean(0)
        e = float((com**2).sum() * 0.01 - 0.05 * (moments[:, 2] ** 2).sum() + 0.001 * moments[:, 2].sum() ** 2)
        return e, -0.02 * com, 0.1 * moments


class EvaluateStateTests(unittest.TestCase):
    def test_moments_and_box_reach_the_model(self):
        model = FakeModel()
        s = state(51, -1.0, [4, 4, -4, 2])
        r = evaluate_state(model, s, 30.0, scale=0.9)
        cell, moments = model.calls[-1]
        np.testing.assert_allclose(cell, np.eye(3) * 30.0)
        np.testing.assert_allclose(moments[:, 2], 0.9 * np.array([4, 4, -4, 2]))
        self.assertAlmostEqual(r["mag_force_transverse_rms"], 0.0)
        r0 = evaluate_state(model, s, None)
        self.assertIsNone(model.calls[-1][0])
        self.assertAlmostEqual(r0["energy_ev"], r["energy_ev"] - 0.05 * (1 - 0.81) * 52 + 0.001 * (1 - 0.81) * 36)

    def test_translation_into_box_does_not_change_fake_energy(self):
        s = state(51, -1.0, [4, 4, -4, 2])
        e = [evaluate_state(FakeModel(), s, L)["energy_ev"] for L in (20.0, 25.0, 30.0)]
        self.assertAlmostEqual(max(e) - min(e), 0.0)


@unittest.skipUnless(os.environ.get("MAGPT_CHECKPOINT"), "set MAGPT_CHECKPOINT to run against the real model")
class RealModelDeterminismTests(unittest.TestCase):
    """Run in the MagPT environment: MAGPT_CHECKPOINT=.../MagPT_399168.pth python -m pytest tests/test_magpt_diagnostic.py"""

    @classmethod
    def setUpClass(cls):
        import torch

        from cluster_mlip.magpt_diagnostic import DeepSpinEvaluator

        torch.set_num_threads(1)  # multi-threaded CPU reductions are not bitwise reproducible
        cls.model = DeepSpinEvaluator(os.environ["MAGPT_CHECKPOINT"])
        # bcc Fe fragment, nearest-neighbour 2.48 A
        a = 2.866
        pos = [[0, 0, 0], [a / 2, a / 2, a / 2], [a, 0, 0], [0, a, 0], [0, 0, a], [a / 2, -a / 2, a / 2]]
        cls.s = state(13, 0.0, [2.2, 2.2, 2.2, 2.2, -2.2, 2.2], positions=pos)

    def test_fe_is_covered(self):
        self.assertIn("Fe", self.model.type_map)

    def test_repeat_inference_is_bitwise_identical(self):
        r1 = evaluate_state(self.model, self.s, 25.0)
        r2 = evaluate_state(self.model, self.s, 25.0)
        self.assertEqual(r1["energy_ev"], r2["energy_ev"])
        np.testing.assert_array_equal(r1["forces"], r2["forces"])
        np.testing.assert_array_equal(r1["mag_forces"], r2["mag_forces"])

    def test_time_reversal_in_bulk_ferromagnet_and_spin_dependence(self):
        # bcc Fe 2x2x2 conventional cell, the model's home domain: global m -> -m must not change E.
        # (MagPT is NOT exactly even in m for general configurations; that is measured in the diagnostic.)
        a = 2.83
        frac = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], dtype=float)
        pos = np.concatenate([frac * a, (frac + 0.5) * a])
        bulk = state(33, 0.0, [2.2] * 16, positions=pos)
        cell = np.eye(3) * 2 * a
        mom = moment_vectors(bulk.local_spins)
        e_up = self.model(pos, cell, mom, bulk.symbols)[0]
        e_down = self.model(pos, cell, -mom, bulk.symbols)[0]
        e_x = self.model(pos, cell, moment_vectors(bulk.local_spins, axis="x"), bulk.symbols)[0]
        self.assertAlmostEqual(e_up, e_down, delta=1e-3)
        self.assertAlmostEqual(e_up, e_x, delta=1e-3)
        e = evaluate_state(self.model, self.s, 25.0)["energy_ev"]
        e_ferro = evaluate_state(self.model, self.s, 25.0, spins=np.abs(self.s.local_spins))["energy_ev"]
        self.assertGreater(abs(e - e_ferro), 1e-3)

if __name__ == "__main__":
    unittest.main()
