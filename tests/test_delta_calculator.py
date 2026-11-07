import unittest

try:
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.calculators.lj import LennardJones
    from ase.optimize import BFGS
except ImportError:  # pragma: no cover - ASE is optional for the package
    Atoms = None


def _system():
    support = Atoms("C4", positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0]],
                    cell=[6, 6, 14], pbc=True)
    cluster = Atoms("Fe2", positions=[[0.75, 0.75, 2.0], [0.75, 0.75, 4.1]])
    atoms = support + cluster
    atoms.arrays["cluster"] = np.array([0, 0, 0, 0, 1, 1])
    atoms.info.update(charge=0, spin=7)
    return atoms


if Atoms is not None:
    class SpinToy(Calculator):
        """E = sum of a harmonic pair term + (M - 5)^2, so each multiplicity is distinct."""

        implemented_properties = ["energy", "free_energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            pos = self.atoms.positions
            e, f = 0.0, np.zeros_like(pos)
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    d = pos[i] - pos[j]
                    r = np.linalg.norm(d)
                    e += (r - 2.0) ** 2
                    g = 2 * (r - 2.0) * d / r
                    f[i] -= g
                    f[j] += g
            e += (self.atoms.info.get("spin", 5) - 5) ** 2
            self.results = {"energy": e, "free_energy": e, "forces": f}


@unittest.skipIf(Atoms is None, "ASE not installed")
class DeltaCalculatorTests(unittest.TestCase):
    def test_all_one_model_reduces_to_that_model(self) -> None:
        from cluster_mlip.delta import DeltaCalculator, SubtractiveInteraction

        lj = LennardJones(sigma=2.0, epsilon=0.1, rc=5.0)
        atoms = _system()
        reference = atoms.copy()
        reference.calc = LennardJones(sigma=2.0, epsilon=0.1, rc=5.0)
        # The reference treats the cluster periodically too; with rc < cell size and
        # the cluster far from its images the free-molecule cluster term is identical.
        atoms.calc = DeltaCalculator(lj, lj, SubtractiveInteraction(lj))
        self.assertAlmostEqual(atoms.get_potential_energy(), reference.get_potential_energy(), places=10)
        np.testing.assert_allclose(atoms.get_forces(), reference.get_forces(), atol=1e-10)
        terms = atoms.calc.terms
        self.assertAlmostEqual(sum(terms.values()), reference.get_potential_energy(), places=10)

    def test_terms_use_cluster_spin_and_closed_shell_support(self) -> None:
        from cluster_mlip.delta import DeltaCalculator, cluster_part, support_part

        atoms = _system()
        mask = atoms.arrays["cluster"].astype(bool)
        self.assertEqual(cluster_part(atoms, mask).info["spin"], 7)
        self.assertFalse(cluster_part(atoms, mask).pbc.any())
        self.assertEqual(support_part(atoms, mask).info["spin"], 1)
        toy = SpinToy()
        zero = LennardJones(sigma=1.0, epsilon=0.0)
        atoms.calc = DeltaCalculator(toy, toy, zero)
        atoms.get_potential_energy()
        # cluster at M=7 pays (7-5)^2 = 4; support at M=1 pays 16
        self.assertAlmostEqual(atoms.calc.terms["cluster"], (np.linalg.norm([0, 0, 2.1]) - 2.0) ** 2 + 4)
        self.assertGreater(atoms.calc.terms["support"], 16)

    def test_same_geometry_other_multiplicity_is_not_served_from_cache(self) -> None:
        from cluster_mlip.delta import _evaluate

        toy = SpinToy()
        a = Atoms("Fe2", positions=[[0, 0, 0], [0, 0, 2.0]])
        a.info["spin"] = 5
        e5, _ = _evaluate(toy, a)
        a.info["spin"] = 9
        e9, _ = _evaluate(toy, a)
        self.assertAlmostEqual(e9 - e5, 16.0)

    def test_relax_over_multiplicities_keeps_lowest_state(self) -> None:
        from cluster_mlip.delta import relax_over_multiplicities

        a = Atoms("Fe2", positions=[[0, 0, 0], [0, 0, 2.5]])
        result = relax_over_multiplicities(
            a, SpinToy(), [3, 5, 7], lambda x: BFGS(x, logfile=None).run(fmax=1e-4, steps=200),
        )
        self.assertEqual(result.multiplicity, 5)
        self.assertEqual(sorted(result.by_multiplicity), [3, 5, 7])
        self.assertAlmostEqual(result.by_multiplicity[3] - result.by_multiplicity[5], 4.0, places=5)
        self.assertEqual(result.atoms.info["spin"], 5)
        self.assertAlmostEqual(result.atoms.get_distance(0, 1), 2.0, places=3)

    def test_mask_fallbacks(self) -> None:
        from cluster_mlip.delta import cluster_mask

        atoms = _system()
        del atoms.arrays["cluster"]
        with self.assertRaises(ValueError):
            cluster_mask(atoms)
        np.testing.assert_array_equal(cluster_mask(atoms, ["Fe"]), [0, 0, 0, 0, 1, 1])
        atoms.set_tags([0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(cluster_mask(atoms), [0, 0, 0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
