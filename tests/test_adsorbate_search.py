import unittest

try:
    import numpy as np
    import scipy  # noqa: F401
    from ase import Atoms
    from ase.build import fcc111
    from ase.cluster import Icosahedron

    from cluster_mlip.adsorbate_search import (
        ROLE_ADSORBATE,
        ROLE_CATALYST,
        ROLE_SUPPORT,
        AdsorbateSearch,
        Gene,
        SearchConfig,
        ToyPairCalculator,
        analyze,
        build_candidate,
        find_cluster_sites,
        native_evaluator,
        observed_genes,
        same_structure,
        sites_from_list,
        validate,
    )
except ImportError:  # pragma: no cover - ASE/scipy are optional for the package
    Atoms = None

COMPOSITION = {"CO": 1, "H": 2}


def _ico():
    atoms = Atoms(Icosahedron("Fe", 2, latticeconstant=3.5))
    atoms.set_array("role", np.full(len(atoms), ROLE_CATALYST))
    return atoms


def _site(sites, kind, without=()):
    return next(s for s in sites if s.kind == kind and not set(s.indices) & set(without))


def _far_apart_genes(report):
    """CO on a top site and two H on hollows that share no atom with it or each other."""
    top = _site(report.sites, "top")
    h1 = _site(report.sites, "hollow3", without=top.indices)
    h2 = _site(report.sites, "hollow3", without=top.indices + h1.indices)
    return [Gene("CO", top.id, "C"), Gene("H", h1.id), Gene("H", h2.id)]


@unittest.skipIf(Atoms is None, "ASE/scipy not installed")
class SiteTests(unittest.TestCase):
    def test_icosahedron_sites_and_interior_atom(self):
        atoms = _ico()
        report = find_cluster_sites(atoms)
        self.assertEqual(report.counts(), {"top": 12, "bridge": 30, "hollow3": 20})
        self.assertEqual(len(report.interior_ids), 1)
        centre = report.interior_ids[0]
        self.assertTrue(all(centre not in s.indices for s in report.sites))
        centroid = atoms.positions.mean(axis=0)
        for s in report.sites:  # normals point away from the cluster
            self.assertGreater(float(s.n @ (s.p - centroid)), 0)
            self.assertAlmostEqual(float(np.linalg.norm(s.n)), 1.0, places=6)

    def test_square_faces_become_fourfold_sites(self):
        a = 2.4
        cube = Atoms("Fe8", positions=[[x, y, z] for x in (0, a) for y in (0, a) for z in (0, a)])
        report = find_cluster_sites(cube, bond_cutoff=2.6)
        self.assertEqual(report.counts(), {"top": 8, "bridge": 12, "hollow4": 6})

    def test_cluster_finder_refuses_periodic_cells(self):
        slab = fcc111("Fe", size=(2, 2, 2), a=3.6, vacuum=8.0)
        with self.assertRaises(ValueError):
            find_cluster_sites(slab)

    def test_manual_site_list_rejects_non_catalyst_atoms(self):
        atoms = _ico()
        role = atoms.arrays["role"].copy()
        role[0] = ROLE_SUPPORT
        atoms.set_array("role", role)
        with self.assertRaises(ValueError):
            sites_from_list(atoms, [{"indices": [0, 1]}])


@unittest.skipIf(Atoms is None, "ASE/scipy not installed")
class BuildAndValidateTests(unittest.TestCase):
    def setUp(self):
        self.cat = _ico()
        self.report = find_cluster_sites(self.cat)
        self.genes = _far_apart_genes(self.report)
        self.atoms = build_candidate(self.cat, self.report.sites, self.genes)

    def test_composition_and_roles_are_preserved(self):
        n0 = len(self.cat)
        self.assertEqual(self.atoms.get_chemical_symbols()[:n0], self.cat.get_chemical_symbols())
        np.testing.assert_allclose(self.atoms.positions[:n0], self.cat.positions)
        self.assertEqual(sorted(self.atoms.get_chemical_symbols()[n0:]), ["C", "H", "H", "O"])
        self.assertTrue((self.atoms.arrays["role"][:n0] == ROLE_CATALYST).all())
        self.assertTrue((self.atoms.arrays["role"][n0:] == ROLE_ADSORBATE).all())
        self.assertEqual(self.atoms.arrays["mol_id"][n0:].tolist(), [0, 0, 1, 2])
        self.assertEqual(self.atoms.arrays["cluster"].tolist(), [1] * n0 + [0] * 4)
        self.assertEqual(validate(self.atoms, self.cat, COMPOSITION), [])

    def test_validate_catches_wrong_composition_and_broken_roles(self):
        self.assertIn("composition_mismatch", validate(self.atoms, self.cat, {"CO": 1, "H": 1}))
        broken = self.atoms.copy()
        role = broken.arrays["role"].copy()
        role[0] = ROLE_ADSORBATE
        broken.set_array("role", role)
        self.assertIn("role_map_broken", validate(broken, self.cat, COMPOSITION))
        moved = self.atoms.copy()
        moved.positions[0] += 0.2
        self.assertIn("catalyst_changed", validate(moved, self.cat, COMPOSITION))

    def test_site_placement_matches_the_requested_site_and_mode(self):
        n0 = len(self.cat)
        top = self.report.sites[self.genes[0].site]
        c, o = self.atoms.positions[n0], self.atoms.positions[n0 + 1]
        self.assertAlmostEqual(float(np.linalg.norm(c - self.cat.positions[top.indices[0]])), 1.78, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(o - c)), 1.15, places=6)
        np.testing.assert_allclose((o - c) / 1.15, top.n, atol=1e-6)  # untilted C-bound CO points outward
        hollow = self.report.sites[self.genes[1].site]
        d = [float(np.linalg.norm(self.atoms.positions[n0 + 2] - self.cat.positions[i])) for i in hollow.indices]
        np.testing.assert_allclose(d, 1.78, atol=1e-6)
        # O-bound mode puts O on the surface, and the analysis reports it
        o_bound = build_candidate(self.cat, self.report.sites, [Gene("CO", top.id, "O")] + self.genes[1:])
        self.assertEqual(o_bound.get_chemical_symbols()[n0:n0 + 2], ["O", "C"])
        self.assertIn("co_o_bound", analyze(o_bound, self.cat, self.report.sites).flags)

    def test_two_adsorbates_on_adjacent_sites_are_rejected(self):
        top = _site(self.report.sites, "top")
        bridge = next(s for s in self.report.sites if s.kind == "bridge" and top.indices[0] in s.indices)
        hollow = next(s for s in self.report.sites if s.kind == "hollow3" and set(bridge.indices) <= set(s.indices))
        far = _site(self.report.sites, "top", without=hollow.indices)
        crowded = build_candidate(self.cat, self.report.sites,
                                  [Gene("CO", far.id, "C"), Gene("H", bridge.id), Gene("H", hollow.id)])
        self.assertIn("adsorbates_too_close", validate(crowded, self.cat, COMPOSITION))

    def test_observed_genes_recover_the_intended_sites(self):
        analysis = analyze(self.atoms, self.cat, self.report.sites)
        self.assertEqual(analysis.topology, "CO+H+H")
        recovered = observed_genes(self.atoms, analysis, self.genes, self.report.sites)
        self.assertEqual([g.site for g in recovered], [g.site for g in self.genes])
        self.assertEqual(recovered[0].mode, "C")


@unittest.skipIf(Atoms is None, "ASE/scipy not installed")
class MolecularIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.cat = _ico()
        self.report = find_cluster_sites(self.cat)
        self.genes = _far_apart_genes(self.report)
        self.atoms = build_candidate(self.cat, self.report.sites, self.genes)
        self.n0 = len(self.cat)

    def _analyze(self, atoms):
        return analyze(atoms, self.cat, self.report.sites)

    def test_dissociated_co_is_labelled_not_hidden(self):
        atoms = self.atoms.copy()
        other = _site(self.report.sites, "hollow3",
                      without=sum((self.report.sites[g.site].indices for g in self.genes), ()))
        atoms.positions[self.n0 + 1] = other.p + 1.3 * other.n  # move O to another hollow
        analysis = self._analyze(atoms)
        self.assertEqual(analysis.topology, "C+H+H+O")
        self.assertIn("co_dissociated", analysis.flags)
        self.assertFalse(analysis.failure)  # same elemental composition: kept, labelled
        self.assertIsNone(observed_genes(atoms, analysis, self.genes, self.report.sites))

    def test_h2_formyl_and_desorption_are_recognised(self):
        h2 = self.atoms.copy()
        h2.positions[self.n0 + 3] = h2.positions[self.n0 + 2] + [0.0, 0.0, 0.75]
        self.assertIn("h2_formed", self._analyze(h2).flags)
        formyl = self.atoms.copy()
        c, o = formyl.positions[self.n0], formyl.positions[self.n0 + 1]
        side = np.cross(o - c, [1.0, 0.3, 0.2])
        formyl.positions[self.n0 + 2] = c + 1.1 * side / np.linalg.norm(side)
        self.assertEqual(self._analyze(formyl).topology, "H+HCO")
        gone = self.atoms.copy()
        top = self.report.sites[self.genes[0].site]
        gone.positions[self.n0:self.n0 + 2] += 6.0 * top.n
        self.assertIn("desorbed_CO", self._analyze(gone).flags)

    def test_atom_overlap_is_a_failure_not_a_low_energy_find(self):
        atoms = self.atoms.copy()
        atoms.positions[self.n0 + 2] = atoms.positions[0] + [0.5, 0.0, 0.0]
        analysis = self._analyze(atoms)
        self.assertTrue(analysis.failure)
        self.assertIn("atom_overlap", analysis.flags)


@unittest.skipIf(Atoms is None, "ASE/scipy not installed")
class DuplicateTests(unittest.TestCase):
    def setUp(self):
        self.cat = _ico()
        self.report = find_cluster_sites(self.cat)
        self.genes = _far_apart_genes(self.report)
        self.atoms = build_candidate(self.cat, self.report.sites, self.genes)

    def test_swapping_identical_adsorbates_is_a_duplicate(self):
        swapped = build_candidate(self.cat, self.report.sites, [self.genes[0], self.genes[2], self.genes[1]])
        self.assertFalse(np.allclose(swapped.positions, self.atoms.positions))
        self.assertTrue(same_structure(self.atoms, swapped))
        self.assertTrue(same_structure(self.atoms, swapped, frame="fixed"))

    def test_moving_one_h_is_not_a_duplicate(self):
        taken = sum((self.report.sites[g.site].indices for g in self.genes), ())
        other = _site(self.report.sites, "bridge", without=taken)
        moved = build_candidate(self.cat, self.report.sites, self.genes[:2] + [Gene("H", other.id)])
        self.assertFalse(same_structure(self.atoms, moved))

    def test_rigid_rotation_is_a_duplicate_only_for_an_isolated_cluster(self):
        rotated = self.atoms.copy()
        rotated.rotate(37.0, (1.0, 2.0, 0.5), center="COP")
        rotated.translate([0.4, -0.2, 0.1])
        self.assertTrue(same_structure(self.atoms, rotated))  # isolated: rigid motion removed
        self.assertFalse(same_structure(self.atoms, rotated, frame="fixed"))  # supported: it is a new structure

    def test_support_atoms_switch_the_default_to_the_fixed_frame(self):
        supported = self.atoms.copy()
        role = supported.arrays["role"].copy()
        role[0] = ROLE_SUPPORT
        supported.set_array("role", role)
        rotated = supported.copy()
        rotated.rotate(37.0, (1.0, 2.0, 0.5), center="COP")
        self.assertFalse(same_structure(supported, rotated))


@unittest.skipIf(Atoms is None, "ASE/scipy not installed")
class PeriodicTests(unittest.TestCase):
    def setUp(self):
        slab = fcc111("Fe", size=(2, 2, 2), a=3.6, vacuum=8.0)
        slab.set_array("role", np.full(len(slab), ROLE_CATALYST))
        self.slab = slab
        top_layer = [i for i in range(len(slab)) if slab.positions[i, 2] > slab.positions[:, 2].mean()]
        self.top_layer = top_layer
        self.report = sites_from_list(slab, [{"indices": [i]} for i in top_layer])

    def test_manual_sites_point_along_z_and_keep_catalyst_ids(self):
        for s, i in zip(self.report.sites, self.top_layer):
            self.assertEqual(s.indices, (i,))
            np.testing.assert_allclose(s.n, [0, 0, 1])

    def test_distance_filter_uses_periodic_images(self):
        # Two H on the same top atom's periodic images: one near x = 0, one moved by a full cell
        # vector minus a little. Without minimum image they look 5 A apart.
        site = self.report.sites[0]
        atoms = build_candidate(self.slab, self.report.sites, [Gene("H", site.id), Gene("H", self.report.sites[1].id)])
        n0 = len(self.slab)
        atoms.positions[n0 + 1] = atoms.positions[n0] + self.slab.cell[0] + [0.6, 0.0, 0.0]
        self.assertIn("adsorbates_too_close", validate(atoms, self.slab, {"H": 2}))
        isolated = atoms.copy()
        isolated.pbc = False
        reference = self.slab.copy()
        reference.pbc = False
        self.assertNotIn("adsorbates_too_close", validate(isolated, reference, {"H": 2}))

    def test_periodic_duplicate_test_ignores_whole_cell_shifts_of_an_adsorbate(self):
        genes = [Gene("H", self.report.sites[0].id), Gene("H", self.report.sites[1].id)]
        a = build_candidate(self.slab, self.report.sites, genes)
        b = a.copy()
        b.positions[len(self.slab)] += self.slab.cell[1]
        self.assertTrue(same_structure(a, b))


@unittest.skipIf(Atoms is None, "ASE/scipy not installed")
class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = _ico()
        cls.report = find_cluster_sites(cls.cat)
        cls.runs = {}
        for arm in ("random", "evolutionary"):
            search = AdsorbateSearch(cls.cat, cls.report.sites, ToyPairCalculator(),
                                     SearchConfig(arm=arm, seed=3, budget=14, population=6, max_steps=80))
            search.run()
            cls.runs[arm] = search

    def test_budget_and_ledger_account_for_every_proposal(self):
        for search in self.runs.values():
            records = search.records
            relaxed = [r for r in records if r["status"] == "relaxed"]
            self.assertEqual(len(relaxed), 14)
            self.assertEqual({r["status"] for r in records} - {"relaxed", "invalid", "duplicate_start"}, set())
            self.assertEqual(sum(r["calc_calls"] for r in records), search.counter.calls)
            self.assertEqual(records[-1]["cum_calc_calls"], search.counter.calls)
            for r in records:
                if r["status"] != "relaxed":
                    self.assertEqual(r["calc_calls"], 0)
                    self.assertTrue(r["reject_reasons"])
            summary = search.summary()
            self.assertEqual(summary["proposals"], summary["invalid"] + summary["duplicate_start"] + summary["relaxations"])
            self.assertEqual(summary["relaxations"], summary["unique"] + summary["duplicates_after_relax"])

    def test_both_arms_share_the_initial_population(self):
        first = {arm: [r["genes"] for r in s.records if r["status"] == "relaxed"][:6] for arm, s in self.runs.items()}
        self.assertEqual(first["random"], first["evolutionary"])
        operators = {r["operator"] for r in self.runs["evolutionary"].records}
        self.assertTrue(operators - {"random"})  # children were proposed
        self.assertEqual({r["operator"] for r in self.runs["random"].records}, {"random"})

    def test_children_record_parents_that_exist(self):
        records = self.runs["evolutionary"].records
        ids = {r["candidate_id"] for r in records}
        for r in records:
            if r["operator"] in ("move_H", "move_CO", "rotate_CO"):
                self.assertEqual(len(r["parent_ids"]), 1)
            if r["operator"] == "recombine":
                self.assertEqual(len(r["parent_ids"]), 2)
            self.assertTrue(set(r["parent_ids"]) <= ids)

    def test_every_relaxed_structure_keeps_catalyst_and_composition(self):
        n0 = len(self.cat)
        for search in self.runs.values():
            for start, final in search.structures.values():
                if final is None:
                    continue
                np.testing.assert_allclose(final.positions[:n0], self.cat.positions)  # Stage A: catalyst fixed
                self.assertEqual(sorted(final.get_chemical_symbols()[n0:]), ["C", "H", "H", "O"])
                self.assertEqual(final.arrays["role"].tolist(), start.arrays["role"].tolist())

    def test_same_seed_gives_the_same_search(self):
        again = AdsorbateSearch(self.cat, self.report.sites, ToyPairCalculator(),
                                SearchConfig(arm="evolutionary", seed=3, budget=14, population=6, max_steps=80))
        again.run()
        self.assertEqual([r["final_sha"] for r in again.records],
                         [r["final_sha"] for r in self.runs["evolutionary"].records])

    def test_batched_evaluation_reproduces_the_sequential_ledger(self):
        keys = ("candidate_id", "operator", "parent_ids", "status", "start_sha", "final_sha", "duplicate_of", "motif")
        for arm, batch in (("random", 5), ("evolutionary", 4)):
            cfg = SearchConfig(arm=arm, seed=3, budget=14, population=6, max_steps=80)
            batched = AdsorbateSearch(self.cat, self.report.sites, ToyPairCalculator(), cfg)
            sizes = []
            evaluate = native_evaluator(ToyPairCalculator(), fmax=cfg.fmax, max_steps=cfg.max_steps)

            def counting(structures, evaluate=evaluate, sizes=sizes):
                sizes.append(len(structures))
                return evaluate(structures)

            batched.run_batched(counting, batch_size=batch)
            sequential = self.runs[arm].records
            self.assertEqual([[r[k] for k in keys] for r in batched.records], [[r[k] for k in keys] for r in sequential])
            self.assertEqual(sum(r["calc_calls"] for r in batched.records), sum(r["calc_calls"] for r in sequential))
            self.assertGreater(max(sizes), 1)  # something was really batched
            if arm == "evolutionary":  # children are never batched: that would change the algorithm
                self.assertEqual(sizes.count(1), 14 - 6)

    def test_batch_evaluator_must_return_one_result_per_structure(self):
        search = AdsorbateSearch(self.cat, self.report.sites, ToyPairCalculator(),
                                 SearchConfig(arm="random", seed=3, budget=4, max_steps=10))
        with self.assertRaises(RuntimeError):
            search.run_batched(lambda structures: [], batch_size=4)


if __name__ == "__main__":
    unittest.main()
