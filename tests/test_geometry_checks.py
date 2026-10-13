import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.cluster import Icosahedron
from ase.io import read, write
from scipy.spatial.transform import Rotation

from cluster_mlip.cli import main
from cluster_mlip.geometry_checks import (
    Partition, compare_isolated, compare_supported, match_atoms, site_report,
    structure_report, unwrapped_positions,
)


def fe13_no(seed=0):
    """Distorted Fe13 icosahedron with NO bound end-on (N down) to one vertex."""
    fe = Icosahedron("Fe", noshells=2, latticeconstant=2.87 * 2 ** 0.5)
    rng = np.random.default_rng(seed)
    pos = fe.positions - fe.positions.mean(0) + rng.normal(0, 0.05, (13, 3))
    vertex = int(np.argmax(np.linalg.norm(pos, axis=1)))
    u = pos[vertex] / np.linalg.norm(pos[vertex])
    n, o = pos[vertex] + 1.70 * u, pos[vertex] + (1.70 + 1.17) * u
    return Atoms(["Fe"] * 13 + ["N", "O"], np.vstack([pos, n, o])), vertex


def mgo_slab():
    slab = bulk("MgO", "rocksalt", a=4.21, cubic=True).repeat((2, 2, 2))
    slab.center(vacuum=7.0, axis=2)
    slab.pbc = True
    return slab


def fe4_on_mgo(shift=(0.0, 0.0)):
    """Fe4 tetrahedron, apex down 2.0 A above a corner surface O, base wrapped across the cell edge."""
    slab = mgo_slab()
    top_z = slab.positions[:, 2].max()
    o_top = [i for i, a in enumerate(slab) if a.symbol == "O" and abs(a.position[2] - top_z) < 0.1]
    anchor = slab.positions[min(o_top, key=lambda i: slab.positions[i, 0] + slab.positions[i, 1])]
    tet = np.array([[0, 0, 0], [-1.2, -0.69, 1.96], [1.2, -0.69, 1.96], [0, 1.39, 1.96]])
    tet = tet + anchor + [0, 0, 2.0]
    system = slab + Atoms("Fe4", tet)
    system.positions[:, :2] += shift
    system.wrap()
    return system, len(slab)


class IsolatedClusterTests(unittest.TestCase):
    def setUp(self):
        self.ref, self.vertex = fe13_no()
        rot = Rotation.from_euler("zyx", [37, -112, 64], degrees=True).as_matrix()
        perm = np.random.default_rng(3).permutation(len(self.ref))
        self.perm = perm
        self.moved = Atoms([self.ref[i].symbol for i in perm],
                           (self.ref.positions @ rot.T + [4.1, -7.3, 2.2])[perm])

    def test_rotated_translated_reordered_copy_is_identical(self):
        cmp = compare_isolated(self.ref, self.moved)
        self.assertLess(cmp["rmsd"], 1e-6)
        self.assertLess(cmp["max_displacement"], 1e-6)
        self.assertLess(cmp["fingerprint_rms"], 1e-6)
        self.assertEqual(cmp["bond_changes"], {"formed": [], "broken": []})
        # the mapping recovers the shuffle: moved[perm_found[i]] is reference atom i
        np.testing.assert_array_equal(self.perm[cmp["perm"]], np.arange(len(self.ref)))

    def test_reports_are_invariant_under_the_same_operation(self):
        def summary(atoms):
            ads = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s in "NO"]
            r = structure_report(atoms, Partition.from_specs(len(atoms), adsorbate=",".join(map(str, ads))),
                                 metal="Fe")
            a = r["adsorbate"]
            return (r["cluster_coordination"]["cn_histogram"], r["contacts"]["min_distance"]["d"],
                    r["cluster_shape"]["radius_of_gyration"], r["cluster_coordination"]["Fe-Fe_bonded"],
                    sorted(b["d"] for b in a["internal_bonds"]), sorted(c["d"] for c in a["host_contacts"]),
                    a["orientation"]["tilt_deg"])
        self.assertEqual(summary(self.ref), summary(self.moved))
        a = structure_report(self.ref, Partition.from_specs(15, adsorbate="13:15"))["adsorbate"]
        self.assertEqual([b["pair"] for b in a["internal_bonds"]], ["N-O"])
        self.assertEqual([(c["pair"], c["j"]) for c in a["host_contacts"]], [("N-Fe", self.vertex)])
        self.assertLess(a["orientation"]["tilt_deg"], 1.0)  # end-on, along the radial direction

    def test_distortion_and_bond_change_are_localized(self):
        broken = self.ref.copy()
        u = broken.positions[13] - broken.positions[self.vertex]
        broken.positions[13:] += 1.5 * u / np.linalg.norm(u)  # pull NO off the vertex
        cmp = compare_isolated(self.ref, broken)
        self.assertEqual([(b["i"], b["j"]) for b in cmp["bond_changes"]["broken"]], [(self.vertex, 13)])
        self.assertEqual({d["i"] for d in cmp["largest_displacements"][:2]}, {13, 14})

    def test_near_spherical_fe16_mapping_is_global(self):
        """Nearly degenerate principal moments once trapped the search in a
        0.72 A local optimum where 0.30 A exists; the optimum must be
        symmetric and unchanged by rotating/shuffling the moving cluster."""
        out = Path(__file__).parents[1] / "examples/supported_fe16_tem/out"
        gas_b1 = read(out / "gas_relaxed.extxyz", 1)
        supported_b4 = read(out / "supported_MgO100.extxyz", 4)
        cluster = Atoms("Fe16", unwrapped_positions(supported_b4, range(192, 208)))
        self.assertGreater(match_atoms(gas_b1, cluster, random_starts=0)["rmsd"], 0.7)  # the old trap
        forward = match_atoms(gas_b1, cluster)["rmsd"]
        self.assertLess(forward, 0.30)
        self.assertAlmostEqual(forward, match_atoms(cluster, gas_b1)["rmsd"], delta=1e-3)
        perm = np.random.default_rng(7).permutation(16)
        rot = Rotation.random(random_state=11).as_matrix()
        shuffled = Atoms("Fe16", (cluster.positions @ rot.T + [3.0, -1.0, 8.0])[perm])
        self.assertAlmostEqual(forward, match_atoms(gas_b1, shuffled)["rmsd"], delta=1e-3)

    def test_mirror_image_is_not_superposed(self):
        chiral = Atoms("FeNOC", [[0, 0, 0], [1.9, 0, 0], [0.3, 1.8, 0.2], [0.2, 0.4, 1.7]])
        mirror = chiral.copy()
        mirror.positions[:, 0] *= -1
        self.assertGreater(match_atoms(chiral, mirror)["rmsd"], 0.1)

    def test_close_contact_flagged(self):
        bad = self.ref.copy()
        bad.positions[14] = bad.positions[13] + [0, 0, 0.6]  # N-O 0.6 A
        contacts = structure_report(bad)["contacts"]["close_contacts"]
        self.assertEqual([(c["i"], c["j"]) for c in contacts], [(13, 14)])


class SupportedClusterTests(unittest.TestCase):
    def test_cluster_unwrapped_across_the_cell_boundary(self):
        system, ns = fe4_on_mgo()
        part = Partition.from_specs(len(system), substrate=f"0:{ns}")
        raw = system.positions[part.cluster]
        self.assertGreater(np.ptp(raw[:, :2], axis=0).max(), 4.0)  # wrapped coordinates look torn apart
        pos = unwrapped_positions(system, part.cluster)
        d = np.linalg.norm(pos[:, None] - pos[None], axis=2)[np.triu_indices(4, 1)]
        np.testing.assert_allclose(sorted(d), sorted(Atoms("Fe4", pos).get_all_distances()[np.triu_indices(4, 1)]))
        self.assertLess(d.max(), 2.5)

    def test_site_assignment_invariant_to_translation_wrap_and_reordering(self):
        def sites(system, ns, order=None):
            if order is not None:
                system = system[order]
                inverse = np.argsort(order)
                substrate = ",".join(str(inverse[i]) for i in range(ns))
            else:
                substrate = f"0:{ns}"
            part = Partition.from_specs(len(system), substrate=substrate)
            rep = site_report(system, part.cluster, part.substrate, site_elements=["O"])
            return (rep["site_counts"], rep["n_in_contact"], rep["min_height"]["h"],
                    rep["closest_contact"]["pair"], rep["closest_contact"]["d"])

        a, ns = fe4_on_mgo()
        b, _ = fe4_on_mgo(shift=(3.3, -5.9))
        order = np.random.default_rng(1).permutation(len(b))
        self.assertEqual(sites(a, ns), sites(b, ns))
        self.assertEqual(sites(a, ns), sites(b, ns, order))
        counts, n, h, pair, d = sites(a, ns)
        self.assertEqual((counts, n, pair), ({"top:O": 1}, 1, "Fe-O"))
        # without restricting to the anion, the four Mg beside the Fe at 2.9 A also count
        part = Partition.from_specs(len(a), substrate=f"0:{ns}")
        self.assertEqual(site_report(a, part.cluster, part.substrate)["site_counts"], {"5-fold:Mg4O": 1})
        self.assertAlmostEqual(d, 2.0, places=3)
        self.assertAlmostEqual(h, 2.0, places=3)

    def test_rigid_shift_across_boundary_is_zero_displacement(self):
        a, ns = fe4_on_mgo()
        b, _ = fe4_on_mgo(shift=(3.3, -5.9))
        part = Partition.from_specs(len(a), substrate=f"0:{ns}")
        cmp = compare_supported(a, b, part)
        self.assertLess(cmp["cluster_displacement"]["max"], 1e-8)
        self.assertLess(cmp["cluster_trajectory_rmsd"], 1e-8)
        self.assertEqual(cmp["bond_changes_cluster_substrate"]["formed"], 0)

    def test_cli_writes_json(self):
        a, ns = fe4_on_mgo()
        b = a.copy()
        b.positions[-1, 2] += 0.4
        with tempfile.TemporaryDirectory() as tmp:
            write(Path(tmp, "a.extxyz"), [a, b])
            out = Path(tmp, "r.json")
            code = main(["geometry-check", f"{tmp}/a.extxyz@0", "--compare", f"{tmp}/a.extxyz@1",
                         "--substrate", f"0:{ns}", "--metal", "Fe", "-o", str(out)])
            self.assertEqual(code, 0)
            r = json.loads(out.read_text())
        self.assertIn("Geometry only", r["reference"]["caveat"])
        self.assertAlmostEqual(r["comparison"]["cluster_displacement"]["max"], 0.4, places=3)


if __name__ == "__main__":
    unittest.main()
