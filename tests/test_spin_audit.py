import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cluster_mlip.cli import main
from cluster_mlip.dataset import write_labeled_extxyz
from cluster_mlip.gaussian import parse_force_frames
from cluster_mlip.models import Atom, Record, LabeledFrame
from cluster_mlip.spin_audit import compare, spins, verify_raw


def frame():
    xyz = [(0, 0, 0), (1.9, 0, 0), (0.3, 2.2, 0), (0.2, 0.4, 2.6)]
    r = Record('a', 'synthetic', [Atom('Fe', *x) for x in xyz], 0, 1, 'test')
    r.metadata['atomic_spins'] = [[i+1, 'Fe', s] for i, s in enumerate([2, 2, -2, -2])]
    return LabeledFrame(r, -10, [(0.1, 0.2, 0.3), (-0.2, 0.1, 0), (0.1, -0.3, 0.2), (0, 0, -0.5)], Path('test.log'))


class SpinAuditTests(unittest.TestCase):
    def compare(self, a, b):
        return compare(a, b, 1e-5, 0.03, 100000)

    def test_rotation_permutation_translation_and_global_reversal(self):
        a, b = frame(), frame()
        p = [2, 0, 3, 1]
        q = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        b.record.atoms = [Atom('Fe', *(np.array([a.record.atoms[i].x, a.record.atoms[i].y, a.record.atoms[i].z]) @ q + 5)) for i in p]
        b.forces_ev_ang = (np.array(a.forces_ev_ang)[p] @ q).tolist()
        b.record.metadata['atomic_spins'] = [[j+1, 'Fe', -spins(a)[i]] for j, i in enumerate(p)]
        c = self.compare(a, b)
        self.assertEqual(c['geometry'], 'exact')
        self.assertLess(c['force_component_rms_eV_A'], 1e-10)
        self.assertLess(c['spin_rms'], 1e-10)

    def test_same_spin_histogram_different_sites(self):
        a, b = frame(), frame()
        b.record.metadata['atomic_spins'] = [[i+1, 'Fe', s] for i, s in enumerate([2, -2, 2, -2])]
        c = self.compare(a, b)
        self.assertGreater(c['spin_rms'], 1)

    def test_symmetry_equivalent_patterns(self):
        a, b = frame(), frame()
        a.record.atoms = b.record.atoms = [Atom('Fe', *x) for x in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]]
        b.record.metadata['atomic_spins'] = [[i+1, 'Fe', s] for i, s in enumerate([2, -2, 2, -2])]
        c = self.compare(a, b)
        self.assertEqual(c['spin_rms'], 0)
        self.assertGreater(c['mappings'], 1)

    def test_near_and_capped(self):
        a, b = frame(), frame()
        atom = b.record.atoms[0]
        b.record.atoms[0] = Atom('Fe', 0.01, atom.y, atom.z)
        self.assertEqual(self.compare(a, b)['geometry'], 'near')
        self.assertTrue(compare(a, b, 1e-5, 0.03, 1)['search_capped'])
        b.record.atoms[0] = Atom('Fe', 10, 0, 0)
        self.assertIsNone(self.compare(a, b))

    def test_partial_spins_are_missing(self):
        a = frame()
        a.record.metadata['atomic_spins'][0][0] = 2
        self.assertIsNone(spins(a))

    def test_no_inheritance_or_forward_fill(self):
        text = (Path(__file__).parent / 'fixtures/force.log').read_text()
        seed = frame().record
        seed.metadata.update(s2_before=12, s2_after=12)
        parsed = parse_force_frames(text, Path('x.log'), seed)
        self.assertNotIn('atomic_spins', parsed[0].record.metadata)
        self.assertNotIn('s2_before', parsed[0].record.metadata)
        block = '\n Mulliken charges and spin densities:\n              1             2\n     1 H  0.0  0.5\n     2 H  0.0 -0.5\n Sum of Mulliken charges = 0.0\n'
        first = text.replace(' Forces (Hartrees/Bohr)', block+' Forces (Hartrees/Bohr)', 1)
        parsed = parse_force_frames(first+text, Path('x.log'))
        self.assertIsNotNone(spins(parsed[0]))
        self.assertIsNone(spins(parsed[1]))

    def test_raw_frame_provenance(self):
        text = (Path(__file__).parent / 'fixtures/force.log').read_text()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'x.log').write_text(text)
            a = parse_force_frames(text, root/'x.log')[0]
            a.record.metadata.update(gaussian_output='x.log', collection_campaign=str(root))
            self.assertEqual(verify_raw(a, None, {}), 'raw_spin_missing')
            a.record.metadata['atomic_spins'] = [[1, 'H', 0.5], [2, 'H', -0.5]]
            self.assertEqual(verify_raw(a, None, {}), 'spin_not_supported_by_raw_frame')
            a.energy_ev += 1
            self.assertEqual(verify_raw(a, None, {}), 'label_mismatch')

    def test_cli_classifies_conflicts_and_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, c = frame(), frame(), frame()
            b.record.record_id = 'b'
            b.energy_ev += 0.2
            b.record.metadata['atomic_spins'] = [[i+1, 'Fe', s] for i, s in enumerate([2, -2, 2, -2])]
            c.record.record_id = 'c'
            c.record.multiplicity = 3
            c.record.metadata['atomic_spins'] = [[i+1, 'Fe', s] for i, s in enumerate([2, 2, -1, -1])]
            write_labeled_extxyz([a, b, c], root/'all.extxyz')
            args = ['audit-spin-labels', str(root/'all.extxyz'), '-o', str(root/'out')]
            self.assertEqual(main(args), 0)
            report = json.loads((root/'out/summary.json').read_text())
            self.assertEqual(report['pair_evidence']['label_conflict_with_distinct_spins'], 1)
            self.assertEqual(report['pair_evidence']['different_multiplicity_control'], 2)
            with self.assertRaises(ValueError):
                main(args)
            self.assertEqual(main(args[:-1]+[str(root/'limited'), '--max-pairs', '1']), 2)

    def test_direct_raw_recursive_logs_and_state_sections(self):
        text = (Path(__file__).parent / 'fixtures/force.log').read_text()
        block = '\n Mulliken charges and spin densities:\n              1             2\n     1 H  0.0  0.5\n     2 H  0.0 -0.5\n Sum of Mulliken charges = 0.0\n'
        first = text.replace(' Forces (Hartrees/Bohr)', block+' Forces (Hartrees/Bohr)', 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root/'logs/batch_0001'
            batch.mkdir(parents=True)
            (batch/'before-restart.log').write_text(first+' Normal termination of Gaussian 09\n'+text+' Error termination\n')
            (batch/'scheduler.out').write_text('Slurm output, not Gaussian')
            args = ['audit-spin-labels', str(root/'logs'), '--gaussian', '--formula', 'H2', '-o', str(root/'out')]
            self.assertEqual(main(args), 0)
            report = json.loads((root/'out/summary.json').read_text())
            self.assertEqual(report['frames'], 2)
            self.assertEqual(report['frames_with_complete_spins'], 1)
            self.assertEqual(report['frames_in_normally_terminated_sections'], 1)
            self.assertEqual(report['raw_logs'], 2)
            self.assertEqual(report['raw_verification']['direct_raw'], 1)
            self.assertTrue((root/'out/input_logs.csv').exists())

    def test_pair_fingerprint_is_not_structural_proof(self):
        a, b = frame(), frame()
        a.record.atoms = [Atom('Fe', i, 0, 0) for i in [0, 1, 4, 10, 12, 17]]
        b.record.atoms = [Atom('Fe', i, 0, 0) for i in [0, 1, 8, 11, 13, 17]]
        self.assertIsNone(self.compare(a, b))

    def test_force_only_conflict_and_rank_deficiency(self):
        a, b = frame(), frame()
        b.forces_ev_ang = (np.array(a.forces_ev_ang) * 4).tolist()
        self.assertGreater(self.compare(a, b)['force_component_rms_eV_A'], 0.05)
        a.record.atoms = b.record.atoms = [Atom('Fe', i, 0, 0) for i in range(4)]
        self.assertIsNone(self.compare(a, b)['force_component_rms_eV_A'])


if __name__ == '__main__':
    unittest.main()
