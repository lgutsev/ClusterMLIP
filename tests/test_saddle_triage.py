import tempfile
import unittest
from pathlib import Path

from cluster_mlip.saddle_triage import (
    step_negative_eigenvalues,
    termination,
    triage_campaign_saddles,
    triage_output,
)
from test_route_audit import BrokenCampaign

ROUTE = ' #p ubpw91/6-311+G* opt=(ts,calcfc,noeigentest) freq\n'


def _step(x: float, force: float, eigenvalues: list[float]) -> str:
    """One Berny step: geometry, SCF, force table, then the Hessian printout."""
    lines = [
        ' Input orientation:',
        ' ' + '-' * 69,
        ' Center     Atomic     Atomic              Coordinates (Angstroms)',
        ' Number     Number      Type              X           Y           Z',
        ' ' + '-' * 69,
        '    1          1             0        0.000000    0.000000    0.000000',
        f'    2          1             0        {x:.6f}    0.000000    0.000000',
        ' ' + '-' * 69,
        f' SCF Done:  E(UBPW91) =  {-1.1 - x / 100:.10f}     A.U.',
        ' Forces (Hartrees/Bohr)',
        ' ' + '-' * 67,
        ' Center     Atomic                   Forces (Hartrees/Bohr)',
        ' Number     Number              X              Y              Z',
        ' ' + '-' * 67,
        f'      1        1           {force:.9f}    0.000000000    0.000000000',
        f'      2        1          {-force:.9f}    0.000000000    0.000000000',
        ' ' + '-' * 67,
        ' Berny optimization.',
    ]
    for start in range(0, len(eigenvalues), 5):
        chunk = eigenvalues[start:start + 5]
        lines.append('     Eigenvalues --- ' + ''.join(f'{v:10.5f}' for v in chunk))
    lines.append(' Search for a saddle point.')
    return '\n'.join(lines) + '\n'


def _log(steps: list[tuple[float, float, list[float]]], tail: str = '') -> str:
    header = ROUTE + ' Charge = 0 Multiplicity = 1\n'
    return header + ''.join(_step(*step) for step in steps) + tail


INDEX1 = [-0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
INDEX0 = [0.001, 0.02, 0.03, 0.04, 0.05, 0.06]
INDEX2 = [-0.02, -0.01, 0.03, 0.04, 0.05, 0.06]
STEPS_EXCEEDED = (' Optimization stopped.\n    -- Number of steps exceeded,  NStep=  20\n'
                  ' Error termination via Lnk1e in /usr/local/g09/l103.exe\n')
FORMBX = (' FormBX had a problem.\n Tors failed for dihedral 1 - 2 - 3 - 4\n'
          ' Error termination via Lnk1e in /usr/local/g09/l103.exe\n')


def _triage(text: str, state: str = 'failed', final=None, config='transition_state'):
    return triage_output(text, Path('job.log'), config, state, final)


class HessianParsingTests(unittest.TestCase):
    def test_one_printout_per_step_spanning_continuation_lines(self):
        text = _log([(0.75, 0.01, INDEX1), (0.76, 0.005, INDEX2 + [0.07, 0.08])])
        self.assertEqual([count for _, count in step_negative_eigenvalues(text)], [1, 2])

    def test_orbital_eigenvalue_lines_are_not_hessian_printouts(self):
        text = ' Alpha  occ. eigenvalues --   -0.50000  -0.30000\n'
        self.assertEqual(step_negative_eigenvalues(text), [])

    def test_termination_kinds(self):
        self.assertEqual(termination(STEPS_EXCEEDED, 'failed')[0], 'steps_exhausted')
        self.assertEqual(termination(FORMBX, 'failed')[0], 'internal_coordinate_failure')
        self.assertEqual(
            termination(' Convergence failure -- run terminated.\n'
                        ' Error termination via Lnk1e in /g09/l502.exe\n', 'failed'),
            ('scf_failure', 'l502'))
        self.assertEqual(termination('', 'incomplete')[0], 'no_termination')
        self.assertEqual(termination(' Normal termination\n', 'complete')[0], 'normal')


class RecommendationTests(unittest.TestCase):
    def test_out_of_steps_near_the_saddle_restarts_from_the_lowest_force_index1_frame(self):
        text = _log([(0.75, 0.02, INDEX1), (0.76, 0.004, INDEX1), (0.77, 0.03, INDEX0)],
                    STEPS_EXCEEDED)
        verdict = _triage(text)
        self.assertEqual(verdict['termination'], 'steps_exhausted')
        self.assertEqual(verdict['recommendation'], 'restart_best_frame')
        self.assertEqual(verdict['best_frame_index'], 1)
        self.assertEqual(verdict['index1_frames'], 2)
        self.assertEqual(verdict['last_negative_eigenvalues'], 0)
        self.assertIn('ended with 0 negative', verdict['reason'])

    def test_broken_internal_coordinates_restart_in_cartesians(self):
        verdict = _triage(_log([(0.75, 0.01, INDEX1)], FORMBX))
        self.assertEqual(verdict['recommendation'], 'restart_best_frame_cartesian')
        self.assertEqual(verdict['failed_torsion_atoms'], '1-2-3-4')

    def test_coordinates_broken_even_in_cartesians_need_a_nudge(self):
        text = _log([(0.75, 0.01, INDEX1)], FORMBX).replace(
            'opt=(ts,calcfc,noeigentest)', 'opt=(ts,calcfc,noeigentest,cartesian)')
        self.assertEqual(_triage(text)['recommendation'], 'mode_displacement')

    def test_surplus_negative_modes_without_index1_ask_for_mode_displacement(self):
        verdict = _triage(_log([(0.75, 0.01, INDEX2), (0.76, 0.01, INDEX2)]), 'incomplete')
        self.assertEqual(verdict['termination'], 'no_termination')
        self.assertEqual(verdict['recommendation'], 'mode_displacement')

    def test_minimum_like_search_needs_an_mlip_guess(self):
        verdict = _triage(_log([(0.75, 0.01, INDEX0), (0.76, 0.01, INDEX0)]), 'incomplete')
        self.assertEqual(verdict['recommendation'], 'mlip_ts_guess')

    def test_scf_failure_before_any_force_is_an_scf_problem(self):
        text = ROUTE + ' Charge = 0 Multiplicity = 1\n Convergence failure -- run terminated.\n' \
            ' Error termination via Lnk1e in /g09/l502.exe\n'
        self.assertEqual(_triage(text)['recommendation'], 'scf_rescue')

    def test_input_rejected_by_gaussian_is_regenerated_not_rescued(self):
        text = ROUTE + ' Charge = 0 Multiplicity = 9\n' \
            ' Atom specifications unexpectedly found in input stream.\n' \
            ' Error termination via Lnk1e in /g09/l101.exe at Wed Sep 23 18:44:04 2026.\n'
        verdict = _triage(text)
        self.assertEqual((verdict['termination'], verdict['error_link']), ('input_error', 'l101'))
        self.assertEqual(verdict['recommendation'], 'regenerate_input')

    def test_unreadable_checkpoint_is_regenerated(self):
        text = ROUTE + ' Wr Pntr         40448\n Error termination in NtrErr:\n'
        verdict = _triage(text)
        self.assertEqual(verdict['termination'], 'checkpoint_error')
        self.assertEqual(verdict['recommendation'], 'regenerate_input')

    def test_converged_saddles(self):
        text = _log([(0.75, 0.0001, INDEX1)], ' Normal termination of Gaussian 09\n')
        self.assertEqual(_triage(text, 'complete', 1)['recommendation'], 'none')
        self.assertEqual(_triage(text, 'complete', None)['recommendation'], 'freq_check')
        self.assertEqual(_triage(text, 'complete', 2)['recommendation'], 'mode_displacement')
        self.assertIn('converged to a minimum',
                      _triage(_log([(0.75, 0.0001, INDEX0)]), 'complete', 0)['reason'])

    def test_judges_the_last_searching_section_not_a_trailing_force_stage(self):
        search = _log([(0.75, 0.01, INDEX2), (0.76, 0.002, INDEX1)])
        force_stage = (' Charge = 0 Multiplicity = 1\n'
                       + _step(0.76, 0.002, []).replace(' Berny optimization.\n', ''))
        verdict = _triage(search + force_stage, 'incomplete')
        self.assertEqual(verdict['section_frames'], 2)
        self.assertEqual(verdict['recommendation'], 'restart_best_frame')
        self.assertEqual(verdict['best_frame_index'], 1)


class CampaignTriageTests(BrokenCampaign, unittest.TestCase):
    def test_campaign_triage_reports_failed_saddles_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, batch, stem = self.campaign(Path(tmp), break_route=False, finished=False)
            (batch / f'{stem}.log').write_text(
                _log([(0.75, 0.02, INDEX1), (0.76, 0.004, INDEX1)], STEPS_EXCEEDED))
            result = triage_campaign_saddles(root, Path(tmp) / 'report')
            summary = result['summary']
            self.assertEqual(summary['saddles_started'], 1)
            self.assertEqual(summary['triaged'], 1)
            self.assertEqual(summary['by_recommendation'], {'restart_best_frame': 1})
            self.assertTrue((Path(tmp) / 'report' / 'saddle_triage.csv').is_file())
            # Read-only: no monitoring/ written into the campaign.
            self.assertFalse((root / 'monitoring').exists())


if __name__ == '__main__':
    unittest.main()
