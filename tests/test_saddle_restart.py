import csv
import io
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from cluster_mlip.cli import build_parser
from cluster_mlip.restart import SaddleStage, _rewrite_saddle_stage, prepare_spin_restarts
from cluster_mlip.routes import checkpoint_stages_with_coordinates
from cluster_mlip.saddle_restart import restart_saddles
from test_irc_force_labels import FE2O2, W2_ROUTE, W2Campaign

TS_ROUTE = W2_ROUTE.replace(' Opt ', ' Opt=(TS,CalcFC,NoEigenTest) Freq ')
TS_LATER = TS_ROUTE.replace(' Pop=Regular', ' Geom=Checkpoint Guess=Read Pop=Regular')
_Z = {'Fe': 26, 'O': 8}
_RULE = ' ' + '-' * 69
INDEX1 = [-0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
INDEX0 = [0.001, 0.02, 0.03, 0.04, 0.05, 0.06]
STEPS_EXCEEDED = (' Optimization stopped.\n    -- Number of steps exceeded,  NStep=  20\n'
                  ' Error termination via Lnk1e in /usr/local/g09/l103.exe\n')
FORMBX = (' FormBX had a problem.\n Tors failed for dihedral 1 - 2 - 3 - 4\n'
          ' Error termination via Lnk1e in /usr/local/g09/l103.exe\n')


def _step(shift: float, force: float, eigenvalues, orientation='Input') -> str:
    """One Berny step of Fe2O2 with the second Fe moved by ``shift``."""
    atoms = [(s, x + (shift if i == 1 else 0.0), y, z) for i, (s, x, y, z) in enumerate(FE2O2)]
    lines = [f'                         {orientation} orientation:', _RULE,
             ' Center     Atomic      Atomic             Coordinates (Angstroms)',
             ' Number     Number       Type             X           Y           Z', _RULE]
    lines += [f' {i + 1:6d} {_Z[s]:10d}           0     {x:11.6f} {y:11.6f} {z:11.6f}'
              for i, (s, x, y, z) in enumerate(atoms)]
    lines += [_RULE, ' SCF Done:  E(UBPW91) =  -2677.5000000000     A.U. after   20 cycles',
              ' Forces (Hartrees/Bohr)', ' ' + '-' * 67,
              ' Center     Atomic                   Forces (Hartrees/Bohr)',
              ' Number     Number              X              Y              Z', ' ' + '-' * 67]
    lines += [f' {i + 1:6d} {_Z[s]:8d}       {force if i == 0 else -force if i == 1 else 0.0:14.9f}'
              f'    0.000000000    0.000000000' for i, (s, *_r) in enumerate(atoms)]
    lines += [' ' + '-' * 67, ' Berny optimization.',
              '     Eigenvalues --- ' + ''.join(f'{v:10.5f}' for v in eigenvalues),
              ' Search for a saddle point.']
    return '\n'.join(lines) + '\n'


def _finished_stage(mult: int) -> str:
    return (f' Charge =  0 Multiplicity = {mult}\n' + _step(0.0, 0.00001, INDEX1)
            + ' Stationary point found.\n Normal termination of Gaussian 09\n')


def _failed_stage(mult: int, tail: str = STEPS_EXCEEDED, orientation='Input') -> str:
    # Frame 1 (shift 0.05) is index 1 with the smallest force; the search then drifts.
    return (f' Charge =  0 Multiplicity = {mult}\n'
            + _step(0.02, 0.020, INDEX1, orientation) + _step(0.05, 0.004, INDEX1, orientation)
            + _step(0.30, 0.030, INDEX0, orientation) + tail)


class RewriteStageTests(unittest.TestCase):
    STAGE = SaddleStage(atoms=[('Fe', 0.0, 0.0, 0.0), ('O', 1.6, 0.0, 0.0)],
                        charge=0, multiplicity=9, frame_index=3, max_step=10)
    SECTION = ('%oldchk=a.chk\n%chk=b.chk\n%mem=24GB\n'
               '#p UBPW91/6-311++G* SCF=(VShift=5,MaxCyc=200) NoSymm '
               'Opt=(TS,RCFC,NoEigenTest,MaxCycles=80) Freq IOP(5/13=1,5/36=1) '
               'Geom=Checkpoint Guess=Read Pop=Regular\n\ntitle\n\n0 9\n\n')

    def test_route_keeps_everything_but_the_search_options(self):
        text = _rewrite_saddle_stage(self.SECTION, oldchk='seed.chk', chk='new.chk',
                                     stage=self.STAGE)
        route = next(line for line in text.splitlines() if line.startswith('#'))
        self.assertIn('Opt=(TS,CalcFC,NoEigenTest,MaxCycles=80,MaxStep=10)', route)
        self.assertNotIn('RCFC', route)
        self.assertNotIn('Geom=', route)
        for kept in ('UBPW91/6-311++G*', 'SCF=(VShift=5,MaxCyc=200)', 'NoSymm', 'Freq',
                     'IOP(5/13=1,5/36=1)', 'Guess=Read'):
            self.assertIn(kept, route)
        self.assertTrue(route.endswith('Pop=Regular'), route)
        self.assertEqual(route.count('Freq'), 1)
        self.assertIn('%oldchk=seed.chk\n%chk=new.chk\n%mem=24GB', text)
        self.assertRegex(text, r'(?m)^0 9\nFe\s+0\.0+\s+0\.0+\s+0\.0+\nO\s+1\.6')

    def test_cartesian_and_a_stage_that_had_its_own_atoms(self):
        section = self.SECTION.replace(' Geom=Checkpoint Guess=Read', '').replace(
            '0 9\n\n', '0 9\nFe 9 9 9\nO 8 8 8\n\n')
        stage = SaddleStage(self.STAGE.atoms, 0, 9, 3, cartesian=True, max_step=None)
        text = _rewrite_saddle_stage(section, oldchk='seed.chk', chk='new.chk', stage=stage)
        route = next(line for line in text.splitlines() if line.startswith('#'))
        self.assertIn('Opt=(TS,CalcFC,NoEigenTest,MaxCycles=80,Cartesian)', route)
        self.assertIn('Guess=Read', route)
        self.assertNotIn('9 9 9', text)
        self.assertEqual(len(re.findall(r'(?m)^(Fe|O)\s', text)), 2)

    def test_a_frame_for_another_multiplicity_is_refused(self):
        with self.assertRaises(ValueError):
            _rewrite_saddle_stage(self.SECTION.replace('0 9', '0 7'), oldchk='a', chk='b',
                                  stage=self.STAGE)


class RestartSaddleTests(W2Campaign, unittest.TestCase):
    NAME = 'ts__fe2o2__transition-state__q0-m11__reference__ttt'

    def _campaign(self, root, log):
        campaign = self.build(root, [
            {'name': self.NAME, 'config_type': 'transition_state', 'batch': 'batch_0006',
             'source': 'warehouse2/Fe2O2_ts_1.log',
             'routes': [TS_ROUTE, TS_LATER, TS_LATER], 'checkpoints': True},
        ])
        batch = campaign / 'slurm_batches' / 'batch_0006'
        (batch / f'{self.NAME}.log').write_text(log)
        return campaign, batch

    def _new_input(self, campaign):
        with (campaign / 'spin_jobs.csv').open(newline='') as handle:
            active = [row for row in csv.DictReader(handle)
                      if row.get('submission_active') != 'false']
        return (campaign / active[0]['input']).read_text(), active

    def test_dry_run_plans_and_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batch = self._campaign(Path(tmp), _failed_stage(11))
            before = sorted((str(p), p.stat().st_size) for p in campaign.rglob('*') if p.is_file())
            result = restart_saddles(campaign, dry_run=True)
            self.assertEqual(result['input_count'], 1)
            self.assertEqual(result['candidates'][0]['frame_index'], 1)
            self.assertEqual(result['plan'][0]['mode'], 'saddle_best_frame_restart')
            after = sorted((str(p), p.stat().st_size) for p in campaign.rglob('*') if p.is_file())
            self.assertEqual(after, before)

    def test_restart_from_the_best_frame_of_the_first_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batch = self._campaign(Path(tmp), _failed_stage(11))
            result = restart_saddles(campaign)
            self.assertEqual((result['input_count'], result['stage_count']), (1, 3))
            text, active = self._new_input(campaign)
            self.assertEqual([row['intended_multiplicity'] for row in active], ['11', '9', '7'])
            stages = re.split(r'(?m)^\s*--Link1--\s*$', text)
            first_route = next(l for l in stages[0].splitlines() if l.startswith('#'))
            self.assertIn('Opt=(TS,CalcFC,NoEigenTest,MaxStep=10)', first_route)
            self.assertNotIn('Geom=', first_route)
            self.assertIn('Guess=Read', first_route)
            # The best frame (second Fe moved by 0.05), not the last one (0.30).
            self.assertRegex(stages[0], r'(?m)^Fe\s+2\.2500')
            self.assertNotRegex(stages[0], r'(?m)^Fe\s+2\.5000')
            self.assertEqual(len(re.findall(r'(?m)^(Fe|O)\s', stages[0])), 4)
            # Later stages continue the ladder from the checkpoint, unchanged otherwise.
            for later in stages[1:]:
                route = next(l for l in later.splitlines() if l.startswith('#'))
                self.assertEqual(route, TS_LATER)
                self.assertEqual(len(re.findall(r'(?m)^(Fe|O)\s', later)), 0)
            self.assertEqual(text.count('IOP(5/13=1,5/36=1,8/11=1)'), 3)
            self.assertEqual(checkpoint_stages_with_coordinates(text), [])
            self.assertTrue(text.endswith('\n\n'))
            # The usual restart bookkeeping happened.
            self.assertFalse((batch / f'{self.NAME}.log').exists())
            self.assertTrue((batch / f'{self.NAME}__before-restart01.log').is_file())
            listed = (batch / 'inputs.txt').read_text().split()
            self.assertEqual(listed, [Path(active[0]['input']).name])
            self.assertTrue((campaign / active[0]['restart_seed_checkpoint']).is_file())

    def test_failure_in_a_later_stage_restarts_that_stage_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _finished_stage(11) + _failed_stage(9)
            campaign, _ = self._campaign(Path(tmp), log)
            result = restart_saddles(campaign)
            self.assertEqual((result['input_count'], result['stage_count']), (1, 2))
            text, active = self._new_input(campaign)
            self.assertEqual([row['intended_multiplicity'] for row in active], ['9', '7'])
            first = re.split(r'(?m)^\s*--Link1--\s*$', text)[0]
            self.assertRegex(first, r'(?m)^0 9$')
            self.assertRegex(first, r'(?m)^Fe\s+2\.2500')

    def test_broken_internal_coordinates_restart_in_cartesians(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _ = self._campaign(Path(tmp), _failed_stage(11, FORMBX))
            result = restart_saddles(campaign, max_step=None)
            self.assertEqual(result['plan'][0]['saddle_cartesian'], 'true')
            text, _ = self._new_input(campaign)
            self.assertIn('Opt=(TS,CalcFC,NoEigenTest,Cartesian)', text)

    def test_frames_in_the_standard_orientation_are_not_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _ = self._campaign(
                Path(tmp), _failed_stage(11, orientation='Standard'))
            result = restart_saddles(campaign, dry_run=True)
            self.assertEqual(result['input_count'], 0)
            self.assertIn('orientation', result['not_chosen'][0]['reason'])

    def _routefixed(self, campaign, batch):
        """Like a W2 route-fix relaunch made before the manifest followed the
        rename: the input writes ``-routefix01`` checkpoints, the manifest still
        names the pre-relaunch files."""
        for path in (campaign / 'inputs' / f'{self.NAME}.gjf', batch / f'{self.NAME}.gjf'):
            text = re.sub(r'(?m)^(%(?:old)?chk=.*)\.chk$', r'\1-routefix01.chk', path.read_text())
            path.write_text(text, newline='\n')
        for stage, mult in enumerate(self.MULTS):
            (batch / f'{self.NAME}-s{stage:02d}-m{mult}.chk').write_bytes(b'before the route fix')

    def test_seed_is_the_checkpoint_the_input_wrote_not_the_manifest_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batch = self._campaign(Path(tmp), _finished_stage(11) + _failed_stage(9))
            self._routefixed(campaign, batch)
            (batch / f'{self.NAME}-s00-m11-routefix01.chk').write_bytes(b'ts m11')
            (batch / f'{self.NAME}-s01-m9-routefix01.chk').write_bytes(b'ts m9')
            result = restart_saddles(campaign)
            self.assertEqual(result['input_count'], 1, result['skipped'])
            text, active = self._new_input(campaign)
            self.assertEqual(Path(active[0]['restart_source_checkpoint']).name,
                             f'{self.NAME}-s01-m9-routefix01.chk')
            seed = campaign / active[0]['restart_seed_checkpoint']
            self.assertEqual(seed.read_bytes(), b'ts m9')
            self.assertEqual(seed.name, f'{self.NAME}-s01-m9-routefix01-seed-r01.chk')
            self.assertEqual([row['checkpoint'] for row in active],
                             [f'{self.NAME}-s01-m9-routefix01-r01.chk',
                              f'{self.NAME}-s02-m7-routefix01-r01.chk'])
            self.assertIn(f'%oldchk={seed.name}\n%chk={active[0]["checkpoint"]}', text)

    def test_predecessor_is_the_previous_stage_of_the_same_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batch = self._campaign(Path(tmp), _finished_stage(11) + _failed_stage(9))
            self._routefixed(campaign, batch)
            (batch / f'{self.NAME}-s00-m11-routefix01.chk').write_bytes(b'ts m11')
            restart_saddles(campaign)
            _, active = self._new_input(campaign)
            self.assertEqual((campaign / active[0]['restart_seed_checkpoint']).read_bytes(),
                             b'ts m11')

    def test_pre_relaunch_checkpoints_are_never_the_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batch = self._campaign(Path(tmp), _finished_stage(11) + _failed_stage(9))
            self._routefixed(campaign, batch)
            result = restart_saddles(campaign, dry_run=True)
            self.assertEqual(result['input_count'], 0)
            self.assertEqual(result['skipped'][0]['reason'], 'saddle_restart_checkpoint_missing')

    def test_restart_names_already_on_disk_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batch = self._campaign(Path(tmp), _failed_stage(11))
            (batch / f'{self.NAME}-s01-m9-r01.chk').write_bytes(b'another attempt')
            with self.assertRaises(FileExistsError):
                restart_saddles(campaign, dry_run=True)

    def test_exclusion_and_the_plain_restart_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _ = self._campaign(Path(tmp), _failed_stage(11))
            name = self.NAME + '.gjf'
            self.assertEqual(restart_saddles(campaign, dry_run=True,
                                             exclude_inputs={name})['input_count'], 0)
            plain = prepare_spin_restarts(campaign, start=6, end=6, dry_run=True,
                                          exclude_inputs={name})
            self.assertEqual(plain['input_count'], 0)
            self.assertEqual(plain['skipped'][0]['reason'], 'excluded')

    def test_cli_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _ = self._campaign(Path(tmp), _failed_stage(11))
            args = build_parser().parse_args(['restart-saddles', str(campaign), '--dry-run'])
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                self.assertEqual(args.func(args), 0)
            self.assertIn('Would restart 1 of 1 candidate(s)', out.getvalue())
            self.assertIn('frame 1', out.getvalue())


if __name__ == '__main__':
    unittest.main()
