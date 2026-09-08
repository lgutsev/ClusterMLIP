import csv
import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from cluster_mlip.cli import build_parser
from cluster_mlip.dataset import read_labeled_extxyz
from cluster_mlip.gaussian import gaussian_job_complete, parse_force_frames, parse_final_force_frame
from cluster_mlip.slurm import _completion_function, SlurmConfig, prepare_slurm_batches

FIXTURE = Path(__file__).parent / 'fixtures' / 'force.log'
NORMAL = '\n Normal termination of Gaussian 09\n'


class GaussianCollectionTests(unittest.TestCase):
    def test_real_force_header_and_energy_geometry_pairing(self):
        text = FIXTURE.read_text().replace(' Forces (Hartrees/Bohr)\n', '', 1)
        # A geometry printed after the energy cannot be paired with that SCF.
        geometry = text[text.index(' Input orientation:'):text.index(' SCF Done:')]
        text = text.replace(' Center     Atomic                   Forces',
                            geometry.replace('0.750000', '9.000000') +
                            ' Center     Atomic                   Forces')
        frame = parse_final_force_frame(text, FIXTURE)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.record.atoms[1].x, 0.75)

    def test_rejects_force_atom_mismatch_and_truncated_final(self):
        text = FIXTURE.read_text()
        self.assertIsNone(parse_final_force_frame(
            text.replace('      2        1', '      2        8'), FIXTURE))
        truncated = text[:text.rfind(' -------------------------------------------------------------------')]
        self.assertIsNone(parse_final_force_frame(text + truncated, FIXTURE))

    def test_frames_are_independent(self):
        frames = parse_force_frames(FIXTURE.read_text() * 2, FIXTURE)
        self.assertEqual(len(frames), 2)
        frames[0].record.metadata['test'] = True
        self.assertNotIn('test', frames[1].record.metadata)

    def test_python_and_shell_completion_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / 'input.gjf'
            inp.write_text('# opt\n\n--Link1--\n# opt guess=read\n')
            out = root / 'output.log'
            cases = [(NORMAL, False), (NORMAL * 2, True),
                     (NORMAL + '\nError termination\n', False),
                     (NORMAL * 2 + '\nSCF Done: E(UHF) = -1\n', False)]
            for text, complete in cases:
                with self.subTest(text=text):
                    out.write_text(text)
                    result = subprocess.run(['bash', '-c', _completion_function() +
                        '\ngaussian_complete "$1" "$2"', 'check', str(inp), str(out)])
                    self.assertEqual(result.returncode == 0, complete)
                    self.assertEqual(gaussian_job_complete(text, 2), complete)
            out.write_text(NORMAL * 2)
            out.with_suffix('.rc').write_text('1\n')
            result = subprocess.run(['bash', '-c', _completion_function() +
                '\ngaussian_complete "$1" "$2"', 'check', str(inp), str(out)])
            self.assertNotEqual(result.returncode, 0)

    def test_spin_manifest_all_frames_and_failed_ladder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / 'spin_jobs.csv').open('w', newline='') as handle:
                fields = ['job_id', 'output', 'parent_record_id', 'intended_charge',
                          'intended_multiplicity', 'checkpoint', 'spin_plan_id']
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for mult in (3, 1):
                    writer.writerow(dict(job_id=f'm{mult}', output='ladder.log',
                        parent_record_id='parent', intended_charge=0,
                        intended_multiplicity=mult, checkpoint=f'm{mult}.chk', spin_plan_id='plan'))
            text = FIXTURE.read_text()
            log = root / 'ladder.log'
            log.write_text(text.replace('Multiplicity = 1', 'Multiplicity = 3') * 2
                           + NORMAL + text + NORMAL)
            args = build_parser().parse_args(['collect', str(root), '-o', str(root / 'data'),
                                             '--frames', 'all'])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(args.func(args), 0)
            frames = read_labeled_extxyz(root / 'data/all.extxyz')
            self.assertEqual([f.record.multiplicity for f in frames], [3, 3, 1])
            self.assertEqual(len({f.record.record_id for f in frames}), 3)
            self.assertTrue(all(f.record.metadata['spin_plan_id'] == 'plan' for f in frames))
            self.assertEqual(frames[-1].record.metadata['checkpoint'], 'm1.chk')
            sizes = [len(read_labeled_extxyz(root / f'data/{split}.extxyz'))
                     for split in ('train', 'valid', 'test')]
            self.assertEqual(sorted(sizes), [0, 0, 3])
            args.frames = 'final'
            with redirect_stdout(io.StringIO()):
                args.func(args)
            self.assertEqual(len(read_labeled_extxyz(root / 'data/all.extxyz')), 2)
            log.write_text(text.replace('Multiplicity = 1', 'Multiplicity = 3')
                           + '\n Stationary point found.\n' + NORMAL
                           + '\nError termination\n')
            with redirect_stdout(io.StringIO()):
                args.func(args)
            self.assertEqual(read_labeled_extxyz(root / 'data/all.extxyz'), [])
            self.assertEqual(json.loads((root / 'data/label_report.json').read_text())['n_frames'], 0)
            self.assertIn('incomplete Gaussian job', (root / 'data/failed_outputs.tsv').read_text())

            args.allow_partial = True
            args.frames = 'converged'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(args.func(args), 0)
            partial = read_labeled_extxyz(root / 'data/all.extxyz')
            self.assertEqual(len(partial), 1)
            self.assertFalse(partial[0].record.metadata['source_job_complete'])
            self.assertTrue(partial[0].record.metadata['spin_stage_normal_termination'])

            args.frames = 'all'
            with self.assertRaisesRegex(ValueError, 'requires --frames converged'):
                args.func(args)

    def test_resume_submits_partial_ladder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'ladder.gjf').write_text('# opt\n--Link1--\n# opt\n')
            (root / 'jobs.csv').write_text('job_id,input,output\nladder,ladder.gjf,ladder.log\n')
            prepare_slurm_batches(root, SlurmConfig())
            batch = root / 'slurm_batches/batch_0001'
            (batch / 'ladder.log').write_text(NORMAL)
            (batch / 'submit.sh').write_text('#!/bin/bash\necho submitted\n')
            (batch / 'submit.sh').chmod(0o755)
            env = dict(os.environ)
            env.pop('SLURM_JOB_ID', None)
            result = subprocess.run(['bash', str(root / 'submit_gaussian_batches.sh')],
                                    capture_output=True, text=True, env=env, check=True)
            self.assertIn('Submitted 1 batch job(s)', result.stdout)
            (batch / 'ladder.log').write_text(NORMAL * 2)
            result = subprocess.run(['bash', str(root / 'submit_gaussian_batches.sh')],
                                    capture_output=True, text=True, env=env, check=True)
            self.assertIn('skipped 1 complete batch(es)', result.stdout)

    def test_srun_failure_overrides_old_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'job.gjf').write_text('# force\n')
            (root / 'jobs.csv').write_text('job_id,input,output\njob,job.gjf,job.log\n')
            prepare_slurm_batches(root, SlurmConfig(gaussian_module='', concurrent_jobs=1))
            batch = root / 'slurm_batches/batch_0001'
            (batch / 'job.log').write_text(NORMAL)
            (root / 'srun').write_text('#!/bin/bash\nexit 42\n')
            (root / 'srun').chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                       RUN_POLICY='all', SLURM_JOB_ID='test',
                       CLUSTER_MLIP_CAMPAIGN_ROOT=str(root), CLUSTER_MLIP_BATCH_DIR=str(batch),
                       GAUSSIAN_SCRATCH_ROOT=str(root / 'scratch'), TMPDIR=str(root))
            result = subprocess.run(['bash', str(batch / 'run_batch.sbatch')],
                                    capture_output=True, text=True, env=env, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((batch / 'job.rc').read_text().strip(), '42')
            self.assertIn('complete=0 incomplete=1', result.stdout)

    def test_electronic_metadata_is_per_frame(self):
        text = FIXTURE.read_text().replace(' Forces (Hartrees/Bohr)',
            ' S**2 before annihilation 2.01 after 2.00\n'
            ' Mulliken charges and spin densities:\n'
            ' 1 H 0.0 0.7\n 2 H 0.0 -0.7\n'
            ' Sum of Mulliken charges = 0.0\n Forces (Hartrees/Bohr)', 1)
        frames = parse_force_frames(text + FIXTURE.read_text(), FIXTURE)
        self.assertEqual(frames[0].record.metadata['s2_after'], 2.0)
        self.assertEqual(frames[0].record.metadata['atomic_spins'], [[1, 'H', 0.7], [2, 'H', -0.7]])
        self.assertNotIn('atomic_spins', frames[1].record.metadata)
