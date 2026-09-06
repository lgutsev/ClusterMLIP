import csv
import os
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.batch_progress import write_batch_progress
from cluster_mlip.cli import build_parser
from cluster_mlip.slurm import SlurmConfig, prepare_slurm_batches

FIXTURE = Path(__file__).parent / 'fixtures/force.log'
NORMAL = '\nNormal termination of Gaussian 09\n'


class BatchProgressTests(unittest.TestCase):
    def campaign(self, root):
        with (root / 'spin_jobs.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['job_id', 'input', 'output', 'intended_charge', 'intended_multiplicity'])
            writer.writeheader()
            for index in range(4):
                name = f'job{index}'
                (root / f'{name}.gjf').write_text('%nprocshared=12\n# opt\n--Link1--\n# opt guess=read\n')
                for mult in (3, 1):
                    writer.writerow(dict(job_id=f'{name}-m{mult}', input=f'{name}.gjf',
                                         output=f'{name}.log', intended_charge=0, intended_multiplicity=mult))
        prepare_slurm_batches(root, SlurmConfig(jobs_per_batch=2, cpus_per_job=12))
        return root / 'slurm_batches'

    def test_pending_mapping_and_stage_vs_input_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batches = self.campaign(root)
            text = FIXTURE.read_text()
            (batches / 'batch_0001/job0.log').write_text(
                text.replace('Multiplicity = 1', 'Multiplicity = 3') + NORMAL + text + NORMAL)
            (batches / 'batch_0001/job1.log').write_text(
                text.replace('Multiplicity = 1', 'Multiplicity = 3') + NORMAL + 'Error termination\n')
            result = write_batch_progress(root, audit=True)
            self.assertEqual(result['batches'][0]['planned'], 2)
            self.assertEqual(result['batches'][0]['complete'], 1)
            self.assertEqual(result['batches'][0]['failed'], 1)
            self.assertEqual(result['batches'][0]['completed_stages'], 3)
            self.assertEqual(result['batches'][1]['not_started'], 2)
            self.assertEqual(result['jobs'][0]['observed_multiplicity'], 1)
            self.assertEqual(result['summary']['errors'], 1)
            selected = write_batch_progress(root, start=2, end=2)
            self.assertEqual(selected['summary']['planned'], 2)
            self.assertEqual(selected['batches'][0]['batch'], 'batch_0002')
            self.assertTrue((root / 'monitoring/job_progress.csv').is_file())

    def test_audit_old_guess_cpu_and_missing_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batches = self.campaign(root)
            (batches / 'batch_0001/job0.gjf').write_text('%nprocshared=16\n# opt Guess=(Read,Always)\n')
            (batches / 'batch_0002/inputs.txt').unlink()
            result = write_batch_progress(root, audit=True)
            details = '\n'.join(i['detail'] for i in result['issues'])
            self.assertIn('Contradictory Guess', details)
            self.assertIn('CPUs/job', details)
            self.assertIn('Missing or empty inputs.txt', details)

    def test_new_attempt_ignores_old_failure_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = self.campaign(root) / 'batch_0001'
            for suffix, text in [('.log', 'Error termination'), ('.rc', '1'), ('.status', 'ERROR 1'), ('.finished', 'yesterday')]:
                path = batch / ('job0' + suffix)
                path.write_text(text)
                os.utime(path, (1, 1))
            (batch / 'job0.started').write_text('now')
            result = write_batch_progress(root)
            self.assertEqual(result['jobs'][0]['state'], 'activity_unconfirmed')
            self.assertFalse(result['summary']['scheduler_queried'])

    def test_cli_and_range(self):
        args = build_parser().parse_args(['campaign-status', 'campaign', '--by-batch', '--audit', '--start', '2', '--end', '4'])
        self.assertTrue(args.audit)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.campaign(root)
            with self.assertRaises(ValueError):
                write_batch_progress(root, start=0)
