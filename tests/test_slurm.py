import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from cluster_mlip.cli import build_parser
from cluster_mlip.slurm import SlurmConfig, prepare_slurm_array


class SlurmPreparationTests(unittest.TestCase):
    def test_prepare_slurm_help_renders(self):
        parser = build_parser()
        prepare_slurm = parser._subparsers._group_actions[0].choices["prepare-slurm"]
        self.assertIn("--allow-nproc-mismatch", prepare_slurm.format_help())

    def _campaign(self, root: Path, count: int = 7) -> Path:
        campaign = root / "campaign"
        campaign.mkdir()
        with (campaign / "jobs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["job_id", "input", "output"])
            writer.writeheader()
            for index in range(count):
                name = f"job_{index:03d}.gjf"
                (campaign / name).write_text(f"%chk=job_{index:03d}.chk\n# force\n", encoding="utf-8")
                writer.writerow({"job_id": f"job_{index:03d}", "input": name, "output": name[:-4] + ".log"})
        return campaign

    def test_prepares_batched_resumable_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._campaign(Path(tmp))
            plan = prepare_slurm_array(
                campaign,
                SlurmConfig(
                    jobs_per_batch=3,
                    concurrent_jobs=2,
                    cpus_per_job=8,
                    array_concurrency=2,
                ),
            )
            self.assertEqual(plan["input_count"], 7)
            self.assertEqual(plan["batch_count"], 3)
            batches = sorted((campaign / "slurm_batches").glob("batch_*.txt"))
            self.assertEqual([len(path.read_text().splitlines()) for path in batches], [3, 3, 1])
            array_text = (campaign / "run_gaussian_array.sbatch").read_text()
            self.assertIn("#SBATCH --array=1-3%2", array_text)
            self.assertIn("#SBATCH --ntasks-per-node=2", array_text)
            self.assertIn("#SBATCH --cpus-per-task=8", array_text)
            self.assertIn("RUN_POLICY:-resume", array_text)
            self.assertNotIn("sed -i", array_text)
            for script in (
                "run_gaussian_array.sbatch",
                "run_gaussian_worker.sh",
                "submit_gaussian_array.sh",
                "gaussian_array_status.sh",
            ):
                subprocess.run(["bash", "-n", str(campaign / script)], check=True)
            saved = json.loads((campaign / "slurm_plan.json").read_text())
            self.assertEqual(saved["manifest"], "jobs.csv")

    def test_spin_manifest_deduplicates_link1_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "spin"
            campaign.mkdir()
            (campaign / "ladder.gjf").write_text("# link1\n", encoding="utf-8")
            with (campaign / "spin_jobs.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["job_id", "stage_index", "input"])
                writer.writeheader()
                writer.writerow({"job_id": "stage0", "stage_index": "0", "input": "ladder.gjf"})
                writer.writerow({"job_id": "stage1", "stage_index": "1", "input": "ladder.gjf"})
            plan = prepare_slurm_array(campaign, SlurmConfig())
            self.assertEqual(plan["input_count"], 1)

    def test_worker_requires_normal_termination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = self._campaign(root, count=1)
            prepare_slurm_array(campaign, SlurmConfig(gaussian_command=str(root / "fake_g16")))
            fake = root / "fake_g16"
            fake.write_text("#!/usr/bin/env bash\necho ' Normal termination of Gaussian 16'\n", encoding="utf-8")
            fake.chmod(0o755)
            output = root / "job.log"
            status = root / "job.status"
            rc_file = root / "job.rc"
            subprocess.run(
                [
                    str(campaign / "run_gaussian_worker.sh"),
                    str(campaign / "job_000.gjf"),
                    str(output),
                    str(status),
                    str(rc_file),
                    str(root / "scratch"),
                ],
                check=True,
                env={"PATH": "/usr/bin:/bin", "SLURM_CPUS_PER_TASK": "2"},
            )
            self.assertEqual(status.read_text().strip(), "OK")
            self.assertFalse((root / "scratch").exists())

    def test_rejects_manifest_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "campaign"
            campaign.mkdir()
            with (campaign / "jobs.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["input"])
                writer.writeheader()
                writer.writerow({"input": "../outside.gjf"})
            (Path(tmp) / "outside.gjf").write_text("# force\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes the campaign"):
                prepare_slurm_array(campaign, SlurmConfig())

    def test_rejects_nproc_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._campaign(Path(tmp), count=1)
            (campaign / "job_000.gjf").write_text("%nprocshared=16\n# force\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                prepare_slurm_array(campaign, SlurmConfig(cpus_per_job=8))

    def test_refuses_batch_reshuffle_after_outputs_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._campaign(Path(tmp), count=4)
            prepare_slurm_array(campaign, SlurmConfig(jobs_per_batch=2))
            outputs = campaign / "slurm_outputs" / "batch_0001"
            outputs.mkdir(parents=True)
            (outputs / "job_000.log").write_text("Normal termination of Gaussian\n")
            with self.assertRaisesRegex(RuntimeError, "refusing to change"):
                prepare_slurm_array(campaign, SlurmConfig(jobs_per_batch=3))


if __name__ == "__main__":
    unittest.main()
