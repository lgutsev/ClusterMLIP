import csv
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import tempfile
import unittest

from cluster_mlip.cli import build_parser
from cluster_mlip.slurm import (
    ExtractSlurmConfig,
    SlurmConfig,
    prepare_extract_slurm,
    prepare_slurm_batches,
)


class SlurmPreparationTests(unittest.TestCase):
    def test_prepare_slurm_help_renders(self):
        parser = build_parser()
        prepare_slurm = parser._subparsers._group_actions[0].choices["prepare-slurm"]
        self.assertIn("--allow-nproc-mismatch", prepare_slurm.format_help())

    def test_extract_slurm_help_renders(self):
        parser = build_parser()
        extract_slurm = parser._subparsers._group_actions[0].choices["extract-slurm"]
        help_text = extract_slurm.format_help()
        self.assertIn("--submit", help_text)
        self.assertIn("--require-elements", help_text)
        self.assertIn("--runtime-env", help_text)

    def test_prepares_reproducible_extract_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "warehouse.zip"
            source.write_bytes(b"test warehouse bytes")
            output = root / "campaign" / "extracted"
            plan = prepare_extract_slurm(
                source,
                output,
                ExtractSlurmConfig(time_limit="08:00:00", job_name="fenom_extract"),
                extract_arguments=["--elements", "Fe,O", "--max-atoms", "20"],
            )
            sbatch = (output / "run_extract.sbatch").read_text()
            self.assertIn("#SBATCH --job-name=fenom_extract", sbatch)
            self.assertIn("#SBATCH --cpus-per-task=1", sbatch)
            self.assertIn("#SBATCH --time=08:00:00", sbatch)
            self.assertIn("module load gaussian/g16-c01", sbatch)
            self.assertIn("conda activate \"$runtime_env\"", sbatch)
            self.assertIn("/project/lgutsev/env/cluster_mlip_runtime", sbatch)
            self.assertIn("--elements Fe,O --max-atoms 20", sbatch)
            self.assertEqual(plan["source"], str(source.resolve()))
            self.assertEqual(
                plan["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest()
            )
            saved = json.loads((output / "extract_slurm_plan.json").read_text())
            self.assertEqual(saved, plan)
            subprocess.run(["bash", "-n", str(output / "run_extract.sbatch")], check=True)
            subprocess.run(["bash", "-n", str(output / "submit_extract.sh")], check=True)

    def test_extract_slurm_rejects_output_inside_source_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "warehouse"
            source.mkdir()
            (source / "record.log").write_text("record", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not be inside"):
                prepare_extract_slurm(
                    source,
                    source / "extracted",
                    ExtractSlurmConfig(),
                )

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

    def test_prepares_separate_monitorable_batch_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._campaign(Path(tmp))
            plan = prepare_slurm_batches(
                campaign,
                SlurmConfig(
                    jobs_per_batch=3,
                    concurrent_jobs=2,
                    cpus_per_job=8,
                ),
            )
            self.assertEqual(plan["input_count"], 7)
            self.assertEqual(plan["batch_count"], 3)
            batches = sorted((campaign / "slurm_batches").glob("batch_*"))
            self.assertEqual(
                [len((path / "inputs.txt").read_text().splitlines()) for path in batches],
                [3, 3, 1],
            )
            # A symlink on Linux/LONI; prepare_slurm_batches falls back to a
            # real copy where unprivileged symlink creation isn't available
            # (default on Windows without admin rights/Developer Mode) --
            # either way the batch directory must have a working, correct
            # copy of the input.
            linked_input = batches[0] / "job_000.gjf"
            self.assertTrue(linked_input.is_symlink() or linked_input.is_file())
            self.assertEqual(linked_input.read_text(), (campaign / "job_000.gjf").read_text())
            first_batch = (batches[0] / "run_batch.sbatch").read_text()
            second_batch = (batches[1] / "run_batch.sbatch").read_text()
            self.assertIn("#SBATCH --job-name=cluster_mlip_g16_0001", first_batch)
            self.assertIn("#SBATCH --job-name=cluster_mlip_g16_0002", second_batch)
            self.assertNotIn("#SBATCH --array", first_batch)
            self.assertIn("#SBATCH --ntasks-per-node=2", first_batch)
            self.assertIn("#SBATCH --cpus-per-task=8", first_batch)
            self.assertIn("RUN_POLICY:-resume", first_batch)
            self.assertNotIn("sed -i", first_batch)
            root_scripts = (
                "run_gaussian_worker.sh",
                "submit_gaussian_batches.sh",
                "gaussian_batch_status.sh",
            )
            for script in root_scripts:
                subprocess.run(["bash", "-n", str(campaign / script)], check=True)
            for batch in batches:
                subprocess.run(["bash", "-n", str(batch / "run_batch.sbatch")], check=True)
                subprocess.run(["bash", "-n", str(batch / "submit.sh")], check=True)
            # Invoke via `bash` explicitly rather than relying on the
            # shebang line + executable bit: that's how Linux/LONI runs it
            # too (nothing here depends on native OS exec of "#!"), and it
            # also works from a native Windows Python process where a
            # direct CreateProcess of a shell script is not a valid Win32
            # executable regardless of the shebang.
            status = subprocess.run(
                ["bash", str(campaign / "gaussian_batch_status.sh")],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("batch_0001", status.stdout)
            self.assertIn("TOTAL", status.stdout)
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
            plan = prepare_slurm_batches(campaign, SlurmConfig())
            self.assertEqual(plan["input_count"], 1)

    @unittest.skipIf(
        platform.system() == "Windows",
        "exercises real POSIX exec/PATH/scratch-cleanup semantics (a Unix-style PATH override, "
        "a fake executable relying on native shebang exec, bash trap-based cleanup) matching the "
        "real LONI runtime -- not a meaningful thing to emulate under native Windows CreateProcess",
    )
    def test_worker_requires_normal_termination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = self._campaign(root, count=1)
            prepare_slurm_batches(campaign, SlurmConfig(gaussian_command=str(root / "fake_g16")))
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
                prepare_slurm_batches(campaign, SlurmConfig())

    def test_rejects_nproc_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._campaign(Path(tmp), count=1)
            (campaign / "job_000.gjf").write_text("%nprocshared=16\n# force\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                prepare_slurm_batches(campaign, SlurmConfig(cpus_per_job=8))

    def test_refuses_batch_reshuffle_after_outputs_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._campaign(Path(tmp), count=4)
            prepare_slurm_batches(campaign, SlurmConfig(jobs_per_batch=2))
            outputs = campaign / "slurm_batches" / "batch_0001"
            (outputs / "job_000.log").write_text("Normal termination of Gaussian\n")
            with self.assertRaisesRegex(RuntimeError, "refusing to change"):
                prepare_slurm_batches(campaign, SlurmConfig(jobs_per_batch=3))


if __name__ == "__main__":
    unittest.main()
