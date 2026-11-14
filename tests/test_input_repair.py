import csv
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.batch_progress import write_batch_progress
from cluster_mlip.input_repair import repair_checkpoint_inputs, repaired_text
from cluster_mlip.route_audit import ladder_rows
from cluster_mlip.routes import checkpoint_stages_with_coordinates
from cluster_mlip.spin import SPIN_MANIFEST_COLUMNS
from test_spin_restart import SpinRestartTests

ATOMS = "Fe 0.000000 0.000000 0.000000\nN 1.650000 0.000000 0.000000\n"


def _broken_stage0(text: str) -> str:
    """Stage 0 as the pre-PR #22 restarts wrote it: Geom=Checkpoint plus atoms."""
    first, rest = text.split("\n--Link1--\n", 1)
    first = first.replace("NoSymm Opt", "NoSymm Opt Geom=Checkpoint Guess=Read")
    first = first.replace("0 55\n", "0 55\n" + ATOMS + "\n")
    return first + "\n--Link1--\n" + rest


class RepairedTextTests(unittest.TestCase):
    TEXT = (
        "%oldchk=seed.chk\n%chk=a.chk\n#p UBPW91/6-311++G* NoSymm IOP(5/13=1) Opt "
        "Geom=Checkpoint Guess=Read\n\nstage 0\n\n0 9\n" + ATOMS + "\n"
        "--Link1--\n%oldchk=a.chk\n%chk=b.chk\n#p UBPW91/6-311++G* NoSymm IOP(5/13=1) Opt "
        "Geom=Checkpoint Guess=Read\n\nstage 1\n\n0 7\n"
    )

    def test_drops_only_the_atom_lines_of_checkpoint_stages(self):
        new, stages, removed = repaired_text(self.TEXT)
        self.assertEqual((stages, removed), ([0], 2))
        self.assertEqual(checkpoint_stages_with_coordinates(new), [])
        self.assertNotIn("Fe 0.000000", new)
        self.assertEqual(new.count("IOP(5/13=1)"), 2)
        self.assertRegex(new, r"(?m)^0 9$")
        self.assertTrue(new.endswith("0 7\n\n"))

    def test_a_correct_input_is_left_alone(self):
        good = self.TEXT.replace(ATOMS, "")
        self.assertEqual(repaired_text(good), (good, [], 0))

    def test_a_stage_with_its_own_coordinates_is_not_touched(self):
        explicit = self.TEXT.replace(
            "Opt Geom=Checkpoint Guess=Read\n\nstage 0", "Opt\n\nstage 0", 1)
        self.assertEqual(repaired_text(explicit)[1], [])


class RepairCampaignTests(unittest.TestCase):
    _campaign = SpinRestartTests._campaign

    def _broken_campaign(self, root: Path, *, ran: bool = False) -> tuple[Path, Path]:
        campaign, log, _ = self._campaign(root)
        source = campaign / "inputs" / "ladder.gjf"
        source.write_text(_broken_stage0(source.read_text()))
        batch = campaign / "slurm_batches" / "batch_0001"
        if not (batch / "ladder.gjf").is_symlink():
            shutil.copy2(source, batch / "ladder.gjf")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = campaign / "spin_jobs.csv"
        with manifest.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["input_sha256"] = digest
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SPIN_MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        if not ran:
            log.unlink()
        return campaign, source

    def test_dry_run_reports_and_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, source = self._broken_campaign(Path(tmp))
            before = source.read_bytes()
            result = repair_checkpoint_inputs(campaign, dry_run=True)
            self.assertEqual(result["input_count"], 1)
            self.assertEqual(result["plan"][0]["repaired_stages"], "0")
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse((campaign / "input_repair_backups").exists())

    def test_repair_rewrites_backs_up_and_updates_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, source = self._broken_campaign(Path(tmp))
            original = source.read_text()
            result = repair_checkpoint_inputs(campaign)
            self.assertEqual(result["input_count"], 1)
            repaired = source.read_text()
            self.assertEqual(checkpoint_stages_with_coordinates(repaired), [])
            batch_copy = campaign / "slurm_batches" / "batch_0001" / "ladder.gjf"
            self.assertEqual(batch_copy.read_text(), repaired)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with (campaign / "spin_jobs.csv").open(newline="") as handle:
                self.assertEqual({row["input_sha256"] for row in csv.DictReader(handle)}, {digest})
            backups = list((campaign / "input_repair_backups").rglob("ladder.gjf"))
            self.assertTrue(backups)
            self.assertEqual(backups[0].read_text(), original)
            self.assertTrue((campaign / "input_repair_plan.csv").is_file())
            # campaign-status no longer flags it, and agrees on the hash.
            progress = write_batch_progress(campaign, audit=True)
            details = "\n".join(issue["detail"] for issue in progress["issues"])
            self.assertNotIn("still carries explicit coordinates", details)
            self.assertNotIn("Input SHA256 differs", details)

    def test_an_input_that_already_ran_is_left_to_the_restart_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, source = self._broken_campaign(Path(tmp), ran=True)
            before = source.read_bytes()
            result = repair_checkpoint_inputs(campaign)
            self.assertEqual(result["input_count"], 0)
            self.assertIn("already ran", result["skipped"][0]["reason"])
            self.assertEqual(source.read_bytes(), before)


class LadderRowTests(unittest.TestCase):
    def _rows(self, active: str, attempt: str) -> list[dict[str, str]]:
        return [{"input": "inputs/x.gjf", "stage_index": str(i), "job_id": f"j{i}-{attempt}",
                 "submission_active": active} for i in range(5)]

    def test_a_from_scratch_rerun_counts_once(self):
        # A rerun keeps the input path: 5 retired rows, then 5 active ones.
        rows = self._rows("false", "old") + self._rows("true", "rerun")
        chosen = ladder_rows(rows)
        self.assertEqual(len(chosen), 5)
        self.assertEqual({row["job_id"].split("-")[1] for row in chosen}, {"rerun"})
        self.assertEqual([row["stage_index"] for row in chosen], ["0", "1", "2", "3", "4"])

    def test_a_fully_retired_lineage_keeps_its_latest_generation(self):
        rows = self._rows("false", "old") + self._rows("false", "rerun")
        self.assertEqual({row["job_id"].split("-")[1] for row in ladder_rows(rows)}, {"rerun"})

    def test_campaign_status_does_not_count_retired_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _, _ = SpinRestartTests._campaign(self, Path(tmp))
            manifest = campaign / "spin_jobs.csv"
            with manifest.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            retired = [dict(row, submission_active="false", job_id=row["job_id"] + "-old",
                            output="ladder__before-rerun01.log") for row in rows]
            fields = list(SPIN_MANIFEST_COLUMNS) + ["submission_active"]
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(retired + rows)
            progress = write_batch_progress(campaign, audit=True)
            details = "\n".join(issue["detail"] for issue in progress["issues"])
            self.assertNotIn("Number of Link1 stages differs", details)


if __name__ == "__main__":
    unittest.main()
