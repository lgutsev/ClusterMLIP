import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.restart import prepare_spin_restarts
from cluster_mlip.spin import SPIN_MANIFEST_COLUMNS, _spin_manifest_audit


class SpinRestartTests(unittest.TestCase):
    def _campaign(self, root: Path, *, active: bool = False) -> tuple[Path, Path, Path]:
        campaign = root / "original"
        inputs = campaign / "inputs"
        batch = campaign / "slurm_batches" / "batch_0001"
        inputs.mkdir(parents=True)
        batch.mkdir(parents=True)
        name = "ladder.gjf"
        sections = []
        rows = []
        previous = ""
        for stage, mult in enumerate((55, 53, 51)):
            chk = f"chain-s{stage:02d}-m{mult}.chk"
            header = (f"%oldchk={previous}\n" if previous else "") + f"%chk={chk}\n%mem=24GB\n%nprocshared=12\n"
            route = "#p UBPW91/6-311++G* NoSymm Opt"
            if stage:
                route += " Geom=Checkpoint Guess=Read"
            sections.append(f"{header}{route}\n\nstage {stage}\n\n0 {mult}\n")
            row = {column: "" for column in SPIN_MANIFEST_COLUMNS}
            row.update({
                "job_id": f"chain-s{stage:02d}", "chain_id": "chain",
                "stage_index": str(stage), "pathway": "multiplicity_ladder",
                "initialization": (
                    "trusted_high_spin_direct" if stage == 0 else "checkpoint_spin_flip"
                ),
                "audit_classification": (
                    "trusted_high_spin_reference" if stage == 0
                    else "sequential_checkpoint_spin_flip"
                ),
                "parent_record_id": "parent", "source": "archive/source.log",
                "formula": "Fe16N2", "source_geometry_sha256": "0" * 64,
                "intended_charge": "0",
                "intended_multiplicity": str(mult), "high_spin_multiplicity": "55",
                "final_target_multiplicity": "51", "checkpoint": chk,
                "predecessor_job_id": "" if stage == 0 else f"chain-s{stage - 1:02d}",
                "predecessor_multiplicity": "" if stage == 0 else str(mult + 2),
                "predecessor_checkpoint": previous,
                "checkpoint_lineage": ">".join(
                    f"m{m}:chain-s{i:02d}-m{m}.chk"
                    for i, m in enumerate((55, 53, 51)[:stage + 1])
                ),
                "input": f"inputs/{name}", "output": "ladder.log",
            })
            rows.append(row)
            previous = chk
        input_path = inputs / name
        input_path.write_text("\n--Link1--\n".join(sections))
        digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
        for row in rows:
            row["input_sha256"] = digest
        with (campaign / "spin_jobs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SPIN_MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        manifest_hash = hashlib.sha256((campaign / "spin_jobs.csv").read_bytes()).hexdigest()
        (campaign / "spin_campaign.json").write_text(json.dumps({
            "schema_version": 1, "strategy": "ladder", "manifest": "spin_jobs.csv",
            "manifest_sha256": manifest_hash,
        }))
        (batch / "inputs.txt").write_text(name + "\n")
        (batch / name).symlink_to("../../inputs/ladder.gjf")
        log = batch / "ladder.log"
        log.write_text(
            " Charge = 0 Multiplicity = 55\n Stationary point found.\n"
            " Normal termination of Gaussian 09\n"
            " Charge = 0 Multiplicity = 53\n SCF Done: E(UHF) = -12.3\n"
        )
        checkpoint = batch / "chain-s01-m53.chk"
        checkpoint.write_bytes(b"live checkpoint")
        if active:
            (batch / "ladder.started").write_text("started\n")
        return campaign, log, checkpoint

    def test_shortens_ladder_and_stages_copied_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign, log, checkpoint = self._campaign(root)
            log_hash = hashlib.sha256(log.read_bytes()).hexdigest()
            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            result = prepare_spin_restarts(campaign)
            self.assertEqual(result["input_count"], 1)
            self.assertEqual(result["stage_count"], 2)
            with (campaign / "spin_jobs.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            active = [row for row in rows if row.get("submission_active") != "false"]
            self.assertEqual([row["intended_multiplicity"] for row in active], ["53", "51"])
            self.assertEqual(active[0]["initialization"], "interrupted_checkpoint_restart")
            generated_path = campaign / active[0]["input"]
            generated = generated_path.read_text()
            self.assertEqual(generated.lower().count("guess=read"), 2)
            self.assertEqual(generated.lower().count("geom=checkpoint"), 2)
            self.assertIn("%oldchk=chain-s01-m53-seed-r01.chk", generated)
            self.assertNotIn("0 55", generated)
            seed = campaign / active[0]["restart_seed_checkpoint"]
            self.assertEqual(hashlib.sha256(seed.read_bytes()).hexdigest(), checkpoint_hash)
            self.assertFalse(log.exists())
            archived = campaign / active[0]["restart_source_output"]
            self.assertEqual(hashlib.sha256(archived.read_bytes()).hexdigest(), log_hash)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), checkpoint_hash)

            _, lineage, errors = _spin_manifest_audit(campaign / "spin_jobs.csv")
            self.assertEqual(errors, [])
            self.assertTrue(lineage)
            self.assertEqual(set(lineage.values()), {"verified"})

            batch = campaign / "slurm_batches/batch_0001"
            self.assertEqual((batch / "inputs.txt").read_text().strip(), generated_path.name)
            self.assertTrue((batch / generated_path.name).is_file())
            self.assertTrue((campaign / "restart_plan.csv").is_file())
            self.assertTrue((campaign / "restart_backups").is_dir())

    def test_requires_confirmation_for_unmatched_started_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign, _, _ = self._campaign(root, active=True)
            with self.assertRaisesRegex(RuntimeError, "activity_unconfirmed"):
                prepare_spin_restarts(campaign)
            result = prepare_spin_restarts(campaign, assume_stopped=True)
            self.assertEqual(result["input_count"], 1)

    def test_dry_run_does_not_modify_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, log, _ = self._campaign(Path(tmp))
            before = hashlib.sha256((campaign / "spin_jobs.csv").read_bytes()).hexdigest()
            result = prepare_spin_restarts(campaign, dry_run=True)
            self.assertEqual(result["input_count"], 1)
            self.assertTrue(log.is_file())
            self.assertEqual(
                hashlib.sha256((campaign / "spin_jobs.csv").read_bytes()).hexdigest(), before
            )

    def test_second_restart_replaces_only_current_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _, _ = self._campaign(Path(tmp))
            prepare_spin_restarts(campaign)
            with (campaign / "spin_jobs.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            active = [row for row in rows if row.get("submission_active") != "false"]
            batch = campaign / "slurm_batches/batch_0001"
            current_stem = Path(active[0]["input"]).stem
            (batch / f"{current_stem}.log").write_text(
                " Charge = 0 Multiplicity = 53\n Stationary point found.\n"
                " Normal termination of Gaussian 09\n"
                " Charge = 0 Multiplicity = 51\n SCF Done: E(UHF) = -12.4\n"
            )
            (batch / active[1]["checkpoint"]).write_bytes(b"second checkpoint")
            result = prepare_spin_restarts(campaign)
            self.assertEqual(result["input_count"], 1)
            self.assertEqual(result["stage_count"], 1)
            with (campaign / "spin_jobs.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            active = [row for row in rows if row.get("submission_active") != "false"]
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]["restart_attempt"], "2")
            self.assertEqual(active[0]["intended_multiplicity"], "51")
            self.assertIn("__restart02-from-m51", active[0]["input"])
            self.assertIn("__before-restart02.log", active[0]["restart_source_output"])
            with (campaign / "restart_plan.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)


if __name__ == "__main__":
    unittest.main()
