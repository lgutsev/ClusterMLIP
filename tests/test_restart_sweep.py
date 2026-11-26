import csv
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.cli import main
from cluster_mlip.restart_sweep import (
    EXIT_CONTINUE,
    EXIT_NOTHING_DONE,
    EXIT_STOPPED,
    LOCK_NAME,
    QueueUnreadable,
    classify_log,
    restart_sweep,
)
from cluster_mlip.spin import SPIN_MANIFEST_COLUMNS

ROUTE = "#p UBPW91/6-311++G* NoSymm IOP(5/13=1) Opt"
SADDLE_ROUTE = "#p UBPW91/6-311++G* NoSymm IOP(5/13=1) Opt=(TS,CalcFC,NoEigenTest)"
MULTS = (55, 53, 51)


def _done(mult: int) -> str:
    return (f" Charge = 0 Multiplicity = {mult}\n SCF Done:  E(UBPW91) = -1.0\n"
            " Stationary point found.\n Normal termination of Gaussian 09\n")


def _step_limit(mult: int, scfs: int = 4, unconverged: int = 0) -> str:
    text = f" Charge = 0 Multiplicity = {mult}\n"
    for index in range(scfs):
        if index < unconverged:
            text += " >>>>>>>>>> Convergence criterion not met.\n"
        text += " SCF Done:  E(UBPW91) = -1.0\n"
    return (text + " Optimization stopped.\n    -- Number of steps exceeded,  NStep=  20\n"
            " Error termination request processed by link 9999.\n")


def _scf_death(mult: int) -> str:
    return (f" Charge = 0 Multiplicity = {mult}\n Convergence failure -- run terminated.\n"
            " Error termination via Lnk1e in /usr/g09/l502.exe\n")


def _cut(mult: int) -> str:
    return f" Charge = 0 Multiplicity = {mult}\n SCF Done:  E(UBPW91) = -1.0\n"


class Campaign:
    """A small spin campaign whose ladders each stopped in a chosen way."""

    def __init__(self, root: Path) -> None:
        self.root = root / "campaign"
        (self.root / "inputs").mkdir(parents=True)
        self.rows: list[dict[str, str]] = []

    def ladder(self, name: str, batch: int, log: str, *, route: str = ROUTE,
               failed_stage: int = 1) -> None:
        folder = self.root / "slurm_batches" / f"batch_{batch:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        sections, previous = [], ""
        for stage, mult in enumerate(MULTS):
            chk = f"{name}-s{stage:02d}-m{mult}.chk"
            header = (f"%oldchk={previous}\n" if previous else "") + f"%chk={chk}\n%mem=24GB\n"
            stage_route = route + (" Geom=Checkpoint Guess=Read" if stage else "")
            sections.append(f"{header}{stage_route}\n\nstage {stage}\n\n0 {mult}\n")
            row = {column: "" for column in SPIN_MANIFEST_COLUMNS}
            row.update({
                "job_id": f"{name}-s{stage:02d}", "chain_id": name, "stage_index": str(stage),
                "pathway": "multiplicity_ladder",
                "initialization": "trusted_high_spin_direct" if stage == 0 else "checkpoint_spin_flip",
                "formula": "Fe3N", "intended_charge": "0", "intended_multiplicity": str(mult),
                "high_spin_multiplicity": str(MULTS[0]), "checkpoint": chk,
                "predecessor_checkpoint": previous, "input": f"inputs/{name}.gjf",
                "output": f"{name}.log",
            })
            self.rows.append(row)
            previous = chk
        (self.root / "inputs" / f"{name}.gjf").write_text("\n--Link1--\n".join(sections))
        shutil.copy2(self.root / "inputs" / f"{name}.gjf", folder / f"{name}.gjf")
        with (folder / "inputs.txt").open("a") as handle:
            handle.write(f"{name}.gjf\n")
        (folder / "run_batch.sbatch").write_text(f"#!/bin/bash\n#SBATCH --job-name=w2_{batch:04d}\n")
        if log:
            (folder / f"{name}.log").write_text(log)
            (folder / f"{name}-s{failed_stage:02d}-m{MULTS[failed_stage]}.chk").write_bytes(b"chk")
            (folder / f"{name}.started").write_text("1\n")
            (folder / f"{name}.finished").write_text("2\n")

    def write(self) -> Path:
        with (self.root / "spin_jobs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SPIN_MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)
        return self.root

    def manifest_hash(self) -> str:
        return hashlib.sha256((self.root / "spin_jobs.csv").read_bytes()).hexdigest()

    def batch(self, number: int) -> Path:
        return self.root / "slurm_batches" / f"batch_{number:04d}"


def _empty_queue() -> list[dict[str, str]]:
    return []


class RestartSweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.campaign = Campaign(Path(self._tmp.name))

    def _one(self, log: str | None = None, **kwargs) -> Path:
        self.campaign.ladder("a", 1, _done(55) + _step_limit(53) if log is None else log, **kwargs)
        return self.campaign.write()

    def _sweep(self, root: Path, **kwargs):
        kwargs.setdefault("max_stages", 100)
        kwargs.setdefault("read_queue", _empty_queue)
        return restart_sweep(root, **kwargs)

    def test_classify_log_reads_only_the_last_section(self):
        self.assertEqual(classify_log(_done(55) + _step_limit(53))["ending"], "step_limit")
        self.assertEqual(classify_log(_step_limit(55) + _done(53))["ending"], "no_error_termination")
        self.assertEqual(classify_log(_done(55) + _scf_death(53))["ending"], "other_error")
        verdict = classify_log(_done(55) + _step_limit(53, scfs=6, unconverged=2))
        self.assertEqual((verdict["scf_total"], verdict["scf_unconverged"]), (6, 2))

    def test_dry_run_plans_the_restart_and_changes_nothing(self):
        root = self._one()
        before = self.campaign.manifest_hash()
        report = self._sweep(root)
        self.assertEqual((report["status"], report["exit_code"]), ("dry_run", EXIT_CONTINUE))
        self.assertEqual([(item["input"], item["multiplicity"], item["stages"])
                          for item in report["plan"]], [("a.gjf", "53", 2)])
        self.assertEqual(Path(report["plan"][0]["source_checkpoint"]).name, "a-s01-m53.chk")
        self.assertEqual(self.campaign.manifest_hash(), before)
        self.assertFalse((root / "restart_sweep").exists())
        self.assertFalse((root / LOCK_NAME).exists())
        self.assertTrue((self.campaign.batch(1) / "a.log").is_file())

    def test_execute_writes_the_restart_with_the_same_route(self):
        root = self._one()
        report = self._sweep(root, execute=True)
        self.assertEqual((report["status"], report["exit_code"]), ("swept", EXIT_CONTINUE))
        self.assertNotEqual(report["manifest_sha256_before"], report["manifest_sha256_after"])
        new_input = (root / report["plan"][0]["new_input"]).read_text()
        self.assertEqual(new_input.count("IOP(5/13=1)"), 2)
        self.assertEqual(new_input.lower().count("geom=checkpoint guess=read"), 2)
        self.assertNotIn("MaxCycle", new_input)
        self.assertIn("%oldchk=a-s01-m53-seed-r01.chk", new_input)
        self.assertEqual((root / "restart_sweep/pending_submit.txt").read_text(), "batch_0001\n")
        ledger = json.loads((root / "restart_sweep/ledger.json").read_text())
        self.assertEqual((ledger["restarted_inputs"], ledger["restarted_stages"]), (1, 2))
        self.assertFalse((root / LOCK_NAME).exists())
        self.assertEqual((self.campaign.batch(1) / "inputs.txt").read_text().strip(),
                         Path(report["plan"][0]["new_input"]).name)
        again = self._sweep(root, execute=True)
        self.assertEqual((again["status"], again["plan"]), ("swept", []))

    def test_a_batch_with_a_job_is_not_touched(self):
        self.campaign.ladder("a", 1, _done(55) + _step_limit(53))
        self.campaign.ladder("b", 2, _done(55) + _step_limit(53))
        root = self.campaign.write()
        jobs = [
            {"job_id": "7", "name": "other", "state": "RUNNING",
             "workdir": str(self.campaign.batch(1))},
            {"job_id": "8", "name": "w2_0002", "state": "PENDING", "workdir": "/somewhere/else"},
        ]
        report = self._sweep(root, execute=True, read_queue=lambda: jobs)
        self.assertEqual(report["plan"], [])
        self.assertEqual(report["counts"]["batch_has_job"], 2)
        self.assertTrue((self.campaign.batch(1) / "a.log").is_file())
        self.assertTrue((self.campaign.batch(2) / "b.log").is_file())

    def test_unreadable_queue_writes_nothing(self):
        root = self._one()
        before = self.campaign.manifest_hash()

        def broken() -> list[dict[str, str]]:
            raise QueueUnreadable("squeue: connection refused")

        report = self._sweep(root, execute=True, read_queue=broken)
        self.assertEqual((report["status"], report["exit_code"]),
                         ("queue_unreadable", EXIT_NOTHING_DONE))
        self.assertEqual(self.campaign.manifest_hash(), before)
        self.assertFalse((root / "restart_sweep/ledger.json").exists())
        self.assertFalse((root / LOCK_NAME).exists())

    def test_a_job_appearing_before_the_write_stops_that_batch(self):
        self.campaign.ladder("a", 1, _done(55) + _step_limit(53))
        self.campaign.ladder("b", 2, _done(55) + _step_limit(53))
        root = self.campaign.write()
        snapshots = iter([[], [{"job_id": "9", "name": "w2_0001", "state": "PENDING",
                                "workdir": str(self.campaign.batch(1))}]])
        report = self._sweep(root, execute=True, read_queue=lambda: next(snapshots))
        self.assertEqual([item["input"] for item in report["plan"]], ["b.gjf"])
        self.assertTrue((self.campaign.batch(1) / "a.log").is_file())
        self.assertIn({"input": "a.gjf", "batch": "batch_0001",
                       "reason": "batch_got_a_job_between_snapshots"}, report["left_alone"])
        self.assertEqual(report["pending_submit"], ["batch_0002"])

    def test_second_queue_failure_writes_no_restart(self):
        root = self._one()
        before = self.campaign.manifest_hash()
        calls = []

        def flaky() -> list[dict[str, str]]:
            calls.append(1)
            if len(calls) > 1:
                raise QueueUnreadable("squeue: timed out")
            return []

        report = self._sweep(root, execute=True, read_queue=flaky)
        self.assertEqual(report["exit_code"], EXIT_NOTHING_DONE)
        self.assertEqual(self.campaign.manifest_hash(), before)

    def test_other_failures_are_left_alone_and_do_not_stop_the_sweep(self):
        self.campaign.ladder("a", 1, _done(55) + _step_limit(53))
        self.campaign.ladder("scf", 1, _done(55) + _scf_death(53))
        self.campaign.ladder("cut", 1, _done(55) + _cut(53))
        self.campaign.ladder("ts", 1, _done(55) + _step_limit(53), route=SADDLE_ROUTE)
        self.campaign.ladder("fresh", 1, "")
        self.campaign.ladder("ok", 1, "".join(_done(mult) for mult in MULTS))
        root = self.campaign.write()
        report = self._sweep(root, execute=True)
        self.assertEqual(report["exit_code"], EXIT_CONTINUE)
        self.assertEqual([item["input"] for item in report["plan"]], ["a.gjf"])
        reasons = {row["input"]: row["reason"] for row in report["left_alone"]}
        self.assertEqual(reasons, {"scf.gjf": "other_error", "cut.gjf": "no_error_termination",
                                   "ts.gjf": "not_a_minimum_search"})
        self.assertEqual((report["counts"]["complete"], report["counts"]["not_started"]), (1, 1))
        for name in ("scf", "cut", "ts"):
            self.assertTrue((self.campaign.batch(1) / f"{name}.log").is_file())
        # A resubmit of batch 1 would rerun these three from scratch.
        self.assertEqual({row["input"] for row in report["rerun_on_resubmit"]},
                         {"scf.gjf", "cut.gjf", "ts.gjf"})

    def test_saddle_named_input_is_out_of_scope_even_with_a_plain_opt_route(self):
        name = "src__fe3n__transition-state__q0-m55__seed__abc"
        self.campaign.ladder(name, 1, _done(55) + _step_limit(53))
        root = self.campaign.write()
        report = self._sweep(root)
        self.assertEqual(report["plan"], [])
        self.assertEqual(report["left_alone"][0]["reason"], "not_a_minimum_search")

    def test_reaction_path_point_is_never_restarted(self):
        name = "src__fe2h2o4__irc-reverse__q0-m55__seed__abc"
        self.campaign.ladder(name, 1, _done(55) + _step_limit(53))
        root = self.campaign.write()
        report = self._sweep(root, execute=True)
        self.assertEqual(report["plan"], [])
        self.assertEqual(report["left_alone"][0]["reason"], "path_point")
        self.assertTrue((self.campaign.batch(1) / f"{name}.log").is_file())

    def test_scf_that_never_printed_its_energy_still_counts(self):
        text = (" Charge = 0 Multiplicity = 53\n SCF Done:  E(UBPW91) = -1.0\n"
                " >>>>>>>>>> Convergence criterion not met.\n"
                " Convergence failure -- run terminated.\n Error termination via Lnk1e in l502.exe\n")
        verdict = classify_log(text)
        self.assertEqual((verdict["scf_total"], verdict["scf_unconverged"]), (2, 2))

    def test_mostly_unconverged_step_is_parked_and_stays_parked(self):
        self.campaign.ladder("bad", 1, _done(55) + _step_limit(53, scfs=10, unconverged=6))
        self.campaign.ladder("edge", 1, _done(55) + _step_limit(53, scfs=10, unconverged=5))
        root = self.campaign.write()
        report = self._sweep(root, execute=True)
        self.assertEqual([item["input"] for item in report["plan"]], ["edge.gjf"])
        self.assertIn("6 of 10 SCFs unconverged", report["left_alone"][0]["reason"])
        self.assertIn("bad.gjf,6 of 10", (root / "restart_sweep/parked.csv").read_text())
        again = self._sweep(root, execute=True)
        self.assertEqual(again["left_alone"][0]["reason"], "parked")
        self.assertTrue((self.campaign.batch(1) / "bad.log").is_file())

    def test_named_parked_inputs_are_never_restarted(self):
        root = self._one()
        report = self._sweep(root, execute=True, parked_inputs={"a.gjf"})
        self.assertEqual(report["plan"], [])
        self.assertEqual(report["left_alone"][0]["reason"], "parked")

    def test_a_restarted_stage_gets_no_second_attempt(self):
        root = self._one()
        first = self._sweep(root, execute=True)
        new_log = root / first["plan"][0]["new_output"]
        new_log.write_text(_step_limit(53))
        stem = new_log.stem
        (new_log.parent / f"{stem}.started").write_text("1\n")
        (new_log.parent / f"{stem}.finished").write_text("2\n")
        (new_log.parent / "a-s01-m53-r01.chk").write_bytes(b"chk")
        before = self.campaign.manifest_hash()
        second = self._sweep(root, execute=True)
        self.assertEqual(second["plan"], [])
        self.assertTrue(second["left_alone"][0]["reason"].startswith("repeat_step_limit"))
        self.assertEqual(second["earlier_restarts"], {"repeat": 1})
        self.assertEqual(self.campaign.manifest_hash(), before)

    def test_a_later_stage_of_a_restart_may_still_be_restarted(self):
        root = self._one()
        first = self._sweep(root, execute=True)
        new_log = root / first["plan"][0]["new_output"]
        new_log.write_text(_done(53) + _step_limit(51))
        (new_log.parent / f"{new_log.stem}.finished").write_text("2\n")
        (new_log.parent / "a-s02-m51-r01.chk").write_bytes(b"chk")
        second = self._sweep(root, execute=True)
        self.assertEqual([(item["multiplicity"], item["stages"]) for item in second["plan"]],
                         [("51", 1)])

    def test_cap_stops_for_good_until_cleared(self):
        self.campaign.ladder("a", 1, _done(55) + _step_limit(53))
        self.campaign.ladder("b", 2, _step_limit(55), failed_stage=0)
        root = self.campaign.write()
        report = self._sweep(root, execute=True, max_stages=2)
        self.assertEqual((report["status"], report["exit_code"]), ("stopped", EXIT_STOPPED))
        self.assertTrue(report["stopped"].startswith("cap_reached"))
        self.assertEqual([item["input"] for item in report["plan"]], ["a.gjf"])
        self.assertTrue((self.campaign.batch(2) / "b.log").is_file())
        held = self._sweep(root, execute=True, max_stages=10)
        self.assertEqual((held["exit_code"], held["plan"]), (EXIT_STOPPED, []))
        self.assertTrue((self.campaign.batch(2) / "b.log").is_file())
        resumed = self._sweep(root, execute=True, max_stages=10, clear_stop="cap raised by LG")
        self.assertEqual((resumed["status"], [item["input"] for item in resumed["plan"]]),
                         ("swept", ["b.gjf"]))
        ledger = json.loads((root / "restart_sweep/ledger.json").read_text())
        self.assertEqual(ledger["cleared_stops"][0]["note"], "cap raised by LG")
        self.assertEqual(ledger["restarted_stages"], 5)

    def test_held_lock_means_nothing_is_done(self):
        root = self._one()
        (root / LOCK_NAME).write_text("manual package q003\n")
        report = self._sweep(root, execute=True)
        self.assertEqual((report["status"], report["exit_code"]), ("lock_held", EXIT_NOTHING_DONE))
        self.assertEqual(report["lock"], "manual package q003")
        self.assertTrue((root / LOCK_NAME).exists())
        self.assertTrue((self.campaign.batch(1) / "a.log").is_file())

    def test_name_collision_stops_the_sweep_and_writes_no_restart(self):
        root = self._one()
        (self.campaign.batch(1) / "a-s01-m53-seed-r01.chk").write_bytes(b"left over")
        before = self.campaign.manifest_hash()
        report = self._sweep(root, execute=True)
        self.assertEqual(report["exit_code"], EXIT_STOPPED)
        self.assertIn("restart target already exists", report["refusal"])
        self.assertEqual(self.campaign.manifest_hash(), before)
        self.assertTrue((self.campaign.batch(1) / "a.log").is_file())
        self.assertFalse((root / LOCK_NAME).exists())

    def test_many_repeat_failures_stop_the_sweep(self):
        root = self._one()
        state = root / "restart_sweep"
        state.mkdir()
        restarts = []
        for index in range(8):
            log = self.campaign.batch(1) / f"old{index}.log"
            log.write_text(_step_limit(53) if index < 3 else _done(53))
            restarts.append({"input": f"old{index}.gjf", "new_output": str(log.relative_to(root))})
        (state / "ledger.json").write_text(json.dumps({
            "schema_version": 1, "stopped": "", "restarted_inputs": 8, "restarted_stages": 8,
            "restarts": restarts, "parked": {}, "sweeps": []}))
        report = self._sweep(root, execute=True)
        self.assertEqual(report["exit_code"], EXIT_STOPPED)
        self.assertTrue(report["stopped"].startswith("repeat_step_limit_rate: 3 of 8"))
        self.assertTrue((self.campaign.batch(1) / "a.log").is_file())

    def test_command_line_dry_run(self):
        root = self._one()
        report_path = Path(self._tmp.name) / "report.json"
        code = main(["restart-sweep", str(root), "--max-stages", "10", "--report", str(report_path),
                     "--queue-command", f'"{sys.executable}" -c pass'.replace("\\", "/")])
        self.assertEqual(code, EXIT_CONTINUE)
        report = json.loads(report_path.read_text())
        self.assertEqual([item["input"] for item in report["plan"]], ["a.gjf"])
        self.assertTrue((self.campaign.batch(1) / "a.log").is_file())

    def test_command_line_reports_an_unreadable_queue(self):
        root = self._one()
        code = main(["restart-sweep", str(root), "--max-stages", "10", "--execute",
                     "--queue-command", "no-such-queue-command-anywhere"])
        self.assertEqual(code, EXIT_NOTHING_DONE)
        self.assertTrue((self.campaign.batch(1) / "a.log").is_file())


if __name__ == "__main__":
    unittest.main()
