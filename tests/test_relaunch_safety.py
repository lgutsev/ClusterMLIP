"""Safety properties of relaunch.py, the only module that mutates a live campaign.

relaunch-routes is deliberately not a checkpoint restart: the collapsed geometry
and its .chk ARE the wrong answer, so it rebuilds from the original guess
geometry with fresh checkpoint names. Every test here pins a property whose
violation silently reintroduces the mislaunched-saddle bug the module exists to
fix. Before this file relaunch.py had no tests at all.
"""

import csv
import hashlib
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import cluster_mlip.relaunch as relaunch_module
from cluster_mlip.jobs import DEFAULT_ROUTE, DEFAULT_SADDLE_ROUTE, expanded_records, write_gaussian_jobs
from cluster_mlip.models import Atom, Record
from cluster_mlip.relaunch import _reseat_rebuilt_row, prepare_route_relaunch
from cluster_mlip.slurm import SlurmConfig, prepare_slurm_batches
from cluster_mlip.spin import DEFAULT_SPIN_ROUTE, write_spin_jobs

ATOMS = [Atom("Fe", 0.0, 0.0, 0.0), Atom("N", 1.7, 0.0, 0.0), Atom("O", 2.85, 0.0, 0.0)]

MINIMUM_LOG = """ Entering Gaussian System
 Charge =  0 Multiplicity = 4
                         Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y          Z
 ---------------------------------------------------------------------
      1         26           0        0.000000    0.000000    0.000000
      2          7           0        1.700000    0.000000    0.000000
      3          8           0        2.850000    0.000000    0.000000
 ---------------------------------------------------------------------
 SCF Done:  E(UBPW91) =  -1400.1234567     A.U. after   12 cycles
 Stationary point found.
 Harmonic frequencies (cm**-1)
   Frequencies --   120.4512   210.3311   455.9922
 Normal termination of Gaussian 09 at Mon Jan  1 00:00:00 2026.
"""


def _record(record_id: str, config_type: str) -> Record:
    return Record(record_id=record_id, source=f"warehouse/{record_id}.txt",
                  atoms=list(ATOMS), charge=0, multiplicity=4, config_type=config_type)


def build_campaign(root: Path) -> tuple[Path, list[str]]:
    """A campaign whose saddle inputs carry the historical bare Opt, across two
    batches, with one of them already finished (normally, at the wrong geometry).
    """
    root.mkdir(parents=True, exist_ok=True)
    seeds = [_record("ts_a", "transition_state"),
             _record("ts_b", "transition_state"),
             _record("fos_c", "first_order_saddle"),
             _record("min_d", "minimum")]
    write_gaussian_jobs(expanded_records(seeds, 1, 0.05, 7), root)

    with (root / "jobs.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    broken = []
    for row in rows:
        path = root / row["input"]
        text = path.read_text(encoding="utf-8")
        if DEFAULT_SADDLE_ROUTE in text:
            path.write_text(text.replace(DEFAULT_SADDLE_ROUTE, DEFAULT_ROUTE), encoding="utf-8")
            row["first_route"] = DEFAULT_ROUTE
            broken.append(Path(row["input"]).stem)
    with (root / "jobs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    prepare_slurm_batches(root, SlurmConfig(jobs_per_batch=4, cpus_per_job=16))

    for batch in sorted((root / "slurm_batches").glob("batch_*")):
        for name in (batch / "inputs.txt").read_text(encoding="utf-8").split():
            stem = Path(name).stem
            if stem == broken[0]:
                (batch / f"{stem}.log").write_text(MINIMUM_LOG, encoding="utf-8")
                (batch / f"{stem}.rc").write_text("0", encoding="utf-8")
                (batch / f"{stem}.status").write_text("COMPLETED", encoding="utf-8")
                (batch / f"{stem}.chk").write_bytes(b"collapsed-minimum-checkpoint")
    return root, broken


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_spin_campaign(root: Path) -> Path:
    """A prepare-spins ladder whose saddle seed carries the historical bare Opt.

    Spin campaigns are the ones with checkpoint columns in their manifest, and
    the ones prepare-spin-restarts later continues from -- so they are where a
    stale `checkpoint` value does real damage.
    """
    root.mkdir(parents=True, exist_ok=True)
    seeds = [Record(record_id="sad_hs", source="warehouse/sad.txt", atoms=list(ATOMS),
                    charge=0, multiplicity=6, config_type="transition_state")]
    saddle_route = DEFAULT_SPIN_ROUTE.replace(" Opt ", " Opt=(TS,CalcFC,NoEigenTest) Freq ")
    # prepare-spins now refuses to launch a saddle seed under a plain Opt, so the
    # historical bad state has to be reproduced the way it actually arose: a
    # correct campaign whose inputs were generated by the older code. Downgrade
    # the route in place after generation.
    write_spin_jobs(seeds, root, high_spin=6, targets=[4], route=saddle_route)
    for gjf in sorted((root / "inputs").glob("*.gjf")):
        gjf.write_text(
            gjf.read_text(encoding="utf-8").replace(
                "Opt=(TS,CalcFC,NoEigenTest) Freq ", "Opt "),
            encoding="utf-8", newline="\n",
        )
    manifest = root / "spin_jobs.csv"
    rows = _rows(manifest)
    fields = list(rows[0])
    for row in rows:
        if row.get("first_route"):
            row["first_route"] = row["first_route"].replace(
                "Opt=(TS,CalcFC,NoEigenTest) Freq ", "Opt ")
        if row.get("route_search_kind") == "saddle":
            row["route_search_kind"] = "minimum"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    # re-stamp the hashes prepare-slurm and the audit verify
    for name in ("spin_campaign.json",):
        path = root / name
        if path.is_file():
            meta = json.loads(path.read_text(encoding="utf-8"))
            for key in ("manifest_sha256", "jobs_csv_sha256"):
                if key in meta:
                    meta[key] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    for row in rows:
        gjf = root / row["input"]
        if gjf.is_file():
            row["input_sha256"] = hashlib.sha256(gjf.read_bytes()).hexdigest()
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    prepare_slurm_batches(root, SlurmConfig(jobs_per_batch=4, cpus_per_job=16))
    return root


def snapshot(root: Path) -> dict[str, str]:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            key = path.relative_to(root).as_posix()  # stable across platforms
            out[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def reference_saddles_missing_ts(campaign: Path) -> list[str]:
    """Reference (non-rattled) saddle inputs the launcher would run without a TS
    search. Rattled variants are deliberately never optimized, so they are excluded."""
    bad = []
    for listing in sorted(campaign.glob("slurm_batches/*/inputs.txt")):
        for name in listing.read_text(encoding="utf-8").split():
            gjf = listing.parent / name
            stem = Path(name).stem
            if not gjf.is_file() or "__reference__" not in stem:
                continue
            if "saddle" not in stem and "transition" not in stem:
                continue
            first = next((l for l in gjf.read_text(encoding="utf-8").splitlines()
                          if l.lstrip().startswith("#")), "")
            if not re.search(r"(?i)Opt\s*=\s*\(\s*TS", first):
                bad.append(name)
    return bad


def orphaned_corrections(campaign: Path) -> list[str]:
    listed = {}
    for listing in sorted(campaign.glob("slurm_batches/*/inputs.txt")):
        listed[listing.parent.name] = set(listing.read_text(encoding="utf-8").split())
    return sorted(
        p.name for p in campaign.rglob("*__routefix*.gjf")
        if p.parent.name.startswith("batch_") and p.name not in listed.get(p.parent.name, set())
    )


class DryRunIsInertTests(unittest.TestCase):
    def test_dry_run_changes_no_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _ = build_campaign(Path(tmp) / "c")
            before = snapshot(campaign)
            result = prepare_route_relaunch(campaign, assume_stopped=True, dry_run=True)
            after = snapshot(campaign)
            self.assertGreater(result["input_count"], 0, "fixture should need a relaunch")
            self.assertEqual(
                sorted(set(before) ^ set(after)), [],
                "dry run created or deleted files",
            )
            changed = [name for name in before if before[name] != after.get(name)]
            self.assertEqual(changed, [], "dry run modified files")


class ResumeAfterInterruptionTests(unittest.TestCase):
    """A crash inside the apply loop must not abandon a job forever.

    The loop writes each replacement input before any inputs.txt is rewritten,
    so an interruption leaves a corrected input on disk with the mislaunched
    original still listed and active. If the next run treats that file as a
    foreign obstacle it skips the job permanently and the saddle is relaxed to a
    minimum again -- the exact defect relaunch-routes exists to repair.
    """

    def _crash_midway(self, campaign):
        real = relaunch_module.write_text_lf
        state = {"n": 0}

        def dying(path, text):
            if path.suffix == ".gjf":
                state["n"] += 1
                if state["n"] == 2:
                    raise KeyboardInterrupt("simulated interruption")
            return real(path, text)

        relaunch_module.write_text_lf = dying
        try:
            prepare_route_relaunch(campaign, assume_stopped=True)
            self.fail("expected the simulated interruption")
        except KeyboardInterrupt:
            pass
        finally:
            relaunch_module.write_text_lf = real

    def test_rerun_completes_an_interrupted_relaunch(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _ = build_campaign(Path(tmp) / "c")
            self._crash_midway(campaign)
            self.assertTrue(orphaned_corrections(campaign),
                            "fixture precondition: the crash should orphan a correction")

            again = prepare_route_relaunch(campaign, assume_stopped=True)
            self.assertEqual(again.get("skipped", []), [],
                             "the re-run must adopt its own half-written work, not skip it")
            self.assertEqual(reference_saddles_missing_ts(campaign), [])
            self.assertEqual(orphaned_corrections(campaign), [])

    def test_interrupted_run_leaves_no_unbacked_listing(self):
        """Nothing may be listed for the launcher without a manifest row: the
        launcher would produce logs collect cannot attribute."""
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _ = build_campaign(Path(tmp) / "c")
            self._crash_midway(campaign)
            rows = list(csv.DictReader((campaign / "jobs.csv").open(newline="", encoding="utf-8")))
            known = {Path(r["input"]).name for r in rows}
            listed = {n for listing in campaign.glob("slurm_batches/*/inputs.txt")
                      for n in listing.read_text(encoding="utf-8").split()}
            self.assertEqual(sorted(listed - known), [])

    def test_second_run_on_a_healthy_campaign_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _ = build_campaign(Path(tmp) / "c")
            prepare_route_relaunch(campaign, assume_stopped=True)
            before = snapshot(campaign)
            with self.assertRaises(RuntimeError):
                prepare_route_relaunch(campaign, assume_stopped=True)
            after = snapshot(campaign)
            # only the run-report files may change
            churn = {n for n in set(before) | set(after)
                     if before.get(n) != after.get(n)}
            self.assertTrue(
                churn <= {"skipped_route_fixes.csv", "monitoring/route_audit.csv",
                          "monitoring/route_audit.json", "monitoring/route_audit.md",
                          "monitoring/route_relaunch_candidates.csv"},
                f"a no-op second run mutated the campaign: {sorted(churn)}",
            )

    def test_already_archived_log_is_adopted_not_dropped(self):
        """If an interrupted run already renamed the poisoned log, the next run
        must adopt that file. Dropping to archived_output=None instead leaves
        the superseded row pointing at a path that no longer exists, orphaning
        the mislaunched result from the manifest that explains it."""
        with tempfile.TemporaryDirectory() as tmp:
            campaign, broken = build_campaign(Path(tmp) / "c")
            finished = broken[0]
            log = next(campaign.glob(f"slurm_batches/*/{finished}.log"))
            archived = log.with_name(f"{finished}__before-routefix01.log")
            log.rename(archived)  # as an interrupted earlier run would leave it

            prepare_route_relaunch(campaign, assume_stopped=True)

            rows = _rows(campaign / "jobs.csv")
            superseded = [r for r in rows
                          if Path(r["input"]).stem == finished
                          and (r.get("submission_active", "") or "").lower() == "false"]
            self.assertTrue(superseded, "the mislaunched row should be retired")
            self.assertEqual(
                superseded[0]["output"], archived.name,
                "the retired row must point at the archived log that exists on disk",
            )
            self.assertTrue(archived.is_file())

    def test_skip_report_is_written_even_when_nothing_to_do(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _ = build_campaign(Path(tmp) / "c")
            prepare_route_relaunch(campaign, assume_stopped=True)
            report = campaign / "skipped_route_fixes.csv"
            report.unlink(missing_ok=True)
            with self.assertRaises(RuntimeError):
                prepare_route_relaunch(campaign, assume_stopped=True)
            self.assertTrue(report.is_file(),
                            "the run report must describe the run that just happened")


class CommitOrderTests(unittest.TestCase):
    def test_manifest_is_committed_before_the_batch_listings(self):
        """A manifest naming a not-yet-listed replacement is inert; a listed
        replacement with no manifest row is data loss. Crash-order matters."""
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _ = build_campaign(Path(tmp) / "c")
            order = []
            real_csv = relaunch_module._write_csv
            real_lf = relaunch_module.write_text_lf

            def note_csv(path, fields, rows):
                if path.name in ("jobs.csv", "spin_jobs.csv"):
                    order.append("manifest")
                return real_csv(path, fields, rows)

            def note_lf(path, text):
                if path.name == "inputs.txt":
                    order.append("inputs.txt")
                return real_lf(path, text)

            relaunch_module._write_csv = note_csv
            relaunch_module.write_text_lf = note_lf
            try:
                prepare_route_relaunch(campaign, assume_stopped=True)
            finally:
                relaunch_module._write_csv = real_csv
                relaunch_module.write_text_lf = real_lf

            self.assertIn("manifest", order)
            self.assertIn("inputs.txt", order)
            self.assertLess(order.index("manifest"), order.index("inputs.txt"))

    def test_campaign_hash_matches_the_manifest_afterwards(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _ = build_campaign(Path(tmp) / "c")
            prepare_route_relaunch(campaign, assume_stopped=True)
            meta_path = campaign / "campaign_manifest.json"
            if not meta_path.is_file():
                self.skipTest("flat campaigns may not carry campaign_manifest.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256((campaign / "jobs.csv").read_bytes()).hexdigest()
            for key in ("jobs_csv_sha256", "manifest_sha256"):
                if key in meta:
                    self.assertEqual(meta[key], digest,
                                     f"{key} does not describe the manifest on disk")


class RebuiltRowProvenanceTests(unittest.TestCase):
    """The rebuilt manifest rows must not point at the discarded checkpoints.

    restart.py resolves `checkpoint` / `predecessor_checkpoint` straight to
    files on disk, where the pre-relaunch checkpoints deliberately still live.
    A rebuilt row carrying the old name makes the next prepare-spin-restarts
    seed from the collapsed geometry.
    """

    def _spin_row(self):
        return {
            "job_id": "j1", "input": "inputs/lad.gjf", "output": "lad.log",
            "checkpoint": "chain-s01-m9.chk",
            "predecessor_checkpoint": "chain-s00-m11.chk",
            "checkpoint_lineage": "m11:chain-s00-m11.chk>m9:chain-s01-m9.chk",
            "restart_attempt": "1",
            "restart_root_input": "inputs/lad.gjf",
            "restart_seed_checkpoint": "chain-s01-m9-seed-r01.chk",
            "restart_seed_sha256": "deadbeef",
        }

    RENAMES = {
        "chain-s01-m9.chk": "chain-s01-m9-routefix01.chk",
        "chain-s00-m11.chk": "chain-s00-m11-routefix01.chk",
    }

    def test_checkpoint_columns_follow_the_input_rename(self):
        row = self._spin_row()
        _reseat_rebuilt_row(row, self.RENAMES)
        self.assertEqual(row["checkpoint"], "chain-s01-m9-routefix01.chk")
        self.assertEqual(row["predecessor_checkpoint"], "chain-s00-m11-routefix01.chk")
        self.assertEqual(
            row["checkpoint_lineage"],
            "m11:chain-s00-m11-routefix01.chk>m9:chain-s01-m9-routefix01.chk",
        )

    def test_restart_provenance_is_cleared(self):
        """A rebuilt job is not a restart of anything. Inheriting the root's
        restart_attempt resets restart.py's -rNN counter, so the next restart
        derives checkpoint names that already exist on disk."""
        row = self._spin_row()
        _reseat_rebuilt_row(row, self.RENAMES)
        for column in ("restart_attempt", "restart_root_input",
                       "restart_seed_checkpoint", "restart_seed_sha256"):
            self.assertEqual(row[column], "", f"{column} was inherited")

    def test_unknown_checkpoint_is_left_alone(self):
        row = self._spin_row()
        row["checkpoint"] = "unrelated.chk"
        _reseat_rebuilt_row(row, self.RENAMES)
        self.assertEqual(row["checkpoint"], "unrelated.chk")

    def test_flat_rows_without_checkpoint_columns_are_unharmed(self):
        row = {"job_id": "j1", "input": "a.gjf", "output": "a.log"}
        _reseat_rebuilt_row(row, self.RENAMES)
        self.assertEqual(row, {"job_id": "j1", "input": "a.gjf", "output": "a.log"})

    def test_relaunched_spin_input_and_manifest_agree_on_checkpoints(self):
        """End-to-end on a SPIN campaign, where the checkpoint columns exist.

        A flat jobs.csv has no checkpoint columns at all, so this property can
        only be exercised against spin_jobs.csv -- which is also the manifest
        restart.py reads.
        """
        with tempfile.TemporaryDirectory() as tmp:
            campaign = build_spin_campaign(Path(tmp) / "s")
            before = {r["job_id"]: dict(r) for r in _rows(campaign / "spin_jobs.csv")}
            prepare_route_relaunch(campaign, assume_stopped=True)
            rows = _rows(campaign / "spin_jobs.csv")
            rebuilt = [r for r in rows if r.get("relaunch_attempt")]
            self.assertTrue(rebuilt, "the spin fixture should have been relaunched")

            for row in rebuilt:
                gjf = campaign / row["input"]
                used = {Path(c).name for c in
                        re.findall(r"(?im)^\s*%(?:old)?chk=(\S+)",
                                   gjf.read_text(encoding="utf-8"))}
                for column in ("checkpoint", "predecessor_checkpoint"):
                    named = (row.get(column) or "").strip()
                    if named:
                        self.assertIn(
                            named, used,
                            f"{column}={named!r} names a checkpoint {gjf.name} never writes; "
                            "prepare-spin-restarts would seed from the collapsed geometry",
                        )
                        self.assertIn("routefix", named)
            # and the pre-relaunch names must be gone from the rebuilt rows
            stale = {(before[r["relaunch_of_job_id"]].get("checkpoint") or "")
                     for r in rebuilt if r.get("relaunch_of_job_id") in before}
            for row in rebuilt:
                self.assertNotIn(row.get("checkpoint", ""), stale - {""})


if __name__ == "__main__":
    unittest.main()
