import csv
import io
import os
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from cluster_mlip.cli import build_parser
from cluster_mlip.hold import hold_inputs, read_input_list


def _campaign(root: Path) -> Path:
    """Two batches: batch_0001 lists a, b, c; batch_0002 lists d."""
    campaign = root / "campaign"
    for batch, names in (("batch_0001", "abc"), ("batch_0002", "d")):
        folder = campaign / "slurm_batches" / batch
        folder.mkdir(parents=True)
        for name in names:
            (folder / f"{name}.gjf").write_text("%chk=x.chk\n#p Opt\n\nt\n\n0 1\nH 0 0 0\n\n")
        (folder / "inputs.txt").write_text("".join(f"{name}.gjf\n" for name in names))
    return campaign


def _listing(campaign: Path, batch: str) -> list[str]:
    return (campaign / "slurm_batches" / batch / "inputs.txt").read_text().split()


class ListFileTests(unittest.TestCase):
    def test_reads_plain_lists_tables_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.tsv"
            path.write_text("# duplicates\ninput\treason\ninputs/a.gjf\tdup of b\n"
                            "b.gjf\n\n\"c.gjf\",x\na.gjf\n")
            self.assertEqual(read_input_list(path), ["a.gjf", "b.gjf", "c.gjf"])


class HoldTests(unittest.TestCase):
    def test_dry_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = _campaign(Path(tmp))
            result = hold_inputs(campaign, ["a.gjf"], reason="duplicate", dry_run=True)
            self.assertEqual(result["input_count"], 1)
            self.assertEqual(_listing(campaign, "batch_0001"), ["a.gjf", "b.gjf", "c.gjf"])
            self.assertFalse((campaign / "held_inputs.csv").exists())
            self.assertFalse((campaign / "hold_backups").exists())

    def test_hold_then_release_restores_the_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = _campaign(Path(tmp))
            batch = campaign / "slurm_batches" / "batch_0001"
            result = hold_inputs(campaign, ["b.gjf", "inputs/a.gjf"], reason="duplicate")
            self.assertEqual((result["input_count"], result["batches_touched"]), (2, 1))
            self.assertEqual(_listing(campaign, "batch_0001"), ["c.gjf"])
            held = (batch / "inputs.held.txt").read_text().splitlines()
            self.assertEqual([line.split("\t")[:2] for line in held],
                             [["b.gjf", "duplicate"], ["a.gjf", "duplicate"]])
            self.assertTrue((batch / "a.gjf").is_file())  # the input itself is untouched
            with (campaign / "held_inputs.csv").open(newline="") as handle:
                self.assertEqual([row["input"] for row in csv.DictReader(handle)],
                                 ["b.gjf", "a.gjf"])
            backups = list((campaign / "hold_backups").rglob("batch_0001.inputs.txt"))
            self.assertEqual(backups[0].read_text().split(), ["a.gjf", "b.gjf", "c.gjf"])

            released = hold_inputs(campaign, ["a.gjf", "b.gjf"], release=True)
            self.assertEqual(released["input_count"], 2)
            self.assertEqual(sorted(_listing(campaign, "batch_0001")), ["a.gjf", "b.gjf", "c.gjf"])
            self.assertFalse((batch / "inputs.held.txt").exists())
            with (campaign / "held_inputs.csv").open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            with (campaign / "hold_history.csv").open(newline="") as handle:
                self.assertEqual([row["action"] for row in csv.DictReader(handle)],
                                 ["hold", "hold", "release", "release"])

    def test_inputs_that_ran_or_may_be_running_are_not_held(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = _campaign(Path(tmp))
            batch = campaign / "slurm_batches" / "batch_0001"
            (batch / "a.log").write_text(" Entering Gaussian System\n")
            (batch / "b.started").write_text("started\n")
            result = hold_inputs(campaign, ["a.gjf", "b.gjf", "zzz.gjf"])
            self.assertEqual(result["input_count"], 0)
            reasons = {row["input"]: row["reason"] for row in result["skipped"]}
            self.assertIn("already has output", reasons["a.gjf"])
            self.assertIn("activity_unconfirmed", reasons["b.gjf"])
            self.assertIn("not listed", reasons["zzz.gjf"])
            self.assertEqual(hold_inputs(campaign, ["b.gjf"], assume_stopped=True)["input_count"], 1)

    def test_a_finished_marker_newer_than_started_is_confirmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = _campaign(Path(tmp))
            batch = campaign / "slurm_batches" / "batch_0001"
            (batch / "b.started").write_text("s\n")
            (batch / "b.finished").write_text("f\n")
            later = time.time() + 5
            os.utime(batch / "b.finished", (later, later))
            self.assertEqual(hold_inputs(campaign, ["b.gjf"])["input_count"], 1)

    def test_a_batch_is_never_emptied_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = _campaign(Path(tmp))
            result = hold_inputs(campaign, ["a.gjf", "b.gjf", "c.gjf", "d.gjf"], dry_run=True)
            self.assertEqual([row["input"] for row in result["plan"]], ["a.gjf", "b.gjf"])
            self.assertEqual(len(result["skipped"]), 2)
            self.assertIn("no inputs", result["skipped"][0]["reason"])
            self.assertEqual(result["empty_batches"], [])
            allowed = hold_inputs(campaign, ["d.gjf"], allow_empty_batches=True)
            self.assertEqual(allowed["empty_batches"], ["batch_0002"])
            self.assertEqual(_listing(campaign, "batch_0002"), [])

    def test_holding_twice_and_releasing_what_is_not_held_are_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = _campaign(Path(tmp))
            hold_inputs(campaign, ["a.gjf"])
            again = hold_inputs(campaign, ["a.gjf"])
            self.assertEqual(again["skipped"][0]["reason"], "already held")
            self.assertEqual(hold_inputs(campaign, ["b.gjf"], release=True)["skipped"][0]["reason"],
                             "not held")

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = _campaign(Path(tmp))
            listing = Path(tmp) / "hold.txt"
            listing.write_text("a.gjf\n")
            args = build_parser().parse_args(
                ["hold-inputs", str(campaign), "--list", str(listing), "--reason", "duplicate"])
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                self.assertEqual(args.func(args), 0)
            self.assertIn("Held 1 of 1 listed input(s) in 1 batch(es)", out.getvalue())
            self.assertEqual(_listing(campaign, "batch_0001"), ["b.gjf", "c.gjf"])


if __name__ == "__main__":
    unittest.main()
