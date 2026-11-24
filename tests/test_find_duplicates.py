import csv
import json
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.duplicates import find_duplicate_inputs
from cluster_mlip.hold import hold_inputs, read_input_list
from test_irc_force_labels import FE2O2, W2Campaign, ladder_input, ladder_routes


def _job(name, batch, config_type="minimum", log=None):
    return {"name": f"{name}__fe2o2__{config_type.replace('_', '-')}__q0-m11__reference__{name}",
            "config_type": config_type, "batch": batch,
            "source": f"warehouse2/{name}.log", "log": log}


def _rewrite(campaign: Path, job: dict, atoms) -> None:
    """Give one job a different stage-0 geometry, in inputs/ and in its batch."""
    text = ladder_input(job["name"], ladder_routes(len(W2Campaign.MULTS)), W2Campaign.MULTS, atoms)
    for folder in (campaign / "inputs", campaign / "slurm_batches" / job["batch"]):
        (folder / f"{job['name']}.gjf").write_text(text, encoding="utf-8", newline="\n")


class FindDuplicateTests(W2Campaign, unittest.TestCase):
    def _campaign(self, root: Path):
        jobs = {
            "done": _job("done", "batch_0001", log="complete"),
            "same": _job("same", "batch_0001"),
            "turned": _job("turned", "batch_0002"),      # rotated, shifted, atoms reordered
            "stretched": _job("stretched", "batch_0002"),  # a different structure
            "saddle": _job("saddle", "batch_0002", config_type="transition_state"),
            # Same geometry and ladder as "done", only the label differs.
            "unlabeled": _job("unlabeled", "batch_0002", config_type="unknown"),
            "pair_a": _job("paira", "batch_0001"),
            "pair_b": _job("pairb", "batch_0002"),
        }
        campaign = self.build(root, list(jobs.values()))
        # 90 degree turn about z, a shift, and the atom order reversed.
        turned = [(s, -y + 3.0, x - 1.0, z + 0.5) for s, x, y, z in reversed(FE2O2)]
        _rewrite(campaign, jobs["turned"], turned)
        stretched = [(s, x * 1.2, y, z) for s, x, y, z in FE2O2]
        _rewrite(campaign, jobs["stretched"], stretched)
        other = [(s, x, y * 1.3, z) for s, x, y, z in FE2O2]
        _rewrite(campaign, jobs["pair_a"], other)
        _rewrite(campaign, jobs["pair_b"], other)
        return campaign, jobs

    def test_lists_only_not_started_repeats_and_keeps_the_best_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, jobs = self._campaign(Path(tmp))
            before = sorted(str(p) for p in campaign.rglob("*"))
            out = Path(tmp) / "report"
            summary = find_duplicate_inputs(campaign, out)
            with (out / "duplicate_inputs.tsv").open(newline="") as handle:
                rows = {row["input"]: row for row in csv.DictReader(handle, delimiter="\t")}

            def name(key):
                return jobs[key]["name"] + ".gjf"

            # Repeats of the finished job, found through rotation and reordering.
            self.assertEqual(set(rows), {name("same"), name("turned"), name("pair_b"),
                                         name("unlabeled")})
            # A plain Opt is the same computation whatever the label says.
            self.assertEqual(rows[name("unlabeled")]["calculation"], "opt")
            self.assertEqual(rows[name("unlabeled")]["config_type"], "unknown")
            self.assertEqual(rows[name("unlabeled")]["kept_config_type"], "minimum")
            self.assertEqual(rows[name("same")]["kept_input"], name("done"))
            self.assertEqual(rows[name("turned")]["kept_state"], "complete")
            # Two not-started copies: the earlier batch is kept.
            self.assertEqual(rows[name("pair_b")]["kept_input"], name("pair_a"))
            self.assertEqual(rows[name("pair_b")]["kept_state"], "not_started")
            # A different geometry and a different kind of job are not duplicates.
            self.assertNotIn(name("stretched"), rows)
            self.assertNotIn(name("saddle"), rows)
            self.assertEqual(summary["not_started_duplicates"], 4)
            self.assertEqual(summary["groups"], 2)
            self.assertEqual(summary["label_of_duplicate_and_kept"]["unknown -> minimum"], 1)
            self.assertEqual(json.loads((out / "duplicate_inputs.json").read_text())["groups"], 2)
            self.assertEqual(sorted(str(p) for p in campaign.rglob("*")), before)

    def test_tolerance_decides_what_counts_as_the_same_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, jobs = self._campaign(Path(tmp))
            nudged = [(s, x + (0.01 if i == 0 else 0.0), y, z)
                      for i, (s, x, y, z) in enumerate(FE2O2)]
            _rewrite(campaign, jobs["same"], nudged)
            loose = find_duplicate_inputs(campaign, Path(tmp) / "loose")
            tight = find_duplicate_inputs(campaign, Path(tmp) / "tight", tolerance=0.001)
            self.assertEqual(loose["not_started_duplicates"], 4)
            self.assertEqual(tight["not_started_duplicates"], 3)

    def test_the_list_feeds_hold_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, jobs = self._campaign(Path(tmp))
            out = Path(tmp) / "report"
            find_duplicate_inputs(campaign, out)
            names = read_input_list(out / "duplicate_inputs.tsv")
            self.assertEqual(len(names), 4)
            result = hold_inputs(campaign, names, reason="duplicate", dry_run=True)
            self.assertEqual(result["input_count"], 4)
            self.assertEqual(result["skipped"], [])


if __name__ == "__main__":
    unittest.main()
