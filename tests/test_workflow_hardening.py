"""Regressions for the Gaussian-workflow hardening pass.

Each test here pins a defect that was live on main and that no existing test
covered. They are grouped by the failure they prevent, not by module.
"""

import csv
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.dataset import _stratum_take, grouped_split, write_labeled_extxyz
from cluster_mlip.io import write_text_lf
from cluster_mlip.models import Atom, LabeledFrame, Record
from cluster_mlip.physical_checks import spin_ordering_check, stationary_point_check
from cluster_mlip.slurm import SlurmConfig, prepare_slurm_batches
from cluster_mlip.spin import SPIN_MANIFEST_COLUMNS, render_ladder_input
from cluster_mlip.stratify import provenance_tier
from cluster_mlip.training import TrainingConfig, scan_dataset, write_training_campaign

FE_O = [Atom("Fe", 0.0, 0.0, 0.0), Atom("O", 1.8, 0.0, 0.0)]
FLAT_ROUTE = "#p UBPW91/Gen SCF=(Tight) NoSymm Opt"
SPIN_ROUTE = "#p UBPW91/6-311++G* SCF=(Tight) NoSymm Opt"


def _record(record_id: str, *, multiplicity: int = 5, config_type: str = "minimum",
            metadata: dict | None = None, atoms: list[Atom] | None = None) -> Record:
    return Record(
        record_id, "src.zip", list(atoms or FE_O), 0, multiplicity, config_type,
        metadata=dict(metadata or {}, parent_record_id=record_id),
    )


def _frame(record: Record, energy: float = -10.0,
           forces: list[tuple[float, float, float]] | None = None) -> LabeledFrame:
    return LabeledFrame(
        record, energy, forces or [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], Path("src.log")
    )


def _dataset(tmp: Path, frames: list[LabeledFrame]) -> Path:
    dataset = tmp / "dataset"
    dataset.mkdir()
    for name in ("all.extxyz", "train.extxyz", "valid.extxyz", "test.extxyz"):
        write_labeled_extxyz(frames, dataset / name)
    return dataset


class ClusterArtifactsUseLfTests(unittest.TestCase):
    """A campaign prepared on Windows is run by bash on Linux.

    `run_batch.sbatch` reads inputs.txt with `mapfile -t`, which strips only
    the trailing newline -- a CRLF listing leaves a carriage return on every
    filename and the batch matches none of its jobs. The same mismatch
    defeated the completion check in submit_gaussian_batches.sh, so a resume
    re-ran finished work.
    """

    def test_write_text_lf_never_emits_crlf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.sh"
            write_text_lf(path, "#!/usr/bin/env bash\nset -euo pipefail\n")
            self.assertNotIn(b"\r\n", path.read_bytes())
            self.assertEqual(path.read_bytes().count(b"\n"), 2)

    def _campaign(self, root: Path, count: int = 3) -> Path:
        campaign = root / "campaign"
        campaign.mkdir()
        with (campaign / "jobs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["job_id", "input", "output"])
            writer.writeheader()
            for index in range(count):
                name = f"job_{index:03d}.gjf"
                write_text_lf(campaign / name, f"%chk=job_{index:03d}.chk\n# force\n")
                writer.writerow({"job_id": f"job_{index:03d}", "input": name,
                                 "output": name[:-4] + ".log"})
        return campaign

    def test_generated_batch_artifacts_are_lf(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._campaign(Path(tmp))
            prepare_slurm_batches(campaign, SlurmConfig(jobs_per_batch=2))

            listings = sorted(campaign.glob("slurm_batches/*/inputs.txt"))
            self.assertTrue(listings, "expected at least one batch listing")
            scripts = sorted(campaign.glob("slurm_batches/*/*.sbatch"))
            scripts += sorted(campaign.glob("slurm_batches/*/submit.sh"))
            scripts += [campaign / "run_gaussian_worker.sh",
                        campaign / "submit_gaussian_batches.sh",
                        campaign / "gaussian_batch_status.sh"]

            for path in listings + scripts:
                with self.subTest(artifact=path.name):
                    self.assertTrue(path.is_file(), f"{path} was not generated")
                    self.assertNotIn(b"\r\n", path.read_bytes())

    def test_batch_listing_names_resolve_verbatim(self):
        """Every inputs.txt line must name a file that exists, byte-for-byte.

        This is the property bash relies on; a trailing \\r breaks it while
        leaving the listing looking correct in an editor.
        """
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._campaign(Path(tmp))
            prepare_slurm_batches(campaign, SlurmConfig(jobs_per_batch=2))
            for listing in sorted(campaign.glob("slurm_batches/*/inputs.txt")):
                raw = listing.read_bytes().decode("utf-8")
                for line in raw.split("\n"):
                    if not line:
                        continue
                    with self.subTest(batch=listing.parent.name, entry=repr(line)):
                        self.assertTrue((listing.parent / line).exists(),
                                        f"{line!r} does not resolve inside {listing.parent}")


class SpinManifestCarriesProvenanceTests(unittest.TestCase):
    """`collect` copies non-empty manifest cells into frame metadata, so a
    column absent from spin_jobs.csv can never reach the dataset."""

    def test_columns_declared(self):
        for column in ("first_route", "state_inference"):
            with self.subTest(column=column):
                self.assertIn(column, SPIN_MANIFEST_COLUMNS)

    def test_ladder_rows_record_route_and_inference(self):
        record = _record("seed1", multiplicity=11,
                         metadata={"state_inference": "filename"})
        _text, rows = render_ladder_input(record, 11, [9], route=SPIN_ROUTE)
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(stage=row["stage_index"]):
                self.assertIn("6-311++G*", row["first_route"])
                self.assertEqual(row["state_inference"], "filename")

    def test_filename_guessed_spin_state_is_not_reported_validated(self):
        guessed = _record("g", metadata={"state_inference": "filename"})
        measured = _record("m")
        self.assertEqual(provenance_tier(guessed), "filename_derived")
        self.assertEqual(provenance_tier(measured), "validated")


class MixedMethodGuardTests(unittest.TestCase):
    """Training one MACE head on two levels of theory is unsound. The guard
    only works if every frame reports the route that labeled it."""

    def _spin(self, i: int) -> LabeledFrame:
        return _frame(_record(f"spin{i}", metadata={
            "chain_id": f"c{i}", "first_route": SPIN_ROUTE}), -200.0 - i)

    def _flat(self, i: int) -> LabeledFrame:
        return _frame(_record(f"flat{i}", metadata={
            "first_route": FLAT_ROUTE,
            "link1_route": "#p UBPW91/Gen Force Guess=Read"}), -100.0 - i)

    def test_spin_only_dataset_reports_its_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _dataset(Path(tmp), [self._spin(i) for i in range(3)])
            self.assertEqual(scan_dataset(dataset).label_routes, {SPIN_ROUTE})

    def test_flat_and_spin_mix_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = [self._flat(0), self._flat(1), self._spin(0), self._spin(1)]
            dataset = _dataset(Path(tmp), frames)
            self.assertEqual(len(scan_dataset(dataset).label_routes), 2)
            with self.assertRaisesRegex(ValueError, "distinct force-label routes"):
                write_training_campaign(
                    TrainingConfig(dataset_dir=dataset, output_dir=Path(tmp) / "out")
                )

    def test_frame_without_any_route_is_refused(self):
        """An absent route is not evidence that the methods match."""
        with tempfile.TemporaryDirectory() as tmp:
            legacy = _frame(_record("old", metadata={"chain_id": "c0"}))
            dataset = _dataset(Path(tmp), [self._flat(0), legacy])
            with self.assertRaisesRegex(ValueError, "no label route"):
                scan_dataset(dataset)

    def test_first_route_wins_over_campaign_default(self):
        """relaunch-routes rewrites first_route; link1_route is only the
        campaign-wide force-stage default, identical across both frames here."""
        with tempfile.TemporaryDirectory() as tmp:
            shared = "#p UBPW91/Gen Force Guess=Read"
            before = _frame(_record("a", metadata={
                "first_route": "#p UBPW91/Gen Opt", "link1_route": shared}))
            after = _frame(_record("b", metadata={
                "first_route": "#p UBPW91/Gen Opt=(TS,CalcFC,NoEigenTest) Freq",
                "link1_route": shared}))
            dataset = _dataset(Path(tmp), [before, after])
            self.assertEqual(len(scan_dataset(dataset).label_routes), 2)


class StationaryPointCheckTests(unittest.TestCase):
    """pes_region strips the `_rattled` suffix, so `minimum_rattled` reports
    as `minimum`. A rattled frame is displaced on purpose and carries a large
    force -- scoring the model for predicting ~0 there penalizes correctness."""

    def test_rattled_frames_are_excluded(self):
        relaxed = _frame(_record("r", config_type="minimum"))
        rattled = _frame(_record("x", config_type="minimum_rattled"))
        quiet = (-10.0, [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
        loud = (-10.0, [(9.0, 0.0, 0.0), (-9.0, 0.0, 0.0)])

        result = stationary_point_check([relaxed, rattled], [quiet, loud])
        self.assertEqual(result["n_frames_considered"], 1)
        self.assertEqual(result["metric_value"], 0.0)
        self.assertIs(result["passed"], True)

    def test_relaxed_frames_still_counted(self):
        relaxed = _frame(_record("r", config_type="minimum"))
        saddle = _frame(_record("s", config_type="first_order_saddle"))
        quiet = (-10.0, [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
        result = stationary_point_check([relaxed, saddle], [quiet, quiet])
        self.assertEqual(result["n_frames_considered"], 2)

    def test_all_rattled_dataset_reports_no_data(self):
        rattled = _frame(_record("x", config_type="minimum_rattled"))
        loud = (-10.0, [(9.0, 0.0, 0.0), (-9.0, 0.0, 0.0)])
        result = stationary_point_check([rattled], [loud])
        self.assertIsNone(result["passed"])


class SpinOrderingCheckTests(unittest.TestCase):
    """Ladder stages are separately optimized at their own multiplicity, and a
    high-/low-spin change moves Fe-ligand bonds ~0.1-0.2 A. Grouping by a 0.05 A
    geometry tolerance put real siblings in separate clusters, so the check
    silently returned "no data" for the campaigns it was written for."""

    def _ladder_pair(self, displacement: float) -> list[LabeledFrame]:
        low = _record("a", multiplicity=5, metadata={"chain_id": "chain1"})
        high = _record("b", multiplicity=11, metadata={"chain_id": "chain1"},
                       atoms=[Atom("Fe", 0.0, 0.0, 0.0),
                              Atom("O", 1.8 + displacement, 0.0, 0.0)])
        return [_frame(low, -10.0), _frame(high, -9.0)]

    def test_relaxed_ladder_siblings_are_compared(self):
        frames = self._ladder_pair(0.15)  # beyond the old 0.05 A tolerance
        agree = [(-10.0, [(0.0, 0.0, 0.0)] * 2), (-9.0, [(0.0, 0.0, 0.0)] * 2)]
        result = spin_ordering_check(frames, agree)
        self.assertEqual(result["n_frames_considered"], 1)
        self.assertEqual(result["metric_value"], 1.0)

    def test_inverted_ground_state_is_caught(self):
        frames = self._ladder_pair(0.15)
        inverted = [(-9.0, [(0.0, 0.0, 0.0)] * 2), (-10.0, [(0.0, 0.0, 0.0)] * 2)]
        result = spin_ordering_check(frames, inverted)
        self.assertEqual(result["n_frames_considered"], 1)
        self.assertEqual(result["metric_value"], 0.0)
        self.assertIs(result["passed"], False)

    def test_geometry_fallback_without_chain_id(self):
        low = _record("a", multiplicity=5)
        high = _record("b", multiplicity=11)
        frames = [_frame(low, -10.0), _frame(high, -9.0)]
        agree = [(-10.0, [(0.0, 0.0, 0.0)] * 2), (-9.0, [(0.0, 0.0, 0.0)] * 2)]
        self.assertEqual(spin_ordering_check(frames, agree)["n_frames_considered"], 1)

    def test_geometry_fallback_spans_a_spin_state_relaxation(self):
        """Without chain_id the tolerance is the only thing grouping siblings,
        so it must be wide enough for a real high-/low-spin bond change. At the
        old 0.05 A these two land in separate clusters and the check goes
        silently to "no data"."""
        low = _record("a", multiplicity=5)
        high = _record("b", multiplicity=11,
                       atoms=[Atom("Fe", 0.0, 0.0, 0.0), Atom("O", 1.95, 0.0, 0.0)])
        frames = [_frame(low, -10.0), _frame(high, -9.0)]
        agree = [(-10.0, [(0.0, 0.0, 0.0)] * 2), (-9.0, [(0.0, 0.0, 0.0)] * 2)]
        result = spin_ordering_check(frames, agree)
        self.assertEqual(result["n_frames_considered"], 1)
        self.assertIsNotNone(result["passed"])

    def test_distinct_species_are_not_merged(self):
        """The wider tolerance must not start comparing unrelated structures."""
        low = _record("a", multiplicity=5)
        far = _record("b", multiplicity=11,
                      atoms=[Atom("Fe", 0.0, 0.0, 0.0), Atom("O", 3.6, 0.0, 0.0)])
        frames = [_frame(low, -10.0), _frame(far, -9.0)]
        preds = [(-10.0, [(0.0, 0.0, 0.0)] * 2), (-9.0, [(0.0, 0.0, 0.0)] * 2)]
        self.assertIsNone(spin_ordering_check(frames, preds)["passed"])

    def test_separate_chains_are_not_compared(self):
        one = _record("a", multiplicity=5, metadata={"chain_id": "c1"})
        two = _record("b", multiplicity=11, metadata={"chain_id": "c2"})
        frames = [_frame(one, -10.0), _frame(two, -9.0)]
        preds = [(-10.0, [(0.0, 0.0, 0.0)] * 2), (-9.0, [(0.0, 0.0, 0.0)] * 2)]
        self.assertIsNone(spin_ordering_check(frames, preds)["passed"])


class StratifiedSplitTests(unittest.TestCase):
    """round(n * 0.10) is 0 for every n <= 5 (Python rounds half to even), so
    small strata were placed entirely in train -- making certain the empty
    bucket that stratified splitting exists to prevent."""

    def test_small_strata_get_held_out_groups(self):
        for n in range(2, 8):
            with self.subTest(groups=n):
                test = _stratum_take(n, 0.10, n - 1)
                valid = _stratum_take(n, 0.10, n - test - 1)
                self.assertGreaterEqual(test, 1)
                if n >= 3:
                    self.assertGreaterEqual(valid, 1)
                self.assertGreater(n - test - valid, 0, "train must not be emptied")

    def test_single_group_stratum_stays_in_train(self):
        self.assertEqual(_stratum_take(1, 0.10, 0), 0)

    def test_zero_fraction_takes_nothing(self):
        self.assertEqual(_stratum_take(50, 0.0, 49), 0)

    def test_large_stratum_stays_proportional(self):
        self.assertEqual(_stratum_take(200, 0.10, 199), 20)

    def test_rare_stratum_reaches_test_split(self):
        frames = [
            _frame(_record(f"min{i}", config_type="minimum")) for i in range(20)
        ] + [
            _frame(_record(f"sad{i}", config_type="higher_order_saddle")) for i in range(3)
        ]
        splits = grouped_split(frames, 0.10, 0.10, seed=7, stratify_by=("pes_region",))
        saddle_splits = {
            name: [f for f in members if f.record.config_type == "higher_order_saddle"]
            for name, members in splits.items()
        }
        self.assertTrue(saddle_splits["test"], "rare stratum absent from test")
        self.assertTrue(saddle_splits["valid"], "rare stratum absent from valid")
        self.assertTrue(saddle_splits["train"], "rare stratum absent from train")

    def test_every_frame_lands_in_exactly_one_split(self):
        frames = [_frame(_record(f"r{i}")) for i in range(37)]
        splits = grouped_split(frames, 0.10, 0.10, seed=3,
                               stratify_by=("pes_region", "charge_spin_class"))
        placed = [f.record.record_id for members in splits.values() for f in members]
        self.assertEqual(sorted(placed), sorted(f.record.record_id for f in frames))


if __name__ == "__main__":
    unittest.main()
