import csv
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.cli import build_parser
from cluster_mlip.dataset import read_labeled_extxyz
from cluster_mlip.jobs import write_gaussian_jobs
from cluster_mlip.models import Atom, Record
from cluster_mlip.progress import write_campaign_progress
from cluster_mlip.spin import write_spin_jobs


FIXTURES = Path(__file__).parent / "fixtures"


class CampaignProgressTests(unittest.TestCase):
    def test_progress_maps_human_output_to_machine_and_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "campaign"
            record = Record(
                "stable-machine-id",
                "Warehouse2.zip/batch/fe2o2_5_12345_LB.log",
                [Atom("H", 0, 0, 0), Atom("H", 0.75, 0, 0)],
                0,
                1,
                "minimum",
                legacy_energy_hartree=-1.0,
            )
            write_gaussian_jobs([record], campaign)
            pending = write_campaign_progress(campaign)
            self.assertEqual(pending["summary"]["by_state"], {"pending": 1})

            with (campaign / "jobs.csv").open(newline="") as handle:
                job = next(csv.DictReader(handle))
            batch = campaign / "slurm_batches" / "batch_0001"
            batch.mkdir(parents=True)
            output = batch / job["output"]
            output.write_text(
                (FIXTURES / "force.log").read_text()
                + "\n Normal termination of Gaussian 16\n"
            )
            (batch / f"{output.stem}.status").write_text("OK\n")
            complete = write_campaign_progress(campaign)
            row = complete["rows"][0]
            self.assertEqual(row["job_id"], "stable-machine-id")
            self.assertEqual(row["source"], "Warehouse2.zip/batch/fe2o2_5_12345_LB.log")
            self.assertEqual(row["campaign_state"], "complete")
            self.assertEqual(row["batch"], "batch_0001")
            self.assertTrue(row["force_frame_parsed"])
            self.assertAlmostEqual(float(row["new_label_energy_hartree"]), -1.1)
            self.assertTrue(complete["csv_path"].is_file())
            self.assertTrue(complete["summary_path"].is_file())

            dataset = Path(tmp) / "dataset"
            parser = build_parser()
            args = parser.parse_args(
                [
                    "collect",
                    str(campaign),
                    "-o",
                    str(dataset),
                    "--valid-fraction",
                    "0",
                    "--test-fraction",
                    "0",
                ]
            )
            self.assertEqual(args.func(args), 0)
            frames = read_labeled_extxyz(dataset / "all.extxyz")
            self.assertEqual(frames[0].record.record_id, "stable-machine-id")
            self.assertEqual(frames[0].record.metadata["human_id"], job["human_id"])
            self.assertEqual(
                frames[0].record.metadata["source_record_id"], "stable-machine-id"
            )
            self.assertEqual(
                frames[0].record.source,
                "Warehouse2.zip/batch/fe2o2_5_12345_LB.log",
            )

    def test_campaign_status_help_renders(self):
        parser = build_parser()
        command = parser._subparsers._group_actions[0].choices["campaign-status"]
        self.assertIn("progress CSV", command.format_help())

    def test_prepare_spins_accepts_exact_repeatable_record_ids(self):
        args = build_parser().parse_args([
            "prepare-spins", "seeds.extxyz", "--high-spin", "29", "--targets", "17",
            "--record-id", "fe10-a", "--record-id", "fe10-b",
        ])
        self.assertEqual(args.record_ids, ["fe10-a", "fe10-b"])

    def test_spin_campaign_progress_reports_each_link1_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "spin_campaign"
            record = Record(
                "fe2-high-spin", "warehouse/fe2_m9.log",
                [Atom("Fe", 0, 0, 0), Atom("Fe", 2, 0, 0)], 0, 9, "minimum",
            )
            write_spin_jobs([record], campaign, 9, [7], strategy="ladder")
            with (campaign / "spin_jobs.csv").open(newline="", encoding="utf-8") as handle:
                jobs = list(csv.DictReader(handle))
            output = campaign / jobs[0]["output"]
            output.write_text("""
 Charge = 0 Multiplicity = 9
 SCF Done: E(UHF) = -100.0 A.U. after 5 cycles
 S**2 before annihilation 20.0, after 20.0
 Mulliken charges and spin densities:
 1 Fe 0.0 4.0
 2 Fe 0.0 4.0
 Sum of Mulliken charges = 0.0 Sum of Mulliken spin densities = 8.0
 Stationary point found.
 The wavefunction is stable under the perturbations considered.
 Charge = 0 Multiplicity = 7
 SCF Done: E(UHF) = -100.1 A.U. after 5 cycles
 S**2 before annihilation 12.0, after 12.0
 Mulliken charges and spin densities:
 1 Fe 0.0 4.0
 2 Fe 0.0 2.0
 Sum of Mulliken charges = 0.0 Sum of Mulliken spin densities = 6.0
 Stationary point found.
 The wavefunction is stable under the perturbations considered.
 Normal termination of Gaussian 16
 """, encoding="utf-8")
            progress = write_campaign_progress(campaign)
            self.assertEqual(progress["summary"]["manifest"], "spin_jobs.csv")
            self.assertEqual(progress["summary"]["by_state"], {"complete": 2})
            self.assertEqual(
                [row["intended_multiplicity"] for row in progress["rows"]], ["9", "7"]
            )
            self.assertTrue(progress["rows"][0]["spin_stage_advanced_to_successor"])
            self.assertEqual(progress["rows"][1]["spin_stage_stability"], "stable")


if __name__ == "__main__":
    unittest.main()
