import csv
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.cli import build_parser
from cluster_mlip.dataset import read_labeled_extxyz

FIXTURES = Path(__file__).parent / "fixtures"


def _collect(campaign: Path, dataset: Path) -> int:
    args = build_parser().parse_args(
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
    return args.func(args)


class CollectQualityGateTests(unittest.TestCase):
    def test_non_converged_scf_output_is_rejected_not_labeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "campaign"
            campaign.mkdir()
            with (campaign / "jobs.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["job_id", "charge", "multiplicity", "output", "input"]
                )
                writer.writeheader()
                writer.writerow(
                    {"job_id": "good", "charge": "0", "multiplicity": "1", "output": "good.log", "input": "good.gjf"}
                )
                writer.writerow(
                    {"job_id": "bad", "charge": "0", "multiplicity": "1", "output": "bad.log", "input": "bad.gjf"}
                )
            (campaign / "good.log").write_text((FIXTURES / "force.log").read_text(), encoding="utf-8")
            (campaign / "bad.log").write_text(
                (FIXTURES / "scf_not_converged.log").read_text(), encoding="utf-8"
            )

            dataset = Path(tmp) / "dataset"
            self.assertEqual(_collect(campaign, dataset), 0)

            frames = read_labeled_extxyz(dataset / "all.extxyz")
            self.assertEqual([frame.record.record_id for frame in frames], ["good"])

            failures = (dataset / "failed_outputs.tsv").read_text(encoding="utf-8")
            self.assertIn("bad.log", failures)
            self.assertIn("did not converge", failures)


if __name__ == "__main__":
    unittest.main()
