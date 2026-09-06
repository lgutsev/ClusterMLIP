import csv
import json
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.cli import build_parser
from cluster_mlip.dataset import read_labeled_extxyz
from cluster_mlip.models import Atom, Record
from cluster_mlip.spin import write_spin_jobs

FIXTURES = Path(__file__).parent / "fixtures"


def _collect(campaign: Path, dataset: Path, *extra: str) -> int:
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
            *extra,
        ]
    )
    return args.func(args)


def _opt_step(charge: int, multiplicity: int, x: float, energy: float) -> str:
    return f"""
 Input orientation:
 ---------------------------------------------------------------------
 Center     Atomic     Atomic              Coordinates (Angstroms)
 Number     Number      Type              X           Y           Z
 ---------------------------------------------------------------------
    1         26             0        0.000000    0.000000    0.000000
    2         26             0        {x:.6f}    0.000000    0.000000
 ---------------------------------------------------------------------
 SCF Done:  E(UBPW91) =  {energy:.7f}     A.U. after  12 cycles
 Forces (Hartrees/Bohr)
 -------------------------------------------------------------------
 Center     Atomic                   Forces (Hartrees/Bohr)
 Number     Number              X              Y              Z
 -------------------------------------------------------------------
      1       26           {0.02 - x/100:.9f}    0.000000000    0.000000000
      2       26          {-(0.02 - x/100):.9f}    0.000000000    0.000000000
 -------------------------------------------------------------------
"""


def _ladder_section(charge: int, multiplicity: int, up: float, down: float) -> str:
    return (
        f"\n Charge = {charge} Multiplicity = {multiplicity}\n"
        + _opt_step(charge, multiplicity, 2.10, -100.0)
        + _opt_step(charge, multiplicity, 2.00, -100.2)
        + f"""
 Mulliken charges and spin densities:
              1          2
     1  Fe   0.000000   {up:.6f}
     2  Fe   0.000000   {down:.6f}
 Sum of Mulliken charges = 0.00000 Sum of Mulliken spin densities = {up + down:.5f}
 Stationary point found.
 The wavefunction is stable under the perturbations considered.
"""
    )


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


class SpinCollectionTests(unittest.TestCase):
    def _campaign(self, tmp: Path) -> Path:
        campaign = tmp / "spin_campaign"
        record = Record(
            "fe2-m9", "warehouse/fe2_m9.log",
            [Atom("Fe", 0.0, 0.0, 0.0), Atom("Fe", 2.0, 0.0, 0.0)], 0, 9, "minimum",
        )
        write_spin_jobs([record], campaign, 9, [7], strategy="ladder")
        with (campaign / "spin_jobs.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        out_name = rows[0]["output"]
        (campaign / out_name).write_text(
            _ladder_section(0, 9, 4.0, 4.0)
            + _ladder_section(0, 7, 3.9, -3.9)
            + "\n Normal termination of Gaussian 16\n",
            encoding="utf-8",
        )
        return campaign

    def test_collect_reads_spin_jobs_and_labels_each_multiplicity(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._campaign(Path(tmp))
            dataset = Path(tmp) / "dataset"
            self.assertEqual(_collect(campaign, dataset), 0)
            frames = read_labeled_extxyz(dataset / "all.extxyz")
            self.assertEqual(sorted(f.record.multiplicity for f in frames), [7, 9])
            by_mult = {f.record.multiplicity: f for f in frames}
            # Only the converged geometry of each rung by default.
            self.assertEqual(len(frames), 2)
            # Provenance carried through.
            meta = by_mult[7].record.metadata
            self.assertEqual(meta["spin_chain_id"], by_mult[9].record.metadata["spin_chain_id"])
            self.assertEqual(meta["parent_record_id"], meta["spin_chain_id"])
            self.assertIn("checkpoint_lineage", meta)
            self.assertEqual(meta["label_route"], json.loads((campaign / "spin_campaign.json").read_text())["route"])
            # Local Fe moments kept as diagnostic metadata (AFM rung m7).
            self.assertEqual(meta["mulliken_spin_per_atom"], "3.9000,-3.9000")

    def test_include_opt_trajectory_keeps_intermediate_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._campaign(Path(tmp))
            dataset = Path(tmp) / "dataset"
            self.assertEqual(_collect(campaign, dataset, "--include-opt-trajectory"), 0)
            frames = read_labeled_extxyz(dataset / "all.extxyz")
            # two rungs x two optimisation steps each
            self.assertEqual(len(frames), 4)
            step_types = sorted({f.record.config_type for f in frames})
            self.assertTrue(any(t.endswith("_opt_step") for t in step_types))


if __name__ == "__main__":
    unittest.main()
