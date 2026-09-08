import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from cluster_mlip.cli import build_parser
from cluster_mlip.dataset import read_labeled_extxyz
from cluster_mlip.gaussian import (
    final_scf_not_converged,
    force_label_quality,
    parse_final_force_frame,
)
from cluster_mlip.jobs import DEFAULT_ROUTE, route_without_scf_continue

FIXTURES = Path(__file__).parent / "fixtures"
NORMAL = "\n Normal termination of Gaussian 16\n"


class ScfConvergenceHelpersTests(unittest.TestCase):
    def test_converged_force_log_passes_the_gate(self):
        text = (FIXTURES / "force.log").read_text()
        self.assertFalse(final_scf_not_converged(text))
        self.assertEqual(force_label_quality(text), (True, ""))

    def test_iop_513_continuation_is_rejected(self):
        text = (FIXTURES / "scf_not_converged.log").read_text()
        # There is still a parseable energy/geometry/force frame ...
        self.assertIsNotNone(
            parse_final_force_frame(text, FIXTURES / "scf_not_converged.log")
        )
        # ... but IOP(5/13=1) let the job finish "Normal termination" on an
        # unconverged final SCF, so it must not become a training label.
        self.assertTrue(final_scf_not_converged(text))
        usable, reason = force_label_quality(text)
        self.assertFalse(usable)
        self.assertIn("did not converge", reason)

    def test_error_termination_is_rejected(self):
        text = (FIXTURES / "force.log").read_text() + "\n Error termination via Lnk1e in l502.exe\n"
        usable, reason = force_label_quality(text)
        self.assertFalse(usable)
        self.assertIn("error termination", reason)

    def test_transient_early_nonconvergence_that_recovered_is_not_held_against_the_frame(self):
        good = (FIXTURES / "force.log").read_text()
        text = " >>>>>>>>>> Convergence criterion not met.\n SCF Done:  E(UBPW91) = -1.0 A.U.\n" + good
        self.assertFalse(final_scf_not_converged(text))


class RouteWithoutScfContinueTests(unittest.TestCase):
    def test_drops_only_the_5_13_term(self):
        stripped = route_without_scf_continue(DEFAULT_ROUTE)
        self.assertIn("IOP(5/13=1", DEFAULT_ROUTE)
        self.assertNotIn("5/13=1", stripped)
        self.assertIn("5/36=1", stripped)
        self.assertIn("8/11=1", stripped)
        self.assertNotIn("  ", stripped)

    def test_drops_the_iop_wrapper_when_5_13_was_its_only_term(self):
        self.assertEqual(
            route_without_scf_continue("#p UBPW91/Gen Force IOP(5/13=1) NoSymm"),
            "#p UBPW91/Gen Force NoSymm",
        )

    def test_prepare_no_scf_continue_removes_the_iop_from_every_stage(self):
        records = build_parser()  # import side effect only; real check below
        del records
        with tempfile.TemporaryDirectory() as tmp:
            seeds = Path(tmp) / "seeds.extxyz"
            seeds.write_text(
                "2\n"
                'Properties=species:S:1:pos:R:3 charge=0 spin=1 config_type=minimum '
                'record_id=r0 source=x\n'
                "Fe 0.0 0.0 0.0\n O 1.6 0.0 0.0\n",
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                ["prepare", str(seeds), "-o", str(Path(tmp) / "jobs"),
                 "--no-scf-continue", "--rattles-per-seed", "1"]
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(args.func(args), 0)
            for gjf in (Path(tmp) / "jobs").glob("*.gjf"):
                self.assertNotIn("5/13=1", gjf.read_text())


class CollectQualityGateTests(unittest.TestCase):
    def _collect(self, campaign: Path, dataset: Path, *extra: str) -> int:
        args = build_parser().parse_args(
            ["collect", str(campaign), "-o", str(dataset),
             "--valid-fraction", "0", "--test-fraction", "0", *extra]
        )
        with redirect_stdout(io.StringIO()):
            return args.func(args)

    def _campaign(self, root: Path, **logs: str) -> Path:
        campaign = root / "campaign"
        campaign.mkdir()
        with (campaign / "jobs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["job_id", "output"])
            writer.writeheader()
            for job_id in logs:
                writer.writerow({"job_id": job_id, "output": f"{job_id}.log"})
        for job_id, text in logs.items():
            (campaign / f"{job_id}.log").write_text(text, encoding="utf-8")
        return campaign

    def test_non_converged_output_is_rejected_not_labeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._campaign(
                Path(tmp),
                good=(FIXTURES / "force.log").read_text() + NORMAL,
                bad=(FIXTURES / "scf_not_converged.log").read_text(),
            )
            dataset = Path(tmp) / "dataset"
            self.assertEqual(self._collect(campaign, dataset), 0)
            frames = read_labeled_extxyz(dataset / "all.extxyz")
            self.assertEqual([f.record.record_id for f in frames], ["good"])
            failures = (dataset / "failed_outputs.tsv").read_text(encoding="utf-8")
            self.assertIn("bad.log", failures)
            self.assertIn("did not converge", failures)

    def test_allow_partial_salvages_only_the_converged_frames(self):
        # First rung converged + optimized, second rung ran under IOP(5/13=1)
        # without converging. --frames converged --allow-partial must keep the
        # first and drop the second rather than salvage a bad label.
        log = (FIXTURES / "scf_not_converged.log").read_text().replace(
            " --Link1--", " Optimization completed.\n Stationary point found.\n --Link1--", 1
        )
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self._campaign(Path(tmp), bad=log)
            dataset = Path(tmp) / "dataset"
            self.assertEqual(
                self._collect(campaign, dataset, "--frames", "converged", "--allow-partial"), 0
            )
            frames = read_labeled_extxyz(dataset / "all.extxyz")
            self.assertEqual(len(frames), 1)
            self.assertAlmostEqual(frames[0].energy_ev, -1.1 * 27.211386245988, places=6)
            self.assertIn(
                "unconverged-SCF", (dataset / "failed_outputs.tsv").read_text(encoding="utf-8")
            )


if __name__ == "__main__":
    unittest.main()
