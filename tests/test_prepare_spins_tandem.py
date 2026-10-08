"""prepare-spins --fragment-layout tandem: the archived two-link fragment tandem.

Draft, opt-in: the single-link Guess=(Fragment=N,Always) layout stays the default
until the fe16_bs_tandem_kit validation run comes back clean.
"""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from cluster_mlip.cli import build_parser
from cluster_mlip.fragment_tandem import (
    ARCHIVED_LEVEL, STAGE_CAMPAIGN, STAGE_FRAGMENT, fragment_route, inspect_tandem_input,
)
from cluster_mlip.models import Atom, Record
from cluster_mlip.routes import input_stage_routes, route_search_kind
from cluster_mlip.spin import (
    DEFAULT_SPIN_ROUTE, SPIN_MANIFEST_COLUMNS, _spin_manifest_audit, render_fragment_tandem_input,
    tandem_campaign_routes, write_spin_jobs,
)

TS_ROUTE = (
    "#p UBPW91/6-311++G* SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) "
    "NoSymm Opt=(TS,CalcFC,NoEigenTest) Freq IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine"
)


def fe2o2n2(config_type: str = "minimum") -> Record:
    # Fe2O2N2_DimAd_AFM_1 link-0 geometry from the archive (docs/fragment-tandem-method.md).
    atoms = [
        Atom("Fe", -1.04914, -0.09571, -0.31538),
        Atom("Fe", 1.08263, 0.08187, -0.21416),
        Atom("O", -0.07622, 1.35317, -0.69354),
        Atom("O", 0.14998, -1.36616, -0.68541),
        Atom("N", 0.52591, 0.0409, 1.58103),
        Atom("N", -0.66478, -0.05829, 1.52448),
    ]
    return Record("fe2o2n2", "legacy/fe2o2n2.log", atoms, 0, 9, config_type)


AFM = {
    "record_id": "fe2o2n2",
    "name": "afm",
    "target_multiplicity": 1,
    "fragments": [
        {"atoms": [1, 3], "charge": 0, "multiplicity": 5, "orientation": "alpha"},
        {"atoms": [2, 4], "charge": 0, "multiplicity": 5, "orientation": "beta"},
        {"atoms": [5, 6], "charge": 0, "multiplicity": 1},
    ],
}


class TandemRouteTests(unittest.TestCase):
    def test_default_route_reproduces_archived_links(self):
        link0, link1 = tandem_campaign_routes(DEFAULT_SPIN_ROUTE, 3)
        # Token for token the archived link-0 route (same template as prepare-bs-pilot).
        self.assertEqual(link0, fragment_route(ARCHIVED_LEVEL, 3))
        self.assertEqual(
            link1,
            "#p UBPW91/6-311++G* SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm Opt "
            "IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine Geom=Checkpoint Guess=Read",
        )
        self.assertNotIn("Always", link0 + link1)
        self.assertNotIn("Stable", link0 + link1)

    def test_ts_route_keeps_its_search_in_link1_only(self):
        link0, link1 = tandem_campaign_routes(TS_ROUTE, 3)
        self.assertIn(" SP ", link0)
        self.assertNotIn("Opt", link0)
        self.assertNotIn("Freq", link0)
        self.assertIn("Opt=(TS,CalcFC,NoEigenTest) Freq IOP(5/13=1,5/36=1,8/11=1)", link1)
        self.assertIn("IOP(5/13=1,5/36=1,8/11=1)", link0)
        self.assertEqual(route_search_kind(link1), "saddle")
        self.assertEqual(route_search_kind(link0), "none")

    def test_iop_5_13_is_added_to_both_links_when_missing(self):
        link0, link1 = tandem_campaign_routes("#p UB3LYP/6-31G* Opt", 2)
        self.assertEqual(link0, "#p UB3LYP/6-31G* SP IOP(5/13=1) Guess=(Fragment=2)")
        self.assertEqual(link1, "#p UB3LYP/6-31G* Opt IOP(5/13=1) Geom=Checkpoint Guess=Read")
        _, link1 = tandem_campaign_routes("#p UB3LYP/6-31G* Opt IOP(5/36=1)", 2)
        self.assertIn("IOP(5/13=1,5/36=1)", link1)

    def test_unsupported_routes_are_refused(self):
        for route, message in (
            ("#p UBPW91/Gen Opt", "Gen"),
            ("#p UBPW91/6-311++G* Opt Stable=Opt", "Stable"),
            ("#p UBPW91/6-311++G* Opt Guess=Mix", "guess"),
            ("#p UBPW91/6-311++G* SP", "needs a job"),
            ("#p UBPW91/6-311++G*", "needs a job"),
        ):
            with self.subTest(route=route), self.assertRaisesRegex(ValueError, message):
                tandem_campaign_routes(route, 2)


class TandemInputTests(unittest.TestCase):
    def test_rendered_tandem_layout(self):
        text, row, job = render_fragment_tandem_input(fe2o2n2(), AFM)
        self.assertEqual(job.stage_kinds(), [STAGE_FRAGMENT, STAGE_CAMPAIGN])
        self.assertEqual(text.count("--Link1--"), 1)
        link0, link1 = text.split("--Link1--\n")
        self.assertEqual(link0.count("%chk=fe2o2n2-fragment-afm-m1.chk"), 1)
        self.assertEqual(link1.count("%chk=fe2o2n2-fragment-afm-m1.chk"), 1)
        self.assertNotIn("%oldchk", text)
        self.assertIn("\n0 1 0 5 0 -5 0 1\n", link0)
        self.assertIn("Fe(Fragment=1)", link0)
        self.assertIn("N(Fragment=3)", link0)
        # Link 1: the same Q M, no coordinates.
        self.assertTrue(link1.rstrip("\n").endswith("\n0 1"))
        self.assertNotIn("Fragment=", link1.split("\n\n", 1)[1])
        routes = input_stage_routes(text)
        self.assertEqual(routes, list(job.routes()))
        self.assertEqual(row["first_route"], routes[1])
        self.assertEqual(row["stage_index"], "1")
        self.assertEqual(row["route_search_kind"], "minimum")
        self.assertEqual(row["fragment_count"], "3")
        self.assertEqual(row["pathway"], "fragment_guess")
        # Both the request-aware and the generic inspection pass.
        self.assertEqual(inspect_tandem_input(text, job).problems, [])
        self.assertEqual(inspect_tandem_input(text).problems, [])

    def test_inspector_rejects_broken_campaign_link1(self):
        text, _, job = render_fragment_tandem_input(fe2o2n2(), AFM)
        broken = {
            "no IOP(5/13=1) in link 1": text.replace("Opt IOP(5/13=1,5/36=1,8/11=1)", "Opt IOP(5/36=1,8/11=1)"),
            "no Guess=Read": text.replace(" Geom=Checkpoint Guess=Read", " Geom=Checkpoint"),
            "no Geom=Checkpoint": text.replace(" Geom=Checkpoint Guess=Read", " Guess=Read"),
            "multiplicity change": text[: text.rstrip("\n").rfind("\n")] + "\n0 3\n\n",
            "single-point link 1": text.replace("NoSymm Opt IOP(5/13", "NoSymm SP IOP(5/13"),
            "Always in link 0": text.replace("Guess=(Fragment=3)", "Guess=(Fragment=3,Always)"),
        }
        for name, bad in broken.items():
            with self.subTest(name):
                self.assertNotEqual(bad, text)
                self.assertTrue(inspect_tandem_input(bad).problems, name)
                self.assertTrue(inspect_tandem_input(bad, job).problems, name)

    def test_strict_fragment_validation_applies(self):
        bad = json.loads(json.dumps(AFM))
        bad["fragments"][2]["orientation"] = "beta"  # a singlet has no orientation
        with self.assertRaisesRegex(ValueError, "singlet"):
            render_fragment_tandem_input(fe2o2n2(), bad)


class TandemCampaignTests(unittest.TestCase):
    def _write(self, output: Path, record: Record, route: str = DEFAULT_SPIN_ROUTE, **kwargs) -> dict:
        count = write_spin_jobs(
            [record], output, 9, [1], route=route,
            fragment_specifications=[AFM], strategy="fragment", **kwargs,
        )
        self.assertEqual(count, 1)
        with (output / "spin_jobs.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, SPIN_MANIFEST_COLUMNS)
            rows = list(reader)
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_tandem_campaign_is_written_inspected_and_auditable(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "jobs"
            row = self._write(output, fe2o2n2(), fragment_layout="tandem")
            written = (output / row["input"]).read_bytes()
            self.assertNotIn(b"\r", written)
            text = written.decode("utf-8")
            self.assertEqual(inspect_tandem_input(text).problems, [])
            self.assertEqual(input_stage_routes(text)[int(row["stage_index"])], row["first_route"])
            self.assertTrue(row["checkpoint_lineage"].startswith("fragment-tandem:afm:m1:"))
            campaign = json.loads((output / "spin_campaign.json").read_text(encoding="utf-8"))
            self.assertEqual(campaign["fragment_layout"], "tandem")
            _, statuses, errors = _spin_manifest_audit(output / "spin_jobs.csv")
            self.assertEqual(errors, [])
            self.assertEqual(statuses[row["job_id"]], "verified")

    def test_saddle_seed_with_ts_route_keeps_opt_ts_in_link1(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "jobs"
            row = self._write(output, fe2o2n2("transition_state"), TS_ROUTE, fragment_layout="tandem")
            self.assertEqual(row["route_search_kind"], "saddle")
            routes = input_stage_routes((output / row["input"]).read_text(encoding="utf-8"))
            self.assertEqual(route_search_kind(routes[0]), "none")
            self.assertEqual(route_search_kind(routes[1]), "saddle")

    def test_default_layout_is_unchanged_single_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "jobs"
            row = self._write(output, fe2o2n2())
            text = (output / row["input"]).read_text(encoding="utf-8")
            self.assertIn("Guess=(Fragment=3,Always)", text)
            self.assertNotIn("--Link1--", text)
            self.assertEqual(row["stage_index"], "0")
            campaign = json.loads((output / "spin_campaign.json").read_text(encoding="utf-8"))
            self.assertNotIn("fragment_layout", campaign)

    def test_tandem_layout_refuses_ladder_only_and_unknown_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "only to fragment jobs"):
                write_spin_jobs([fe2o2n2()], Path(tmp) / "a", 9, [1], strategy="ladder",
                                fragment_layout="tandem")
            with self.assertRaisesRegex(ValueError, "fragment layout"):
                write_spin_jobs([fe2o2n2()], Path(tmp) / "b", 9, [1], fragment_specifications=[AFM],
                                strategy="fragment", fragment_layout="always")
            self.assertFalse((Path(tmp) / "a").exists())

    def test_cli_flag_defaults_to_single_link(self):
        parser = build_parser()
        base = ["prepare-spins", "seeds.extxyz", "--high-spin", "9", "--targets", "1"]
        self.assertEqual(parser.parse_args(base).fragment_layout, "single-link")
        self.assertEqual(
            parser.parse_args(base + ["--fragment-layout", "tandem"]).fragment_layout, "tandem"
        )


if __name__ == "__main__":
    unittest.main()
