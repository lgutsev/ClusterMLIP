import csv
from pathlib import Path
import tempfile
import unittest
import zipfile

from cluster_mlip.analysis import write_analysis
from cluster_mlip.audit import run_private_audit
from cluster_mlip.gaussian import (
    extract_document_records,
    extract_records,
    extract_warehouse_record,
    parse_final_force_frame,
)
from cluster_mlip.io import read_document, read_extxyz, write_extxyz
from cluster_mlip.jobs import (
    DEFAULT_LINK1_ROUTE,
    DEFAULT_RATTLE_ROUTE,
    DEFAULT_ROUTE,
    expanded_records,
    human_job_stem,
    write_gaussian_jobs,
)
from cluster_mlip.models import Atom, Record
from cluster_mlip.spin import (
    geometry_distance,
    parse_spin_diagnostics,
    render_fragment_input,
    render_ladder_input,
    validate_spin_campaign,
    write_spin_jobs,
)


FIXTURES = Path(__file__).parent / "fixtures"


class PipelineTests(unittest.TestCase):
    def test_minimum(self):
        records = extract_records((FIXTURES / "minimum.log").read_text(), "minimum.log")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].config_type, "minimum")
        self.assertEqual(records[0].multiplicity, 3)
        self.assertEqual(records[0].formula, "FeO")

    def test_irc_split(self):
        records = extract_records((FIXTURES / "irc.log").read_text(), "irc.log")
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r.config_type == "irc_forward" for r in records))
        self.assertEqual([r.irc_point for r in records], [1, 2])

    def test_force_parse(self):
        frame = parse_final_force_frame((FIXTURES / "force.log").read_text(), FIXTURES / "force.log")
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(len(frame.forces_ev_ang), 2)
        self.assertAlmostEqual(frame.forces_ev_ang[0][0], 0.5142206748, places=7)

    def test_extxyz_and_jobs(self):
        records = extract_records((FIXTURES / "minimum.log").read_text(), "minimum.log")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            xyz = tmp_path / "seeds.extxyz"
            write_extxyz(records, xyz)
            loaded = read_extxyz(xyz)
            self.assertEqual(loaded[0].formula, "FeO")
            jobs = expanded_records(loaded, 2, 0.05, 7)
            write_gaussian_jobs(jobs, tmp_path / "jobs")
            self.assertEqual(len(list((tmp_path / "jobs").glob("*.gjf"))), 3)
            with (tmp_path / "jobs" / "jobs.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            seed_input = (tmp_path / "jobs" / rows[0]["input"]).read_text()
            rattle_input = (tmp_path / "jobs" / rows[1]["input"]).read_text()
            self.assertIn("minimum__q0-m3__reference", rows[0]["human_id"])
            self.assertIn("minimum__q0-m3__r01", rows[1]["human_id"])
            self.assertEqual(rows[0]["source_record_id"], jobs[0].record_id)
            self.assertEqual(rows[1]["source_record_id"], jobs[0].record_id)
            self.assertEqual(len(rows[1]["input_geometry_sha256"]), 64)
            self.assertEqual(len(rows[1]["input_sha256"]), 64)
            self.assertTrue((tmp_path / "jobs" / "campaign_manifest.json").is_file())
            self.assertIn(DEFAULT_ROUTE, seed_input)
            self.assertIn(DEFAULT_RATTLE_ROUTE, rattle_input)
            self.assertNotIn(" Opt Freq ", rattle_input.split("--Link1--", 1)[0])
            for text in (seed_input, rattle_input):
                self.assertEqual(text.count("--Link1--"), 1)
                self.assertIn(DEFAULT_LINK1_ROUTE, text)
                self.assertIn("UBPW91/6-311++G* Force", text)
                self.assertIn("Guess=Read Geom=Checkpoint", text)
                self.assertNotIn("wB97M-V", text)

    def test_rattles_are_stable_when_seed_order_changes(self):
        first = Record("first", "archive/first.log", [Atom("H", 0, 0, 0)], 0, 1, "minimum")
        second = Record("second", "archive/second.log", [Atom("H", 1, 0, 0)], 0, 1, "minimum")
        forward = expanded_records([first, second], 1, 0.05, 41)
        reverse = expanded_records([second, first], 1, 0.05, 41)
        forward_rattles = {
            record.metadata["source_record_id"]: (record.record_id, record.atoms)
            for record in forward
            if "rattle_index" in record.metadata
        }
        reverse_rattles = {
            record.metadata["source_record_id"]: (record.record_id, record.atoms)
            for record in reverse
            if "rattle_index" in record.metadata
        }
        self.assertEqual(forward_rattles, reverse_rattles)
        different_seed = expanded_records([first], 1, 0.05, 42)[1]
        self.assertNotEqual(forward[1].record_id, different_seed.record_id)
        self.assertNotEqual(human_job_stem(forward[1]), human_job_stem(different_seed))

    def test_native_warehouse(self):
        text = (FIXTURES / "warehouse.txt").read_text()
        records = extract_document_records(text, "fe2o2_5_12345_LB.txt")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].formula, "Fe2O2")
        self.assertEqual(records[0].multiplicity, 5)
        self.assertEqual(records[0].config_type, "warehouse_structure")

    def test_native_warehouse_survives_crlf_line_endings(self):
        # read_document() reads raw bytes with no universal-newline
        # translation, so a CRLF-encoded warehouse file must be exercised
        # through it directly -- Path.read_text() would silently normalize
        # the endings and hide a regression here.
        crlf_text = (FIXTURES / "warehouse.txt").read_text().replace("\n", "\r\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fe2o2_5_12345_LB.txt"
            path.write_bytes(crlf_text.encode("utf-8"))
            text = read_document(path)
            records = extract_document_records(text, path.name)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].formula, "Fe2O2")
        self.assertEqual(len(records[0].atoms), 4)

    def test_native_warehouse_unmatched_filename_is_flagged_not_guessed_silently(self):
        text = (FIXTURES / "warehouse.txt").read_text()
        matched = extract_warehouse_record(text, "fe2o2_5_12345_LB.txt")
        self.assertEqual(matched[0].multiplicity, 5)
        self.assertEqual(matched[0].metadata["state_inference"], "filename")

        unmatched = extract_warehouse_record(text, "no_convention_here.txt")
        # A filename that does not match the multiplicity convention still
        # gets a usable default, but it must never be mislabeled as if it
        # came from the filename convention -- that would hide a silent
        # wrong-multiplicity guess from anyone auditing the dataset.
        self.assertEqual(unmatched[0].multiplicity, 1)
        self.assertEqual(unmatched[0].metadata["state_inference"], "default_unmatched_singlet")

    def test_metadata_roundtrips_through_extxyz_and_manifest(self):
        text = (FIXTURES / "warehouse.txt").read_text()
        records = extract_warehouse_record(text, "no_convention_here.txt")
        with tempfile.TemporaryDirectory() as tmp:
            xyz = Path(tmp) / "seeds.extxyz"
            write_extxyz(records, xyz)
            loaded = read_extxyz(xyz)
            self.assertEqual(
                loaded[0].metadata["state_inference"], "default_unmatched_singlet"
            )

    def test_formatted_checkpoint(self):
        text = (FIXTURES / "example.fchk").read_text()
        records = extract_document_records(text, "example.fchk")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].formula, "FeO")
        self.assertEqual(records[0].multiplicity, 3)
        self.assertAlmostEqual(records[0].atoms[1].x, 1.5875316327, places=8)

        ts_records = extract_document_records(text, "reaction_TS.fchk")
        self.assertEqual(ts_records[0].config_type, "transition_state")

        irc_records = extract_document_records(text, "reaction_irc_forward_point12.fchk")
        self.assertEqual(irc_records[0].config_type, "irc_forward")
        self.assertEqual(irc_records[0].irc_point, 12)

    def test_ts_gaussian_input(self):
        text = (FIXTURES / "ts.gjf").read_text()
        records = extract_document_records(text, "ts.gjf")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].config_type, "transition_state")
        self.assertEqual(records[0].multiplicity, 4)

    def test_analyze_nested_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inner = root / "inner.zip"
            with zipfile.ZipFile(inner, "w") as archive:
                archive.write(FIXTURES / "minimum.log", "runs/minimum.log")
                archive.writestr("notes/readme.xyz", "unsupported")
            outer = root / "warehouse.zip"
            with zipfile.ZipFile(outer, "w") as archive:
                archive.write(inner, "batch/inner.zip")
            summary = write_analysis(outer, root / "analysis")
            self.assertEqual(summary["structures"]["records"], 1)
            self.assertEqual(summary["files"]["by_status"]["parsed"], 1)
            self.assertEqual(summary["files"]["by_status"]["unsupported"], 2)
            self.assertTrue((root / "analysis" / "report.md").exists())
            self.assertTrue((root / "analysis" / "records.csv").exists())

    def test_analyze_record_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "analysis"
            summary = write_analysis(
                FIXTURES / "minimum.log",
                output,
                record_filter=lambda record: "N" in {atom.symbol for atom in record.atoms},
            )
            self.assertEqual(summary["files"]["by_status"]["parsed"], 1)
            self.assertEqual(summary["structures"]["records"], 0)

    def test_private_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "private_audits" / "fixture"
            result = run_private_audit(
                FIXTURES / "minimum.log",
                output,
                elements={"Fe", "O"},
                required_elements={"Fe"},
                max_atoms=20,
            )
            self.assertEqual(result["full"]["structures"]["records"], 1)
            self.assertEqual(result["selection"]["structures"]["records"], 1)
            self.assertTrue((output / "full" / "report.md").exists())
            self.assertTrue((output / "selection" / "records.csv").exists())
            self.assertTrue((output / "provenance.json").exists())

    def test_spin_ladder_inserts_every_single_flip(self):
        record = Record("fe2", "legacy", [Atom("Fe", 0, 0, 0), Atom("Fe", 2, 0, 0)], 0, 9, "minimum")
        text, rows = render_ladder_input(record, 9, [5, 1])
        self.assertEqual([int(row["intended_multiplicity"]) for row in rows], [9, 7, 5, 3, 1])
        self.assertEqual(text.count("--Link1--"), 4)
        self.assertIn("Geom=Checkpoint Guess=(Read,Always)", text)
        self.assertIn("%oldchk=fe2-ladder-m9-m1-s00-m9.chk", text)

    def test_fragment_afm_input_is_explicit_and_partitioned(self):
        record = Record("fe2", "legacy", [Atom("Fe", 0, 0, 0), Atom("Fe", 2, 0, 0)], 0, 9, "minimum")
        specification = {
            "name": "afm-opposed",
            "target_multiplicity": 1,
            "fragments": [
                {"atoms": [1], "charge": 0, "multiplicity": 5, "orientation": "alpha"},
                {"atoms": [2], "charge": 0, "multiplicity": 5, "orientation": "beta"},
            ],
        }
        text, row = render_fragment_input(record, specification)
        self.assertIn("Guess=(Fragment=2,Always)", text)
        self.assertIn("0 1 0 5 0 -5", text)
        self.assertIn("Fe(Fragment=1)", text)
        self.assertIn("Fe(Fragment=2)", text)
        self.assertEqual(row["pathway"], "fragment_guess")

    def test_fragment_spins_must_reproduce_total_multiplicity(self):
        record = Record("fe2", "legacy", [Atom("Fe", 0, 0, 0), Atom("Fe", 2, 0, 0)], 0, 9, "minimum")
        inconsistent = {
            "name": "bad-afm",
            "target_multiplicity": 1,
            "fragments": [
                {"atoms": [1], "charge": 0, "multiplicity": 5, "orientation": "alpha"},
                {"atoms": [2], "charge": 0, "multiplicity": 5, "orientation": "alpha"},
            ],
        }
        with self.assertRaisesRegex(ValueError, "inconsistent with the total multiplicity"):
            render_fragment_input(record, inconsistent)

    def test_prepare_spins_bad_fragment_spec_writes_nothing(self):
        record = Record("fe2", "legacy", [Atom("Fe", 0, 0, 0), Atom("Fe", 2, 0, 0)], 0, 9, "minimum")
        good = {
            "record_id": "fe2",
            "name": "good",
            "target_multiplicity": 1,
            "fragments": [
                {"atoms": [1], "charge": 0, "multiplicity": 5, "orientation": "alpha"},
                {"atoms": [2], "charge": 0, "multiplicity": 5, "orientation": "beta"},
            ],
        }
        bad = {
            "record_id": "fe2",
            "name": "bad",
            "target_multiplicity": 1,
            "fragments": [
                {"atoms": [1], "charge": 0, "multiplicity": 5, "orientation": "alpha"},
                {"atoms": [2], "charge": 0, "multiplicity": 5, "orientation": "alpha"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "jobs"
            with self.assertRaises(ValueError):
                write_spin_jobs([record], output, 9, [1], fragment_specifications=[good, bad])
            # A malformed spec later in the list must not leave a half-written
            # job directory with no manifest to explain what is/isn't there.
            self.assertFalse(output.exists())

    def test_spin_diagnostics_distinguish_afm_root(self):
        text = """
 Charge = 0 Multiplicity = 1
 SCF Done:  E(UHF) =  -100.250000 A.U. after 10 cycles
 S**2 before annihilation  4.1000, after  3.9500
 Mulliken charges and spin densities:
              1          2
     1  Fe   0.100000   3.800000
     2  Fe  -0.100000  -3.800000
 Sum of Mulliken charges = 0.00000 Sum of Mulliken spin densities = 0.00000
 Stationary point found.
 The wavefunction is stable under the perturbations considered.
 Normal termination of Gaussian 16
 """
        diagnostic = parse_spin_diagnostics(text)[0]
        self.assertEqual(diagnostic.spin_pattern, "compensated_afm_like")
        self.assertEqual(diagnostic.root_signature != "", True)
        self.assertAlmostEqual(diagnostic.expected_s2, 0.0)
        self.assertTrue(diagnostic.optimized)
        self.assertEqual(diagnostic.stability, "stable")

    def test_geometry_fingerprint_is_translation_and_order_invariant(self):
        left = Record("a", "", [Atom("Fe", 0, 0, 0), Atom("O", 2, 0, 0)], 0, 1, "minimum")
        right = Record("b", "", [Atom("O", 7, 4, 3), Atom("Fe", 5, 4, 3)], 0, 1, "minimum")
        self.assertAlmostEqual(geometry_distance(left, right), 0.0)

    def test_validator_retains_alternative_root(self):
        def output(spin_a: float, spin_b: float) -> str:
            return f"""
 Charge = 0 Multiplicity = 1
 Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
 1 26 0 0.000000 0.000000 0.000000
 2 26 0 2.000000 0.000000 0.000000
 ---------------------------------------------------------------------
 SCF Done:  E(UHF) =  -100.250000 A.U. after 10 cycles
 Mulliken charges and spin densities:
              1          2
 1 Fe 0.0 {spin_a}
 2 Fe 0.0 {spin_b}
 Sum of Mulliken charges = 0.0 Sum of Mulliken spin densities = 0.0
 Stationary point found.
 Normal termination of Gaussian 16
 """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "legacy.log"
            legacy.write_text(output(3.8, -3.8))
            new = root / "new"
            new.mkdir()
            (new / "candidate.log").write_text(output(1.2, -1.2))
            summary = validate_spin_campaign(legacy, new, root / "validation", spin_tolerance=0.25)
            self.assertEqual(summary["alternative_root"], 1)
            self.assertTrue((root / "validation" / "new_states.csv").exists())


if __name__ == "__main__":
    unittest.main()
