import csv
import json
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
    DEFAULT_SPIN_ROUTE,
    _fragment_spin_alignment,
    _spin_manifest_audit,
    fe_spin_summary,
    geometry_distance,
    infer_automatic_fe_spin_plans,
    parse_spin_diagnostics,
    render_fragment_input,
    render_ladder_input,
    validate_spin_campaign,
    write_automatic_fe_spin_jobs,
    write_spin_inventory,
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
                self.assertIn("UBPW91/Gen Force", text)
                self.assertIn("Guess=Read Geom=Checkpoint", text)
                self.assertNotIn("wB97M-V", text)
                self.assertEqual(text.count("Fe     0"), 2)
                self.assertEqual(text.count("6-31G*"), 2)
                self.assertNotIn("6-311G*", text)
                self.assertNotIn("6-311++G*", text)

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

    def test_spin_inventory_preserves_multiplicity_validation_in_extxyz(self):
        text = (FIXTURES / "warehouse.txt").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "fe2o2_4_12345_LB.txt"
            source.write_text(text, encoding="utf-8")
            output = Path(tmp) / "inventory"
            self.assertEqual(write_spin_inventory(source, output), 1)
            record = read_extxyz(output / "seeds.extxyz")[0]
        self.assertEqual(record.metadata["electron_count"], "68")
        self.assertEqual(record.metadata["multiplicity_parity_valid"], "false")

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
        self.assertIn("UBPW91/6-311++G*", DEFAULT_SPIN_ROUTE)
        self.assertIn("VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc", DEFAULT_SPIN_ROUTE)
        self.assertIn("Pop=Regular", DEFAULT_SPIN_ROUTE)
        self.assertNotIn("Pop=Full", DEFAULT_SPIN_ROUTE)
        self.assertNotIn("SpinDensity", DEFAULT_SPIN_ROUTE)
        self.assertNotIn("wB97M-V", DEFAULT_SPIN_ROUTE)
        self.assertEqual(text.count("6-311++G*"), 5)
        self.assertNotIn("Fe     0", text)
        first_coordinate = [
            line
            for line in text.splitlines()
            if line.startswith("Fe ") and len(line.split()) == 4
        ][-1]
        self.assertIn(f"{first_coordinate}\n\n", text)
        self.assertIn("0 7\n\n", text)
        self.assertNotIn("6-31G*", text)
        self.assertNotIn("6-311G*", text)

    def test_spin_ladder_keeps_gen_override_compatible(self):
        record = Record("feo", "legacy", [Atom("Fe", 0, 0, 0), Atom("O", 2, 0, 0)], 0, 5, "minimum")
        route = DEFAULT_SPIN_ROUTE.replace("/6-311++G*", "/Gen")
        text, _ = render_ladder_input(record, 5, [3], route=route)
        self.assertEqual(text.count("O     0"), 2)
        self.assertEqual(text.count("Fe     0"), 2)

    def test_fe10_spin_flip_ladder_has_complete_checkpoint_lineage(self):
        record = Record(
            "fe10-high-spin",
            "warehouse/fe10_m29.log",
            [Atom("Fe", float(index), 0.0, 0.0) for index in range(10)],
            0,
            29,
            "minimum",
        )
        unsafe_low_spin_seed = Record(
            "fe10-direct-m17",
            "warehouse/fe10_m17.log",
            list(record.atoms),
            0,
            17,
            "minimum",
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "jobs"
            count = write_spin_jobs(
                [record, unsafe_low_spin_seed], output, 29, [17], strategy="ladder"
            )
            self.assertEqual(count, 7)
            with (output / "spin_jobs.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [int(row["intended_multiplicity"]) for row in rows],
                [29, 27, 25, 23, 21, 19, 17],
            )
            self.assertEqual(rows[0]["audit_classification"], "trusted_high_spin_reference")
            for previous, current in zip(rows, rows[1:]):
                self.assertEqual(current["audit_classification"], "sequential_checkpoint_spin_flip")
                self.assertEqual(current["predecessor_job_id"], previous["job_id"])
                self.assertEqual(current["predecessor_checkpoint"], previous["checkpoint"])
                self.assertNotEqual(current["checkpoint"], previous["checkpoint"])
                self.assertTrue(current["checkpoint_lineage"].startswith(previous["checkpoint_lineage"] + ">"))
            input_name = rows[0]["input"]
            self.assertIn("fe10-m29", input_name)
            self.assertEqual(Path(input_name).parent, Path("inputs"))
            text = (output / input_name).read_text(encoding="utf-8")
            self.assertEqual(text.count("--Link1--"), 6)
            self.assertIn(f"%oldchk={rows[0]['checkpoint']}\n%chk={rows[1]['checkpoint']}", text)
            manifest_rows, statuses, errors = _spin_manifest_audit(output / "spin_jobs.csv")
            self.assertEqual(len(manifest_rows), 7)
            self.assertEqual(errors, [])
            self.assertEqual(set(statuses.values()), {"verified"})
            with (output / "skipped_spin_seeds.csv").open(newline="", encoding="utf-8") as handle:
                skipped = list(csv.DictReader(handle))
            self.assertEqual(skipped[0]["record_id"], "fe10-direct-m17")
            self.assertEqual(skipped[0]["reason"], "direct_low_spin_initialization_prohibited")

    def test_automatic_oxide_plan_uses_real_fe10_m29_not_idealized_m41(self):
        atoms = [Atom("Fe", float(index), 0.0, 0.0) for index in range(10)] + [
            Atom("O", float(index), 2.0, 0.0) for index in range(10)
        ]
        high = Record(
            "fe10o10-m29", "warehouse/fe10o10_29_10000.txt", atoms, 0, 29,
            "warehouse_structure", metadata={"state_inference": "filename"},
        )
        low = Record(
            "fe10o10-m17", "warehouse/fe10o10_17_10001.txt", atoms, 0, 17,
            "warehouse_structure", metadata={"state_inference": "filename"},
        )
        plans, skipped = infer_automatic_fe_spin_plans([high, low])
        self.assertEqual(skipped, [])
        self.assertEqual({plan.high_spin_multiplicity for plan in plans}, {29})
        self.assertEqual(
            {plan.inference for plan in plans}, {"highest_observed_group_multiplicity"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "auto"
            stages = write_automatic_fe_spin_jobs([high, low], output)
            self.assertEqual(stages, 8)  # one m29 reference + the seven-stage 29 -> 17 ladder
            with (output / "spin_plan.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            low_plan = next(row for row in rows if row["record_id"] == low.record_id)
            self.assertEqual(low_plan["inferred_high_spin_multiplicity"], "29")
            self.assertEqual(low_plan["observed_group_multiplicities"], "29;17")
            self.assertNotIn("41", low_plan.values())
            summary = json.loads((output / "spin_plan_summary.json").read_text())
            self.assertEqual(summary["by_ladder"], {"m29->m17": 1, "m29->m29": 1})
            self.assertEqual(summary["skipped_records"], 0)
            _, statuses, errors = _spin_manifest_audit(output / "spin_jobs.csv")
            self.assertEqual(errors, [])
            self.assertEqual(set(statuses.values()), {"verified"})

    def test_automatic_oxide_plan_skips_unsupported_singleton(self):
        record = Record(
            "fe10o10-only-m17", "warehouse/fe10o10_17_10001.txt",
            [Atom("Fe", float(index), 0.0, 0.0) for index in range(10)]
            + [Atom("O", float(index), 2.0, 0.0) for index in range(10)],
            0, 17, "warehouse_structure", metadata={"state_inference": "filename"},
        )
        plans, skipped = infer_automatic_fe_spin_plans([record])
        self.assertEqual(plans, [])
        self.assertEqual(skipped[0].reason, "insufficient_real_data_no_parallel_reference")

    def test_automatic_campaign_collapses_duplicate_inventory_records(self):
        atoms = [Atom("Fe", float(index), 0.0, 0.0) for index in range(2)] + [
            Atom("O", float(index), 2.0, 0.0) for index in range(2)
        ]
        high = Record(
            "fe2o2-m9", "warehouse/fe2o2_9.log", atoms, 0, 9,
            "minimum", metadata={"state_inference": "filename"},
        )
        low = Record(
            "fe2o2-m1", "warehouse/fe2o2_1.log", atoms, 0, 1,
            "minimum", metadata={"state_inference": "filename"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "auto"
            stages = write_automatic_fe_spin_jobs([high, low, high, low], output)
            self.assertEqual(stages, 6)  # m9 plus the five-stage m9 -> m1 ladder
            self.assertEqual(len(list(output.glob("*.gjf"))), 0)
            self.assertEqual(len(list((output / "inputs").glob("*.gjf"))), 2)
            with (output / "spin_plan.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(Path(row["input"]).parent == Path("inputs") for row in rows))
            summary = json.loads((output / "spin_plan_summary.json").read_text())
            self.assertEqual(summary["total_archive_records"], 4)
            self.assertEqual(summary["unique_archive_records"], 2)
            self.assertEqual(summary["duplicate_planned_records_collapsed"], 2)
            self.assertEqual(summary["duplicate_skipped_records_collapsed"], 0)

    def test_automatic_oxide_plan_skips_invalid_filename_multiplicity_without_aborting(self):
        atoms = [Atom("Fe", float(index), 0.0, 0.0) for index in range(10)] + [
            Atom("O", float(index), 2.0, 0.0) for index in range(10)
        ]
        high = Record(
            "fe10o10-m39", "warehouse/fe10o10_39_10000.txt", atoms, 0, 39,
            "warehouse_structure", metadata={"state_inference": "filename"},
        )
        low = Record(
            "fe10o10-m17", "warehouse/fe10o10_17_10002.txt", atoms, 0, 17,
            "warehouse_structure", metadata={"state_inference": "filename"},
        )
        invalid = Record(
            "fe10o10-m16", "warehouse/fe10o10_16_10001.txt", atoms, 0, 16,
            "warehouse_structure", metadata={"state_inference": "filename"},
        )
        plans, skipped = infer_automatic_fe_spin_plans([high, low, invalid])
        self.assertEqual(
            [plan.record.record_id for plan in plans], [high.record_id, low.record_id]
        )
        self.assertEqual(skipped[0].record.record_id, invalid.record_id)
        self.assertIn("target_multiplicity_not_physically_valid", skipped[0].reason)
        self.assertIn("wrong parity", skipped[0].reason)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "auto"
            stages = write_automatic_fe_spin_jobs([high, low, invalid], output)
            self.assertEqual(stages, 13)
            with (output / "spin_plan.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            invalid_row = next(row for row in rows if row["record_id"] == invalid.record_id)
            self.assertEqual(invalid_row["status"], "skipped")
            self.assertIn("wrong parity", invalid_row["reason"])

    def test_actual_parallel_fe_moments_can_support_singleton_reference(self):
        record = Record(
            "fe2o2-m9", "fe2o2.log",
            [Atom("Fe", 0, 0, 0), Atom("Fe", 2, 0, 0), Atom("O", 0, 2, 0), Atom("O", 2, 2, 0)],
            0, 9, "minimum",
        )
        diagnostic = parse_spin_diagnostics("""
 Charge = 0 Multiplicity = 9
 Mulliken charges and spin densities:
 1 Fe 0.0 3.7
 2 Fe 0.0 3.6
 3 O  0.0 0.3
 4 O  0.0 0.4
 Sum of Mulliken charges = 0.0 Sum of Mulliken spin densities = 8.0
 """)[0]
        record.metadata.update(fe_spin_summary(record, diagnostic))
        plans, skipped = infer_automatic_fe_spin_plans([record])
        self.assertEqual(skipped, [])
        self.assertEqual(plans[0].high_spin_multiplicity, 9)
        self.assertEqual(plans[0].inference, "observed_all_resolved_fe_parallel")

    def test_manual_fragment_pathway_is_locked_and_auditable(self):
        record = Record(
            "fe10-high-spin",
            "warehouse/fe10_m29.log",
            [Atom("Fe", float(index), 0.0, 0.0) for index in range(10)],
            0,
            29,
            "minimum",
        )
        specification = {
            "record_id": record.record_id,
            "name": "seven-up-three-down",
            "target_multiplicity": 17,
            "fragments": [
                {
                    "atoms": [index],
                    "charge": 0,
                    "multiplicity": 5,
                    "orientation": "alpha" if index <= 7 else "beta",
                }
                for index in range(1, 11)
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "jobs"
            count = write_spin_jobs(
                [record], output, 29, [17],
                fragment_specifications=[specification], strategy="fragment",
            )
            self.assertEqual(count, 1)
            self.assertTrue((output / "fragment_specifications.lock.json").is_file())
            with (output / "spin_jobs.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["initialization"], "explicit_manual_fragment_map")
            self.assertEqual(row["audit_classification"], "manual_fragment_preparation")
            self.assertEqual(len(row["fragment_spec_sha256"]), 64)
            _, statuses, errors = _spin_manifest_audit(output / "spin_jobs.csv")
            self.assertEqual(errors, [])
            self.assertEqual(statuses[row["job_id"]], "verified")

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
        last_coordinate = next(
            line for line in text.splitlines() if line.startswith("Fe(Fragment=2)")
        )
        self.assertIn(f"{last_coordinate}\n\n", text)
        self.assertNotIn("Fe     0", text)
        self.assertIn("UBPW91/6-311++G*", text)
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

    def test_fragment_audit_checks_converged_orientation(self):
        specification = {
            "fragments": [
                {"atoms": [1], "multiplicity": 5, "orientation": "alpha"},
                {"atoms": [2], "multiplicity": 5, "orientation": "beta"},
            ]
        }
        matched = parse_spin_diagnostics("""
 Charge = 0 Multiplicity = 1
 Mulliken charges and spin densities:
 1 Fe 0.0 3.8
 2 Fe 0.0 -3.8
 Sum of Mulliken charges = 0.0 Sum of Mulliken spin densities = 0.0
 """)[0]
        mismatched = parse_spin_diagnostics("""
 Charge = 0 Multiplicity = 1
 Mulliken charges and spin densities:
 1 Fe 0.0 3.8
 2 Fe 0.0 3.8
 Sum of Mulliken charges = 0.0 Sum of Mulliken spin densities = 7.6
 """)[0]
        self.assertEqual(_fragment_spin_alignment(matched, specification), "matched")
        self.assertEqual(_fragment_spin_alignment(mismatched, specification), "mismatch")

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
