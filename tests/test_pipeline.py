from pathlib import Path
import tempfile
import unittest
import zipfile

from cluster_mlip.analysis import write_analysis
from cluster_mlip.gaussian import extract_document_records, extract_records, parse_final_force_frame
from cluster_mlip.io import read_extxyz, write_extxyz
from cluster_mlip.jobs import expanded_records, write_gaussian_jobs


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

    def test_native_warehouse(self):
        text = (FIXTURES / "warehouse.txt").read_text()
        records = extract_document_records(text, "fe2o2_5_12345_LB.txt")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].formula, "Fe2O2")
        self.assertEqual(records[0].multiplicity, 5)
        self.assertEqual(records[0].config_type, "warehouse_structure")

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


if __name__ == "__main__":
    unittest.main()
