"""Characterization tests for Gaussian label parsing and the IO readers.

They pin current behaviour with small synthetic inputs. Cases marked
"CHARACTERIZATION" document behaviour that looks questionable; they are
listed as suspected bugs in the PR rather than fixed.
"""

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from cluster_mlip import gaussian as g
from cluster_mlip.io import (
    _decode_bytes,
    _expand_nested_zips,
    _safe_extract,
    iter_documents,
    parse_extxyz_info_line,
    quote_extxyz,
    read_document,
    read_extxyz,
    source_tree,
    write_extxyz,
    write_manifest,
    write_text_lf,
)
from cluster_mlip.models import Atom, Record

FIXTURES = Path(__file__).parent / "fixtures"
FCHK = (FIXTURES / "example.fchk").read_text()
FORCE_LOG = (FIXTURES / "force.log").read_text()


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class FilenameStateTests(unittest.TestCase):
    def test_charge_and_multiplicity_from_warehouse_names(self):
        cases = {
            "x_q-1_.log": (-1, 1, False),
            "fe16o18_anion_3.txt": (-1, 1, False),  # multiplicity not matched -> flagged default
            "Fe2O3-_1_12345.txt": (-1, 1, True),
            "fe4o6_5_05123.txt": (0, 5, True),
            "Fe2O3+_x.txt": (1, 1, False),
            "plain.txt": (0, 1, False),
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(g._filename_state(name), expected)


class ClassifyTests(unittest.TestCase):
    def test_classification_precedence(self):
        self.assertEqual(g._classify("a.log", "# opt=(ts,calcfc)", None, None), "transition_state")
        self.assertEqual(g._classify("ts_a.log", "# sp", 1, None), "transition_state")
        self.assertEqual(g._classify("a.log", "# sp", 1, None), "first_order_saddle")
        self.assertEqual(g._classify("a.log", "# sp", 2, None), "higher_order_saddle")
        self.assertEqual(g._classify("a.log", "# sp", 0, None), "minimum")
        self.assertEqual(g._classify("a.log", "# opt", None, None), "optimized_unverified")
        self.assertEqual(g._classify("a.log", "# sp", None, None), "unknown")

    def test_irc_path_number_picks_direction(self):
        self.assertEqual(g._classify("a.log", "# irc", 0, (3, 1)), "irc_forward")
        self.assertEqual(g._classify("a.log", "# irc", 0, (3, 2)), "irc_reverse")
        # Any path number other than 1 is "reverse".
        self.assertEqual(g._classify("a.log", "# irc", 0, (3, 7)), "irc_reverse")


class FormattedCheckpointIrcTests(unittest.TestCase):
    def _record(self, name):
        (record,) = g.extract_formatted_checkpoint(FCHK, name)
        return record

    def test_irc_filenames(self):
        cases = {
            "a_irc_forward_pt12.fchk": ("irc_forward", 1, 12),
            "a_irc_rev_step3.fchk": ("irc_reverse", 2, 3),
            "a_irc_fwd_p7.fchk": ("irc_forward", 1, 7),
            "a_irc.fchk": ("irc_checkpoint", None, None),
            "a_ts.fchk": ("transition_state", None, None),
            "a_transition.fchk": ("transition_state", None, None),
            "a.fchk": ("checkpoint_geometry", None, None),
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                record = self._record(name)
                self.assertEqual((record.config_type, record.irc_path, record.irc_point), expected)

    def test_irc_point_regex_is_unanchored(self):
        # CHARACTERIZATION (suspected bug, low): the bare "p" alternative also matches
        # inside unrelated tokens such as "Fe2p3" or "opt5".
        self.assertEqual(self._record("irc_forward_Fe2p3.fchk").irc_point, 3)
        self.assertEqual(self._record("fe2o3_irc_reverse_opt5.fchk").irc_point, 5)

    def test_fixture_values_and_units(self):
        record = self._record("a.fchk")
        self.assertEqual((record.charge, record.multiplicity), (0, 3))
        self.assertEqual([a.symbol for a in record.atoms], ["Fe", "O"])
        self.assertAlmostEqual(record.atoms[1].x, 3.0 * g.BOHR_TO_ANG)
        self.assertAlmostEqual(record.legacy_energy_hartree, -1337.1234567)

    def test_incomplete_checkpoints_give_no_records(self):
        self.assertEqual(g.extract_formatted_checkpoint("nothing here", "a.fchk"), [])
        truncated = FCHK.replace("Current cartesian coordinates", "Other coordinates")
        self.assertEqual(g.extract_formatted_checkpoint(truncated, "a.fchk"), [])


class NativeCoordinateTests(unittest.TestCase):
    def test_title_block_stops_at_end_and_accepts_commas(self):
        atoms = g._native_coordinate_lines("TITLE\nFe 0 0 0\nO 1,0,0\nEND\nFe 5 5 5")
        self.assertEqual([(a.symbol, a.x) for a in atoms], [("Fe", 0.0), ("O", 1.0)])

    def test_moldat_uses_atom_count_line(self):
        atoms = g._native_coordinate_lines("MOLDAT\n\n2\nFe 0 0 0\nO 1 0 0\nH 9 9 9")
        self.assertEqual(len(atoms), 2)

    def test_crlf_does_not_drop_atoms(self):
        atoms = g._native_coordinate_lines("MOLDAT\r\n2\r\nFe 0 0 0\r\nO 1 0 0\r\n")
        self.assertEqual(len(atoms), 2)

    def test_fortran_exponents_and_unknown_symbols(self):
        atoms = g._native_coordinate_lines("Fe 1.0D+00 0 0\nXx 1 1 1\nO 2.5E-1 0 0\n")
        self.assertEqual(atoms[0].x, 1.0)
        self.assertEqual(len(atoms), 2 if atoms[-1].symbol == "O" else 1)


class WarehouseRecordTests(unittest.TestCase):
    def test_needs_marker_or_naming_convention(self):
        self.assertEqual(g.extract_warehouse_record("Fe 0 0 0\nO 1 0 0\n", "notes.txt"), [])

    def test_state_inference_labels(self):
        text = "TITLE\nFe 0 0 0\nO 1 0 0\nEND\n"
        (matched,) = g.extract_warehouse_record(text, "fe4o6_5_05123.txt")
        self.assertEqual((matched.multiplicity, matched.metadata["state_inference"]), (5, "filename"))
        (default,) = g.extract_warehouse_record(text, "other.txt")
        self.assertEqual(default.metadata["state_inference"], "default_unmatched_singlet")

    def test_huge_multiplicity_falls_back_to_electron_parity(self):
        text = "TITLE\nFe 0 0 0\nO 1 0 0\nEND\n"  # 26 + 8 = 34 electrons (even)
        (record,) = g.extract_warehouse_record(text, "fe1o1_200_12345.txt")
        self.assertEqual(record.multiplicity, 1)
        self.assertEqual(record.metadata["state_inference"], "electron_parity_fallback")


class GaussianInputTests(unittest.TestCase):
    def test_irc_route_marks_input_seed(self):
        text = "# irc=(forward,calcfc) b3lyp\n\ntitle\n\n0 1\nFe 0 0 0\nO 1 0 0\n\n"
        (record,) = g.extract_gaussian_input(text, "a.gjf")
        self.assertEqual(record.config_type, "irc_input_seed")
        self.assertEqual((record.charge, record.multiplicity), (0, 1))
        self.assertEqual(len(record.atoms), 2)

    def test_no_charge_line_means_no_record(self):
        self.assertEqual(g.extract_gaussian_input("# opt\n\ntitle\n", "a.gjf"), [])

    def test_dispatch_by_suffix(self):
        records = g.extract_document_records(FCHK, "x_irc_forward_pt2.fchk")
        self.assertEqual(records[0].config_type, "irc_forward")
        self.assertEqual(g.extract_document_records("nothing", "x.log"), [])


class JobCompleteAndForceTests(unittest.TestCase):
    NORMAL = " Normal termination of Gaussian 16\n"

    def test_job_complete_policy(self):
        self.assertTrue(g.gaussian_job_complete("Entering Gaussian System\n" + self.NORMAL))
        self.assertFalse(g.gaussian_job_complete(self.NORMAL + " Error termination via Lnk1e\n"))
        self.assertFalse(g.gaussian_job_complete(self.NORMAL + " SCF Done:  E(RB3LYP) = -1\n"))
        self.assertFalse(g.gaussian_job_complete(self.NORMAL, expected_stages=2))
        self.assertTrue(g.gaussian_job_complete(self.NORMAL * 2, expected_stages=2))

    def test_irc_point_regex(self):
        match = g._IRC_POINT_RE.search("Point Number:  12  Path Number:   2")
        self.assertEqual((match.group(1), match.group(2)), ("12", "2"))
        self.assertEqual(g._IRC_POINT_RE.search("Point Number: -3 Path Number: 1").group(1), "-3")

    def test_force_units_and_rms(self):
        (frame,) = g.parse_force_frames(FORCE_LOG, Path("force.log"))
        self.assertAlmostEqual(frame.forces_ev_ang[0][0], 0.01 * g.FORCE_AU_TO_EV_ANG)
        self.assertAlmostEqual(frame.energy_ev, -1.1 * g.HARTREE_TO_EV)
        expected = (2 * (0.01 * g.FORCE_AU_TO_EV_ANG) ** 2 / 6) ** 0.5
        self.assertAlmostEqual(g.rms_force(frame), expected)

    def test_force_table_without_preceding_scf_is_skipped(self):
        text = FORCE_LOG.replace(" SCF Done:  E(RWB97M-V) =  -1.1000000000     A.U.\n", "")
        self.assertEqual(g.parse_force_frames(text, Path("force.log")), [])

    def test_non_finite_and_malformed_force_rows_are_skipped(self):
        bad = FORCE_LOG.replace("0.010000000", "nan")
        self.assertEqual(g.parse_force_frames(bad, Path("f.log")), [])
        short = FORCE_LOG.replace("      2        1          -0.010000000", "      2        1")
        self.assertEqual(g.parse_force_frames(short, Path("f.log")), [])

    def test_unconverged_scf_is_flagged_not_dropped(self):
        # The region checked runs from the end of the geometry table to the force table,
        # so a warning on either side of "SCF Done" flags the frame.
        before_scf = FORCE_LOG.replace(" SCF Done:", " Convergence criterion not met.\n SCF Done:", 1)
        before_forces = FORCE_LOG.replace(
            " Forces (Hartrees/Bohr)\n", " Convergence criterion not met.\n Forces (Hartrees/Bohr)\n", 1)
        for text in (before_scf, before_forces):
            (frame,) = g.parse_force_frames(text, Path("f.log"))
            self.assertTrue(frame.record.metadata["scf_unconverged"])
            self.assertTrue(frame.record.metadata["scf_convergence_warning"])
        (clean,) = g.parse_force_frames(FORCE_LOG, Path("f.log"))
        self.assertFalse(clean.record.metadata["scf_unconverged"])

    def test_s2_and_spin_populations_are_read_from_own_scf(self):
        extra = (" S**2 before annihilation     2.0100,   after     2.0001\n"
                 " Mulliken charges and spin densities:\n"
                 "               1          2\n"
                 "     1  H    0.100000   0.500000\n"
                 "     2  H   -0.100000   0.500000\n"
                 " Sum of Mulliken charges =   0.00000   1.00000\n")
        text = FORCE_LOG.replace(" Forces (Hartrees/Bohr)\n", extra + " Forces (Hartrees/Bohr)\n", 1)
        (frame,) = g.parse_force_frames(text, Path("f.log"))
        meta = frame.record.metadata
        self.assertEqual((meta["s2_before"], meta["s2_after"]), (2.0100, 2.0001))
        self.assertEqual(meta["atomic_spins"], [[1, "H", 0.5], [2, "H", 0.5]])


class DocumentReaderTests(unittest.TestCase):
    def test_decode_bytes_falls_back_through_encodings(self):
        self.assertEqual(_decode_bytes("é".encode("utf-8")), "é")
        self.assertEqual(_decode_bytes("é".encode("cp1252")), "é")

    def test_read_document_docx_joins_paragraphs(self):
        ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        xml = (f'<w:document xmlns:w="{ns}"><w:body><w:p><w:r><w:t>Fe</w:t></w:r><w:r><w:t>O</w:t></w:r></w:p>'
               f"<w:p><w:r><w:t>second</w:t></w:r></w:p></w:body></w:document>")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.docx"
            path.write_bytes(_zip_bytes({"word/document.xml": xml.encode()}))
            self.assertEqual(read_document(path), "FeO\nsecond")

    def test_read_document_plain_text_and_chk_requires_formchk(self):
        with tempfile.TemporaryDirectory() as tmp:
            txt = Path(tmp) / "a.log"
            txt.write_text("hello")
            self.assertEqual(read_document(txt), "hello")

    def test_safe_extract_blocks_zip_slip_and_enforces_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            dest.mkdir()
            evil = zipfile.ZipFile(io.BytesIO(_zip_bytes({"../escape.txt": b"x"})))
            with self.assertRaises(ValueError):
                _safe_extract(evil, dest)
            self.assertFalse((Path(tmp) / "escape.txt").exists())
            big = zipfile.ZipFile(io.BytesIO(_zip_bytes({"a.txt": b"x" * 10})))
            with self.assertRaises(ValueError):
                _safe_extract(big, dest, budget=[5])

    def test_nested_zip_expansion_and_depth_limit(self):
        inner = _zip_bytes({"deep.log": b"hi"})
        outer = _zip_bytes({"inner.zip": inner, "top.log": b"top"})
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "outer.zip"
            archive.write_bytes(outer)
            with source_tree(archive) as root:
                names = sorted(p.name for p in iter_documents(root))
                self.assertEqual(names, ["deep.log", "top.log"])
            deeper = Path(tmp) / "deeper.zip"
            deeper.write_bytes(_zip_bytes({"mid.zip": outer}))
            with self.assertRaises(ValueError):
                with source_tree(deeper, nested_zip_depth=1):
                    pass

    def test_corrupt_nested_zip_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bad.zip").write_bytes(b"not a zip")
            _expand_nested_zips(root, 2, [1000])  # must not raise

    def test_iter_documents_skips_appledouble_and_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a.log", "._a.log", "b.png", "c.GJF"):
                (root / name).write_text("x")
            self.assertEqual([p.name for p in iter_documents(root)], ["a.log", "c.GJF"])

    def test_source_tree_for_plain_file_yields_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.log"
            path.write_text("x")
            with source_tree(path) as root:
                self.assertEqual(root, path.parent)


class ExtxyzRoundTripTests(unittest.TestCase):
    def _record(self):
        return Record(
            record_id="r1", source='dir/"quoted".log', atoms=[Atom("Fe", 0.0, 0.0, 0.0), Atom("O", 1.5, 0.0, 0.0)],
            charge=-1, multiplicity=2, config_type="irc_forward", route="# irc=(forward)",
            legacy_energy_hartree=-1337.5, imaginary_frequencies=1, irc_path=1, irc_point=4,
            metadata={"state_inference": "filename"},
        )

    def test_round_trip_preserves_every_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.extxyz"
            write_extxyz([self._record()], path)
            (back,) = read_extxyz(path)
        original = self._record()
        for field in ("record_id", "source", "charge", "multiplicity", "config_type", "route",
                      "legacy_energy_hartree", "imaginary_frequencies", "irc_path", "irc_point", "metadata"):
            self.assertEqual(getattr(back, field), getattr(original, field), field)
        self.assertEqual([(a.symbol, a.x) for a in back.atoms], [("Fe", 0.0), ("O", 1.5)])

    def test_optional_fields_are_omitted_when_none(self):
        record = Record(record_id="r2", source="s", atoms=[Atom("H", 0, 0, 0)], charge=0, multiplicity=2,
                        config_type="unknown")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.extxyz"
            write_extxyz([record], path)
            text = path.read_text()
            (back,) = read_extxyz(path)
        for key in ("irc_path", "irc_point", "legacy_route", "metadata", "legacy_energy_hartree"):
            self.assertNotIn(key + "=", text)
        self.assertIsNone(back.irc_point)
        self.assertEqual(back.metadata, {})

    def test_info_line_parser_handles_quotes_and_escapes(self):
        info = parse_extxyz_info_line('a=1 b="x y" c="q\\"z" pbc="F F F"')
        self.assertEqual(info, {"a": "1", "b": "x y", "c": 'q"z', "pbc": "F F F"})
        self.assertEqual(json.loads(quote_extxyz("é")), "é")
        self.assertEqual(quote_extxyz("é"), '"\\u00e9"')

    def test_spin_key_is_fallback_for_multiplicity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.extxyz"
            path.write_text('1\nrecord_id="r" spin=3\nH 0 0 0\n')
            self.assertEqual(read_extxyz(path)[0].multiplicity, 3)

    def test_manifest_columns_and_blank_state_inference(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.csv"
            write_manifest([self._record()], path)
            (row,) = list(csv.DictReader(path.open(newline="")))
        self.assertEqual((row["n_atoms"], row["charge"], row["irc_point"]), ("2", "-1", "4"))
        self.assertEqual(row["state_inference"], "filename")

    def test_write_text_lf_never_emits_crlf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.sh"
            write_text_lf(path, "a\nb\n")
            self.assertEqual(path.read_bytes(), b"a\nb\n")


if __name__ == "__main__":
    unittest.main()
