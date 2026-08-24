from pathlib import Path
import contextlib
import io
import json
import tempfile
import unittest

from cluster_mlip.batch_inventory import build_inventory
from cluster_mlip.cli import build_parser
from cluster_mlip.literature import (
    build_gap_report,
    classify_paper,
    extract_compositions,
    load_known_formulas,
    normalize_author_ids,
    write_gap_report,
)

FIXTURES = Path(__file__).parent / "fixtures"


class AuthorIdTests(unittest.TestCase):
    def test_normalizes_urls_and_removes_duplicates(self):
        self.assertEqual(
            normalize_author_ids(
                ["https://openalex.org/a5029253658/", "A5029253658", "A123"]
            ),
            ("A5029253658", "A123"),
        )

    def test_requires_an_explicit_valid_author_id(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            normalize_author_ids([])
        with self.assertRaisesRegex(ValueError, "invalid OpenAlex author id"):
            normalize_author_ids(["Gennady Gutsev"])

    def test_cli_has_no_implicit_author(self):
        parser = build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["literature-gap", "inventory"])
        args = parser.parse_args(
            ["literature-gap", "inventory", "--author-id", "A123"]
        )
        self.assertEqual(args.author_id, ["A123"])


class ExtractCompositionsTests(unittest.TestCase):
    def test_extracts_real_formulas_with_counts(self):
        text = "Structure and properties of iron oxide clusters: From Fe6 to Fe6O20 and from Fe7 to Fe7O24"
        self.assertEqual(
            extract_compositions(text), ["Fe6", "Fe6O20", "Fe7", "Fe7O24"]
        )

    def test_extracts_all_one_count_multi_element_formula(self):
        self.assertEqual(extract_compositions("DFT study of NiO and CoO nanoclusters"), ["NiO", "CoO"])

    def test_does_not_match_element_symbol_embedded_in_an_ordinary_word(self):
        # "Fe" inside "Ferromagnetic"/"Fermi-level" must not spuriously match.
        text = "Ferromagnetic ordering was observed in Fermi-level calculations"
        self.assertEqual(extract_compositions(text), [])

    def test_does_not_match_bare_single_element_symbol_colliding_with_a_word(self):
        # "As" (Arsenic) is also an ordinary English word; a bare, digit-less
        # single-element token is rejected to avoid this class of false
        # positive, even though it means missing genuine bare mentions.
        text = "As shown in Figure 1, the result was notable"
        self.assertEqual(extract_compositions(text), [])

    def test_general_series_notation_is_not_treated_as_a_specific_formula(self):
        # "FenO" (literal lowercase n standing in for an unspecified size) is
        # not a concrete composition to compare against the local inventory;
        # correctly not extracting it (rather than guessing a size) is the
        # point, not a bug -- the paper still surfaces via classify_paper's
        # "unclear" bucket for human review.
        self.assertEqual(extract_compositions("Structural properties of FenO and FenO- clusters"), [])


class ClassifyPaperTests(unittest.TestCase):
    def test_all_compositions_known_is_on_file(self):
        work = {"title": "Fe6O20 revisited", "publication_year": 2020, "doi": "10.1/x"}
        paper = classify_paper(work, {"Fe6O20"})
        self.assertEqual(paper["status"], "on_file")
        self.assertEqual(paper["compositions"], ["Fe6O20"])

    def test_partial_match_is_possible_gap_not_on_file(self):
        # Fe6O20 is known but Fe9O12 is not -- a paper covering both must not
        # be silently marked "on file" just because part of it matched.
        work = {"title": "Fe9O12 and Fe6O20 clusters compared", "publication_year": 2021, "doi": None}
        paper = classify_paper(work, {"Fe6O20"})
        self.assertEqual(paper["status"], "possible_gap")

    def test_no_extractable_formula_is_unclear(self):
        work = {"title": "A general study of iron clusters", "publication_year": 2019, "doi": None}
        paper = classify_paper(work, {"Fe6O20"})
        self.assertEqual(paper["status"], "unclear")
        self.assertEqual(paper["compositions"], [])

    def test_reconstructs_abstract_from_openalex_inverted_index(self):
        # {"Fe2O2": [0], "clusters": [1]} -> "Fe2O2 clusters"
        work = {
            "title": "Untitled",
            "publication_year": 2018,
            "doi": None,
            "abstract_inverted_index": {"Fe2O2": [0], "clusters": [1], "studied": [2]},
        }
        paper = classify_paper(work, set())
        self.assertEqual(paper["compositions"], ["Fe2O2"])


class GapReportTests(unittest.TestCase):
    def test_write_gap_report_groups_by_status_with_missing_first(self):
        papers = build_gap_report(
            [
                {"title": "Have it", "publication_year": 2020, "doi": "10.1/a"},
                {"title": "Fe9O12 missing", "publication_year": 2021, "doi": "10.1/b"},
                {"title": "No formula here", "publication_year": 2022, "doi": None},
            ],
            known_formulas=set(),
        )
        # Rig one paper to be genuinely "on_file" for the grouping test.
        papers[0] = {**papers[0], "compositions": ["Fe6O20"], "status": "on_file"}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "gap"
            summary = write_gap_report(papers, output)
            self.assertEqual(summary["n_papers"], 3)
            text = (output / "literature_gap.md").read_text(encoding="utf-8")
            # The actionable "please send these" section must appear before
            # "we already have" -- that's the whole point of the ordering.
            # (Search the section headings specifically, not the summary
            # paragraph above them, which also happens to say "already have".)
            self.assertLess(
                text.index("## Papers we may be missing"),
                text.index("## Papers we already have"),
            )
            self.assertTrue((output / "literature_gap.json").exists())
            data = json.loads((output / "literature_gap.json").read_text(encoding="utf-8"))
            self.assertEqual(data["counts"]["possible_gap"], 1)
            self.assertEqual(data["counts"]["unclear"], 1)
            self.assertEqual(data["counts"]["on_file"], 1)
            self.assertEqual(data["query"]["author_ids"], [])

    def test_readable_format_is_plain_blocks_not_a_dense_table(self):
        papers = build_gap_report(
            [{"title": "Fe9O12 clusters", "publication_year": 2021, "doi": "10.1/x"}], set()
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "gap"
            write_gap_report(papers, output)
            text = (output / "literature_gap.md").read_text(encoding="utf-8")
            self.assertIn("1. **Fe9O12 clusters** (2021)", text)
            self.assertIn("Formulas mentioned: Fe9O12", text)
            self.assertIn("doi.org/10.1/x", text)

    def test_report_is_author_agnostic_and_records_query(self):
        papers = build_gap_report(
            [{"title": "Ni4O4 clusters", "publication_year": 2024, "doi": None}], set()
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "gap"
            write_gap_report(
                papers,
                output,
                author_ids=("A123",),
                author_name="Example Researcher",
                keywords=("nickel cluster",),
            )
            text = (output / "literature_gap.md").read_text(encoding="utf-8")
            self.assertIn("# Literature gap report for Example Researcher", text)
            self.assertNotIn("Gennady", text)
            data = json.loads((output / "literature_gap.json").read_text(encoding="utf-8"))
            self.assertEqual(data["query"]["author_ids"], ["A123"])
            self.assertEqual(data["query"]["keywords"], ["nickel cluster"])


class LoadKnownFormulasTests(unittest.TestCase):
    def test_reads_existing_inventory_output(self):
        import zipfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "warehouses"
            folder.mkdir()
            with zipfile.ZipFile(folder / "d1.zip", "w") as archive:
                archive.write(FIXTURES / "minimum.log", "minimum.log")
            inventory_dir = root / "inventory_out"
            build_inventory(folder, inventory_dir)

            formulas = load_known_formulas(inventory_dir, root / "gap_out")
            self.assertIn("FeO", formulas)

    def test_builds_inventory_inline_from_a_raw_zip_folder(self):
        import zipfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "warehouses"
            folder.mkdir()
            with zipfile.ZipFile(folder / "d1.zip", "w") as archive:
                archive.write(FIXTURES / "minimum.log", "minimum.log")

            gap_output = root / "gap_out"
            formulas = load_known_formulas(folder, gap_output)
            self.assertIn("FeO", formulas)
            self.assertTrue((gap_output / "inventory" / "inventory.json").exists())


if __name__ == "__main__":
    unittest.main()
