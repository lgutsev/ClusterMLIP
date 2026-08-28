from pathlib import Path
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import urllib.parse

from cluster_mlip.batch_inventory import build_inventory
from cluster_mlip.cli import build_parser, main
from cluster_mlip.literature import (
    build_gap_report,
    classify_paper,
    extract_compositions,
    fetch_openalex_works,
    filter_relevant_works,
    load_known_formulas,
    normalize_author_ids,
    normalize_orcids,
    target_elements_from_formulas,
    write_gap_report,
)

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeResponse:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


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

    def test_normalizes_and_validates_orcid(self):
        self.assertEqual(
            normalize_orcids(
                ["https://orcid.org/0000-0002-1825-0097/", "0000-0002-1825-0097"]
            ),
            ("0000-0002-1825-0097",),
        )
        with self.assertRaisesRegex(ValueError, "check digit"):
            normalize_orcids(["0000-0002-1825-0098"])

    def test_cli_has_no_implicit_author(self):
        parser = build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["literature-gap", "inventory"])
        args = parser.parse_args(
            ["literature-gap", "inventory", "--author-id", "A123"]
        )
        self.assertEqual(args.author_id, ["A123"])
        args = parser.parse_args(
            ["literature-gap", "inventory", "--orcid", "0000-0002-1825-0097"]
        )
        self.assertEqual(args.orcid, ["0000-0002-1825-0097"])

    def test_orcid_is_sent_as_the_documented_openalex_authorship_filter(self):
        payload = {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "title": "Ni4 cluster",
                    "publication_year": 2024,
                    "doi": None,
                }
            ]
        }
        with patch(
            "cluster_mlip.literature.urllib.request.urlopen",
            return_value=_FakeResponse(payload),
        ) as urlopen:
            works = fetch_openalex_works(
                orcids=("0000-0002-1825-0097",),
                keywords=("nickel cluster",),
            )
        self.assertEqual(len(works), 1)
        request = urlopen.call_args.args[0]
        decoded_url = urllib.parse.unquote(request.full_url)
        self.assertIn(
            "authorships.author.orcid:https://orcid.org/0000-0002-1825-0097",
            decoded_url,
        )


class FetchPaginationAndFilterTests(unittest.TestCase):
    def test_fetch_does_not_restrict_the_query_by_keyword_by_default(self):
        # The earlier default filtered the OpenAlex query itself by literal
        # English phrases ("iron cluster", "iron oxide cluster"), which
        # silently dropped most papers in this field since their titles are
        # written as formulas ("Fe6O20"), not those phrases -- confirmed
        # against a real author query returning far fewer papers than the
        # author's true output. Fetching must not filter by keyword unless
        # explicitly asked to.
        payload = {"results": [{"id": "https://openalex.org/W1", "title": "Fe6O20 structure"}]}
        with patch(
            "cluster_mlip.literature.urllib.request.urlopen", return_value=_FakeResponse(payload)
        ) as urlopen:
            works = fetch_openalex_works(author_ids=("A5029253658",))
        self.assertEqual(len(works), 1)
        decoded_url = urllib.parse.unquote(urlopen.call_args.args[0].full_url)
        self.assertNotIn("title_and_abstract.search", decoded_url)

    def test_fetch_paginates_through_a_large_bibliography(self):
        # per_page below is deliberately small so two pages are exercised
        # without needing to fabricate 200 fake works.
        page1 = {"results": [{"id": f"https://openalex.org/W{i}", "title": f"Fe{i}"} for i in range(3)]}
        page2 = {"results": [{"id": "https://openalex.org/W3", "title": "Fe3"}]}
        page3 = {"results": []}
        responses = [_FakeResponse(page1), _FakeResponse(page2), _FakeResponse(page3)]
        with patch(
            "cluster_mlip.literature.urllib.request.urlopen", side_effect=responses
        ):
            works = fetch_openalex_works(author_ids=("A5029253658",), per_page=3)
        self.assertEqual(len(works), 4)

    def test_fetch_deduplicates_across_author_id_and_orcid_pages(self):
        payload = {"results": [{"id": "https://openalex.org/W1", "title": "Fe6O20"}]}
        with patch(
            "cluster_mlip.literature.urllib.request.urlopen", return_value=_FakeResponse(payload)
        ):
            works = fetch_openalex_works(
                author_ids=("A5029253658",), orcids=("0000-0002-1825-0097",)
            )
        self.assertEqual(len(works), 1)


class RelevanceFilterTests(unittest.TestCase):
    def test_target_elements_from_formulas(self):
        self.assertEqual(target_elements_from_formulas({"Fe2O2", "FeN"}), {"Fe", "O", "N"})
        self.assertEqual(target_elements_from_formulas(set()), set())

    def test_keeps_a_paper_whose_formula_shares_a_local_element(self):
        # This is the exact case the earlier keyword filter missed: a title
        # that never says "iron" at all.
        works = [{"title": "Structure of Fe6O20 clusters", "publication_year": 2016}]
        relevant = filter_relevant_works(works, known_formulas={"FeO", "Fe2O2"})
        self.assertEqual(len(relevant), 1)

    def test_drops_a_paper_about_unrelated_chemistry(self):
        works = [{"title": "Vibrational spectroscopy of calcium fluoride complexes"}]
        relevant = filter_relevant_works(works, known_formulas={"FeO"}, keywords=("cluster",))
        self.assertEqual(relevant, [])

    def test_keyword_is_a_fallback_when_no_formula_is_extracted(self):
        works = [{"title": "A general survey of iron cluster chemistry"}]
        relevant = filter_relevant_works(works, known_formulas={"FeO"}, keywords=("cluster",))
        self.assertEqual(len(relevant), 1)

    def test_any_composition_counts_when_local_inventory_is_empty(self):
        works = [{"title": "NiO nanoclusters"}]
        relevant = filter_relevant_works(works, known_formulas=set(), keywords=())
        self.assertEqual(len(relevant), 1)


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


class OpenAccessTests(unittest.TestCase):
    def test_oa_url_present_when_openalex_reports_is_oa(self):
        work = {
            "title": "Fe6O20 revisited",
            "publication_year": 2020,
            "doi": "10.1/x",
            "open_access": {"is_oa": True, "oa_url": "https://example.edu/repo/fe6o20.pdf"},
        }
        paper = classify_paper(work, set())
        self.assertEqual(paper["oa_url"], "https://example.edu/repo/fe6o20.pdf")

    def test_oa_url_is_none_when_closed_access(self):
        # Most of this literature predates open-access norms and has no
        # free copy at all -- must report that honestly as None, not fall
        # back to a paywalled link mislabeled as free.
        work = {
            "title": "Fe6O20 revisited",
            "doi": "10.1/x",
            "open_access": {"is_oa": False, "oa_url": None},
        }
        self.assertIsNone(classify_paper(work, set())["oa_url"])

    def test_oa_url_is_none_when_field_is_absent(self):
        # Real OpenAlex payloads always include open_access, but every
        # existing hand-built test work dict in this file predates that
        # field -- must not crash on a missing key.
        paper = classify_paper({"title": "Fe6O20", "doi": "10.1/x"}, set())
        self.assertIsNone(paper["oa_url"])

    def test_free_pdf_line_appears_before_publisher_page_when_open_access(self):
        papers = build_gap_report(
            [
                {
                    "title": "Ni4O4 clusters",
                    "publication_year": 2024,
                    "doi": "10.1/oa",
                    "open_access": {"is_oa": True, "oa_url": "https://example.edu/ni4o4.pdf"},
                }
            ],
            set(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "gap"
            write_gap_report(papers, output)
            text = (output / "literature_gap.md").read_text(encoding="utf-8")
            self.assertIn("Free PDF: https://example.edu/ni4o4.pdf", text)
            self.assertIn("Publisher page: https://doi.org/10.1/oa", text)
            self.assertLess(text.index("Free PDF"), text.index("Publisher page"))

    def test_publisher_page_notes_possible_paywall_when_no_free_copy(self):
        papers = build_gap_report(
            [{"title": "Fe6O20 clusters", "publication_year": 2016, "doi": "10.1/x"}], set()
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "gap"
            write_gap_report(papers, output)
            text = (output / "literature_gap.md").read_text(encoding="utf-8")
            self.assertNotIn("Free PDF", text)
            self.assertIn("may require a subscription or the author directly", text)


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
            self.assertIn("Formulas mentioned (title/abstract): Fe9O12", text)
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
                orcids=("0000-0002-1825-0097",),
            )
            text = (output / "literature_gap.md").read_text(encoding="utf-8")
            self.assertIn("# Literature gap report for Example Researcher", text)
            self.assertNotIn("Gennady", text)
            data = json.loads((output / "literature_gap.json").read_text(encoding="utf-8"))
            self.assertEqual(data["query"]["author_ids"], ["A123"])
            self.assertEqual(data["query"]["keywords"], ["nickel cluster"])
            self.assertEqual(data["query"]["orcids"], ["0000-0002-1825-0097"])


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
