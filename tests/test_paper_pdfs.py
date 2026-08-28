from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
import zipfile

from cluster_mlip.cli import build_parser, main
from cluster_mlip.literature import normalize_doi

_HAVE_PYPDF = importlib.util.find_spec("pypdf") is not None

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES / "sample_paper.pdf"


@unittest.skipUnless(_HAVE_PYPDF, "requires the optional `pdf` extra (pypdf) for text extraction")
class PaperPdfTests(unittest.TestCase):
    def test_extracts_doi_and_compositions_from_real_pdf_text(self):
        from cluster_mlip.paper_pdfs import extract_pdf_text, find_doi

        text = extract_pdf_text(SAMPLE_PDF.read_bytes())
        self.assertIn("Fe6O20", text)
        self.assertEqual(find_doi(text), "10.1021/example.2024")

    def test_unreadable_pdf_is_reported_not_raised(self):
        from cluster_mlip.paper_pdfs import index_paper_pdfs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "garbage.pdf").write_bytes(b"not actually a pdf")
            papers = index_paper_pdfs(root)
            self.assertEqual(len(papers), 1)
            self.assertEqual(papers[0]["status"], "unreadable")
            self.assertIsNone(papers[0]["doi"])

    def test_indexes_a_zip_of_pdfs_and_writes_report(self):
        from cluster_mlip.paper_pdfs import load_pdf_compositions, write_pdf_index

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zip_path = root / "papers.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.write(SAMPLE_PDF, "sample_paper.pdf")
            output = root / "out"
            result = write_pdf_index(zip_path, output)
            self.assertEqual(result["n_pdfs"], 1)
            self.assertEqual(result["counts"], {"ok": 1})
            payload = json.loads((output / "pdf_index.json").read_text())
            self.assertEqual(len(payload["papers"]), 1)
            self.assertEqual(payload["papers"][0]["doi"], "10.1021/example.2024")
            self.assertIn("Fe6O20", payload["papers"][0]["compositions"])

            mapping = load_pdf_compositions(output / "pdf_index.json")
            self.assertEqual(mapping["10.1021/example.2024"], ["Fe6O20"])

    def test_pdf_index_cli_command_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "cli_out"
            self.assertEqual(main(["pdf-index", str(SAMPLE_PDF), "-o", str(output)]), 0)
            self.assertTrue((output / "pdf_index.json").is_file())

    def test_literature_gap_merges_full_text_compositions(self):
        from cluster_mlip.literature import classify_paper

        work = {
            "title": "A short note",
            "abstract_inverted_index": None,
            "doi": "https://doi.org/10.1021/example.2024",
            "publication_year": 2024,
        }
        pdf_compositions = {"10.1021/example.2024": ["Fe6O20"]}
        paper = classify_paper(work, known_formulas=set(), pdf_compositions=pdf_compositions)
        self.assertEqual(paper["compositions"], ["Fe6O20"])
        self.assertTrue(paper["full_text_checked"])
        self.assertEqual(paper["status"], "possible_gap")


class DoiNormalizationTests(unittest.TestCase):
    def test_normalizes_url_and_bare_forms_identically(self):
        self.assertEqual(normalize_doi("https://doi.org/10.1021/Example.2024"), "10.1021/example.2024")
        self.assertEqual(normalize_doi("10.1021/Example.2024"), "10.1021/example.2024")

    def test_strips_trailing_prose_punctuation(self):
        self.assertEqual(normalize_doi("10.1021/example.2024)."), "10.1021/example.2024")


class PdfIndexHelpTests(unittest.TestCase):
    def test_pdf_index_help_renders(self):
        parser = build_parser()
        pdf_index = parser._subparsers._group_actions[0].choices["pdf-index"]
        self.assertIn("ZIP of PDFs", pdf_index.format_help())


if __name__ == "__main__":
    unittest.main()
