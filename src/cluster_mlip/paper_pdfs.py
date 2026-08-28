"""Deep, offline composition extraction from a local corpus of paper PDFs.

`literature.py`'s gap report matches formulas out of each OpenAlex work's
title/abstract only -- OpenAlex doesn't give us full text, and a paper's
title/abstract can easily omit a composition that's only discussed in the
body. When the actual PDFs are on hand (a delivery like Gennady's), this
module does a slower, more thorough pass: extract each PDF's full text,
pull out its DOI so it can be matched back to the right OpenAlex work, and
mine the whole document for formulas instead of just two sentences.

This is deliberately a separate, offline step (`cluster-mlip pdf-index`,
see scripts/run_pdf_index_slurm.sh) rather than folded into `literature-gap`
directly: parsing a large PDF corpus is slow and CPU-bound but needs no
network, the opposite profile of `literature-gap`'s OpenAlex fetch, which
needs network but is fast. Keeping them separate means the slow part can run
as an unattended single-core batch job while the network part stays a quick
interactive command from a login node.

`pypdf` is required for this module only -- installed via the optional
`pdf` extra (`pip install -e '.[pdf]'`) -- and is imported lazily inside
`extract_pdf_text`, the same isolation pattern `mace_glue.py` uses for
torch, so nothing else in this package needs it importable.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Iterator, TypedDict

from .io import source_tree
from .literature import extract_compositions, normalize_doi

# A DOI's own syntax (ISO 26324) is just "10." + a registrant prefix + "/" +
# a registrant-chosen suffix; the suffix has no universally fixed shape, so
# this stops at the first character that's essentially never part of one in
# practice (whitespace or the punctuation/quoting a PDF layout tends to glue
# on immediately after: a trailing period, closing bracket/paren, comma, or
# angle bracket from surrounding prose or a hyperlink).
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+")


class PdfPaper(TypedDict):
    source_pdf: str
    doi: str | None
    compositions: list[str]
    text_length: int
    status: str  # "ok" | "no_doi_found" | "unreadable"
    error: str | None


def find_doi(text: str) -> str | None:
    """First DOI-shaped token in the text, or None if the PDF doesn't
    contain one findable this way (e.g. a scanned image with no text layer,
    or a preprint with the DOI only as a hyperlink target rather than
    visible text)."""
    match = _DOI_RE.search(text)
    if not match:
        return None
    return normalize_doi(match.group(0))


def extract_pdf_text(data: bytes) -> str:
    """Full text of every page, concatenated. Raises on a PDF pypdf cannot
    parse at all (encrypted with an unrecoverable password, corrupt); the
    caller is responsible for turning that into a per-paper "unreadable"
    status rather than aborting the whole corpus over one bad file.
    """
    import io

    import pypdf  # noqa: PLC0415 -- deliberately lazy, see module docstring

    reader = pypdf.PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        # Many publisher-distributed reprints (Elsevier, ACS, Wiley, ...)
        # are "encrypted" only to restrict printing/copying in a viewer,
        # with an empty user password -- pypdf still refuses to read pages
        # until decrypt() is called explicitly, even though no real
        # password is needed. This is worth trying before giving up; a
        # genuinely password-protected PDF still raises once this fails.
        reader.decrypt("")
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _iter_pdf_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.pdf")):
        if path.is_file() and not path.name.startswith("._"):
            yield path


def iter_pdf_sources(source: Path) -> Iterator[tuple[str, bytes]]:
    """Yield (display_name, pdf_bytes) for every PDF under ``source``.

    ``source`` may be a single ZIP of PDFs (Gennady's delivery format), a
    directory of PDFs (already unpacked), or a single ``.pdf`` file.
    Reuses `io.source_tree`'s safe/size-bounded ZIP extraction rather than
    re-implementing ZIP handling here.
    """
    if source.is_file() and source.suffix.lower() == ".pdf":
        yield source.name, source.read_bytes()
        return
    with source_tree(source) as root:
        for path in _iter_pdf_files(root):
            yield str(path.relative_to(root)), path.read_bytes()


def _require_pypdf() -> None:
    """Fail once, clearly, if pypdf isn't installed -- rather than letting
    every single PDF in the corpus fail identically inside the per-file
    try/except below and get mislabeled "unreadable", which looks like a
    corrupt-file problem but is actually a missing-dependency problem (this
    exact failure mode was hit for real: a full run reported 405/405 PDFs
    "unreadable" with 0 ok and 0 no_doi_found, which is what a missing
    import looks like, not what 405 genuinely bad PDFs look like).
    """
    try:
        import pypdf  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pdf-index requires pypdf, which isn't installed in this Python "
            "environment. Install the optional 'pdf' extra in the same "
            "environment you run `cluster-mlip` from: "
            "pip install -e '.[pdf]' (or `pip install pypdf` directly)."
        ) from exc


def index_paper_pdfs(source: Path) -> list[PdfPaper]:
    """Extract text, DOI, and formula mentions from every PDF under
    ``source``. Every PDF gets an entry -- one that couldn't be read, or
    read but had no DOI found, is reported with that status rather than
    silently dropped, so a person can see what's missing from the index
    rather than assuming full coverage.
    """
    _require_pypdf()
    papers: list[PdfPaper] = []
    for name, data in iter_pdf_sources(source):
        try:
            text = extract_pdf_text(data)
        except Exception as exc:  # noqa: BLE001 -- any one bad PDF must not abort the batch
            papers.append({
                "source_pdf": name,
                "doi": None,
                "compositions": [],
                "text_length": 0,
                "status": "unreadable",
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        doi = find_doi(text)
        papers.append({
            "source_pdf": name,
            "doi": doi,
            "compositions": sorted(set(extract_compositions(text))),
            "text_length": len(text),
            "status": "ok" if doi else "no_doi_found",
            "error": None,
        })
    return papers


def write_pdf_index(source: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    papers = index_paper_pdfs(source)
    counts: dict[str, int] = {}
    for paper in papers:
        counts[paper["status"]] = counts.get(paper["status"], 0) + 1
    payload = {"source": str(source), "papers": papers, "counts": counts}
    (output / "pdf_index.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # The most common distinct error messages among unreadable PDFs, most
    # frequent first -- surfaced directly to the CLI so a systemic problem
    # (e.g. every PDF hitting the same exception) is obvious without having
    # to go dig through pdf_index.json by hand.
    error_counts: dict[str, int] = {}
    for paper in papers:
        if paper["error"]:
            error_counts[paper["error"]] = error_counts.get(paper["error"], 0) + 1
    top_errors = sorted(error_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "n_pdfs": len(papers),
        "counts": counts,
        "output": str(output / "pdf_index.json"),
        "top_errors": top_errors,
    }


def load_pdf_compositions(path: Path) -> dict[str, list[str]]:
    """Load a `pdf_index.json` into {normalized_doi: compositions}, for
    literature.py to merge into its title/abstract-only extraction. PDFs
    with no findable DOI aren't usable here (there's nothing to match them
    to an OpenAlex work by) and are simply absent from the returned map --
    `pdf_index.json` itself still records them under "no_doi_found" for a
    human to notice and reconcile by hand.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for paper in data.get("papers", []):
        doi = paper.get("doi")
        if doi:
            result[doi] = list(paper.get("compositions", []))
    return result
