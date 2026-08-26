from __future__ import annotations

import collections
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, TypedDict

from .batch_inventory import build_inventory
from .batch_inventory import known_formulas as inventory_known_formulas
from .gaussian import ATOMIC_SYMBOLS

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

# Used only as a *local*, secondary relevance signal now (see
# filter_relevant_works) -- never as the OpenAlex query filter. An earlier
# version filtered the API query by these literal English phrases, which
# silently dropped most of a prolific cluster-science author's papers:
# titles in this field are overwhelmingly written as formulas ("Fe6O20",
# "Fe2O4-6+ Clusters") rather than the words "iron cluster"/"iron oxide
# cluster", so the phrase filter never saw them. Confirmed against a real
# author query: the phrase filter returned 41 works when the true count of
# relevant papers is known to be much larger.
DEFAULT_KEYWORDS = ("cluster", "iron", "oxide")

_ELEMENT_SYMBOLS = set(ATOMIC_SYMBOLS.values())
# A run of 1-4 element-symbol(+count) tokens, anchored on word boundaries so
# it can't match a chemical-symbol-like prefix embedded in an ordinary word
# (e.g. "Fe" inside "Ferromagnetic": nothing follows the match but a
# lowercase word character, so the trailing \b fails and the whole attempt
# is rejected).
_TOKEN_RE = re.compile(r"\b(?:[A-Z][a-z]?\d*){1,4}\b")
_PART_RE = re.compile(r"[A-Z][a-z]?\d*")


class Paper(TypedDict):
    title: str
    year: int | None
    doi: str | None
    oa_url: str | None
    compositions: list[str]
    status: str  # "on_file" | "possible_gap" | "unclear"


def normalize_author_ids(author_ids: Iterable[str]) -> tuple[str, ...]:
    """Return unique OpenAlex author IDs in canonical ``A<digits>`` form.

    Accept full OpenAlex author URLs as a convenience, but never resolve a
    person from a name: author disambiguation is a scientific/provenance
    decision and must remain explicit at the command line.
    """
    normalized: list[str] = []
    for value in author_ids:
        author_id = value.strip().rstrip("/").rsplit("/", 1)[-1].upper()
        if not re.fullmatch(r"A\d+", author_id):
            raise ValueError(
                f"invalid OpenAlex author id {value!r}; expected A followed by digits "
                "(for example A5029253658)"
            )
        if author_id not in normalized:
            normalized.append(author_id)
    if not normalized:
        raise ValueError("at least one OpenAlex author id is required")
    return tuple(normalized)


def normalize_orcids(orcids: Iterable[str]) -> tuple[str, ...]:
    """Return unique ORCID iDs in canonical hyphenated form.

    Both bare iDs and ``https://orcid.org/...`` URLs are accepted. In
    addition to checking the shape, validate the ISO 7064 MOD 11-2 check
    digit so a typo cannot silently produce an empty literature query.
    """
    normalized: list[str] = []
    for value in orcids:
        orcid = value.strip().rstrip("/").rsplit("/", 1)[-1].upper()
        if not re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", orcid):
            raise ValueError(
                f"invalid ORCID {value!r}; expected 0000-0000-0000-0000 "
                "(the final character may be X)"
            )
        digits = orcid.replace("-", "")
        total = 0
        for digit in digits[:-1]:
            total = (total + int(digit)) * 2
        result = (12 - total % 11) % 11
        expected = "X" if result == 10 else str(result)
        if digits[-1] != expected:
            raise ValueError(f"invalid ORCID {value!r}; check digit does not match")
        if orcid not in normalized:
            normalized.append(orcid)
    return tuple(normalized)


def extract_compositions(text: str) -> list[str]:
    """Heuristic chemical-formula extraction from free text (paper titles/
    abstracts). Deliberately conservative: a bare single element symbol with
    no count digit (e.g. "As", "In", "He", "No") is rejected because it
    collides too easily with ordinary English words; a multi-element run
    (with or without counts, e.g. "FeO") or any single element *with* a
    count (e.g. "Fe6") is accepted. This is a first-pass text-mining
    heuristic, not a chemistry parser -- it will both miss real formulas
    (e.g. the "Fe_n" general-series notation many cluster papers use, which
    isn't a specific composition to compare against anyway) and occasionally
    over-match; results are for human review, not automatic decisions.
    """
    results: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        parts = _PART_RE.findall(token)
        symbols = [part.rstrip("0123456789") for part in parts]
        if not parts or any(symbol not in _ELEMENT_SYMBOLS for symbol in symbols):
            continue
        has_digit = any(char.isdigit() for char in token)
        if len(parts) < 2 and not has_digit:
            continue
        results.append(token)
    return results


def _elements_in_token(token: str) -> set[str]:
    return {part.rstrip("0123456789") for part in _PART_RE.findall(token)}


def target_elements_from_formulas(known_formulas: Iterable[str]) -> set[str]:
    """Which elements the *local* warehouse actually contains -- used to
    decide whether a fetched paper is topically relevant, since we no longer
    ask OpenAlex to pre-filter by topic (see fetch_openalex_works)."""
    elements: set[str] = set()
    for formula in known_formulas:
        elements |= _elements_in_token(formula)
    return elements


def is_relevant_work(
    title: str, abstract: str, compositions: list[str], target_elements: set[str], keywords: Iterable[str]
) -> bool:
    """Decide whether a fetched paper belongs in the report at all.

    An author with a large body of work (hundreds of papers, in Gutsev's
    case) writes about plenty of things other than the clusters this
    warehouse cares about; showing every one of them would bury the
    actionable list in noise. A paper is kept if it mentions a formula built
    from an element the local warehouse actually has (the precise signal),
    or -- when that can't be determined, e.g. an empty local inventory, or
    the paper's formula didn't parse -- if the plain keyword list matches
    anywhere in the title/abstract (the loose fallback signal). Both checks
    run client-side, over the full fetched bibliography; see
    fetch_openalex_works for why this isn't done as an OpenAlex query filter.
    """
    composition_elements = {element for composition in compositions for element in _elements_in_token(composition)}
    if target_elements and composition_elements & target_elements:
        return True
    if compositions and not target_elements:
        return True
    lowered = f"{title} {abstract}".lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def filter_relevant_works(
    works: list[dict[str, object]],
    known_formulas: set[str],
    keywords: Iterable[str] = DEFAULT_KEYWORDS,
) -> list[dict[str, object]]:
    target_elements = target_elements_from_formulas(known_formulas)
    keyword_values = tuple(keywords)
    relevant = []
    for work in works:
        title = str(work.get("title") or "")
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
        compositions = extract_compositions(title) + extract_compositions(abstract)
        if is_relevant_work(title, abstract, compositions, target_elements, keyword_values):
            relevant.append(work)
    return relevant


def _reconstruct_abstract(inverted_index: object) -> str:
    """OpenAlex represents an abstract as {word: [positions]} rather than
    plain text (a side effect of how they're allowed to redistribute
    copyrighted abstracts); this puts it back in reading order.
    """
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    max_position = max(
        (position for positions in inverted_index.values() for position in positions), default=-1
    )
    words: list[str] = [""] * (max_position + 1)
    for word, positions in inverted_index.items():
        for position in positions:
            words[position] = str(word)
    return " ".join(words)


def fetch_openalex_works(
    author_ids: Iterable[str] = (),
    orcids: Iterable[str] = (),
    keywords: Iterable[str] = (),
    contact_email: str | None = None,
    per_page: int = 200,
    timeout: float = 30.0,
    max_pages: int = 50,
) -> list[dict[str, object]]:
    """Fetch every work by the given author(s) from the free, keyless
    OpenAlex Works API, fully paginated (an author with a large body of
    work -- Gutsev has ~300 -- would otherwise be silently truncated to one
    page). `keywords`, if given, additionally restricts the *query itself*
    to works whose title/abstract match one of the phrases; the default is
    no such restriction, since the earlier default did exactly this and
    silently dropped most of a prolific author's cluster papers -- this
    field's titles are overwhelmingly written as formulas ("Fe6O20") rather
    than the English phrases a keyword filter needs ("iron oxide cluster").
    Topic relevance is decided client-side instead, see
    filter_relevant_works, which runs over this function's full, unfiltered
    result and is what actually keeps the report focused.

    The only function in this package that talks to the network, isolated
    here the same way mace_glue.py isolates the one function needing torch,
    so everything else (extraction, relevance, classification, report
    writing) is unit-testable without a real network call. `contact_email`
    fills OpenAlex's documented "polite pool" `mailto` parameter for better
    rate limits; omit it and the request still works, just at a lower
    priority. `max_pages` (default 50, i.e. 10,000 works at per_page=200) is
    a safety cap matching OpenAlex's basic-pagination limit.
    """
    author_id_values = tuple(author_ids)
    orcid_values = tuple(orcids)
    keyword_values = tuple(keywords)
    normalized_author_ids = normalize_author_ids(author_id_values) if author_id_values else ()
    normalized_orcids = normalize_orcids(orcid_values)
    if not normalized_author_ids and not normalized_orcids:
        raise ValueError("at least one OpenAlex author id or ORCID is required")

    author_filters: list[str] = []
    if normalized_author_ids:
        author_filters.append(f"authorships.author.id:{'|'.join(normalized_author_ids)}")
    if normalized_orcids:
        canonical_orcids = [f"https://orcid.org/{orcid}" for orcid in normalized_orcids]
        author_filters.append(f"authorships.author.orcid:{'|'.join(canonical_orcids)}")

    works: list[dict[str, object]] = []
    seen: set[str] = set()
    for author_filter in author_filters:
        filter_value = author_filter
        if keyword_values:
            filter_value += f",title_and_abstract.search:{'|'.join(keyword_values)}"
        for page in range(1, max_pages + 1):
            params = {
                "filter": filter_value,
                "per-page": str(per_page),
                "page": str(page),
                "select": "id,title,publication_year,doi,abstract_inverted_index,open_access",
            }
            if contact_email:
                params["mailto"] = contact_email
            url = f"{OPENALEX_WORKS_URL}?{urllib.parse.urlencode(params)}"
            request = urllib.request.Request(
                url, headers={"User-Agent": "cluster-mlip literature-gap (mailto: none)"}
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 -- fixed https API host
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError as exc:
                # This is the one command in the pipeline that needs live internet --
                # give a clear message instead of a raw urllib/ssl traceback, since
                # every other command in this project is designed to run fully
                # offline on an HPC compute node that may have no route out at all.
                raise RuntimeError(
                    f"could not reach {OPENALEX_WORKS_URL}: {exc}. This command needs internet "
                    "access -- run it from a login node or local machine, not an offline HPC "
                    "compute node. If this is a TLS/certificate error, it may be your local "
                    "network's proxy rather than this tool."
                ) from exc
            results = payload.get("results", [])
            if not isinstance(results, list) or not results:
                break
            for work in results:
                if not isinstance(work, dict):
                    continue
                identity = str(
                    work.get("id")
                    or work.get("doi")
                    or (work.get("title"), work.get("publication_year"))
                )
                if identity not in seen:
                    seen.add(identity)
                    works.append(work)
            if len(results) < per_page:
                break
    return works


def _open_access_url(work: dict[str, object]) -> str | None:
    """A genuinely free copy of the paper (green/gold open access, e.g. an
    institutional repository or a fully-OA journal), per OpenAlex's own
    open_access tracking -- not the publisher's DOI, which is frequently
    paywalled. Most of this literature predates open-access norms and
    simply has no free copy anywhere; this returns None in that case rather
    than falling back to a paywalled link mislabeled as free.
    """
    oa = work.get("open_access")
    if not isinstance(oa, dict) or not oa.get("is_oa"):
        return None
    url = oa.get("oa_url")
    return str(url) if url else None


def classify_paper(work: dict[str, object], known_formulas: set[str]) -> Paper:
    title = str(work.get("title") or "(untitled)")
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    compositions = sorted(set(extract_compositions(title) + extract_compositions(abstract)))
    doi = work.get("doi")
    year = work.get("publication_year")
    if not compositions:
        status = "unclear"
    elif all(composition in known_formulas for composition in compositions):
        status = "on_file"
    else:
        # Even one uncovered formula in a multi-composition paper is worth a
        # human look -- silently calling the whole paper "on file" because
        # *something* in it matched would hide a real gap.
        status = "possible_gap"
    return {
        "title": title,
        "year": year if isinstance(year, int) else None,
        "doi": str(doi) if doi else None,
        "oa_url": _open_access_url(work),
        "compositions": compositions,
        "status": status,
    }


def build_gap_report(works: list[dict[str, object]], known_formulas: set[str]) -> list[Paper]:
    return [classify_paper(work, known_formulas) for work in works]


def load_known_formulas(source: Path, output: Path, jobs: int = 1) -> set[str]:
    """`source` is either an existing `cluster-mlip inventory` output
    directory (has inventory.json at its top level) or a raw folder of
    warehouse ZIPs, in which case the inventory is built inline into
    `output/inventory/` so this command is self-contained.
    """
    existing = source / "inventory.json"
    if existing.is_file():
        data = json.loads(existing.read_text(encoding="utf-8"))
        return {row["formula"] for row in data["master"]}
    result = build_inventory(source, output / "inventory", jobs=jobs)
    return inventory_known_formulas(result)


def write_gap_report(
    papers: list[Paper],
    output: Path,
    *,
    author_ids: Iterable[str] = (),
    orcids: Iterable[str] = (),
    keywords: Iterable[str] = (),
    author_name: str | None = None,
    n_fetched: int | None = None,
) -> dict[str, object]:
    """Written for a human reader who is not going to parse a dense table --
    a plain-English count up top, then one short numbered block per paper,
    grouped with the actionable "please send these" group first.

    `n_fetched`, when given, is the number of papers OpenAlex returned for
    the author *before* the local relevance filter -- shown so a reader can
    see the funnel (e.g. "292 papers by this author, 63 mention a relevant
    formula") rather than only ever seeing the narrowed count and wondering
    whether something was missed.
    """
    output.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter(paper["status"] for paper in papers)
    query = {
        "author_ids": list(author_ids),
        "author_name": author_name,
        "keywords": list(keywords),
        "orcids": list(orcids),
    }
    (output / "literature_gap.json").write_text(
        json.dumps(
            {"papers": papers, "counts": dict(counts), "query": query, "n_fetched": n_fetched},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    def paper_block(paper: Paper, number: int) -> list[str]:
        year = f" ({paper['year']})" if paper["year"] else ""
        block = [f"{number}. **{paper['title']}**{year}"]
        if paper["compositions"]:
            block.append(f"   - Formulas mentioned: {', '.join(paper['compositions'])}")
        else:
            block.append("   - No specific formula found in the title/abstract -- worth a quick look.")
        if paper["oa_url"]:
            block.append(f"   - Free PDF: {paper['oa_url']}")
        if paper["doi"]:
            note = "" if paper["oa_url"] else " (may require a subscription or the author directly)"
            block.append(
                f"   - Publisher page{note}: https://doi.org/{paper['doi'].removeprefix('https://doi.org/')}"
            )
        block.append("")
        return block

    heading = (
        f"# Literature gap report for {author_name}"
        if author_name
        else "# Literature gap report"
    )
    if n_fetched is not None and n_fetched != len(papers):
        summary_line = (
            f"OpenAlex returned {n_fetched} papers by this author; {len(papers)} of them "
            "mention a relevant formula or keyword and are listed below."
        )
    else:
        summary_line = f"We looked at {len(papers)} cluster papers returned by OpenAlex for the requested author query."
    lines = [
        heading,
        "",
        summary_line,
        "",
        f"- We already have data for **{counts.get('on_file', 0)}** of them.",
        f"- We may be **missing data for {counts.get('possible_gap', 0)}** of them -- see below.",
        f"- We were **not sure about {counts.get('unclear', 0)}** of them -- please check these too.",
        "",
        "This list was built automatically by matching formulas mentioned in each ",
        "paper's title/abstract against our files, so it can make mistakes -- please ",
        "double check before assuming something is really missing.",
        "",
    ]
    sections = [
        ("possible_gap", "## Papers we may be missing (please send these)"),
        ("unclear", "## Papers we are not sure about (please check)"),
        ("on_file", "## Papers we already have"),
    ]
    for status, heading in sections:
        group = [paper for paper in papers if paper["status"] == status]
        if not group:
            continue
        lines.append(heading)
        lines.append("")
        for index, paper in enumerate(group, start=1):
            lines.extend(paper_block(paper, index))
    (output / "literature_gap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"counts": dict(counts), "n_papers": len(papers), "n_fetched": n_fetched}


def run_literature_gap(
    source: Path,
    output: Path,
    author_ids: Iterable[str] = (),
    orcids: Iterable[str] = (),
    keywords: Iterable[str] = DEFAULT_KEYWORDS,
    contact_email: str | None = None,
    jobs: int = 1,
    author_name: str | None = None,
) -> dict[str, object]:
    author_id_values = tuple(author_ids)
    orcid_values = tuple(orcids)
    normalized_author_ids = normalize_author_ids(author_id_values) if author_id_values else ()
    normalized_orcids = normalize_orcids(orcid_values)
    if not normalized_author_ids and not normalized_orcids:
        raise ValueError("at least one OpenAlex author id or ORCID is required")
    normalized_keywords = tuple(keywords)
    known_formulas = load_known_formulas(source, output, jobs=jobs)
    # Fetch the author's full bibliography unfiltered by topic -- a
    # query-level keyword filter silently drops most cluster papers in this
    # field, since their titles are written as formulas, not English phrases
    # (see fetch_openalex_works) -- then narrow to relevant papers
    # client-side against the local warehouse's own elements.
    works = fetch_openalex_works(normalized_author_ids, normalized_orcids, contact_email=contact_email)
    relevant_works = filter_relevant_works(works, known_formulas, keywords=normalized_keywords)
    papers = build_gap_report(relevant_works, known_formulas)
    return write_gap_report(
        papers,
        output,
        author_ids=normalized_author_ids,
        orcids=normalized_orcids,
        keywords=normalized_keywords,
        author_name=author_name,
        n_fetched=len(works),
    )
