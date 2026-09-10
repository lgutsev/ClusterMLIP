"""Gaussian route intent: does a job actually search for the stationary point
its label claims?

A saddle point is not found by an ordinary ``Opt``. A bare ``Opt`` follows the
energy downhill and lands on the nearest *minimum*, silently discarding the
transition state its input geometry was a guess for. The job terminates
normally and produces a perfectly parseable optimized geometry, so nothing in
the completion-oriented audit notices; only the route reveals the mistake. A
saddle search needs, in the route:

* ``Opt=(TS,...)`` (or ``QST2``/``QST3``/``Saddle=N``) to select the search;
* an analytic starting Hessian (``CalcFC``/``CalcAll``/``ReadFC``), because the
  estimated Hessian rarely has usable curvature for a saddle;
* ``NoEigenTest``, or Gaussian aborts a TS search whose initial Hessian does
  not already have exactly one negative eigenvalue -- routine for a guess
  carried over from a different level of theory;
* ``Freq``, the only way to confirm the converged point has the intended
  number of imaginary modes.

Everything here is read-only text analysis. ``relaunch.py`` consumes it to
rebuild the affected inputs.
"""
from __future__ import annotations

import re
from pathlib import Path

from .stratify import pes_region


# Opt options that select a saddle search rather than a minimization.
_SADDLE_OPTIONS = ("ts", "qst2", "qst3", "saddle")
# Options that supply a computed starting Hessian.
_HESSIAN_OPTIONS = {"calcfc", "calcall", "readfc", "rcfc", "readfchk", "readcartesianfc"}

_LINK1_SPLIT_RE = re.compile(r"^\s*--\s*link1\s*--\s*$", re.IGNORECASE | re.MULTILINE)
# Gaussian accepts Opt, Opt=TS, Opt=(TS,CalcFC) and Opt(TS,CalcFC) alike, but
# options only ever attach through "=" or a parenthesised list -- never across
# whitespace. Allowing a bare following token would read "Opt Freq" as
# Opt=Freq and "Opt IOP(5/13=1)" as an IOP option, so a plain minimization
# would look like it carried options it does not have.
_OPT_RE = re.compile(
    r"\bopt(?:imize)?\b(?:\s*=\s*(\([^)]*\)|[^\s,]+)|\s*(\([^)]*\)))?",
    re.IGNORECASE,
)
_FREQ_RE = re.compile(r"\bfreq(?:uency)?\b", re.IGNORECASE)
_HARMONIC_RE = re.compile(r"Harmonic frequencies\s*\(cm\*\*-1\)", re.IGNORECASE)
_FREQ_LINE_RE = re.compile(r"Frequencies\s+--\s+([^\n\r]+)", re.IGNORECASE)


def _opt_option_tokens(route: str) -> list[str]:
    """Each ``Opt`` option as written, across every ``Opt`` keyword in a route."""
    tokens: list[str] = []
    for match in _OPT_RE.finditer(route):
        raw = (match.group(1) or match.group(2) or "").strip()
        if not raw:
            continue
        for item in raw.strip("()").split(","):
            item = item.strip()
            if item:
                tokens.append(item)
    return tokens


def opt_options(route: str) -> set[str]:
    """Lowercased ``Opt`` option keywords, with any ``=value`` suffix dropped."""
    return {token.split("=")[0].strip().lower() for token in _opt_option_tokens(route)}


def route_optimizes(route: str) -> bool:
    return _OPT_RE.search(route) is not None


def route_has_frequency(route: str) -> bool:
    return _FREQ_RE.search(route) is not None


def _selects_saddle(options: set[str]) -> bool:
    return any(option.startswith(_SADDLE_OPTIONS) for option in options)


def route_search_kind(route: str) -> str:
    """``saddle``, ``minimum`` or ``none`` -- what this route actually looks for."""
    if not route_optimizes(route):
        return "none"
    return "saddle" if _selects_saddle(opt_options(route)) else "minimum"


def intended_stationary_point(config_type: str) -> str:
    """What a job's *label* says its geometry is.

    Rattled variants are deliberately displaced and run as single points, so
    they carry no stationary-point intent even when their parent was a saddle.
    Types whose curvature was never established (``warehouse_structure``,
    ``optimized_unverified``, ``checkpoint_geometry``, IRC points) are
    ``unconstrained``: a plain ``Opt`` on them is a legitimate choice, not a
    mistake, and the audit must not manufacture work for them.
    """
    if config_type.endswith("_rattled"):
        return "unconstrained"
    region = pes_region(config_type)
    if region == "saddle":
        return "saddle"
    if region == "minimum":
        return "minimum"
    return "unconstrained"


def expected_imaginary_modes(config_type: str) -> int | None:
    base = config_type.removesuffix("_rattled")
    if base in ("transition_state", "first_order_saddle"):
        return 1
    if base == "minimum":
        return 0
    return None


def saddle_route_gaps(route: str) -> list[str]:
    """Missing ingredients of a route that is already a saddle search."""
    options = opt_options(route)
    gaps: list[str] = []
    if not any(option.startswith("noeigen") for option in options):
        gaps.append("noeigentest")
    if not options & _HESSIAN_OPTIONS:
        gaps.append("hessian")
    if not route_has_frequency(route):
        gaps.append("freq")
    return gaps


def corrected_saddle_route(route: str, order: int = 1) -> str:
    """Rewrite an optimizing route into a complete saddle search, in place.

    Existing ``Opt`` options are preserved (a campaign may legitimately carry
    ``ModRedundant`` or ``MaxCycles=``); only the missing saddle keywords are
    added, so a relaunch changes the search and nothing else about the
    protocol. Routes that do not optimize come back untouched -- the Link1
    force stage of a generated job must stay a plain ``Force``.
    """
    if not route_optimizes(route):
        return route
    tokens = _opt_option_tokens(route)
    options = opt_options(route)
    additions: list[str] = []
    if not _selects_saddle(options):
        additions.append("TS" if order <= 1 else f"Saddle={order}")
    if not options & _HESSIAN_OPTIONS:
        additions.append("CalcFC")
    if not any(option.startswith("noeigen") for option in options):
        additions.append("NoEigenTest")
    merged = list(dict.fromkeys(tokens + additions))
    corrected = _replace_opt_keyword(route, f"Opt=({','.join(merged)})")
    if not route_has_frequency(corrected):
        corrected = f"{corrected.rstrip()} Freq"
    return corrected


def corrected_minimum_route(route: str) -> str:
    """Strip saddle-search keywords from a route that should minimize.

    ``CalcFC`` is left alone: it costs an extra Hessian but changes no
    stationary point, and removing protocol settings we were not asked to
    change would make a relaunch harder to compare against its predecessor.
    """
    if not route_optimizes(route):
        return route
    kept = [
        token for token in _opt_option_tokens(route)
        if not token.split("=")[0].strip().lower().startswith(_SADDLE_OPTIONS)
        and not token.split("=")[0].strip().lower().startswith("noeigen")
    ]
    replacement = f"Opt=({','.join(kept)})" if kept else "Opt"
    return _replace_opt_keyword(route, replacement)


def _replace_opt_keyword(route: str, replacement: str) -> str:
    """Substitute the first ``Opt`` keyword and drop any later duplicates.

    A route with two ``Opt`` keywords has already had its options merged into
    ``replacement``; leaving the second one in place would let a stale bare
    ``Opt`` override the corrected search.
    """
    pieces: list[str] = []
    cursor = 0
    for index, match in enumerate(_OPT_RE.finditer(route)):
        pieces.append(route[cursor:match.start()])
        if index == 0:
            pieces.append(replacement)
        cursor = match.end()
    pieces.append(route[cursor:])
    return re.sub(r"[ \t]{2,}", " ", "".join(pieces)).strip()


def stage_route(section: str) -> str:
    """The route of one Gaussian input stage, continuation lines joined.

    Gaussian's route section starts at the first ``#`` line and ends at the
    first blank line, so a route wrapped over several lines is read whole.
    """
    collected: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not collected:
            if stripped.startswith("#"):
                collected.append(stripped)
            continue
        if not stripped:
            break
        collected.append(stripped)
    return " ".join(collected)


def input_stage_routes(text: str) -> list[str]:
    """One route per Link1 stage of a Gaussian input, in order."""
    return [stage_route(section) for section in _LINK1_SPLIT_RE.split(text)]


def log_routes(text: str) -> list[str]:
    """Every route Gaussian echoed into an output, in order."""
    from .gaussian import _routes

    return [route for _, route in _routes(text)]


def final_imaginary_count(text: str) -> int | None:
    """Imaginary modes in the *last* frequency analysis of an output.

    Counting over the whole file would add up every stage of a Link1 job, so a
    ladder that passed through a saddle at one multiplicity would look like a
    saddle here no matter what it finished on.

    Anchored on the ``Harmonic frequencies`` banner that precedes each real
    analysis. Where that banner is absent, the trailing run of ``Frequencies
    --`` lines is used instead: the mode table of one analysis is contiguous,
    so a gap far larger than its own line spacing marks the previous analysis.
    """
    lines = [
        (match.start(), match.end(), match.group(1))
        for match in _FREQ_LINE_RE.finditer(text)
    ]
    if not lines:
        return None
    headers = list(_HARMONIC_RE.finditer(text))
    if headers:
        start = headers[-1].end()
        block = [item for item in lines if item[0] >= start]
    else:
        block = [lines[-1]]
        for previous, current in zip(reversed(lines[:-1]), reversed(lines[1:])):
            if current[0] - previous[1] > 4000:
                break
            block.insert(0, previous)
    values: list[float] = []
    for _, _, group in block:
        for token in group.split():
            try:
                values.append(float(token.replace("D", "E").replace("d", "e")))
            except ValueError:
                pass
    if not values:
        return None
    return sum(value < -1.0 for value in values)


# Finding code -> (severity, results unusable / must relaunch, explanation).
FINDINGS: dict[str, tuple[str, bool, str]] = {
    "saddle_launched_as_minimum_search": (
        "error", True,
        "labeled a saddle point but launched with a plain Opt: the optimizer "
        "walked downhill to a minimum and the saddle was discarded",
    ),
    "minimum_launched_as_saddle_search": (
        "error", True,
        "labeled a verified minimum but launched as a TS/QST saddle search",
    ),
    "saddle_route_missing_noeigentest": (
        "error", True,
        "saddle search without NoEigenTest: Gaussian accepts or aborts the "
        "search on the guess Hessian's eigenvalues instead of optimizing",
    ),
    "saddle_converged_to_minimum": (
        "error", True,
        "finished output's last frequency analysis has no imaginary mode: the "
        "geometry is a minimum, not the labeled saddle",
    ),
    "input_route_disagrees_with_output": (
        "error", True,
        "the route Gaussian executed differs from the route in the current "
        "input; the output belongs to another protocol",
    ),
    "saddle_route_missing_hessian": (
        "warning", False,
        "saddle search without CalcFC/CalcAll/ReadFC: the estimated Hessian is "
        "rarely usable for a TS",
    ),
    "saddle_order_unverified": (
        "warning", False,
        "saddle search without Freq: the converged point's number of imaginary "
        "modes cannot be confirmed",
    ),
    "saddle_order_mismatch": (
        "warning", False,
        "finished output's imaginary-mode count differs from the labeled "
        "saddle order",
    ),
}

MUST_RELAUNCH = frozenset(code for code, (_, relaunch, _) in FINDINGS.items() if relaunch)


def inspect_job(config_type: str, input_text: str, output_text: str = "") -> dict[str, object]:
    """Classify one job's route intent and, where it finished, its result."""
    intent = intended_stationary_point(config_type)
    routes = [route for route in input_stage_routes(input_text) if route]
    optimizing = [route for route in routes if route_optimizes(route)]
    kinds = [route_search_kind(route) for route in optimizing]
    findings: set[str] = set()

    if intent == "saddle" and optimizing:
        # A saddle geometry run as a pure single point (no Opt at all) is a
        # deliberate label job, not a broken saddle search.
        if "saddle" not in kinds:
            findings.add("saddle_launched_as_minimum_search")
        else:
            gaps: set[str] = set()
            for route, kind in zip(optimizing, kinds):
                if kind == "saddle":
                    gaps.update(saddle_route_gaps(route))
            if "noeigentest" in gaps:
                findings.add("saddle_route_missing_noeigentest")
            if "hessian" in gaps:
                findings.add("saddle_route_missing_hessian")
            if "freq" in gaps:
                findings.add("saddle_order_unverified")
    elif intent == "minimum" and "saddle" in kinds:
        findings.add("minimum_launched_as_saddle_search")

    imaginary: int | None = None
    executed_kinds: list[str] = []
    if output_text:
        executed = [route for route in log_routes(output_text) if route_optimizes(route)]
        executed_kinds = [route_search_kind(route) for route in executed]
        if executed_kinds and kinds and set(executed_kinds) != set(kinds):
            findings.add("input_route_disagrees_with_output")
        if intent == "saddle" and executed and "saddle" not in executed_kinds:
            findings.add("saddle_launched_as_minimum_search")
        imaginary = final_imaginary_count(output_text)
        expected = expected_imaginary_modes(config_type)
        if intent == "saddle" and imaginary == 0:
            findings.add("saddle_converged_to_minimum")
        elif (intent == "saddle" and imaginary is not None
                and expected is not None and imaginary != expected):
            findings.add("saddle_order_mismatch")

    ordered = [code for code in FINDINGS if code in findings]
    return {
        "intent": intent,
        "search_kinds": ",".join(kinds),
        "executed_search_kinds": ",".join(executed_kinds),
        "optimizing_routes": optimizing,
        "final_imaginary_modes": imaginary,
        "findings": ordered,
        "severity": (
            "error" if any(FINDINGS[code][0] == "error" for code in ordered)
            else "warning" if ordered else "ok"
        ),
        "must_relaunch": any(code in MUST_RELAUNCH for code in ordered),
    }


_STEM_CONFIG_TYPES = {
    "transition-state": "transition_state",
    "first-order-saddle": "first_order_saddle",
    "higher-order-saddle": "higher_order_saddle",
    "minimum": "minimum",
    "optimized-unverified": "optimized_unverified",
    "irc-forward": "irc_forward",
    "irc-reverse": "irc_reverse",
    "irc-checkpoint": "irc_checkpoint",
    "irc-input-seed": "irc_input_seed",
    "warehouse-structure": "warehouse_structure",
    "checkpoint-geometry": "checkpoint_geometry",
    "labeled": "labeled",
}


def config_type_from_stem(stem: str) -> str:
    """Recover the config_type ``jobs.human_job_stem`` encoded in a filename.

    Both flat and spin campaigns name every input
    ``source__formula__configtype__qN-mM__variant__id...``, so the structural
    label survives even for a spin campaign, whose manifest had no
    config_type column before this change.
    """
    parts = Path(stem).stem.split("__")
    if len(parts) < 3:
        return ""
    slug = parts[2]
    resolved = _STEM_CONFIG_TYPES.get(slug, slug.replace("-", "_"))
    variant = parts[4] if len(parts) > 4 else ""
    if re.fullmatch(r"r\d+", variant):
        resolved += "_rattled"
    return resolved
