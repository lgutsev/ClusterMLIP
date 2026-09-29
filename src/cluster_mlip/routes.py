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

A frame taken from a reaction path is different again. An IRC point is a
fixed point along the path, not a stationary point of any kind, so *no*
search is right for it: ``Opt`` walks it off the path and ``Freq`` reports
curvature that means nothing at a non-stationary geometry. What an MLIP needs
from such a frame is its energy and forces at exactly the archived geometry,
which is a fixed-geometry ``Force`` job. A higher-order saddle candidate gets
the same treatment unless an ``Opt=(Saddle=N)`` search with a known N was
explicitly requested. Which of these applies is the job's *geometry role*
(``GEOMETRY_ROLES``), recorded next to -- not inferred from -- its
``config_type``: an IRC frame whose frequency analysis happens to show several
imaginary modes is still an IRC frame.

Everything here is read-only text analysis. ``relaunch.py`` consumes it to
rebuild the affected inputs.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypedDict

from .stratify import pes_region


# Opt options that select a saddle search rather than a minimization.
_SADDLE_OPTIONS = ("ts", "qst2", "qst3", "saddle")
# Options that supply a computed starting Hessian.
_HESSIAN_OPTIONS = {"calcfc", "calcall", "readfc", "rcfc", "readfchk", "readcartesianfc"}

_LINK1_SPLIT_RE = re.compile(r"^\s*--\s*link1\s*--\s*$", re.IGNORECASE | re.MULTILINE)
_HARMONIC_RE = re.compile(r"Harmonic frequencies\s*\(cm\*\*-1\)", re.IGNORECASE)
_FREQ_LINE_RE = re.compile(r"Frequencies\s+--\s+([^\n\r]+)", re.IGNORECASE)


def route_tokens(route: str) -> list[str]:
    """A route's top-level keywords, each with its options attached.

    Gaussian separates route keywords with spaces, tabs or commas, while a
    parenthesised option list keeps its own commas: ``Opt=(TS,CalcFC)`` and
    ``Opt(TS,CalcFC)`` are one keyword each, and ``#p B3LYP/6-31G*,Opt,Freq``
    is three. Options attach only through ``=`` or parentheses, never across a
    bare separator, so ``Opt Freq`` is two keywords rather than ``Opt=Freq``.
    Keyword identity is decided on these tokens, never by searching the raw
    text: ``Stable=Opt`` is a Stable keyword whose value happens to read "Opt",
    not a geometry optimization.
    """
    pieces: list[str] = []
    depth = 0
    current = ""
    for char in route:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        if depth == 0 and char in " \t\r\n,":
            if current:
                pieces.append(current)
            current = ""
            continue
        current += char
    if current:
        pieces.append(current)
    # "Opt = TS" and "Opt (TS)" still name one keyword.
    merged: list[str] = []
    for piece in pieces:
        if merged and (
            merged[-1].endswith("=") or piece.startswith("=")
            or (piece.startswith("(") and not re.search(r"[=(]", merged[-1]))
        ):
            merged[-1] += piece
        else:
            merged.append(piece)
    return merged


def keyword_name(token: str) -> str:
    return re.split(r"[=(]", token, maxsplit=1)[0].strip().lower()


def keyword_options(token: str) -> list[str]:
    """The options of one keyword token: ``Opt=(TS,CalcFC)`` -> ``[TS, CalcFC]``."""
    match = re.match(r"[^=(]*(?:=\s*)?(.*)$", token, re.DOTALL)
    raw = (match.group(1) if match else "").strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1]
    options: list[str] = []
    depth = 0
    current = ""
    for char in raw:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        if char == "," and depth == 0:
            if current.strip():
                options.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        options.append(current.strip())
    return options


def _keywords(route: str, *names: str) -> list[str]:
    return [token for token in route_tokens(route) if keyword_name(token) in names]


def route_keyword_options(route: str, name: str) -> list[str]:
    """Options of every ``name`` keyword in a route, lowercased."""
    return [option.lower() for token in _keywords(route, name) for option in keyword_options(token)]


def _opt_option_tokens(route: str) -> list[str]:
    """Each ``Opt`` option as written, across every ``Opt`` keyword in a route."""
    return [option for token in _keywords(route, "opt", "optimize")
            for option in keyword_options(token)]


def opt_options(route: str) -> set[str]:
    """Lowercased ``Opt`` option keywords, with any ``=value`` suffix dropped."""
    return {token.split("=")[0].strip().lower() for token in _opt_option_tokens(route)}


def route_optimizes(route: str) -> bool:
    return bool(_keywords(route, "opt", "optimize"))


def route_has_frequency(route: str) -> bool:
    return bool(_keywords(route, "freq", "frequency"))


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


# What a geometry *is*, independent of how it was labeled or which Gaussian
# job produced it. ``config_type`` mixes the two -- ``irc_forward`` names a
# calculation, ``higher_order_saddle`` a frequency count -- so it cannot say
# on its own which search, if any, is legitimate for the frame.
GEOMETRY_ROLES = (
    "stationary_minimum",
    "transition_state",
    "irc_point",
    "reaction_path_endpoint",
    "higher_order_candidate",
    "unconstrained_geometry",
)
PATH_ROLES = frozenset({"irc_point", "reaction_path_endpoint"})
# Roles whose stationarity was established by the calculation that produced
# them. A higher-order *candidate* is not among them: its curvature is exactly
# what nobody has confirmed.
STATIONARY_ROLES = frozenset({"stationary_minimum", "transition_state"})
IRC_CONFIG_TYPES = frozenset({
    "irc_forward", "irc_reverse", "irc_point", "irc_checkpoint", "irc_input_seed",
})


def geometry_role_for_config_type(config_type: str) -> str:
    """The role a bare ``config_type`` implies, for records that carry no role.

    ``irc_input_seed`` is the geometry an IRC calculation was *started* from,
    i.e. its transition state. Every other IRC type is a point on the path;
    whether it is the path's end is not recorded in the type, so it defaults
    to ``irc_point`` (both roles get the same fixed-geometry treatment).
    """
    if config_type.endswith("_rattled"):
        return "unconstrained_geometry"
    if config_type == "minimum":
        return "stationary_minimum"
    if config_type in ("transition_state", "first_order_saddle", "irc_input_seed"):
        return "transition_state"
    if config_type == "higher_order_saddle":
        return "higher_order_candidate"
    if config_type in IRC_CONFIG_TYPES:
        return "irc_point"
    return "unconstrained_geometry"


class FrameLabel(TypedDict):
    frame_role: str
    config_type: str
    geometry_role: str


_SADDLE_CONFIG_TYPES = ("transition_state", "first_order_saddle", "higher_order_saddle")


def frame_label(
    *,
    position: int,
    count: int,
    search_kind: str,
    converged: bool,
    job_config_type: str,
    job_geometry_role: str,
    stage_index: int = 0,
    initialization: str = "",
    same_state_as_label: bool = True,
) -> FrameLabel:
    """What one force frame of a job *is* -- not always what the job was labeled.

    A job's ``config_type`` describes the geometry it was started from. An
    optimization then moves the atoms, so only some of its frames are that
    geometry:

    * ``fixed_geometry`` -- a Force/single-point stage (``search_kind``
      ``none``): the frame is the labeled geometry.
    * ``input_geometry`` -- the first step of a stage that started from the
      job's own coordinates. It keeps the job's label; if the stage runs a
      different charge/multiplicity than the label was established at, a
      stationary role is dropped (a verified minimum at M=49 is not stationary
      at M=53).
    * ``spin_flip_start`` / ``restart_start`` -- the first step of a stage
      seeded from a checkpoint: the previous stage's geometry at a new
      multiplicity, or wherever an interrupted run stopped.
    * ``optimization_step`` -- every intermediate step, and the last one of a
      search that did not converge: ``optimization_path``, no stationarity.
    * ``optimized_endpoint`` -- the last step of a converged search. A
      converged saddle search is a saddle; a converged minimum search is a
      ``minimum`` only when it re-found a verified minimum in its own state,
      otherwise ``optimized_unverified`` (no frequency check was read), the
      warehouse's own term for that.

    Without this, every step of an IRC point run as ``Opt`` stratifies as an
    IRC frame, and every step of a minimum search counts as a stationary
    point in the physical checks.
    """
    def path(frame_role: str) -> FrameLabel:
        return {"frame_role": frame_role, "config_type": "optimization_path",
                "geometry_role": "unconstrained_geometry"}

    if search_kind == "none":
        return {"frame_role": "fixed_geometry", "config_type": job_config_type,
                "geometry_role": job_geometry_role}
    if position == count - 1 and converged:
        if search_kind == "saddle":
            higher = job_config_type == "higher_order_saddle"
            return {
                "frame_role": "optimized_endpoint",
                "config_type": (job_config_type if job_config_type in _SADDLE_CONFIG_TYPES
                                else "transition_state"),
                "geometry_role": "higher_order_candidate" if higher else "transition_state",
            }
        if job_config_type == "minimum" and same_state_as_label:
            return {"frame_role": "optimized_endpoint", "config_type": "minimum",
                    "geometry_role": "stationary_minimum"}
        return {"frame_role": "optimized_endpoint", "config_type": "optimized_unverified",
                "geometry_role": "unconstrained_geometry"}
    if position == 0:
        if initialization == "interrupted_checkpoint_restart":
            return path("restart_start")
        if stage_index > 0 or initialization == "checkpoint_spin_flip":
            return path("spin_flip_start")
        role = job_geometry_role
        if role in STATIONARY_ROLES and not same_state_as_label:
            role = "unconstrained_geometry"
        return {"frame_role": "input_geometry", "config_type": job_config_type,
                "geometry_role": role}
    return path("optimization_step")


def route_policy(
    config_type: str, geometry_role: str = "", source_calculation_type: str = ""
) -> str:
    """Which searches are legitimate for a job, as one of:

    * ``fixed_geometry`` -- a reaction-path frame. Only ``Force`` labels are
      valid; any ``Opt`` moves the atoms off the path point the label claims.
      Anything extracted from an IRC calculation counts, including the
      transition state the path started from: the record *is* a path frame.
    * ``higher_order`` -- a higher-order saddle candidate. ``Force`` by
      default; ``Opt=(Saddle=N)`` only with an explicitly requested N.
    * ``saddle`` / ``minimum`` -- a verified stationary point: its own search,
      or a single point.
    * ``unconstrained`` -- curvature never established (and rattled frames,
      which carry their own single-point route): nothing to enforce.
    """
    if config_type.endswith("_rattled"):
        return "unconstrained"
    role = geometry_role or geometry_role_for_config_type(config_type)
    if (role in PATH_ROLES or source_calculation_type == "irc"
            or config_type in IRC_CONFIG_TYPES):
        return "fixed_geometry"
    if role == "higher_order_candidate":
        return "higher_order"
    if role == "transition_state":
        return "saddle"
    if role == "stationary_minimum":
        return "minimum"
    return "unconstrained"


# A legacy archive often records that a geometry came from an IRC only in its
# file or folder name. That is weaker evidence than an IRC route or Gaussian's
# own "Point Number/Path Number" markers, so callers use it only as a fallback
# and say so. Matched as a token: "irc" inside "circle" is not evidence.
_IRC_NAME_RE = re.compile(r"(?:^|[^a-z])irc(?:[^a-z]|$|fwd|rev|forward|reverse|back)")
# A bare "for" only counts right after "irc" ("irc_for"): "ts_for_irc" is
# English, not a direction.
_FORWARD_NAME_RE = re.compile(r"forward|fwd|irc[_-]?for(?:[^a-z]|$)")
_REVERSE_NAME_RE = re.compile(r"reverse|backward|bwd|(?:^|[^a-z]|irc)(?:rev|back)(?:[^a-z]|$)")


def irc_name_evidence(*names: str) -> str:
    """The first path component naming an IRC origin, or ``""``."""
    for name in names:
        for part in re.split(r"[\\/]", name or ""):
            if _IRC_NAME_RE.search(part.lower()):
                return part
    return ""


def irc_direction_from_name(name: str) -> str:
    lowered = name.lower()
    if _FORWARD_NAME_RE.search(lowered):
        return "forward"
    if _REVERSE_NAME_RE.search(lowered):
        return "reverse"
    return ""


# Manifest columns that carry a job's geometry role and path provenance. They
# are optional: campaigns prepared before they existed resolve the role from
# config_type, the seed record, or -- reported as such -- the source filename.
GEOMETRY_COLUMNS = [
    "geometry_role", "geometry_role_source", "source_calculation_type", "route_policy",
    "requested_saddle_order", "irc_direction", "irc_point_index", "irc_path_position",
    "irc_parent_record_id", "original_charge", "original_multiplicity",
]


class GeometryRoleResolution(TypedDict):
    geometry_role: str
    geometry_role_source: str
    source_calculation_type: str
    evidence: str


def resolve_geometry_role(
    config_type: str,
    metadata: dict[str, Any] | None = None,
    *,
    source: str = "",
    route: str = "",
    running_route: str = "",
) -> GeometryRoleResolution:
    """The geometry role of one record or manifest row, and where it came from.

    ``route`` is the archived calculation's route; ``running_route`` is the
    first optimizing route of the job the campaign runs *now*, which is how a
    saddle search that an earlier relaunch already corrected is recognized.

    Evidence in order of strength:

    1. a role recorded with the record/row itself;
    2. an IRC route (``route`` is the archived calculation's route), or an IRC
       ``config_type`` -- both decisive;
    3. ``filename_fallback``: the source path names an IRC. Legacy extraction
       classified by frequency count first, so an IRC frame with several
       imaginary modes became ``higher_order_saddle``; its file name is often
       the only surviving evidence. It is never applied to a geometry its own
       calculation established as a minimum or a transition state, and the
       source is always reported so an operator can check it;
    4. the role ``config_type`` implies.
    """
    metadata = metadata or {}
    recorded = str(metadata.get("geometry_role") or "").strip()
    calculation = str(metadata.get("source_calculation_type") or "").strip()
    if recorded:
        return GeometryRoleResolution(
            geometry_role=recorded,
            geometry_role_source=str(metadata.get("geometry_role_source") or "recorded"),
            source_calculation_type=calculation,
            evidence=str(metadata.get("geometry_role_evidence") or ""),
        )
    if config_type.endswith("_rattled"):
        return GeometryRoleResolution(
            geometry_role="unconstrained_geometry", geometry_role_source="config_type",
            source_calculation_type=calculation, evidence=config_type,
        )
    if route and route_has_irc(route):
        return GeometryRoleResolution(
            geometry_role=(
                "transition_state" if config_type == "irc_input_seed" else "irc_point"
            ),
            geometry_role_source="irc_route", source_calculation_type="irc", evidence=route,
        )
    if config_type in IRC_CONFIG_TYPES:
        return GeometryRoleResolution(
            geometry_role=geometry_role_for_config_type(config_type),
            geometry_role_source="config_type", source_calculation_type="irc",
            evidence=config_type,
        )
    name = irc_name_evidence(source)
    # A legacy ``transition_state`` came from an explicit TS route or name and
    # a ``minimum`` from an optimization that found no imaginary mode; a file
    # name must not overturn either (nor re-flag a TS search an earlier
    # relaunch already corrected). The labels a name may correct are the ones
    # the old extraction derived from a frequency count, or never set at all.
    established = (
        config_type == "transition_state"
        or (config_type == "minimum" and (not route or route_optimizes(route)))
        or (config_type == "first_order_saddle" and route_search_kind(route) == "saddle")
    )
    # The job already runs a saddle search for its label (the TS relaunch put
    # it there, or an explicitly requested Saddle=N): a name hint alone does
    # not undo finished or running work. Reported, so the decision is visible.
    running_search = (
        config_type in ("first_order_saddle", "higher_order_saddle")
        and route_search_kind(running_route) == "saddle"
    )
    if name and not established and running_search:
        return GeometryRoleResolution(
            geometry_role=geometry_role_for_config_type(config_type),
            geometry_role_source="running_saddle_search", source_calculation_type="",
            evidence=f"source name suggests an IRC ({name}) but the job runs {running_route}",
        )
    if name and not established:
        return GeometryRoleResolution(
            geometry_role="irc_point", geometry_role_source="filename_fallback",
            source_calculation_type="irc", evidence=name,
        )
    return GeometryRoleResolution(
        geometry_role=geometry_role_for_config_type(config_type),
        geometry_role_source="config_type", source_calculation_type=calculation,
        evidence=config_type,
    )


def saddle_order_of_route(route: str) -> int | None:
    """The saddle order a route searches for: 1 for TS/QST, N for Saddle=N."""
    for token in _opt_option_tokens(route):
        name, _, value = token.partition("=")
        name = name.strip().lower()
        if name == "saddle":
            try:
                return int(value.strip())
            except ValueError:
                return None
        if name in ("ts", "qst2", "qst3"):
            return 1
    return None


def recorded_saddle_order(row: dict[str, str]) -> int | None:
    """The saddle order a campaign row was *explicitly* asked to search for.

    ``requested_saddle_order`` is written by prepare/relaunch when an operator
    asks for a higher-order search. A row relaunched by an earlier
    relaunch-routes, before that column existed, recorded the request only in
    ``relaunch_route_after``; a ``Saddle=N`` there was also explicit (that
    tool refused a higher-order saddle unless given --saddle-order or
    --saddle-order-from), so it counts.
    """
    raw = (row.get("requested_saddle_order") or "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            return None
    if (row.get("relaunch_attempt") or "").strip():
        order = saddle_order_of_route(row.get("relaunch_route_after") or "")
        if order is not None and order >= 2:
            return order
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


# Gaussian's Berny optimizer works in redundant internal coordinates by
# default. FormBX builds the Wilson B matrix for that transformation, and a
# torsion whose three defining atoms are near-collinear has no well-defined
# value, which makes the matrix singular. Both markers report the same
# breakdown; neither is a route mistake, and neither is affected by the choice
# of GEDIIS/GDIIS step, because the failure is in forming the coordinates
# rather than in taking a step.
_INTERNAL_COORDINATE_FAILURE_RE = re.compile(
    r"FormBX had a problem|Tors failed for dihedral|"
    r"Error in internal coordinate system|Linear angle in (?:Tors|Bend)",
    re.IGNORECASE,
)


# Gaussian names the offending atoms: "Tors failed for dihedral 1 - 2 - 3 - 4".
# Those centre numbers are what a rebuild in a builder needs, so carry them
# through rather than making someone reopen every log to find them.
_FAILED_TORSION_RE = re.compile(
    r"Tors failed for dihedral\s+(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)",
    re.IGNORECASE,
)


def internal_coordinate_failure(text: str) -> str:
    """The internal-coordinate breakdown marker in an output, or ``""``."""
    match = _INTERNAL_COORDINATE_FAILURE_RE.search(text)
    return match.group(0) if match else ""


def failed_torsion_atoms(text: str) -> str:
    """The atom centres of the degenerate dihedral, e.g. ``1-2-3-4``.

    Empty when the output reports the breakdown without naming a torsion.
    """
    match = _FAILED_TORSION_RE.search(text)
    return "-".join(match.groups()) if match else ""


def route_uses_cartesian(route: str) -> bool:
    return "cartesian" in opt_options(route)


def corrected_cartesian_route(route: str) -> str:
    """Move an optimizing route off internal coordinates.

    ``Opt=Cartesian`` optimizes in Cartesians, bypassing the B-matrix
    transformation that failed. It typically needs more steps than the
    redundant-internal default, but it cannot hit a degenerate torsion, and it
    keeps the input's existing geometry rather than requiring the structure to
    be rebuilt as a Z-matrix with dummy atoms to dodge the same collinearity.
    """
    if not route_optimizes(route) or route_uses_cartesian(route):
        return route
    tokens = _opt_option_tokens(route)
    merged = list(dict.fromkeys(tokens + ["Cartesian"]))
    return _replace_opt_keyword(route, f"Opt=({','.join(merged)})")


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


def route_guess_always(route: str) -> bool:
    return "always" in route_keyword_options(route, "guess")


def route_has_irc(route: str) -> bool:
    return bool(_keywords(route, "irc"))


def forbidden_fixed_geometry_keywords(route: str) -> list[str]:
    """Everything a fixed-geometry Force stage must not carry.

    ``Opt`` moves the atoms. ``Freq`` reports curvature that is meaningless
    away from a stationary point. ``Stable=Opt`` may re-optimize the
    wavefunction onto a different electronic state than the ladder is
    tracking, and ``Guess=Always`` throws away the checkpoint orbitals a spin
    flip is supposed to start from. ``IRC`` would walk a new path.
    """
    found: list[str] = []
    if route_optimizes(route):
        found.append("Opt")
    if route_has_frequency(route):
        found.append("Freq")
    if _keywords(route, "stable"):
        found.append("Stable")
    if route_guess_always(route):
        found.append("Guess=Always")
    if route_has_irc(route):
        found.append("IRC")
    return found


def route_is_force_only(route: str) -> bool:
    """A Force job at a fixed geometry, with nothing that could move or reinterpret it."""
    return bool(_keywords(route, "force")) and not forbidden_fixed_geometry_keywords(route)


def _guess_token(options: list[str]) -> str:
    if not options:
        return ""
    if len(options) == 1 and "=" not in options[0]:
        return f"Guess={options[0]}"
    return f"Guess=({','.join(options)})"


def force_only_route(route: str) -> str:
    """Rewrite one stage's route into the fixed-geometry ``Force`` protocol.

    The ``Opt`` keyword is replaced by ``Force`` *where it stood*, so the
    method, SCF settings, IOps, grid and any population request survive
    untouched and a repaired stage differs from its predecessor only in what
    it asks Gaussian to do with the geometry. ``Freq``, ``Stable``, ``IRC``
    and ``Guess=Always`` are removed; ``SP`` gives way to ``Force``.

    A stage that reads its geometry from the checkpoint keeps
    ``Geom=Checkpoint`` and must also read the orbitals, so ``Guess=Read`` is
    added when it is missing: a spin flip starts from its predecessor's
    wavefunction, and the geometry is the same fixed geometry throughout.
    """
    tokens = route_tokens(route)
    placed = any(keyword_name(token) == "force" for token in tokens)
    rebuilt: list[str] = []
    for token in tokens:
        name = keyword_name(token)
        if name in ("stable", "freq", "frequency", "irc"):
            continue
        if name in ("opt", "optimize", "sp"):
            if not placed:
                rebuilt.append("Force")
                placed = True
            continue
        if name == "guess":
            token = _guess_token([
                option for option in keyword_options(token) if option.lower() != "always"
            ])
            if not token:
                continue
        rebuilt.append(token)
    if not placed:
        rebuilt.append("Force")
    result = " ".join(rebuilt)
    if route_reads_checkpoint_geometry(result) and "read" not in route_keyword_options(
            result, "guess"):
        guesses = [index for index, token in enumerate(rebuilt) if keyword_name(token) == "guess"]
        if guesses:
            rebuilt[guesses[0]] = _guess_token(["Read", *keyword_options(rebuilt[guesses[0]])])
        else:
            rebuilt.append("Guess=Read")
        result = " ".join(rebuilt)
    return result


def stage_is_force_only(input_text: str, stage_index: int) -> bool:
    """Whether Link1 stage ``stage_index`` of an input is a fixed-geometry Force job."""
    routes = input_stage_routes(input_text)
    return 0 <= stage_index < len(routes) and route_is_force_only(routes[stage_index])


def _replace_opt_keyword(route: str, replacement: str) -> str:
    """Substitute the first ``Opt`` keyword and drop any later duplicates.

    A route with two ``Opt`` keywords has already had its options merged into
    ``replacement``; leaving the second one in place would let a stale bare
    ``Opt`` override the corrected search.
    """
    rebuilt: list[str] = []
    replaced = False
    for token in route_tokens(route):
        if keyword_name(token) in ("opt", "optimize"):
            if not replaced and replacement:
                rebuilt.append(replacement)
            replaced = True
            continue
        rebuilt.append(token)
    return " ".join(rebuilt)


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


def route_reads_checkpoint_geometry(route: str) -> bool:
    """Whether this stage takes its geometry from a checkpoint file.

    ``Guess=Read`` alone only reuses orbitals; it is ``Geom=Checkpoint`` (or
    ``AllCheck``/``Check``) that replaces the molecular specification.
    """
    return any("check" in option for option in route_keyword_options(route, "geom"))


def geometry_source(text: str) -> str:
    """Where a multi-stage input's *first* stage gets its geometry.

    ``checkpoint`` means the input carries no coordinates of its own: a
    spin-ladder restart produced by ``prepare-spin-restarts`` begins with
    ``%oldchk`` plus ``Geom=Checkpoint``. Such an input cannot be corrected in
    place -- there is no geometry in it to correct, and the checkpoint it would
    read was written by the run being discarded -- so a route relaunch has to
    go back to the root input that still holds real coordinates.
    """
    sections = _LINK1_SPLIT_RE.split(text)
    if not sections:
        return "unknown"
    first = sections[0]
    if route_reads_checkpoint_geometry(stage_route(first)):
        return "checkpoint"
    return "input_coordinates" if _has_coordinates(first) else "unknown"


# Intra-line whitespace only: "\s" would match newlines and let a Gen basis
# shell header plus its first exponent row ("S 2 1.00\n 10.47 -0.229") pass as
# a coordinate line, so an input carrying only a basis block would look like it
# still had a geometry.
_COORDINATE_RE = re.compile(
    r"^[ \t]*[A-Z][a-z]?(?:\(Fragment=\d+\))?(?:[ \t]+-?\d+)?"
    r"(?:[ \t]+[+-]?(?:\d+(?:\.\d*)?|\.\d+)){3}[ \t]*$",
    re.MULTILINE,
)


# The charge/multiplicity line: one or more "charge multiplicity" integer pairs
# (a fragment guess supplies several). It is the anchor for the molecular
# specification that follows, which is the only place a geometry can live --
# looking anywhere else in the stage means a Gen basis block's element-led
# shell lines get mistaken for atoms.
_CHARGE_MULT_LINE_RE = re.compile(r"[ \t]*-?\d+[ \t]+\d+(?:[ \t]+-?\d+[ \t]+\d+)*[ \t]*")


def _molecular_specification(section: str) -> list[str]:
    """The atom lines between a stage's charge/multiplicity line and the next blank."""
    lines = section.splitlines()
    for index, line in enumerate(lines):
        if not _CHARGE_MULT_LINE_RE.fullmatch(line):
            continue
        block: list[str] = []
        for following in lines[index + 1:]:
            if not following.strip():
                break
            block.append(following)
        return block
    return []


def _has_coordinates(section: str) -> bool:
    """Whether a stage carries its own molecular specification, in any form."""
    return bool(_molecular_specification(section))


def checkpoint_stages_with_coordinates(text: str) -> list[int]:
    """Stages that read ``Geom=Checkpoint`` yet still carry atom lines.

    Under ``Geom=Checkpoint`` Gaussian takes the structure from the checkpoint
    and expects only the charge/multiplicity line before the terminating blank
    line, so leftover atom lines are read as the *next* input section. No
    generated input may contain such a stage.
    """
    return [
        index for index, section in enumerate(_LINK1_SPLIT_RE.split(text))
        if route_reads_checkpoint_geometry(stage_route(section)) and _has_coordinates(section)
    ]


def drop_molecule_coordinates(text: str) -> str:
    """Keep only the charge/multiplicity line of a stage's molecule specification.

    Everything after the terminating blank line -- a Gen basis block, say --
    is kept, and so is everything before the charge/multiplicity line.
    """
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if not _CHARGE_MULT_LINE_RE.fullmatch(line.rstrip("\r")):
            continue
        end = index + 1
        while end < len(lines) and lines[end].strip():
            end += 1
        return "\n".join(lines[:index + 1] + lines[end:])
    raise ValueError("stage has no charge/multiplicity line")


def input_coordinates(text: str) -> list[tuple[str, float, float, float]] | None:
    """The first stage's Cartesian atoms, or ``None`` when it has none.

    ``None`` covers a checkpoint-seeded stage (no molecular specification)
    and a Z-matrix (no Cartesians to compare against without rebuilding it).
    Fragment tags and freeze flags are dropped; dummy atoms are skipped, as
    Gaussian's orientation tables skip them.
    """
    sections = _LINK1_SPLIT_RE.split(text) if text else []
    if (not sections or route_reads_checkpoint_geometry(stage_route(sections[0]))
            or input_is_zmatrix(text)):
        return None
    block = _molecular_specification(sections[0])
    atoms: list[tuple[str, float, float, float]] = []
    for line in block:
        parts = line.split()
        if len(parts) < 4:
            return None
        symbol = parts[0].split("(")[0]
        try:
            x, y, z = (float(value.replace("D", "E")) for value in parts[-3:])
        except ValueError:
            return None
        if symbol.upper() in ("X", "BQ"):
            continue
        atoms.append((symbol, x, y, z))
    return atoms or None


def running_search_route(text: str) -> str:
    """The first optimizing stage route of an input, or ``""`` for a Force job."""
    return next((route for route in input_stage_routes(text) if route_optimizes(route)), "")


def input_atom_symbols(text: str) -> list[str] | None:
    """Element symbols of the first stage's atoms, Cartesian or Z-matrix.

    The fallback for checking a fixed-geometry label whose coordinates cannot
    be compared directly: the atom count and order must still agree. Dummy
    and ghost centres are skipped, as Gaussian's orientation tables skip them.
    """
    sections = _LINK1_SPLIT_RE.split(text) if text else []
    if not sections or route_reads_checkpoint_geometry(stage_route(sections[0])):
        return None
    symbols: list[str] = []
    for line in _molecular_specification(sections[0]):
        token = line.split()[0].split("(")[0]
        symbol = re.match(r"[A-Za-z]{1,2}", token)
        if symbol is None:
            return None
        name = symbol.group(0)
        name = name[0].upper() + name[1:].lower()
        if name in ("X", "Bq") or token.startswith("-"):
            continue
        symbols.append(name)
    return symbols or None


def input_is_zmatrix(text: str) -> bool:
    """Whether the first stage's geometry is a Z-matrix rather than Cartesians.

    Decided on the first atom line, which is the one unambiguous difference: a
    Z-matrix opens with a bare element symbol, because the first atom has
    nothing to reference, whereas a Cartesian line always carries three
    coordinates. Judging individual later lines cannot work -- a Z-matrix line
    such as ``X 1 1.0 2 90.0`` has exactly the shape of a Cartesian one.

    This matters because a Z-matrix built with dummy atoms is itself a
    deliberate fix for a degenerate internal coordinate, so such a job must
    keep optimizing in internals: adding ``Opt=Cartesian`` would discard the
    very construction that makes its torsions well defined.
    """
    block = _molecular_specification(_LINK1_SPLIT_RE.split(text)[0] if text else "")
    return bool(block) and len(block[0].split()) == 1


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
    "path_point_launched_as_optimization": (
        "error", True,
        "IRC/reaction-path frame launched with Opt: an IRC point is a fixed point "
        "along the path, not a stationary point, so optimizing it moves the atoms "
        "away from the archived geometry. Path frames need fixed-geometry Force labels",
    ),
    "higher_order_launched_as_minimum_search": (
        "error", True,
        "higher-order saddle candidate launched with an ordinary Opt: the optimizer "
        "walks downhill and the archived geometry is lost. Label it with a "
        "fixed-geometry Force job, or request Opt=(Saddle=N) explicitly with a known N",
    ),
    "higher_order_saddle_search_unrequested": (
        "error", True,
        "higher-order saddle candidate launched as a saddle search whose order was "
        "never explicitly requested and recorded (requested_saddle_order), or whose "
        "Saddle=N disagrees with it; Opt=TS is a first-order search",
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
    "internal_coordinate_failure": (
        "error", True,
        "Gaussian could not build internal coordinates (FormBX/Tors failed), so "
        "the optimizer never took a step: a near-linear angle leaves a torsion "
        "undefined. Relaunching with Opt=Cartesian avoids that transformation",
    ),
    "internal_coordinate_failure_in_cartesian": (
        "error", False,
        "the coordinate system failed even though the route already optimizes in "
        "Cartesians; this one needs the geometry looked at, not a route change",
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
    "path_point_route_has_extra_keywords": (
        "warning", False,
        "fixed-geometry path frame whose route also carries Freq, Stable, IRC or "
        "Guess=Always; the atoms did not move, so the energy/force label is usable, "
        "but the route is not the force-only protocol",
    ),
}

MUST_RELAUNCH = frozenset(code for code, (_, relaunch, _) in FINDINGS.items() if relaunch)


class RouteVerdict(TypedDict):
    """One job's route intent and, where it finished, its result.

    Typed rather than ``dict[str, object]`` so callers can join
    ``findings`` and test ``must_relaunch`` without mypy losing the
    element types -- these fields drive the collect route gate and the
    campaign audit, so a silent type error here is a silent data-quality
    error.
    """

    intent: str
    search_kinds: str
    executed_search_kinds: str
    optimizing_routes: list[str]
    final_imaginary_modes: int | None
    failed_torsion_atoms: str
    findings: list[str]
    severity: str
    must_relaunch: bool


def _saddle_gap_findings(optimizing: list[str], kinds: list[str]) -> set[str]:
    gaps: set[str] = set()
    for route, kind in zip(optimizing, kinds):
        if kind == "saddle":
            gaps.update(saddle_route_gaps(route))
    codes = {
        "noeigentest": "saddle_route_missing_noeigentest",
        "hessian": "saddle_route_missing_hessian",
        "freq": "saddle_order_unverified",
    }
    return {codes[gap] for gap in gaps}


def inspect_job(
    config_type: str,
    input_text: str,
    output_text: str = "",
    *,
    geometry_role: str = "",
    source_calculation_type: str = "",
    requested_saddle_order: int | None = None,
) -> RouteVerdict:
    """Classify one job's route intent and, where it finished, its result.

    ``intent`` in the verdict is the job's ``route_policy``; for the saddle and
    minimum labels that is the same value ``intended_stationary_point`` gives.
    """
    intent = route_policy(config_type, geometry_role, source_calculation_type)
    routes = [route for route in input_stage_routes(input_text) if route]
    optimizing = [route for route in routes if route_optimizes(route)]
    kinds = [route_search_kind(route) for route in optimizing]
    findings: set[str] = set()

    if intent == "fixed_geometry":
        if optimizing:
            findings.add("path_point_launched_as_optimization")
        if any(
            keyword != "Opt"
            for route in routes for keyword in forbidden_fixed_geometry_keywords(route)
        ):
            findings.add("path_point_route_has_extra_keywords")
    elif intent == "higher_order" and optimizing:
        if "saddle" not in kinds:
            findings.add("higher_order_launched_as_minimum_search")
        else:
            orders = {
                saddle_order_of_route(route)
                for route, kind in zip(optimizing, kinds) if kind == "saddle"
            }
            if requested_saddle_order is None or orders != {requested_saddle_order}:
                findings.add("higher_order_saddle_search_unrequested")
            else:
                findings |= _saddle_gap_findings(optimizing, kinds)
    elif intent == "saddle" and optimizing:
        # A saddle geometry run as a pure single point (no Opt at all) is a
        # deliberate label job, not a broken saddle search.
        if "saddle" not in kinds:
            findings.add("saddle_launched_as_minimum_search")
        else:
            findings |= _saddle_gap_findings(optimizing, kinds)
    elif intent == "minimum" and "saddle" in kinds:
        findings.add("minimum_launched_as_saddle_search")

    imaginary: int | None = None
    executed_kinds: list[str] = []
    failed_torsion = ""
    if output_text:
        executed = [route for route in log_routes(output_text) if route_optimizes(route)]
        executed_kinds = [route_search_kind(route) for route in executed]
        if executed_kinds and kinds and set(executed_kinds) != set(kinds):
            findings.add("input_route_disagrees_with_output")
        if intent == "saddle" and executed and "saddle" not in executed_kinds:
            findings.add("saddle_launched_as_minimum_search")
        # What ran is what the label describes: a force-only input whose log
        # shows an optimization belongs to some other attempt, and its frame
        # is not at the archived path geometry.
        if intent == "fixed_geometry" and executed:
            findings.add("path_point_launched_as_optimization")
        if intent == "higher_order" and executed and "saddle" not in executed_kinds:
            findings.add("higher_order_launched_as_minimum_search")
        # Independent of intent: a minimization can break the coordinate
        # system just as a saddle search can.
        if internal_coordinate_failure(output_text) and optimizing:
            findings.add(
                "internal_coordinate_failure_in_cartesian"
                if all(route_uses_cartesian(route) for route in optimizing)
                else "internal_coordinate_failure"
            )
        failed_torsion = failed_torsion_atoms(output_text)
        imaginary = final_imaginary_count(output_text)
        expected = expected_imaginary_modes(config_type)
        searched_order = intent == "higher_order" and requested_saddle_order is not None and bool(
            optimizing
        )
        if searched_order:
            expected = requested_saddle_order
        if (intent == "saddle" or searched_order) and imaginary == 0:
            findings.add("saddle_converged_to_minimum")
        elif ((intent == "saddle" or searched_order) and imaginary is not None
                and expected is not None and imaginary != expected):
            findings.add("saddle_order_mismatch")

    ordered = [code for code in FINDINGS if code in findings]
    return RouteVerdict(
        intent=intent,
        search_kinds=",".join(kinds),
        executed_search_kinds=",".join(executed_kinds),
        optimizing_routes=optimizing,
        final_imaginary_modes=imaginary,
        failed_torsion_atoms=failed_torsion,
        findings=ordered,
        severity=(
            "error" if any(FINDINGS[code][0] == "error" for code in ordered)
            else "warning" if ordered else "ok"
        ),
        must_relaunch=any(code in MUST_RELAUNCH for code in ordered),
    )


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
    "irc-point": "irc_point",
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
