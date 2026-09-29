from __future__ import annotations

import hashlib
import copy
import math
import re
from pathlib import Path

from .models import Atom, LabeledFrame, Record, geometry_signature
from .routes import (
    geometry_role_for_config_type,
    irc_direction_from_name,
    irc_name_evidence,
    route_has_frequency,
    route_has_irc,
    route_keyword_options,
    route_optimizes,
    route_search_kind,
)


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
FORCE_AU_TO_EV_ANG = HARTREE_TO_EV / BOHR_TO_ANG

ATOMIC_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
    44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La",
    58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd",
    65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu",
    72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt",
    79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At",
    86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa", 92: "U",
}

_CM_RE = re.compile(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", re.I)
_HF_RE = re.compile(r"\bHF\s*=\s*(-?\d+(?:\.\d+)?(?:[DEde][+-]?\d+)?)")
_SCF_RE = re.compile(r"SCF Done:\s+E\([^)]*\)\s*=\s*(-?\d+(?:\.\d+)?(?:[DEde][+-]?\d+)?)", re.I)
_FREQ_RE = re.compile(r"Frequencies\s+--\s+([^\n\r]+)", re.I)
_IRC_POINT_RE = re.compile(r"Point\s+Number:\s*(-?\d+)\s+Path\s+Number:\s*(\d+)", re.I)
_STATE_RE = re.compile(r"State\s*=\s*([^\\\s]+)", re.I)


_S2_RE = re.compile(
    r"S\*\*2\s+before\s+annihilation\s+([+-]?[\d.]+).*?after\s+([+-]?[\d.]+)",
    re.I | re.S,
)
_MULLIKEN_SPIN_RE = re.compile(
    r"Mulliken\s+charges\s+and\s+spin\s+densities:(.*?)(?:Sum\s+of\s+Mulliken|\n\s*\n)",
    re.I | re.S,
)
_MULLIKEN_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([A-Z][a-z]?)\s+[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[EeDd][+-]?\d+)?\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)\s*$",
    re.M,
)


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _orientation_tables(text: str) -> list[tuple[int, int, list[Atom], str]]:
    starts = list(re.finditer(r"(?:Standard|Input|Z-Matrix) orientation:\s*", text, re.I))
    tables: list[tuple[int, int, list[Atom], str]] = []
    for start in starts:
        cursor = start.end()
        # Gaussian orientation tables have two header separators before atoms.
        separators = list(re.finditer(r"^\s*-{10,}\s*$", text[cursor:], re.M))
        if len(separators) < 2:
            continue
        atom_start = cursor + separators[1].end()
        tail = text[atom_start:]
        atom_end_match = re.search(r"^\s*-{10,}\s*$", tail, re.M)
        if atom_end_match is None:
            continue
        atom_end = atom_start + atom_end_match.start()
        atoms: list[Atom] = []
        for line in text[atom_start:atom_end].splitlines():
            parts = line.split()
            if len(parts) < 6 or not parts[0].lstrip("+-").isdigit():
                continue
            try:
                atomic_number = int(parts[1])
                x, y, z = map(_float, parts[-3:])
            except (ValueError, KeyError):
                continue
            if atomic_number == 0:
                continue  # Gaussian dummy atom
            symbol = ATOMIC_SYMBOLS.get(atomic_number)
            if symbol is None:
                continue
            atoms.append(Atom(symbol, x, y, z))
        if atoms:
            tables.append((start.start(), atom_end, atoms, start.group(0).split()[0].lower()))
    return tables


def _routes(text: str) -> list[tuple[int, str]]:
    results: list[tuple[int, str]] = []
    for match in re.finditer(r"^\s*#([^\n\r]*(?:[\n\r]+\s+[^\n\r#-][^\n\r]*){0,6})", text, re.M):
        route = " ".join(part.strip() for part in match.group(0).splitlines())
        route = re.split(r"\s*-{10,}", route)[0].strip()
        if route.lstrip().startswith("##") or _IRC_SUMMARY_LINE_RE.match(route):
            continue
        results.append((match.start(), route))
    return results


def _last_before(items, position: int):
    candidates = [item for item in items if item[0] < position]
    return candidates[-1] if candidates else None


def _imaginary_count(text: str, start: int, end: int) -> int | None:
    values: list[float] = []
    for match in _FREQ_RE.finditer(text, start, end):
        for token in match.group(1).split():
            try:
                values.append(_float(token))
            except ValueError:
                pass
    return sum(value < -1.0 for value in values) if values else None


_IRC_PATH_DONE_RE = re.compile(r"Calculation of (FORWARD|REVERSE) path complete", re.I)
_IRC_ALL_DONE_RE = re.compile(r"Reaction path calculation complete", re.I)
_IRC_PATH_BOUNDARY_RE = re.compile(
    r"Calculation of (?:FORWARD|REVERSE) path complete|"
    r"Beginning calculation of the (?:FORWARD|REVERSE) path|"
    r"Reaction path calculation complete",
    re.I,
)
_IRC_REVERSE_START_RE = re.compile(r"Beginning calculation of the REVERSE path", re.I)
# Gaussian 09/16 print the Point/Path summary *after* a point has converged,
# followed by these lines; a log carrying them numbers each geometry by the
# marker that closes it. Without them (a condensed or hand-edited log) the
# marker is taken to introduce the geometry that follows it.
_IRC_MARKER_AFTER_RE = re.compile(
    r"Calculating another point on the path|#\s*OF\s+POINTS\s+ALONG\s+THE\s+PATH|"
    r"Optimized point\s*#",
    re.I,
)
_IRC_SUMMARY_LINE_RE = re.compile(r"#\s*OF\s+(?:POINTS|STEPS)\b")


def _calculation_type(route: str) -> str:
    """What kind of Gaussian job produced a frame, from its route."""
    if not route:
        return ""
    if route_has_irc(route):
        return "irc"
    if route_optimizes(route):
        return "optimization"
    if route_has_frequency(route):
        return "frequency"
    return "single_point"


def _classify(source: str, route: str, imag: int | None, irc: tuple[int, int] | None) -> str:
    return _classify_with_evidence(source, route, imag, irc)[0]


def _classify_with_evidence(
    source: str, route: str, imag: int | None, irc: tuple[int, int] | None
) -> tuple[str, str, str]:
    """``(config_type, geometry_role_source, evidence)`` for one frame.

    Evidence is ranked. An IRC route or Gaussian's own Point/Path markers are
    decisive, and they come before the frequency count on purpose: a
    frequency analysis at an IRC point is taken away from a stationary point,
    so "several imaginary modes" there says nothing about a higher-order
    saddle. An explicit TS route comes next. A file or folder name containing
    ``irc`` is only a *fallback* -- it is reported as such, and it never
    overrides a calculation that itself established a minimum.
    """
    if irc is not None:
        return ("irc_forward" if irc[1] == 1 else "irc_reverse",
                "irc_point_marker", f"Point Number {irc[0]} Path Number {irc[1]}")
    filename = Path(source).stem.lower()
    if route_search_kind(route) == "saddle":
        return "transition_state", "route", route
    # A TS name is explicit evidence too, and resolve_geometry_role never lets
    # an "irc" name overturn a transition_state label -- the two must agree.
    if re.search(r"(?:^|[_\-.])ts(?:[_\-.]|$)", filename):
        return "transition_state", "filename", Path(source).name
    name = irc_name_evidence(source)
    optimized_minimum = route_optimizes(route) and imag == 0
    if name and not optimized_minimum:
        direction = irc_direction_from_name(source)
        config_type = {"forward": "irc_forward", "reverse": "irc_reverse"}.get(
            direction, "irc_point"
        )
        return config_type, "filename_fallback", name
    if imag == 1:
        return "first_order_saddle", "frequency_analysis", "1 imaginary mode"
    if imag is not None and imag > 1:
        return "higher_order_saddle", "frequency_analysis", f"{imag} imaginary modes"
    if imag == 0:
        return "minimum", "frequency_analysis", "0 imaginary modes"
    if route_optimizes(route):
        return "optimized_unverified", "route", route
    return "unknown", "", ""


def _irc_directions(route: str) -> set[str]:
    """Directions an IRC route asked for; both when it does not restrict them."""
    requested: set[str] = {
        name for name in ("forward", "reverse") if name in route_keyword_options(route, "irc")
    }
    return requested or {"forward", "reverse"}


def _irc_frame(
    position: int,
    calc_start: int,
    markers: list[tuple[int, int, int]],
    frame_positions: list[int],
    boundaries: list[int],
    reverse_starts: list[int],
    marker_after: bool,
) -> tuple[tuple[int, int] | None, str]:
    """``((point, path) or None, frame kind)`` for one energy of an IRC calculation.

    ``markers`` and ``frame_positions`` belong to this calculation only. The
    first frame is the geometry the path starts from (kind ``ts``). With
    Gaussian's own layout (``marker_after``) a geometry belongs to the point
    whose marker *follows* it: the frame right before a marker is that point's
    ``converged_point`` and earlier frames are its ``optimization_step``s; a
    frame after the last marker of a path is an ``unconverged_step`` of the
    next point, which the job never finished. In the marker-first layout the
    nearest preceding marker numbers the frame (``path_point``).
    """
    first = bool(frame_positions) and position == frame_positions[0]

    def crosses(start: int, end: int) -> bool:
        return any(start < boundary < end for boundary in boundaries)

    if not marker_after:
        before = [marker for marker in markers if marker[0] < position]
        if before and position - before[-1][0] < 100_000:
            return (before[-1][1], before[-1][2]), "path_point"
        return None, "ts" if first else "unnumbered_step"
    if first:
        return None, "ts"
    closing = next((marker for marker in markers if marker[0] > position), None)
    if closing is not None and not crosses(position, closing[0]):
        later = any(position < other < closing[0] for other in frame_positions)
        return (closing[1], closing[2]), "optimization_step" if later else "converged_point"
    previous = next((marker for marker in reversed(markers) if marker[0] < position), None)
    if previous is not None and not crosses(previous[0], position):
        return (previous[1] + 1, previous[2]), "unconverged_step"
    path = 2 if any(calc_start < start < position for start in reverse_starts) else 1
    return (1, path), "unconverged_step"


def _irc_direction(route: str, path_number: int | None) -> str:
    """Gaussian numbers the forward path 1 and the reverse path 2.

    A one-directional IRC has a single path, which is the requested one
    whatever number Gaussian printed for it.
    """
    requested = _irc_directions(route)
    if len(requested) == 1:
        return next(iter(requested))
    if path_number is None:
        return ""
    return "forward" if path_number == 1 else "reverse"


def _irc_metadata(
    text: str,
    route: str,
    irc: tuple[int, int] | None,
    irc_markers: list[tuple[int, int, int]],
    parent_record_id: str,
    source: str,
    kind: str = "path_point",
) -> dict[str, object]:
    """Path provenance for one frame of an IRC calculation.

    ``irc_path_position`` is ``ts`` for the geometry the path starts from,
    ``endpoint`` for the last point of a path Gaussian reports as complete,
    and ``intermediate`` otherwise -- including the last point of a path cut
    short, which is not a path end.
    """
    if irc is None:
        return {
            "source_calculation_type": "irc",
            "irc_direction": "",
            "irc_path_position": "ts",
            "irc_frame_kind": "ts",
            "irc_parent_source": source,
        }
    point, path = irc
    direction = _irc_direction(route, path)
    last_point = max((p for _, p, n in irc_markers if n == path), default=point)
    completed = {match.group(1).lower() for match in _IRC_PATH_DONE_RE.finditer(text)}
    if _IRC_ALL_DONE_RE.search(text):
        completed |= {"forward", "reverse"}
    position = (
        "endpoint"
        if kind in ("converged_point", "path_point")
        and point == last_point and direction in completed
        else "intermediate"
    )
    return {
        "source_calculation_type": "irc",
        "irc_direction": direction,
        "irc_path_number": path,
        "irc_path_position": position,
        "irc_frame_kind": kind,
        "irc_parent_record_id": parent_record_id,
        "irc_parent_source": source,
    }


def _geometry_role(config_type: str, metadata: dict[str, object]) -> str:
    position = metadata.get("irc_path_position")
    if position == "ts":
        return "transition_state"
    if position == "endpoint":
        return "reaction_path_endpoint"
    if position == "intermediate":
        return "irc_point"
    return geometry_role_for_config_type(config_type)


def _geometry_hash(atoms: list[Atom]) -> str:
    return hashlib.sha1(geometry_signature(atoms).encode()).hexdigest()[:16]


def extract_records(text: str, source: str) -> list[Record]:
    """Extract unique final/state geometries and split explicit IRC points."""
    tables = _orientation_tables(text)
    if not tables:
        return []
    routes = _routes(text)
    cms = [(m.start(), int(m.group(1)), int(m.group(2))) for m in _CM_RE.finditer(text)]
    states = [(m.start(), m.group(1)) for m in _STATE_RE.finditer(text)]
    ircs = [(m.start(), int(m.group(1)), int(m.group(2))) for m in _IRC_POINT_RE.finditer(text)]
    energies = [(m.start(), _float(m.group(1)), "HF") for m in _HF_RE.finditer(text)]
    energies.extend((m.start(), _float(m.group(1)), "SCF") for m in _SCF_RE.finditer(text))
    energies.sort()

    candidates: list[tuple[int, float | None]] = [(p, e) for p, e, _ in energies]
    if not candidates:
        # Preserve the last geometry from each charge/multiplicity section even if
        # its energy summary was not retained in the legacy document.
        candidates = [(end, None) for _, end, _, _ in tables]
    scf_positions = [p for p, _, kind in energies if kind == "SCF"]
    boundaries = [m.start() for m in _IRC_PATH_BOUNDARY_RE.finditer(text)]
    reverse_starts = [m.start() for m in _IRC_REVERSE_START_RE.finditer(text)]
    marker_after = _IRC_MARKER_AFTER_RE.search(text) is not None
    route_starts = [p for p, _ in routes]

    records: list[Record] = []
    seen: set[tuple] = set()
    # The first frame of each IRC calculation is the geometry the path starts
    # from; its record id is the parent every later path point refers to.
    irc_origins: dict[int, str] = {}
    # One record per printed geometry within an IRC: the archive summary's
    # HF= energy re-describes the last geometry and must not become a phantom
    # "next point".
    irc_tables: set[int] = set()
    for position, energy in candidates:
        table = _last_before(tables, position + 1)
        if table is None:
            continue
        t_start, t_end, atoms, orientation = table
        # Avoid pairing an energy with a geometry from a remote prior calculation.
        if position - t_end > 250_000:
            continue
        cm = _last_before(cms, position + 1)
        charge, multiplicity = (cm[1], cm[2]) if cm else (0, 1)
        route_item = _last_before(routes, position + 1)
        route = route_item[1] if route_item else ""
        route_key = route_item[0] if route_item else -1
        # Markers and frames of *this* calculation: a marker left over from an
        # IRC in an earlier Link1 step does not make a later calculation a
        # path point.
        calc_end = min((p for p in route_starts if p > route_key), default=len(text) + 1)
        calc_markers = [m for m in ircs if route_key < m[0] < calc_end]
        irc_calculation = route_has_irc(route) or bool(calc_markers)
        irc = None
        frame_kind = ""
        if irc_calculation:
            if t_start in irc_tables:
                continue
            irc, frame_kind = _irc_frame(
                position, route_key, calc_markers,
                [p for p in scf_positions if route_key < p < calc_end] or [position],
                boundaries, reverse_starts, marker_after,
            )
        next_boundary = min((p for p, *_ in cms if p > t_end), default=position + 200_000)
        imag = _imaginary_count(text, t_end, min(next_boundary, position + 200_000))
        config_type, role_source, evidence = _classify_with_evidence(source, route, imag, irc)
        irc_metadata: dict[str, object] = {}
        if irc_calculation:
            role_source = "irc_point_marker" if calc_markers else "irc_route"
            if irc is None and frame_kind == "ts":
                config_type = "transition_state"
                evidence = route
                irc_metadata = _irc_metadata(text, route, None, calc_markers, "", source)
            elif irc is None:
                # A path frame nothing numbers: the marker-first layout with
                # no marker before it.
                config_type = "irc_point"
                evidence = route
                irc_metadata = {
                    "source_calculation_type": "irc", "irc_direction": "",
                    "irc_path_position": "intermediate", "irc_frame_kind": frame_kind,
                    "irc_parent_record_id": irc_origins.get(route_key, ""),
                    "irc_parent_source": source,
                }
            else:
                evidence = f"Point Number {irc[0]} Path Number {irc[1]}"
                irc_metadata = _irc_metadata(
                    text, route, irc, calc_markers, irc_origins.get(route_key, ""), source,
                    frame_kind,
                )
                # The label follows the direction the route asked for, not
                # just the path number Gaussian printed for a one-way IRC.
                direction = irc_metadata.get("irc_direction")
                config_type = f"irc_{direction}" if direction in ("forward", "reverse") else (
                    "irc_forward" if irc[1] == 1 else "irc_reverse")
            irc_metadata["irc_marker_layout"] = (
                "marker_after_point" if marker_after else "marker_before_point"
            )
            irc_tables.add(t_start)
        state_item = _last_before(states, position + 1)
        state = state_item[1] if state_item and position - state_item[0] < 100_000 else ""
        geom_hash = _geometry_hash(atoms)
        key = (geom_hash, charge, multiplicity, config_type, irc)
        if key in seen:
            continue
        seen.add(key)
        seed = f"{source}|{geom_hash}|{charge}|{multiplicity}|{config_type}|{irc}"
        rec_id = hashlib.sha1(seed.encode()).hexdigest()[:20]
        if irc_metadata.get("irc_path_position") == "ts":
            irc_origins.setdefault(route_key, rec_id)
        metadata: dict[str, object] = {"orientation": orientation}
        metadata.update(irc_metadata)
        metadata.setdefault("source_calculation_type", _calculation_type(route))
        metadata.update({
            "geometry_role": _geometry_role(config_type, metadata),
            "geometry_role_source": role_source,
            "geometry_role_evidence": evidence,
            "original_charge": charge,
            "original_multiplicity": multiplicity,
        })
        if role_source == "filename_fallback":
            metadata["source_calculation_type"] = "irc"
            metadata.setdefault("irc_direction", {
                "irc_forward": "forward", "irc_reverse": "reverse",
            }.get(config_type, ""))
            metadata.setdefault("irc_path_position", "")
            metadata.setdefault("irc_parent_source", source)
        records.append(Record(
            record_id=rec_id,
            source=source,
            atoms=list(atoms),
            charge=charge,
            multiplicity=multiplicity,
            config_type=config_type,
            route=route,
            legacy_energy_hartree=energy,
            imaginary_frequencies=imag,
            irc_point=irc[0] if irc else None,
            irc_path=irc[1] if irc else None,
            electronic_state=state,
            metadata=metadata,
        ))
    return records


def _native_coordinate_lines(text: str) -> list[Atom]:
    """Parse warehouse TITLE/END, MOLDAT, or simple element/XYZ blocks."""
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
    atom_re = re.compile(
        rf"^\s*([A-Z][a-z]?)\s+({number})\s+({number})\s+({number})(?:\s|$)"
    )
    # Collapse CRLF pairs to a single \n before handling lone \r (old Mac
    # line endings). Replacing bare "\r" first would turn every "\r\n" into
    # "\n\n", inserting a spurious blank line after each real line; the
    # MOLDAT/TITLE readers below count a fixed number of *lines* for the atom
    # block, so that extra blank line silently discards every other atom.
    # read_document() reads raw bytes with no universal-newline translation,
    # so this matters for any real CRLF-encoded (Windows-authored) warehouse
    # file, even though Path.read_text() in a quick script would hide it.
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.splitlines()
    start = 0
    stop = len(lines)
    for i, line in enumerate(lines):
        if line.strip().upper() == "TITLE":
            start = i + 1
            for j in range(start, len(lines)):
                if lines[j].strip().upper().startswith("END"):
                    stop = j
                    break
            break
        if "MOLDAT" in line.strip().upper():
            start = i + 1
            while start < len(lines) and not lines[start].strip():
                start += 1
            if start < len(lines) and lines[start].strip().isdigit():
                count = int(lines[start].strip())
                start += 1
                stop = min(len(lines), start + count)
            break
    atoms: list[Atom] = []
    for line in lines[start:stop]:
        match = atom_re.match(line.replace(",", " "))
        if not match or match.group(1) not in ATOMIC_SYMBOLS.values():
            if atoms and line.strip():
                break
            continue
        atoms.append(Atom(match.group(1), *(_float(v) for v in match.groups()[1:])))
    return atoms


def _filename_state(source: str) -> tuple[int, int, bool]:
    name = Path(source).name.lower()
    charge = 0
    qmatch = re.search(r"(?:^|[_-])q([+-]?\d+)(?:[_-]|\.)", name)
    if qmatch:
        charge = int(qmatch.group(1))
    elif re.search(r"anion|negative|(?:^|[_-])neg(?:[_-]|\.)", name):
        charge = -1
    elif re.search(r"cation|positive|(?:^|[_-])pos(?:[_-]|\.)", name):
        charge = 1
    elif re.search(r"[a-z]+\d+(?:[a-z]+\d+)+-[_-]", name):
        charge = -1
    elif re.search(r"[a-z]+\d+(?:[a-z]+\d+)+\+[_-]", name):
        charge = 1
    multiplicity = 1
    # The stable warehouse convention ends in _<multiplicity>_<5-digit energy>.
    # It also works when a dopant token occurs between formula and multiplicity.
    mmatch = re.search(r"_(\d{1,3})_[.]?\d{4,5}(?:_|\.|\s|$)", name)
    if mmatch is None:
        mmatch = re.search(r"fe\d+o\d+[+-]?[_-](\d{1,3})(?:[_-]|\.)", name)
    if mmatch:
        multiplicity = int(mmatch.group(1))
    return charge, multiplicity, mmatch is not None


def extract_warehouse_record(text: str, source: str) -> list[Record]:
    atoms = _native_coordinate_lines(text)
    if not atoms:
        return []
    # Native warehouse files carry an explicit TITLE/MOLDAT marker or the
    # composition/multiplicity naming convention. Avoid treating arbitrary text
    # documents containing a few coordinate-looking lines as structures.
    warehouse_name = bool(re.search(r"[A-Za-z]+\d+[A-Za-z]+\d+[_-]\d+", Path(source).name))
    if not re.search(r"^\s*(?:TITLE|\S*MOLDAT)\b", text, re.I | re.M) and not warehouse_name:
        return []
    charge, multiplicity, multiplicity_matched = _filename_state(source)
    # A record whose filename does not match the multiplicity convention still
    # gets a singlet default so downstream code has a value, but that default
    # must never be reported as if it came from the filename convention -- an
    # unflagged wrong guess here is exactly what the spin-safety workflow
    # exists to prevent.
    state_inference = "filename" if multiplicity_matched else "default_unmatched_singlet"
    if multiplicity > 100:
        electron_count = sum(next(z for z, symbol in ATOMIC_SYMBOLS.items() if symbol == a.symbol) for a in atoms) - charge
        multiplicity = 1 if electron_count % 2 == 0 else 2
        state_inference = "electron_parity_fallback"
    geom_hash = _geometry_hash(atoms)
    seed = f"warehouse|{source}|{geom_hash}|{charge}|{multiplicity}"
    return [Record(
        record_id=hashlib.sha1(seed.encode()).hexdigest()[:20],
        source=source,
        atoms=atoms,
        charge=charge,
        multiplicity=multiplicity,
        config_type="warehouse_structure",
        metadata={"format": "native_coordinate", "state_inference": state_inference},
    )]


def extract_formatted_checkpoint(text: str, source: str) -> list[Record]:
    def scalar(label: str, kind: str = "I") -> str | None:
        match = re.search(rf"^{re.escape(label)}\s+{kind}\s+(.+?)\s*$", text, re.M)
        return match.group(1).strip() if match else None

    nat_raw = scalar("Number of atoms")
    if nat_raw is None:
        return []
    n_atoms = int(nat_raw.split()[0])

    def array(label: str, kind: str, count: int) -> list[str]:
        match = re.search(rf"^{re.escape(label)}\s+{kind}\s+N=\s*{count}\s*$", text, re.M)
        if not match:
            return []
        tokens = text[match.end():].split()
        return tokens[:count]

    numbers = [int(v) for v in array("Atomic numbers", "I", n_atoms)]
    coords = [_float(v) for v in array("Current cartesian coordinates", "R", 3 * n_atoms)]
    if len(numbers) != n_atoms or len(coords) != 3 * n_atoms:
        return []
    atoms = [
        Atom(ATOMIC_SYMBOLS[z], coords[3*i] * BOHR_TO_ANG, coords[3*i+1] * BOHR_TO_ANG, coords[3*i+2] * BOHR_TO_ANG)
        for i, z in enumerate(numbers)
    ]
    charge = int((scalar("Charge") or "0").split()[0])
    multiplicity = int((scalar("Multiplicity") or "1").split()[0])
    energy_raw = scalar("Total Energy", "R")
    filename = Path(source).stem.lower()
    config_type = "checkpoint_geometry"
    irc_path = None
    irc_point = None
    metadata: dict[str, object] = {
        "format": "formatted_checkpoint",
        "original_charge": charge,
        "original_multiplicity": multiplicity,
    }
    # A checkpoint carries no route, so its name is the only IRC evidence.
    # It is checked before the TS pattern: an IRC run's checkpoint holds the
    # last point of the path, not the transition state it started from, even
    # when the name mentions both.
    irc_name = irc_name_evidence(Path(source).name)
    if irc_name:
        direction = irc_direction_from_name(filename)
        if direction == "forward":
            config_type, irc_path = "irc_forward", 1
        elif direction == "reverse":
            config_type, irc_path = "irc_reverse", 2
        else:
            config_type = "irc_checkpoint"
        point_match = re.search(r"(?:point|pt|step|p)[_-]?(\d+)", filename)
        if point_match:
            irc_point = int(point_match.group(1))
        metadata.update({
            "source_calculation_type": "irc",
            "irc_direction": direction,
            "irc_parent_source": source,
            "geometry_role_source": "filename_fallback",
            "geometry_role_evidence": irc_name,
        })
    elif re.search(r"(?:^|[_\-.])(?:ts|qst[23])(?:[_\-.]|$)|transition", filename):
        config_type = "transition_state"
        metadata["geometry_role_source"] = "filename"
    metadata["geometry_role"] = geometry_role_for_config_type(config_type)
    geom_hash = _geometry_hash(atoms)
    seed = f"fchk|{source}|{geom_hash}|{charge}|{multiplicity}|{config_type}|{irc_point}"
    return [Record(
        record_id=hashlib.sha1(seed.encode()).hexdigest()[:20],
        source=source,
        atoms=atoms,
        charge=charge,
        multiplicity=multiplicity,
        config_type=config_type,
        legacy_energy_hartree=_float(energy_raw.split()[0]) if energy_raw else None,
        irc_path=irc_path,
        irc_point=irc_point,
        metadata=metadata,
    )]


def extract_gaussian_input(text: str, source: str) -> list[Record]:
    cm = _CM_RE.search(text)
    if cm:
        charge, multiplicity = int(cm.group(1)), int(cm.group(2))
        coordinate_start = cm.end()
    else:
        # Gaussian input convention: charge/multiplicity line follows title and blanks.
        match = re.search(r"^\s*(-?\d+)\s+(\d+)\s*$", text, re.M)
        if not match:
            return []
        charge, multiplicity = int(match.group(1)), int(match.group(2))
        coordinate_start = match.end()
    atoms = _native_coordinate_lines(text[coordinate_start:])
    if not atoms:
        return []
    route_items = _routes(text)
    route = route_items[0][1] if route_items else ""
    config_type, role_source, evidence = _classify_with_evidence(source, route, None, None)
    metadata: dict[str, object] = {
        "format": "gaussian_input",
        "source_calculation_type": _calculation_type(route),
        "original_charge": charge,
        "original_multiplicity": multiplicity,
    }
    if route_has_irc(route):
        # The geometry an IRC is started from: its transition state.
        config_type, role_source, evidence = "irc_input_seed", "irc_route", route
        metadata.update({
            "irc_direction": ",".join(sorted(_irc_directions(route))),
            "irc_path_position": "ts",
            "irc_parent_source": source,
        })
    elif role_source == "filename_fallback":
        metadata.update({
            "source_calculation_type": "irc",
            "irc_direction": irc_direction_from_name(evidence),
            "irc_parent_source": source,
        })
    metadata.update({
        "geometry_role": _geometry_role(config_type, metadata),
        "geometry_role_source": role_source,
        "geometry_role_evidence": evidence,
    })
    geom_hash = _geometry_hash(atoms)
    seed = f"gaussian_input|{source}|{geom_hash}|{charge}|{multiplicity}|{config_type}"
    return [Record(
        record_id=hashlib.sha1(seed.encode()).hexdigest()[:20],
        source=source,
        atoms=atoms,
        charge=charge,
        multiplicity=multiplicity,
        config_type=config_type,
        route=route,
        metadata=metadata,
    )]


def extract_document_records(text: str, source: str) -> list[Record]:
    """Dispatch Gaussian outputs, inputs, checkpoints, and native warehouse files."""
    suffix = Path(source).suffix.lower()
    if suffix in {".fchk", ".fch", ".chk"}:
        records = extract_formatted_checkpoint(text, source)
        if records:
            return records
    if suffix in {".com", ".gjf"}:
        records = extract_gaussian_input(text, source)
        if records:
            return records
    records = extract_records(text, source)
    if records:
        return records
    if suffix == ".txt":
        return extract_warehouse_record(text, source)
    return []


_FORCE_HEADER_RE = re.compile(
    r"^\s*Center\s+Atomic\s+Forces \(Hartrees/Bohr\)\s*$", re.I | re.M
)


def gaussian_job_complete(text: str, expected_stages: int = 1) -> bool:
    """Match the generated shell launchers' full-job completion policy."""
    normal = 0
    finished = False
    for line in text.splitlines():
        if any(token in line for token in (
            "Entering Gaussian System", "Link1:  Proceeding", "SCF Done:",
        )) or re.search(r"Link1: *Proceeding", line):
            finished = False
        if "Error termination" in line:
            return False
        if "Normal termination of Gaussian" in line:
            normal += 1
            finished = True
    return normal >= expected_stages and finished


def parse_force_frames(text: str, source: Path, seed: Record | None = None) -> list[LabeledFrame]:
    """Read force-bearing steps, pairing each table with its preceding SCF/geometry.

    Structural extraction alone is not a force label. In particular, never attach
    a newly printed geometry to an energy/force pair from the previous step.
    """
    tables = _orientation_tables(text)
    energies = list(_SCF_RE.finditer(text))
    cms = list(_CM_RE.finditer(text))
    frames = []
    for index, header in enumerate(_FORCE_HEADER_RE.finditer(text)):
        preceding_energy = [match for match in energies if match.end() < header.start()]
        if not preceding_energy:
            continue
        energy = preceding_energy[-1]
        preceding_cm = [match for match in cms if match.end() < header.start()]
        cm = preceding_cm[-1] if preceding_cm else None
        # A force table in a new Link1 section cannot borrow an earlier SCF.
        if cm is not None and energy.start() < cm.start():
            continue
        geometries = [table for table in tables if table[1] < energy.start()
                      and (cm is None or table[0] > cm.start())]
        if not geometries:
            continue
        _, geometry_end, atoms, orientation = geometries[-1]
        tail = text[header.end():]
        separators = list(re.finditer(r"^\s*-{10,}\s*$", tail, re.M))
        if len(separators) < 2:
            continue
        force_lines = tail[separators[0].end():separators[1].start()].strip().splitlines()
        if len(force_lines) != len(atoms):
            continue
        forces = []
        for center, (line, atom) in enumerate(zip(force_lines, atoms), 1):
            parts = line.split()
            if len(parts) != 5:
                break
            try:
                if int(parts[0]) != center or ATOMIC_SYMBOLS.get(int(parts[1])) != atom.symbol:
                    break
                fx, fy, fz = (_float(value) * FORCE_AU_TO_EV_ANG for value in parts[2:])
                force = (fx, fy, fz)
                if not all(math.isfinite(value) for value in force):
                    break
                forces.append(force)
            except ValueError:
                break
        if len(forces) != len(atoms):
            continue
        energy_h = _float(energy.group(1))
        if not math.isfinite(energy_h):
            continue
        charge, multiplicity = ((int(cm.group(1)), int(cm.group(2))) if cm else
                                (seed.charge, seed.multiplicity) if seed else (0, 1))
        record = copy.deepcopy(seed) if seed else Record(
            record_id=_geometry_hash(atoms), source=str(source), atoms=[],
            charge=charge, multiplicity=multiplicity, config_type="labeled",
        )
        record.atoms = atoms
        record.charge = charge
        record.multiplicity = multiplicity
        record.metadata.update({
            "force_frame_index": index, "orientation": orientation,
            "scf_convergence_warning": "convergence failure" in
                text[geometry_end:header.start()].lower(),
        })
        electronic = text[energy.end():header.start()]
        s2 = list(_S2_RE.finditer(electronic))
        if s2:
            record.metadata["s2_before"] = _float(s2[-1].group(1))
            record.metadata["s2_after"] = _float(s2[-1].group(2))
        spin_blocks = list(_MULLIKEN_SPIN_RE.finditer(electronic))
        if spin_blocks:
            record.metadata["atomic_spins"] = [
                [int(match.group(1)), match.group(2), _float(match.group(3))]
                for match in _MULLIKEN_ROW_RE.finditer(spin_blocks[-1].group(1))
            ]
        frames.append(LabeledFrame(record, energy_h * HARTREE_TO_EV, forces, source))
    return frames


def parse_final_force_frame(text: str, source: Path, seed: Record | None = None) -> LabeledFrame | None:
    frames = parse_force_frames(text, source, seed)
    headers = list(_FORCE_HEADER_RE.finditer(text))
    if not frames or frames[-1].record.metadata["force_frame_index"] != len(headers) - 1:
        return None
    return frames[-1]


def rms_force(frame: LabeledFrame) -> float:
    return math.sqrt(sum(x*x + y*y + z*z for x, y, z in frame.forces_ev_ang) / (3 * len(frame.forces_ev_ang)))
