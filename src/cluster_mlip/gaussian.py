from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from .models import Atom, LabeledFrame, Record, geometry_signature


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
        if route.lstrip().startswith("##"):
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


def _classify(source: str, route: str, imag: int | None, irc: tuple[int, int] | None) -> str:
    if irc is not None:
        return "irc_forward" if irc[1] == 1 else "irc_reverse"
    route_l = route.lower()
    filename = Path(source).stem.lower()
    explicit_ts = bool(
        re.search(r"(?:opt\s*=\s*\([^)]*\bts\b|\bqst[23]\b|\bsaddle\b)", route_l)
        or re.search(r"(?:^|[_\-.])ts(?:[_\-.]|$)", filename)
    )
    if explicit_ts:
        return "transition_state"
    if imag == 1:
        return "first_order_saddle"
    if imag is not None and imag > 1:
        return "higher_order_saddle"
    if imag == 0:
        return "minimum"
    if "opt" in route_l:
        return "optimized_unverified"
    return "unknown"


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

    records: list[Record] = []
    seen: set[tuple] = set()
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
        irc_item = _last_before(ircs, position + 1)
        irc = None
        if irc_item is not None and position - irc_item[0] < 100_000:
            irc = (irc_item[1], irc_item[2])
        next_boundary = min((p for p, *_ in cms if p > t_end), default=position + 200_000)
        imag = _imaginary_count(text, t_end, min(next_boundary, position + 200_000))
        config_type = _classify(source, route, imag, irc)
        state_item = _last_before(states, position + 1)
        state = state_item[1] if state_item and position - state_item[0] < 100_000 else ""
        geom_hash = _geometry_hash(atoms)
        key = (geom_hash, charge, multiplicity, config_type, irc)
        if key in seen:
            continue
        seen.add(key)
        seed = f"{source}|{geom_hash}|{charge}|{multiplicity}|{config_type}|{irc}"
        rec_id = hashlib.sha1(seed.encode()).hexdigest()[:20]
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
            metadata={"orientation": orientation},
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
    if re.search(r"(?:^|[_\-.])(?:ts|qst[23])(?:[_\-.]|$)|transition", filename):
        config_type = "transition_state"
    elif "irc" in filename:
        if re.search(r"forward|fwd|(?:^|[_-])for(?:[_-]|$)", filename):
            config_type, irc_path = "irc_forward", 1
        elif re.search(r"reverse|rev|backward|bwd", filename):
            config_type, irc_path = "irc_reverse", 2
        else:
            config_type = "irc_checkpoint"
        point_match = re.search(r"(?:point|pt|step|p)[_-]?(\d+)", filename)
        if point_match:
            irc_point = int(point_match.group(1))
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
        metadata={"format": "formatted_checkpoint"},
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
    config_type = _classify(source, route, None, None)
    if "irc" in route.lower():
        config_type = "irc_input_seed"
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
        metadata={"format": "gaussian_input"},
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


def parse_final_force_frame(text: str, source: Path, seed: Record | None = None) -> LabeledFrame | None:
    force_headers = list(re.finditer(r"^\s*Forces \(Hartrees/Bohr\)\s*$", text, re.I | re.M))
    if not force_headers:
        return None
    header = force_headers[-1]
    force_tail = text[header.end():]
    sep = list(re.finditer(r"^\s*-{10,}\s*$", force_tail, re.M))
    if len(sep) < 2:
        return None
    start = sep[1].end()
    end_match = re.search(r"^\s*-{10,}\s*$", force_tail[start:], re.M)
    if end_match is None:
        return None
    forces: list[tuple[float, float, float]] = []
    for line in force_tail[start:start + end_match.start()].splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            fx, fy, fz = map(_float, parts[-3:])
        except ValueError:
            continue
        forces.append((fx * FORCE_AU_TO_EV_ANG, fy * FORCE_AU_TO_EV_ANG, fz * FORCE_AU_TO_EV_ANG))

    tables = _orientation_tables(text[:header.start()])
    if not tables:
        return None
    _, _, atoms, _ = tables[-1]
    energy_matches = list(_SCF_RE.finditer(text[:header.start()]))
    if not energy_matches:
        energy_matches = list(_HF_RE.finditer(text[:header.start()]))
    if not energy_matches or len(atoms) != len(forces):
        return None
    energy_h = _float(energy_matches[-1].group(1))
    cm = list(_CM_RE.finditer(text[:header.start()]))
    charge, multiplicity = (int(cm[-1].group(1)), int(cm[-1].group(2))) if cm else (0, 1)
    if seed is None:
        geom_hash = _geometry_hash(atoms)
        seed = Record(
            record_id=geom_hash,
            source=str(source),
            atoms=atoms,
            charge=charge,
            multiplicity=multiplicity,
            config_type="labeled",
        )
    else:
        seed.atoms = atoms
        seed.charge = charge
        seed.multiplicity = multiplicity
    return LabeledFrame(seed, energy_h * HARTREE_TO_EV, forces, source)


def rms_force(frame: LabeledFrame) -> float:
    return math.sqrt(sum(x*x + y*y + z*z for x, y, z in frame.forces_ev_ang) / (3 * len(frame.forces_ev_ang)))
