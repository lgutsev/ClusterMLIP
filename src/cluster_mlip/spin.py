from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .gaussian import ATOMIC_SYMBOLS, _CM_RE, _SCF_RE, _float, extract_document_records
from .io import iter_documents, read_document, read_extxyz, source_tree, write_extxyz
from .models import Atom, Record


DEFAULT_SPIN_ROUTE = (
    "#p uwB97M-V/def2TZVPP Opt SCF=(XQC,Tight,MaxCycle=512) "
    "Integral=UltraFine NoSymm Pop=(Full,SpinDensity)"
)

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


@dataclass(frozen=True)
class SpinDiagnostics:
    charge: int | None
    multiplicity: int | None
    energy_hartree: float | None
    s2_before: float | None
    s2_after: float | None
    atomic_spins: tuple[tuple[int, str, float], ...]
    normal_termination: bool
    optimized: bool
    stability: str

    @property
    def expected_s2(self) -> float | None:
        if self.multiplicity is None:
            return None
        spin = (self.multiplicity - 1) / 2
        return spin * (spin + 1)

    @property
    def s2_delta(self) -> float | None:
        observed = self.s2_after if self.s2_after is not None else self.s2_before
        expected = self.expected_s2
        return None if observed is None or expected is None else observed - expected

    @property
    def spin_pattern(self) -> str:
        significant = [spin for _, _, spin in self.atomic_spins if abs(spin) >= 0.10]
        if not significant:
            return "unavailable" if not self.atomic_spins else "weak_or_unresolved"
        if any(spin > 0 for spin in significant) and any(spin < 0 for spin in significant):
            return "compensated_afm_like"
        return "ferro_like"

    @property
    def root_signature(self) -> str:
        if not self.atomic_spins or self.multiplicity is None:
            return ""
        canonical = ";".join(
            f"{symbol}:{spin:.3f}"
            for _, symbol, spin in sorted(self.atomic_spins, key=lambda item: (item[1], item[2]))
        )
        return hashlib.sha1(f"{self.charge}|{self.multiplicity}|{canonical}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class StateObservation:
    record: Record
    diagnostics: SpinDiagnostics


def electron_count(record: Record) -> int:
    symbol_to_z = {symbol: number for number, symbol in ATOMIC_SYMBOLS.items()}
    return sum(symbol_to_z[atom.symbol] for atom in record.atoms) - record.charge


def validate_multiplicity(record: Record, multiplicity: int) -> None:
    electrons = electron_count(record)
    if multiplicity < 1:
        raise ValueError("multiplicity must be positive")
    if multiplicity > electrons + 1:
        raise ValueError(
            f"multiplicity {multiplicity} exceeds the electron-count limit for {record.record_id}"
        )
    if (electrons + multiplicity - 1) % 2:
        parity = "odd" if electrons % 2 else "even"
        raise ValueError(
            f"multiplicity {multiplicity} has the wrong parity for {electrons} ({parity}) electrons"
        )


def multiplicity_ladder(high_spin: int, targets: Iterable[int]) -> list[int]:
    requested = sorted(set(targets), reverse=True)
    if not requested:
        return [high_spin]
    if any(target >= high_spin for target in requested):
        raise ValueError("all low-spin targets must be below the high-spin multiplicity")
    if any((high_spin - target) % 2 for target in requested):
        raise ValueError("each ladder step changes 2S+1 by two; target parity is inconsistent")
    lowest = min(requested)
    return list(range(high_spin, lowest - 1, -2))


def _route_with(route: str, *keywords: str) -> str:
    route = route.strip()
    if not route.startswith("#"):
        raise ValueError("Gaussian route must start with '#'")
    lower = route.lower()
    conflicts = [key.split("=")[0].lower() for key in keywords]
    for conflict in conflicts:
        if re.search(rf"\b{re.escape(conflict)}\s*=", lower):
            raise ValueError(f"route already defines {conflict}; keep state-control keywords tool-managed")
    return f"{route} {' '.join(keywords)}".strip()


def _link_header(checkpoint: str, memory: str, nproc: int, old_checkpoint: str | None = None) -> list[str]:
    lines = []
    if old_checkpoint:
        lines.append(f"%oldchk={old_checkpoint}")
    lines.extend([f"%chk={checkpoint}", f"%mem={memory}", f"%nprocshared={nproc}"])
    return lines


def _coordinates(atoms: list[Atom], fragments: dict[int, int] | None = None) -> list[str]:
    lines = []
    for index, atom in enumerate(atoms, start=1):
        label = atom.symbol if fragments is None else f"{atom.symbol}(Fragment={fragments[index]})"
        lines.append(f"{label:18s} {atom.x: .12f} {atom.y: .12f} {atom.z: .12f}")
    return lines


def render_ladder_input(
    record: Record,
    high_spin: int,
    targets: Iterable[int],
    route: str = DEFAULT_SPIN_ROUTE,
    memory: str = "16GB",
    nproc: int = 16,
) -> tuple[str, list[dict[str, str]]]:
    """Render a checkpoint-preserving one-spin-flip-at-a-time Link1 ladder."""
    sequence = multiplicity_ladder(high_spin, targets)
    for multiplicity in sequence:
        validate_multiplicity(record, multiplicity)
    chain = f"{record.record_id}-ladder-m{high_spin}-m{sequence[-1]}"
    parts: list[str] = []
    rows: list[dict[str, str]] = []
    previous_checkpoint: str | None = None
    for stage, multiplicity in enumerate(sequence):
        checkpoint = f"{chain}-s{stage:02d}-m{multiplicity}.chk"
        if stage:
            parts.extend(["", "--Link1--"])
        parts.extend(_link_header(checkpoint, memory, nproc, previous_checkpoint))
        stage_route = route if stage == 0 else _route_with(route, "Geom=Checkpoint", "Guess=(Read,Always)")
        parts.extend([
            stage_route,
            "",
            f"ClusterMLIP multiplicity ladder; record={record.record_id}; stage={stage}; multiplicity={multiplicity}",
            "",
            f"{record.charge} {multiplicity}",
        ])
        if stage == 0:
            parts.extend(_coordinates(record.atoms))
        parts.append("")
        rows.append({
            "job_id": f"{chain}-s{stage:02d}",
            "chain_id": chain,
            "stage_index": str(stage),
            "pathway": "multiplicity_ladder",
            "parent_record_id": record.record_id,
            "source": record.source,
            "formula": record.formula,
            "intended_charge": str(record.charge),
            "intended_multiplicity": str(multiplicity),
            "predecessor_multiplicity": "" if stage == 0 else str(sequence[stage - 1]),
            "checkpoint": checkpoint,
            "fragment_label": "",
        })
        previous_checkpoint = checkpoint
    return "\n".join(parts) + "\n", rows


_FRAGMENT_ORIENTATIONS = (1, -1, "+", "-", "alpha", "beta", "up", "down")


def validate_fragment_specification_shape(specifications: object) -> list[str]:
    """Structural pre-check for a parsed --fragment-spec payload.

    Mirrors examples/spin_fragments.schema.json. This exists so a malformed
    spec fails once, up front, with every problem listed -- rather than as a
    bare KeyError/TypeError raised from deep inside _validated_fragments (or
    from write_spin_jobs's dict-keying) that only reports the first mistake
    and gives no line to fix it at. It does not check cross-record semantics
    (atom coverage, spin-sum consistency); that stays in _validated_fragments
    since it needs the target Record to check against.
    """
    errors: list[str] = []
    if not isinstance(specifications, list):
        return ["fragment specification must be a list of guesses (or an object with a 'guesses' list)"]
    for index, spec in enumerate(specifications):
        label = f"guesses[{index}]"
        if not isinstance(spec, dict):
            errors.append(f"{label}: must be an object, got {type(spec).__name__}")
            continue
        name = spec.get("name")
        if name:
            label = f"guesses[{index}] ({name!r})"
        for key, kind in (("record_id", str), ("target_multiplicity", int)):
            if key not in spec:
                errors.append(f"{label}: missing required key '{key}'")
            elif not isinstance(spec[key], kind):
                errors.append(f"{label}: '{key}' must be a {kind.__name__}, got {type(spec[key]).__name__}")
        fragments = spec.get("fragments")
        if "fragments" not in spec:
            errors.append(f"{label}: missing required key 'fragments'")
        elif not isinstance(fragments, list) or len(fragments) < 2:
            errors.append(f"{label}: 'fragments' must be a list of at least two fragments")
        else:
            for f_index, fragment in enumerate(fragments):
                f_label = f"{label}.fragments[{f_index}]"
                if not isinstance(fragment, dict):
                    errors.append(f"{f_label}: must be an object, got {type(fragment).__name__}")
                    continue
                for key in ("atoms", "charge", "multiplicity"):
                    if key not in fragment:
                        errors.append(f"{f_label}: missing required key '{key}'")
                atoms = fragment.get("atoms")
                if "atoms" in fragment and (not isinstance(atoms, list) or not atoms):
                    errors.append(f"{f_label}: 'atoms' must be a non-empty list of 1-based indices")
                for key in ("charge", "multiplicity"):
                    if key in fragment and not isinstance(fragment[key], int):
                        errors.append(f"{f_label}: '{key}' must be an integer")
                orientation = fragment.get("orientation", "alpha")
                if orientation not in _FRAGMENT_ORIENTATIONS:
                    errors.append(
                        f"{f_label}: 'orientation' must be one of {_FRAGMENT_ORIENTATIONS}, got {orientation!r}"
                    )
    return errors


def _orientation_sign(value: object) -> int:
    if value in (1, "+", "alpha", "up"):
        return 1
    if value in (-1, "-", "beta", "down"):
        return -1
    raise ValueError(f"unknown fragment orientation {value!r}; use alpha/beta or +1/-1")


def _validated_fragments(record: Record, specification: dict) -> tuple[dict[int, int], list[tuple[int, int]]]:
    fragments = specification.get("fragments")
    if not isinstance(fragments, list) or len(fragments) < 2:
        raise ValueError("a fragment guess requires at least two fragments")
    atom_map: dict[int, int] = {}
    states: list[tuple[int, int]] = []
    symbol_to_z = {symbol: number for number, symbol in ATOMIC_SYMBOLS.items()}
    signed_unpaired_electrons = 0
    for fragment_index, fragment in enumerate(fragments, start=1):
        atoms = fragment.get("atoms", [])
        if not atoms:
            raise ValueError(f"fragment {fragment_index} has no atoms")
        charge = int(fragment["charge"])
        multiplicity = int(fragment["multiplicity"])
        orientation = _orientation_sign(fragment.get("orientation", "alpha"))
        if multiplicity < 1:
            raise ValueError("fragment multiplicity must be positive; orientation carries the sign")
        atom_indices = [int(index) for index in atoms]
        for atom_index in atom_indices:
            if atom_index < 1 or atom_index > len(record.atoms):
                raise ValueError(f"atom index {atom_index} is outside 1..{len(record.atoms)}")
            if atom_index in atom_map:
                raise ValueError(f"atom {atom_index} occurs in more than one fragment")
            atom_map[atom_index] = fragment_index
        states.append((charge, orientation * multiplicity))
        fragment_electrons = sum(symbol_to_z[record.atoms[index - 1].symbol] for index in atom_indices) - charge
        if multiplicity > fragment_electrons + 1:
            raise ValueError(
                f"fragment {fragment_index} multiplicity {multiplicity} exceeds its electron-count limit"
            )
        if (fragment_electrons + multiplicity - 1) % 2:
            raise ValueError(
                f"fragment {fragment_index} multiplicity {multiplicity} has the wrong parity "
                f"for {fragment_electrons} electrons"
            )
        signed_unpaired_electrons += orientation * (multiplicity - 1)
    missing = sorted(set(range(1, len(record.atoms) + 1)) - set(atom_map))
    if missing:
        raise ValueError(f"fragment map does not cover atoms: {missing}")
    if sum(charge for charge, _ in states) != record.charge:
        raise ValueError("fragment charges do not sum to the molecular charge")
    target = int(specification["target_multiplicity"])
    if signed_unpaired_electrons != target - 1:
        raise ValueError(
            "signed fragment spins are inconsistent with the total multiplicity: "
            f"sum[orientation * (fragment_multiplicity - 1)]={signed_unpaired_electrons}, "
            f"but total_multiplicity - 1={target - 1}"
        )
    return atom_map, states


def render_fragment_input(
    record: Record,
    specification: dict,
    route: str = DEFAULT_SPIN_ROUTE,
    memory: str = "16GB",
    nproc: int = 16,
) -> tuple[str, dict[str, str]]:
    target = int(specification["target_multiplicity"])
    validate_multiplicity(record, target)
    atom_map, states = _validated_fragments(record, specification)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(specification.get("name", "afm"))).strip("-") or "afm"
    job_id = f"{record.record_id}-fragment-{label}-m{target}"
    checkpoint = f"{job_id}.chk"
    fragment_state = " ".join(f"{charge} {multiplicity}" for charge, multiplicity in states)
    lines = _link_header(checkpoint, memory, nproc)
    lines.extend([
        _route_with(route, f"Guess=(Fragment={len(states)},Always)"),
        "",
        f"ClusterMLIP fragment AFM candidate; record={record.record_id}; label={label}; multiplicity={target}",
        "",
        f"{record.charge} {target} {fragment_state}",
    ])
    lines.extend(_coordinates(record.atoms, atom_map))
    lines.extend(["", ""])
    row = {
        "job_id": job_id,
        "chain_id": job_id,
        "stage_index": "0",
        "pathway": "fragment_guess",
        "parent_record_id": record.record_id,
        "source": record.source,
        "formula": record.formula,
        "intended_charge": str(record.charge),
        "intended_multiplicity": str(target),
        "predecessor_multiplicity": "",
        "checkpoint": checkpoint,
        "fragment_label": label,
    }
    return "\n".join(lines), row


def write_spin_jobs(
    records: list[Record],
    output: Path,
    high_spin: int,
    targets: Iterable[int],
    route: str = DEFAULT_SPIN_ROUTE,
    memory: str = "16GB",
    nproc: int = 16,
    fragment_specifications: list[dict] | None = None,
) -> int:
    high_spin_records = [record for record in records if record.multiplicity == high_spin]
    if not high_spin_records:
        raise ValueError(f"no seed has the requested high-spin multiplicity {high_spin}")
    specs_by_record: dict[str, list[dict]] = {}
    for specification in fragment_specifications or []:
        specs_by_record.setdefault(str(specification["record_id"]), []).append(specification)
    unknown = sorted(set(specs_by_record) - {record.record_id for record in high_spin_records})
    if unknown:
        raise ValueError(f"fragment specifications reference unselected records: {unknown}")

    # Render every ladder/fragment job in memory first. render_ladder_input
    # and render_fragment_input validate as they go (multiplicity parity,
    # fragment atom coverage, signed-spin consistency, ...); rendering
    # everything before writing anything means a bad spec later in the list
    # raises cleanly instead of leaving a half-written job directory with no
    # spin_jobs.csv to explain what is and is not there.
    files: list[tuple[str, str]] = []
    rows: list[dict[str, str]] = []
    for record in high_spin_records:
        text, chain_rows = render_ladder_input(record, high_spin, targets, route, memory, nproc)
        filename = f"{chain_rows[0]['chain_id']}.gjf"
        files.append((filename, text))
        for row in chain_rows:
            row["input"] = filename
            row["output"] = f"{Path(filename).stem}.log"
        rows.extend(chain_rows)
        for specification in specs_by_record.get(record.record_id, []):
            text, row = render_fragment_input(record, specification, route, memory, nproc)
            filename = f"{row['job_id']}.gjf"
            files.append((filename, text))
            row["input"] = filename
            row["output"] = f"{Path(filename).stem}.log"
            rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    for filename, text in files:
        (output / filename).write_text(text, encoding="utf-8")
    columns = [
        "job_id", "chain_id", "stage_index", "pathway", "parent_record_id", "source", "formula",
        "intended_charge", "intended_multiplicity", "predecessor_multiplicity", "checkpoint",
        "fragment_label", "input", "output",
    ]
    with (output / "spin_jobs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output / "run_one.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ninput=$1\noutput=${input%.gjf}.log\ng16 \"$input\" > \"$output\"\n",
        encoding="utf-8",
    )
    (output / "run_one.sh").chmod(0o755)
    return len(rows)


def _diagnostics_from_section(text: str, charge: int | None, multiplicity: int | None) -> SpinDiagnostics:
    energies = list(_SCF_RE.finditer(text))
    s2 = list(_S2_RE.finditer(text))
    spin_blocks = list(_MULLIKEN_SPIN_RE.finditer(text))
    atomic_spins: tuple[tuple[int, str, float], ...] = ()
    if spin_blocks:
        atomic_spins = tuple(
            (int(match.group(1)), match.group(2), _float(match.group(3)))
            for match in _MULLIKEN_ROW_RE.finditer(spin_blocks[-1].group(1))
        )
    lower = text.lower()
    if "wavefunction is stable" in lower or "stable under the perturbations" in lower:
        stability = "stable"
    elif "wavefunction has an internal instability" in lower or "wavefunction is unstable" in lower:
        stability = "unstable"
    else:
        stability = "not_tested"
    return SpinDiagnostics(
        charge=charge,
        multiplicity=multiplicity,
        energy_hartree=_float(energies[-1].group(1)) if energies else None,
        s2_before=_float(s2[-1].group(1)) if s2 else None,
        s2_after=_float(s2[-1].group(2)) if s2 else None,
        atomic_spins=atomic_spins,
        normal_termination="normal termination of gaussian" in lower,
        optimized="stationary point found" in lower or "optimization completed" in lower,
        stability=stability,
    )


def parse_spin_diagnostics(text: str) -> list[SpinDiagnostics]:
    """Parse every charge/multiplicity section, including Link1 spin ladders."""
    cms = list(_CM_RE.finditer(text))
    if not cms:
        return [_diagnostics_from_section(text, None, None)]
    diagnostics = []
    for index, cm in enumerate(cms):
        end = cms[index + 1].start() if index + 1 < len(cms) else len(text)
        diagnostics.append(
            _diagnostics_from_section(text[cm.start():end], int(cm.group(1)), int(cm.group(2)))
        )
    return diagnostics


def _pair_fingerprint(atoms: list[Atom]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for left in range(len(atoms)):
        for right in range(left + 1, len(atoms)):
            a, b = atoms[left], atoms[right]
            key = "-".join(sorted((a.symbol, b.symbol)))
            distance = math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
            result.setdefault(key, []).append(distance)
    for values in result.values():
        values.sort()
    return result


def geometry_distance(left: Record, right: Record) -> float:
    if left.formula != right.formula or len(left.atoms) != len(right.atoms):
        return math.inf
    first, second = _pair_fingerprint(left.atoms), _pair_fingerprint(right.atoms)
    if first.keys() != second.keys() or any(len(first[key]) != len(second[key]) for key in first):
        return math.inf
    differences = [a-b for key in first for a, b in zip(first[key], second[key])]
    return math.sqrt(sum(value * value for value in differences) / len(differences)) if differences else 0.0


def spin_density_distance(left: SpinDiagnostics, right: SpinDiagnostics) -> float | None:
    if not left.atomic_spins or not right.atomic_spins:
        return None
    first = sorted((symbol, spin) for _, symbol, spin in left.atomic_spins)
    second = sorted((symbol, spin) for _, symbol, spin in right.atomic_spins)
    if [item[0] for item in first] != [item[0] for item in second]:
        return None
    return math.sqrt(sum((a[1]-b[1])**2 for a, b in zip(first, second)) / len(first))


def s2_distance(left: SpinDiagnostics, right: SpinDiagnostics) -> float | None:
    first = left.s2_after if left.s2_after is not None else left.s2_before
    second = right.s2_after if right.s2_after is not None else right.s2_before
    return None if first is None or second is None else abs(first - second)


def _load_observations(source: Path, outputs_only: bool = False) -> tuple[list[StateObservation], list[tuple[str, str]]]:
    if source.suffix.lower() in {".extxyz", ".xyz"}:
        empty = SpinDiagnostics(None, None, None, None, None, (), False, False, "not_tested")
        return [StateObservation(record, empty) for record in read_extxyz(source)], []
    observations: list[StateObservation] = []
    errors: list[tuple[str, str]] = []
    with source_tree(source) as root:
        documents = [source] if source.is_file() and source.suffix.lower() != ".zip" else list(iter_documents(root))
        for document in documents:
            if outputs_only and document.suffix.lower() not in {".log", ".out", ".fchk", ".fch"}:
                continue
            try:
                relative = str(document.relative_to(root)) if document.is_relative_to(root) else document.name
                text = read_document(document)
                cms = list(_CM_RE.finditer(text))
                if cms:
                    # A geometry optimization contains many SCF cycles. Retain the
                    # final geometry/root from each Gaussian or Link1 state section,
                    # rather than misidentifying intermediate optimization steps as
                    # independent electronic states.
                    for index, cm in enumerate(cms):
                        end = cms[index + 1].start() if index + 1 < len(cms) else len(text)
                        section = text[cm.start():end]
                        records = extract_document_records(section, relative)
                        if not records:
                            continue
                        record = records[-1]
                        diagnostic = _diagnostics_from_section(
                            section, int(cm.group(1)), int(cm.group(2))
                        )
                        observations.append(StateObservation(record, diagnostic))
                else:
                    records = extract_document_records(text, relative)
                    diagnostics = parse_spin_diagnostics(text)
                    diagnostic = diagnostics[-1]
                    for record in records:
                        observations.append(StateObservation(record, diagnostic))
            except Exception as exc:
                errors.append((str(document), str(exc)))
    return observations, errors


def write_spin_inventory(source: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    observations, errors = _load_observations(source)
    write_extxyz([observation.record for observation in observations], output / "seeds.extxyz")
    columns = [
        "record_id", "source", "formula", "n_atoms", "charge", "multiplicity", "electron_count",
        "multiplicity_parity_valid", "config_type", "state_inference", "energy_hartree", "expected_s2",
        "s2_before", "s2_after", "s2_delta", "spin_pattern", "root_signature", "normal_termination",
        "optimized", "stability",
    ]
    with (output / "spin_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for observation in observations:
            record, diagnostic = observation.record, observation.diagnostics
            try:
                validate_multiplicity(record, record.multiplicity)
                parity_valid = True
            except ValueError:
                parity_valid = False
            writer.writerow({
                "record_id": record.record_id, "source": record.source, "formula": record.formula,
                "n_atoms": len(record.atoms), "charge": record.charge, "multiplicity": record.multiplicity,
                "electron_count": electron_count(record), "multiplicity_parity_valid": parity_valid,
                "config_type": record.config_type,
                "state_inference": record.metadata.get("state_inference", ""),
                "energy_hartree": diagnostic.energy_hartree,
                "expected_s2": diagnostic.expected_s2, "s2_before": diagnostic.s2_before,
                "s2_after": diagnostic.s2_after, "s2_delta": diagnostic.s2_delta,
                "spin_pattern": diagnostic.spin_pattern, "root_signature": diagnostic.root_signature,
                "normal_termination": diagnostic.normal_termination, "optimized": diagnostic.optimized,
                "stability": diagnostic.stability,
            })
    with (output / "errors.tsv").open("w", encoding="utf-8") as handle:
        for name, message in errors:
            handle.write(f"{name}\t{message}\n")
    return len(observations)


def validate_spin_campaign(
    original: Path,
    new_outputs: Path,
    output: Path,
    geometry_tolerance: float = 0.05,
    spin_tolerance: float = 0.25,
    s2_tolerance: float = 0.25,
) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    old, old_errors = _load_observations(original)
    new, new_errors = _load_observations(new_outputs, outputs_only=True)
    coverage_rows: list[dict[str, object]] = []
    matched_new: set[int] = set()
    counts = {
        "matched_same_root": 0,
        "matched_state_uncharacterized": 0,
        "alternative_root": 0,
        "new_calculation_incomplete": 0,
        "missing": 0,
    }
    for legacy in old:
        candidates = [
            (index, current, geometry_distance(legacy.record, current.record))
            for index, current in enumerate(new)
            if current.record.formula == legacy.record.formula
            and current.record.charge == legacy.record.charge
            and current.record.multiplicity == legacy.record.multiplicity
        ]
        candidates.sort(key=lambda item: item[2])
        if not candidates or candidates[0][2] > geometry_tolerance:
            status = "missing"
            current = None
            distance = None
            spin_distance = None
            s2_difference = None
        else:
            index, current, distance = candidates[0]
            matched_new.add(index)
            spin_distance = spin_density_distance(legacy.diagnostics, current.diagnostics)
            s2_difference = s2_distance(legacy.diagnostics, current.diagnostics)
            if not current.diagnostics.normal_termination or not current.diagnostics.optimized:
                status = "new_calculation_incomplete"
            elif (
                (spin_distance is not None and spin_distance > spin_tolerance)
                or (s2_difference is not None and s2_difference > s2_tolerance)
            ):
                status = "alternative_root"
            elif spin_distance is None:
                status = "matched_state_uncharacterized"
            else:
                status = "matched_same_root"
        counts[status] += 1
        coverage_rows.append({
            "legacy_record_id": legacy.record.record_id,
            "legacy_source": legacy.record.source,
            "formula": legacy.record.formula,
            "charge": legacy.record.charge,
            "multiplicity": legacy.record.multiplicity,
            "status": status,
            "new_record_id": "" if current is None else current.record.record_id,
            "new_source": "" if current is None else current.record.source,
            "geometry_fingerprint_rms_ang": distance,
            "spin_density_rms": spin_distance,
            "s2_absolute_difference": s2_difference,
            "energy_difference_hartree": (
                None if current is None
                or legacy.diagnostics.energy_hartree is None
                or current.diagnostics.energy_hartree is None
                else current.diagnostics.energy_hartree - legacy.diagnostics.energy_hartree
            ),
            "legacy_spin_pattern": legacy.diagnostics.spin_pattern,
            "new_spin_pattern": "" if current is None else current.diagnostics.spin_pattern,
            "legacy_root_signature": legacy.diagnostics.root_signature,
            "new_root_signature": "" if current is None else current.diagnostics.root_signature,
        })
    with (output / "legacy_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]) if coverage_rows else ["status"])
        writer.writeheader()
        writer.writerows(coverage_rows)
    new_rows = []
    for index, observation in enumerate(new):
        diagnostic = observation.diagnostics
        new_rows.append({
            "record_id": observation.record.record_id,
            "source": observation.record.source,
            "formula": observation.record.formula,
            "charge": observation.record.charge,
            "multiplicity": observation.record.multiplicity,
            "energy_hartree": diagnostic.energy_hartree,
            "spin_pattern": diagnostic.spin_pattern,
            "root_signature": diagnostic.root_signature,
            "s2_delta": diagnostic.s2_delta,
            "normal_termination": diagnostic.normal_termination,
            "optimized": diagnostic.optimized,
            "stability": diagnostic.stability,
            "novel_or_unmatched_candidate": index not in matched_new,
        })
    with (output / "new_states.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(new_rows[0]) if new_rows else ["record_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(new_rows)
    planned_rows: list[dict[str, object]] = []
    manifest = new_outputs / "spin_jobs.csv"
    if manifest.exists():
        observed = {
            (Path(observation.record.source).name, observation.record.charge, observation.record.multiplicity)
            for observation in new
        }
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (Path(row["output"]).name, int(row["intended_charge"]), int(row["intended_multiplicity"]))
                planned_rows.append({
                    "job_id": row["job_id"],
                    "pathway": row["pathway"],
                    "output": row["output"],
                    "intended_charge": row["intended_charge"],
                    "intended_multiplicity": row["intended_multiplicity"],
                    "observed": key in observed,
                })
        with (output / "planned_state_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(planned_rows[0]) if planned_rows else ["job_id"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(planned_rows)
    planned_missing = sum(not bool(row["observed"]) for row in planned_rows)
    summary = {
        "legacy_records": len(old), "new_records": len(new), **counts,
        "novel_or_unmatched_new": len(new) - len(matched_new),
        "planned_stages": len(planned_rows), "planned_stages_missing": planned_missing,
        "legacy_parse_errors": len(old_errors), "new_parse_errors": len(new_errors),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Spin-state validation",
        "",
        f"- Legacy records checked: {len(old)}",
        f"- Same local-spin root: {counts['matched_same_root']}",
        f"- Geometry/state matched but legacy root uncharacterized: {counts['matched_state_uncharacterized']}",
        f"- Alternative local-spin root: {counts['alternative_root']}",
        f"- Matching state with incomplete new calculation: {counts['new_calculation_incomplete']}",
        f"- Missing legacy states: {counts['missing']}",
        f"- Planned spin stages without an observed state: {planned_missing}",
        f"- Novel or unmatched new candidates retained: {summary['novel_or_unmatched_new']}",
        "",
        "`alternative_root` is not discarded. It marks a geometrically matching charge/multiplicity state with a materially different local-spin distribution.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "errors.tsv").open("w", encoding="utf-8") as handle:
        for category, errors in (("legacy", old_errors), ("new", new_errors)):
            for name, message in errors:
                handle.write(f"{category}\t{name}\t{message}\n")
    return summary
