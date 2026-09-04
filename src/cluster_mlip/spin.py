from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .basis import render_gen_basis
from .gaussian import ATOMIC_SYMBOLS, _CM_RE, _SCF_RE, _float, extract_document_records
from .io import iter_documents, read_document, read_extxyz, source_tree, write_extxyz
from .jobs import human_job_stem
from .models import Atom, Record


# Basis is Gen (see basis.py): 6-31G* for light elements, an explicit
# def2-TZVP-without-f contraction for Fe.
DEFAULT_SPIN_ROUTE = (
    "#p UBPW91/Gen SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) "
    "NoSymm Opt Freq IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine "
    "Stable=Opt Pop=(Full,SpinDensity)"
)

SPIN_MANIFEST_COLUMNS = [
    "job_id", "chain_id", "stage_index", "pathway", "initialization", "audit_classification",
    "spin_plan_id", "spin_group_key", "target_record_multiplicity", "high_spin_inference",
    "high_spin_evidence_record_ids",
    "parent_record_id", "source", "formula", "source_geometry_sha256", "high_spin_multiplicity",
    "final_target_multiplicity", "intended_charge", "intended_multiplicity", "spin_flip_index",
    "predecessor_job_id", "predecessor_multiplicity", "predecessor_checkpoint", "checkpoint",
    "checkpoint_lineage", "fragment_label", "fragment_count", "fragment_spec_sha256", "input",
    "input_sha256", "output",
]

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


@dataclass(frozen=True)
class AutomaticSpinPlan:
    plan_id: str
    record: Record
    group_key: str
    target_multiplicity: int
    high_spin_multiplicity: int
    inference: str
    evidence_record_ids: tuple[str, ...]
    observed_multiplicities: tuple[int, ...]
    fe_count: int


@dataclass(frozen=True)
class SkippedAutomaticSpinPlan:
    record: Record
    group_key: str
    reason: str
    observed_multiplicities: tuple[int, ...]
    fe_count: int


def _metadata_true(record: Record, key: str) -> bool:
    return str(record.metadata.get(key, "")).lower() == "true"


def fe_spin_summary(
    record: Record,
    diagnostic: SpinDiagnostics,
    threshold: float = 0.25,
) -> dict[str, str]:
    """Summarize actual Fe Mulliken moments without imposing an ionic spin model."""
    fe_indices = [index for index, atom in enumerate(record.atoms, start=1) if atom.symbol == "Fe"]
    spins = {index: spin for index, symbol, spin in diagnostic.atomic_spins if symbol == "Fe"}
    complete = bool(fe_indices) and all(index in spins for index in fe_indices)
    resolved = [spins[index] for index in fe_indices if index in spins and abs(spins[index]) >= threshold]
    positive = sum(spin > 0 for spin in resolved)
    negative = sum(spin < 0 for spin in resolved)
    weak = len(fe_indices) - len(resolved) if complete else 0
    parallel = (
        complete
        and len(resolved) == len(fe_indices)
        and bool(resolved)
        and not (positive and negative)
    )
    mean_local_s = sum(abs(spin) / 2 for spin in resolved) / len(resolved) if resolved else None
    return {
        "fe_count": str(len(fe_indices)),
        "fe_spin_density_complete": str(complete).lower(),
        "fe_resolved_spin_count": str(len(resolved)),
        "fe_positive_spin_count": str(positive),
        "fe_negative_spin_count": str(negative),
        "fe_weak_spin_count": str(weak),
        "fe_all_resolved_parallel": str(parallel).lower(),
        "fe_mean_abs_local_s": "" if mean_local_s is None else f"{mean_local_s:.6g}",
    }


def infer_automatic_fe_spin_plans(
    records: list[Record],
) -> tuple[list[AutomaticSpinPlan], list[SkippedAutomaticSpinPlan]]:
    """Infer oxide spin ladders strictly from archived state evidence.

    A formula/charge group's highest reliably observed multiplicity is accepted
    when its Fe Mulliken moments are all parallel, or as a clearly labeled
    fallback when the group contains multiple observed multiplicities. A
    singleton without parallel-moment evidence is deliberately unplannable; no
    idealized 4*N(Fe)+1 state is invented.
    """
    grouped: dict[tuple[str, int], list[Record]] = {}
    skipped: list[SkippedAutomaticSpinPlan] = []
    for record in records:
        fe_count = sum(atom.symbol == "Fe" for atom in record.atoms)
        group_key = f"{record.formula}|q{record.charge:+d}"
        if not fe_count:
            skipped.append(SkippedAutomaticSpinPlan(
                record, group_key, "no_fe_atoms", (record.multiplicity,), 0
            ))
            continue
        grouped.setdefault((record.formula, record.charge), []).append(record)

    plans: list[AutomaticSpinPlan] = []
    unreliable_inferences = {"default_unmatched_singlet", "electron_parity_fallback"}
    for (formula, charge), members in sorted(grouped.items()):
        group_key = f"{formula}|q{charge:+d}"
        observed = tuple(sorted({record.multiplicity for record in members}, reverse=True))
        multiplicity_errors: dict[str, str] = {}
        for record in members:
            try:
                validate_multiplicity(record, record.multiplicity)
            except ValueError as exc:
                multiplicity_errors[record.record_id] = str(exc)
        parallel = [record for record in members if _metadata_true(record, "fe_all_resolved_parallel")]
        reliable = [
            record for record in members
            if record.metadata.get("state_inference", "") not in unreliable_inferences
            and record.record_id not in multiplicity_errors
        ]
        reliable_multiplicities = sorted(
            {record.multiplicity for record in reliable}, reverse=True
        )
        if not reliable_multiplicities:
            for record in members:
                reason = (
                    "target_multiplicity_not_physically_valid:"
                    f"{multiplicity_errors[record.record_id]}"
                    if record.record_id in multiplicity_errors
                    else "insufficient_real_data_no_parallel_reference"
                )
                skipped.append(SkippedAutomaticSpinPlan(
                    record,
                    group_key,
                    reason,
                    observed,
                    sum(atom.symbol == "Fe" for atom in record.atoms),
                ))
            continue
        high_spin = reliable_multiplicities[0]
        parallel_at_high = [record for record in parallel if record.multiplicity == high_spin]
        if parallel_at_high:
            evidence = tuple(sorted({record.record_id for record in parallel_at_high}))
            inference = "observed_all_resolved_fe_parallel"
        elif len(reliable_multiplicities) >= 2:
            evidence = tuple(sorted({
                record.record_id for record in reliable if record.multiplicity == high_spin
            }))
            inference = "highest_observed_group_multiplicity"
        else:
            for record in members:
                skipped.append(SkippedAutomaticSpinPlan(
                    record,
                    group_key,
                    "insufficient_real_data_no_parallel_reference",
                    observed,
                    sum(atom.symbol == "Fe" for atom in record.atoms),
                ))
            continue

        for record in members:
            fe_count = sum(atom.symbol == "Fe" for atom in record.atoms)
            if record.record_id in multiplicity_errors:
                skipped.append(SkippedAutomaticSpinPlan(
                    record,
                    group_key,
                    "target_multiplicity_not_physically_valid:"
                    f"{multiplicity_errors[record.record_id]}",
                    observed,
                    fe_count,
                ))
                continue
            if record.metadata.get("state_inference", "") in unreliable_inferences:
                skipped.append(SkippedAutomaticSpinPlan(
                    record, group_key, "target_multiplicity_not_reliably_observed", observed, fe_count
                ))
                continue
            try:
                validate_multiplicity(record, high_spin)
                if record.multiplicity > high_spin:
                    raise ValueError("target exceeds inferred high-spin multiplicity")
            except ValueError as exc:
                skipped.append(SkippedAutomaticSpinPlan(
                    record, group_key, f"invalid_data_inferred_ladder:{exc}", observed, fe_count
                ))
                continue
            seed = (
                f"{record.record_id}|{group_key}|m{high_spin}|m{record.multiplicity}|{inference}"
            )
            plans.append(AutomaticSpinPlan(
                plan_id=hashlib.sha1(seed.encode()).hexdigest()[:20],
                record=record,
                group_key=group_key,
                target_multiplicity=record.multiplicity,
                high_spin_multiplicity=high_spin,
                inference=inference,
                evidence_record_ids=evidence,
                observed_multiplicities=observed,
                fe_count=fe_count,
            ))
    return plans, skipped


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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _source_geometry_sha256(record: Record) -> str:
    return _canonical_sha256([
        [atom.symbol, f"{atom.x:.12f}", f"{atom.y:.12f}", f"{atom.z:.12f}"]
        for atom in record.atoms
    ])


def _manifest_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def render_ladder_input(
    record: Record,
    high_spin: int,
    targets: Iterable[int],
    route: str = DEFAULT_SPIN_ROUTE,
    memory: str = "16GB",
    nproc: int = 16,
    provenance: str = "",
) -> tuple[str, list[dict[str, str]]]:
    """Render a checkpoint-preserving one-spin-flip-at-a-time Link1 ladder."""
    sequence = multiplicity_ladder(high_spin, targets)
    for multiplicity in sequence:
        validate_multiplicity(record, multiplicity)
    chain = f"{record.record_id}-ladder-m{high_spin}-m{sequence[-1]}"
    gen_basis = render_gen_basis({atom.symbol for atom in record.atoms}).rstrip("\n")
    parts: list[str] = []
    rows: list[dict[str, str]] = []
    previous_checkpoint: str | None = None
    previous_job_id = ""
    lineage: list[str] = []
    for stage, multiplicity in enumerate(sequence):
        checkpoint = f"{chain}-s{stage:02d}-m{multiplicity}.chk"
        job_id = f"{chain}-s{stage:02d}"
        if stage:
            parts.extend(["", "--Link1--"])
        parts.extend(_link_header(checkpoint, memory, nproc, previous_checkpoint))
        stage_route = route if stage == 0 else _route_with(route, "Geom=Checkpoint", "Guess=(Read,Always)")
        parts.extend([
            stage_route,
            "",
            "ClusterMLIP spin pathway; "
            f"strategy=sequential_spin_flip; record={record.record_id}; stage={stage}; "
            f"multiplicity={multiplicity}; predecessor={previous_job_id or 'none'}"
            f"{'; ' + provenance if provenance else ''}",
            "",
            f"{record.charge} {multiplicity}",
        ])
        if stage == 0:
            parts.extend(_coordinates(record.atoms))
        parts.append(gen_basis)
        parts.append("")
        lineage.append(f"m{multiplicity}:{checkpoint}")
        rows.append({
            "job_id": job_id,
            "chain_id": chain,
            "stage_index": str(stage),
            "pathway": "multiplicity_ladder",
            "initialization": "trusted_high_spin_direct" if stage == 0 else "checkpoint_spin_flip",
            "audit_classification": (
                "trusted_high_spin_reference" if stage == 0 else "sequential_checkpoint_spin_flip"
            ),
            "parent_record_id": record.record_id,
            "source": record.source,
            "formula": record.formula,
            "source_geometry_sha256": _source_geometry_sha256(record),
            "high_spin_multiplicity": str(high_spin),
            "final_target_multiplicity": str(sequence[-1]),
            "intended_charge": str(record.charge),
            "intended_multiplicity": str(multiplicity),
            "spin_flip_index": str(stage),
            "predecessor_job_id": previous_job_id,
            "predecessor_multiplicity": "" if stage == 0 else str(sequence[stage - 1]),
            "predecessor_checkpoint": previous_checkpoint or "",
            "checkpoint": checkpoint,
            "checkpoint_lineage": ">".join(lineage),
            "fragment_label": "",
            "fragment_count": "",
            "fragment_spec_sha256": "",
        })
        previous_checkpoint = checkpoint
        previous_job_id = job_id
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
        "ClusterMLIP spin pathway; strategy=manual_fragment_guess; "
        f"record={record.record_id}; label={label}; multiplicity={target}",
        "",
        f"{record.charge} {target} {fragment_state}",
    ])
    lines.extend(_coordinates(record.atoms, atom_map))
    lines.append(render_gen_basis({atom.symbol for atom in record.atoms}).rstrip("\n"))
    lines.append("")
    row = {
        "job_id": job_id,
        "chain_id": job_id,
        "stage_index": "0",
        "pathway": "fragment_guess",
        "initialization": "explicit_manual_fragment_map",
        "audit_classification": "manual_fragment_preparation",
        "parent_record_id": record.record_id,
        "source": record.source,
        "formula": record.formula,
        "source_geometry_sha256": _source_geometry_sha256(record),
        "high_spin_multiplicity": str(record.multiplicity),
        "final_target_multiplicity": str(target),
        "intended_charge": str(record.charge),
        "intended_multiplicity": str(target),
        "spin_flip_index": "",
        "predecessor_job_id": "",
        "predecessor_multiplicity": "",
        "predecessor_checkpoint": "",
        "checkpoint": checkpoint,
        "checkpoint_lineage": f"fragment:{label}:m{target}:{checkpoint}",
        "fragment_label": label,
        "fragment_count": str(len(states)),
        "fragment_spec_sha256": _canonical_sha256(specification),
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
    strategy: str = "auto",
) -> int:
    if strategy not in {"auto", "ladder", "fragment", "both"}:
        raise ValueError("spin preparation strategy must be auto, ladder, fragment, or both")
    if strategy == "auto":
        strategy = "both" if fragment_specifications else "ladder"
    requested_targets = set(targets)
    if not requested_targets:
        raise ValueError("at least one low-spin target multiplicity is required")
    high_spin_records = [record for record in records if record.multiplicity == high_spin]
    skipped_records = [record for record in records if record.multiplicity != high_spin]
    if not high_spin_records:
        raise ValueError(f"no seed has the requested high-spin multiplicity {high_spin}")
    specs_by_record: dict[str, list[dict]] = {}
    for specification in fragment_specifications or []:
        specs_by_record.setdefault(str(specification["record_id"]), []).append(specification)
    unknown = sorted(set(specs_by_record) - {record.record_id for record in high_spin_records})
    if unknown:
        raise ValueError(f"fragment specifications reference unselected records: {unknown}")
    if strategy in {"fragment", "both"} and not fragment_specifications:
        raise ValueError(f"--strategy {strategy} requires --fragment-spec")
    if strategy == "fragment":
        missing_specs: list[str] = []
        for record in high_spin_records:
            defined = {
                int(specification["target_multiplicity"])
                for specification in specs_by_record.get(record.record_id, [])
            }
            for target in sorted(requested_targets - defined, reverse=True):
                missing_specs.append(f"{record.record_id}:m{target}")
        if missing_specs:
            raise ValueError(
                "fragment-only preparation requires an explicit fragment guess for every requested state; "
                f"missing {missing_specs}"
            )
    if output.exists():
        existing = sorted(path.name for path in output.iterdir())
        if existing:
            preview = ", ".join(existing[:5])
            raise RuntimeError(
                "refusing to overwrite an existing spin campaign; use a fresh output directory "
                f"so checkpoint and audit lineage remain immutable: {preview}"
            )

    # Render every ladder/fragment job in memory first. render_ladder_input
    # and render_fragment_input validate as they go (multiplicity parity,
    # fragment atom coverage, signed-spin consistency, ...); rendering
    # everything before writing anything means a bad spec later in the list
    # raises cleanly instead of leaving a half-written job directory with no
    # spin_jobs.csv to explain what is and is not there.
    files: list[tuple[str, str]] = []
    rows: list[dict[str, str]] = []
    for record in high_spin_records:
        readable = human_job_stem(record)
        if strategy in {"ladder", "both"}:
            text, chain_rows = render_ladder_input(record, high_spin, requested_targets, route, memory, nproc)
            filename = f"{readable}__spin-ladder-m{high_spin}-to-m{min(requested_targets)}.gjf"
            files.append((filename, text))
            for row in chain_rows:
                row["input"] = filename
                row["output"] = f"{Path(filename).stem}.log"
            rows.extend(chain_rows)
        if strategy in {"fragment", "both"}:
            for specification in specs_by_record.get(record.record_id, []):
                text, row = render_fragment_input(record, specification, route, memory, nproc)
                filename = (
                    f"{readable}__spin-fragment-{row['fragment_label']}"
                    f"-m{row['intended_multiplicity']}.gjf"
                )
                files.append((filename, text))
                row["input"] = filename
                row["output"] = f"{Path(filename).stem}.log"
                rows.append(row)

    filenames = [filename for filename, _ in files]
    if len(set(filenames)) != len(filenames):
        duplicates = sorted({name for name in filenames if filenames.count(name) > 1})
        raise ValueError(f"spin specifications produce duplicate input filenames: {duplicates}")
    job_ids = [row["job_id"] for row in rows]
    if len(set(job_ids)) != len(job_ids):
        duplicates = sorted({name for name in job_ids if job_ids.count(name) > 1})
        raise ValueError(f"spin specifications produce duplicate job IDs: {duplicates}")

    output.mkdir(parents=True, exist_ok=True)
    for filename, text in files:
        (output / filename).write_text(text, encoding="utf-8")
    # Hash what was actually written, not the pre-write string: write_text's
    # universal-newline translation turns "\n" into the platform line
    # separator on disk (a no-op on Linux, but "\r\n" on Windows), which
    # would silently desync a hash computed from the in-memory string
    # beforehand from what _spin_manifest_audit reads back later. Reading
    # the file back after writing, like jobs.py's _file_sha256 already does,
    # is correct by construction on every platform.
    written_hashes = {filename: hashlib.sha256((output / filename).read_bytes()).hexdigest() for filename, _ in files}
    for row in rows:
        row["input_sha256"] = written_hashes[row["input"]]
    with (output / "spin_jobs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPIN_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "skipped_spin_seeds.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["record_id", "source", "formula", "charge", "multiplicity", "reason"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in skipped_records:
            writer.writerow({
                "record_id": record.record_id,
                "source": record.source,
                "formula": record.formula,
                "charge": record.charge,
                "multiplicity": record.multiplicity,
                "reason": "direct_low_spin_initialization_prohibited",
            })
    if fragment_specifications:
        locked = {"guesses": fragment_specifications}
        (output / "fragment_specifications.lock.json").write_text(
            json.dumps(locked, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    spin_manifest_sha256 = hashlib.sha256((output / "spin_jobs.csv").read_bytes()).hexdigest()
    campaign = {
        "schema_version": 1,
        "strategy": strategy,
        "trusted_high_spin_multiplicity": high_spin,
        "requested_target_multiplicities": sorted(requested_targets, reverse=True),
        "trusted_high_spin_seed_count": len(high_spin_records),
        "skipped_non_high_spin_seed_count": len(skipped_records),
        "route": route,
        "memory": memory,
        "nproc": nproc,
        "manifest": "spin_jobs.csv",
        "manifest_sha256": spin_manifest_sha256,
        "fragment_specifications": (
            "fragment_specifications.lock.json" if fragment_specifications else None
        ),
        "fragment_specifications_sha256": (
            _canonical_sha256({"guesses": fragment_specifications})
            if fragment_specifications else None
        ),
    }
    (output / "spin_campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "run_one.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ninput=$1\noutput=${input%.gjf}.log\ng16 \"$input\" > \"$output\"\n",
        encoding="utf-8",
    )
    (output / "run_one.sh").chmod(0o755)
    return len(rows)


def write_automatic_fe_spin_jobs(
    records: list[Record],
    output: Path,
    route: str = DEFAULT_SPIN_ROUTE,
    memory: str = "16GB",
    nproc: int = 16,
) -> int:
    """Prepare one data-inferred high-spin-to-archived-target ladder per Fe record."""
    inferred_plans, inferred_skipped = infer_automatic_fe_spin_plans(records)

    # A warehouse often contains the same Gaussian state more than once: for
    # example, duplicate document/log exports or repeated charge/multiplicity
    # sections. The spin inventory intentionally retains those observations,
    # but an automatic campaign must prepare only one deterministic ladder for
    # an identical record/target. Collapsing by plan_id is safe because plan_id
    # includes the record identity, formula/charge group, inferred high spin,
    # target multiplicity, and inference method. Distinct geometries, charges,
    # multiplicities, or preparation decisions therefore remain separate.
    plans_by_id: dict[str, AutomaticSpinPlan] = {}
    for plan in inferred_plans:
        previous = plans_by_id.get(plan.plan_id)
        if previous is not None:
            if _source_geometry_sha256(previous.record) != _source_geometry_sha256(plan.record):
                raise ValueError(
                    f"spin plan identity collision for different geometries: {plan.plan_id}"
                )
            continue
        plans_by_id[plan.plan_id] = plan
    plans = list(plans_by_id.values())

    skipped_by_identity: dict[tuple[str, str], SkippedAutomaticSpinPlan] = {}
    for item in inferred_skipped:
        skipped_by_identity.setdefault((item.record.record_id, item.reason), item)
    skipped = list(skipped_by_identity.values())
    duplicate_plans_collapsed = len(inferred_plans) - len(plans)
    duplicate_skips_collapsed = len(inferred_skipped) - len(skipped)
    if output.exists():
        existing = sorted(path.name for path in output.iterdir())
        if existing:
            preview = ", ".join(existing[:5])
            raise RuntimeError(
                "refusing to overwrite an existing spin campaign; use a fresh output directory "
                f"so checkpoint and audit lineage remain immutable: {preview}"
            )

    files: list[tuple[str, str]] = []
    rows: list[dict[str, str]] = []
    plan_rows: list[dict[str, object]] = []
    for plan in plans:
        targets = [] if plan.target_multiplicity == plan.high_spin_multiplicity else [plan.target_multiplicity]
        text, chain_rows = render_ladder_input(
            plan.record,
            plan.high_spin_multiplicity,
            targets,
            route,
            memory,
            nproc,
            provenance=(
                f"spin_plan_id={plan.plan_id}; high_spin_inference={plan.inference}"
            ),
        )
        readable = human_job_stem(plan.record)
        filename = (
            f"{readable}__auto-spin-m{plan.high_spin_multiplicity}"
            f"-to-m{plan.target_multiplicity}.gjf"
        )
        evidence_ids = ";".join(plan.evidence_record_ids)
        for row in chain_rows:
            row.update({
                "spin_plan_id": plan.plan_id,
                "spin_group_key": plan.group_key,
                "target_record_multiplicity": str(plan.target_multiplicity),
                "high_spin_inference": plan.inference,
                "high_spin_evidence_record_ids": evidence_ids,
                "input": filename,
                "output": f"{Path(filename).stem}.log",
            })
            if row["stage_index"] == "0":
                row["initialization"] = "data_inferred_high_spin_direct_on_target_geometry"
                row["audit_classification"] = "data_inferred_high_spin_reference"
        files.append((filename, text))
        rows.extend(chain_rows)
        plan_rows.append({
            "spin_plan_id": plan.plan_id,
            "status": "planned",
            "record_id": plan.record.record_id,
            "source": plan.record.source,
            "formula": plan.record.formula,
            "charge": plan.record.charge,
            "fe_count": plan.fe_count,
            "target_multiplicity": plan.target_multiplicity,
            "inferred_high_spin_multiplicity": plan.high_spin_multiplicity,
            "high_spin_inference": plan.inference,
            "high_spin_evidence_record_ids": evidence_ids,
            "observed_group_multiplicities": ";".join(map(str, plan.observed_multiplicities)),
            "input": filename,
            "reason": "",
        })
    for item in skipped:
        plan_rows.append({
            "spin_plan_id": "",
            "status": "skipped",
            "record_id": item.record.record_id,
            "source": item.record.source,
            "formula": item.record.formula,
            "charge": item.record.charge,
            "fe_count": item.fe_count,
            "target_multiplicity": item.record.multiplicity,
            "inferred_high_spin_multiplicity": "",
            "high_spin_inference": "",
            "high_spin_evidence_record_ids": "",
            "observed_group_multiplicities": ";".join(map(str, item.observed_multiplicities)),
            "input": "",
            "reason": item.reason,
        })

    filenames = [filename for filename, _ in files]
    if len(set(filenames)) != len(filenames):
        duplicates = sorted({name for name in filenames if filenames.count(name) > 1})
        raise ValueError(f"automatic spin planning produced duplicate filenames: {duplicates}")
    job_ids = [row["job_id"] for row in rows]
    if len(set(job_ids)) != len(job_ids):
        duplicates = sorted({name for name in job_ids if job_ids.count(name) > 1})
        raise ValueError(f"automatic spin planning produced duplicate job IDs: {duplicates}")

    output.mkdir(parents=True, exist_ok=True)
    for filename, text in files:
        (output / filename).write_text(text, encoding="utf-8")
    # See write_spin_jobs for why this hashes the file after writing rather
    # than the pre-write string.
    written_hashes = {filename: hashlib.sha256((output / filename).read_bytes()).hexdigest() for filename, _ in files}
    for row in rows:
        row["input_sha256"] = written_hashes[row["input"]]
    with (output / "spin_jobs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPIN_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    plan_columns = [
        "spin_plan_id", "status", "record_id", "source", "formula", "charge", "fe_count",
        "target_multiplicity", "inferred_high_spin_multiplicity", "high_spin_inference",
        "high_spin_evidence_record_ids", "observed_group_multiplicities", "input", "reason",
    ]
    with (output / "spin_plan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=plan_columns)
        writer.writeheader()
        writer.writerows(plan_rows)
    plan_summary = {
        "total_archive_records": len(records),
        "unique_archive_records": len(plan_rows),
        "duplicate_planned_records_collapsed": duplicate_plans_collapsed,
        "duplicate_skipped_records_collapsed": duplicate_skips_collapsed,
        "planned_records": len(plans),
        "skipped_records": len(skipped),
        "by_inference": dict(sorted(collections.Counter(
            plan.inference for plan in plans
        ).items())),
        "by_skip_reason": dict(sorted(collections.Counter(
            item.reason for item in skipped
        ).items())),
        "by_formula_charge_group": dict(sorted(collections.Counter(
            plan.group_key for plan in plans
        ).items())),
        "by_ladder": dict(sorted(collections.Counter(
            f"m{plan.high_spin_multiplicity}->m{plan.target_multiplicity}"
            for plan in plans
        ).items())),
    }
    (output / "spin_plan_summary.json").write_text(
        json.dumps(plan_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "skipped_spin_seeds.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["record_id", "source", "formula", "charge", "multiplicity", "reason"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in skipped:
            writer.writerow({
                "record_id": item.record.record_id,
                "source": item.record.source,
                "formula": item.record.formula,
                "charge": item.record.charge,
                "multiplicity": item.record.multiplicity,
                "reason": item.reason,
            })
    manifest_hash = hashlib.sha256((output / "spin_jobs.csv").read_bytes()).hexdigest()
    plan_hash = hashlib.sha256((output / "spin_plan.csv").read_bytes()).hexdigest()
    plan_summary_hash = hashlib.sha256(
        (output / "spin_plan_summary.json").read_bytes()
    ).hexdigest()
    campaign = {
        "schema_version": 2,
        "strategy": "automatic_data_inferred_ladder",
        "automatic_from_real_data": True,
        "planned_parent_count": len(plans),
        "skipped_parent_count": len(skipped),
        "duplicate_planned_records_collapsed": duplicate_plans_collapsed,
        "duplicate_skipped_records_collapsed": duplicate_skips_collapsed,
        "high_spin_inference_precedence": [
            "observed_all_resolved_fe_parallel",
            "highest_observed_group_multiplicity",
            "skip_if_insufficient_real_data",
        ],
        "idealized_per_fe_high_spin_used": False,
        "route": route,
        "memory": memory,
        "nproc": nproc,
        "manifest": "spin_jobs.csv",
        "manifest_sha256": manifest_hash,
        "spin_plan": "spin_plan.csv",
        "spin_plan_sha256": plan_hash,
        "spin_plan_summary": "spin_plan_summary.json",
        "spin_plan_summary_sha256": plan_summary_hash,
    }
    (output / "spin_campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
    for observation in observations:
        record = observation.record
        try:
            validate_multiplicity(record, record.multiplicity)
            parity_valid = True
        except ValueError:
            parity_valid = False
        record.metadata.update({
            **fe_spin_summary(record, observation.diagnostics),
            "electron_count": str(electron_count(record)),
            "multiplicity_parity_valid": str(parity_valid).lower(),
        })
    write_extxyz([observation.record for observation in observations], output / "seeds.extxyz")
    columns = [
        "record_id", "source", "formula", "n_atoms", "charge", "multiplicity", "electron_count",
        "multiplicity_parity_valid", "config_type", "state_inference", "energy_hartree", "expected_s2",
        "s2_before", "s2_after", "s2_delta", "spin_pattern", "root_signature", "normal_termination",
        "optimized", "stability", "fe_count", "fe_spin_density_complete",
        "fe_resolved_spin_count", "fe_positive_spin_count", "fe_negative_spin_count",
        "fe_weak_spin_count", "fe_all_resolved_parallel", "fe_mean_abs_local_s",
    ]
    with (output / "spin_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for observation in observations:
            record, diagnostic = observation.record, observation.diagnostics
            writer.writerow({
                "record_id": record.record_id, "source": record.source, "formula": record.formula,
                "n_atoms": len(record.atoms), "charge": record.charge, "multiplicity": record.multiplicity,
                "electron_count": electron_count(record),
                "multiplicity_parity_valid": record.metadata["multiplicity_parity_valid"] == "true",
                "config_type": record.config_type,
                "state_inference": record.metadata.get("state_inference", ""),
                "energy_hartree": diagnostic.energy_hartree,
                "expected_s2": diagnostic.expected_s2, "s2_before": diagnostic.s2_before,
                "s2_after": diagnostic.s2_after, "s2_delta": diagnostic.s2_delta,
                "spin_pattern": diagnostic.spin_pattern, "root_signature": diagnostic.root_signature,
                "normal_termination": diagnostic.normal_termination, "optimized": diagnostic.optimized,
                "stability": diagnostic.stability,
                **{
                    key: record.metadata.get(key, "")
                    for key in (
                        "fe_count", "fe_spin_density_complete", "fe_resolved_spin_count",
                        "fe_positive_spin_count", "fe_negative_spin_count", "fe_weak_spin_count",
                        "fe_all_resolved_parallel", "fe_mean_abs_local_s",
                    )
                },
            })
    with (output / "errors.tsv").open("w", encoding="utf-8") as handle:
        for name, message in errors:
            handle.write(f"{name}\t{message}\n")
    return len(observations)


def _spin_manifest_audit(
    manifest: Path,
) -> tuple[list[dict[str, str]], dict[str, str], list[str]]:
    """Validate preparation provenance without trusting filenames or outputs."""
    errors: list[str] = []
    campaign: dict[str, object] = {}
    if not manifest.is_file():
        return [], {}, [f"missing spin manifest: {manifest}"]
    campaign_path = manifest.parent / "spin_campaign.json"
    if not campaign_path.is_file():
        errors.append(f"missing spin campaign metadata: {campaign_path}")
    else:
        try:
            campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
            actual_manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            if campaign.get("manifest_sha256") != actual_manifest_hash:
                errors.append("spin_jobs.csv SHA-256 does not match spin_campaign.json")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read spin campaign metadata {campaign_path}: {exc}")
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(set(SPIN_MANIFEST_COLUMNS) - fields)
        if missing:
            errors.append(f"spin manifest is missing audit columns: {missing}")
        rows = list(reader)

    planned_spin_ids: set[str] = set()
    spin_plan_name = campaign.get("spin_plan")
    if isinstance(spin_plan_name, str):
        spin_plan = manifest.parent / spin_plan_name
        if not spin_plan.is_file():
            errors.append(f"missing automatic spin plan: {spin_plan}")
        else:
            actual_plan_hash = hashlib.sha256(spin_plan.read_bytes()).hexdigest()
            if campaign.get("spin_plan_sha256") != actual_plan_hash:
                errors.append("spin_plan.csv SHA-256 does not match spin_campaign.json")
            with spin_plan.open(newline="", encoding="utf-8") as handle:
                planned_spin_ids = {
                    row.get("spin_plan_id", "")
                    for row in csv.DictReader(handle)
                    if row.get("status") == "planned"
                }
    plan_summary_name = campaign.get("spin_plan_summary")
    if isinstance(plan_summary_name, str):
        plan_summary = manifest.parent / plan_summary_name
        if not plan_summary.is_file():
            errors.append(f"missing automatic spin plan summary: {plan_summary}")
        elif campaign.get("spin_plan_summary_sha256") != hashlib.sha256(
            plan_summary.read_bytes()
        ).hexdigest():
            errors.append("spin_plan_summary.json SHA-256 does not match spin_campaign.json")

    by_job = {row.get("job_id", ""): row for row in rows}
    if "" in by_job:
        errors.append("spin manifest contains a row without job_id")
    if len(by_job) != len(rows):
        errors.append("spin manifest contains duplicate job_id values")

    locked_hashes: set[str] = set()
    lock = manifest.parent / "fragment_specifications.lock.json"
    if lock.is_file():
        try:
            payload = json.loads(lock.read_text(encoding="utf-8"))
            guesses = payload.get("guesses", payload) if isinstance(payload, dict) else payload
            if isinstance(guesses, list):
                locked_hashes = {_canonical_sha256(specification) for specification in guesses}
            else:
                errors.append(f"fragment lock has no guesses list: {lock}")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read fragment lock {lock}: {exc}")

    status_by_job: dict[str, str] = {}
    ordered = sorted(
        rows,
        key=lambda row: (row.get("chain_id", ""), _manifest_int(row.get("stage_index"))),
    )
    for row in ordered:
        job_id = row.get("job_id", "")
        problems: list[str] = []
        if len(row.get("source_geometry_sha256", "")) != 64:
            problems.append("source_geometry_hash_missing")
        input_path = manifest.parent / row.get("input", "")
        expected_input_hash = row.get("input_sha256", "")
        if not input_path.is_file():
            problems.append("input_file_missing")
        elif len(expected_input_hash) != 64:
            problems.append("input_hash_missing")
        elif hashlib.sha256(input_path.read_bytes()).hexdigest() != expected_input_hash:
            problems.append("input_hash_mismatch")
        if row.get("high_spin_inference") and row.get("spin_plan_id") not in planned_spin_ids:
            problems.append("spin_plan_crosswalk_missing")
        pathway = row.get("pathway", "")
        stage = _manifest_int(row.get("stage_index"))
        if pathway == "fragment_guess":
            if row.get("initialization") != "explicit_manual_fragment_map":
                problems.append("fragment_initialization_unmarked")
            if row.get("audit_classification") != "manual_fragment_preparation":
                problems.append("fragment_audit_classification_unmarked")
            spec_hash = row.get("fragment_spec_sha256", "")
            if len(spec_hash) != 64:
                problems.append("fragment_spec_hash_missing")
            elif spec_hash not in locked_hashes:
                problems.append("fragment_spec_not_in_lock")
            if row.get("predecessor_job_id") or row.get("predecessor_checkpoint"):
                problems.append("fragment_has_checkpoint_predecessor")
        elif pathway == "multiplicity_ladder":
            intended = _manifest_int(row.get("intended_multiplicity"))
            high = _manifest_int(row.get("high_spin_multiplicity"))
            predecessor_id = row.get("predecessor_job_id", "")
            if stage == 0:
                if intended != high:
                    problems.append("direct_state_is_not_trusted_high_spin")
                inferred = row.get("initialization") == "data_inferred_high_spin_direct_on_target_geometry"
                if inferred:
                    if row.get("audit_classification") != "data_inferred_high_spin_reference":
                        problems.append("inferred_high_spin_audit_classification_unmarked")
                    if row.get("spin_plan_id") not in planned_spin_ids:
                        problems.append("inferred_high_spin_plan_missing")
                    if row.get("high_spin_inference") not in {
                        "observed_all_resolved_fe_parallel",
                        "highest_observed_group_multiplicity",
                    }:
                        problems.append("inferred_high_spin_evidence_unrecognized")
                    if not row.get("high_spin_evidence_record_ids"):
                        problems.append("inferred_high_spin_evidence_missing")
                else:
                    if row.get("initialization") != "trusted_high_spin_direct":
                        problems.append("high_spin_initialization_unmarked")
                    if row.get("audit_classification") != "trusted_high_spin_reference":
                        problems.append("high_spin_audit_classification_unmarked")
                if predecessor_id or row.get("predecessor_checkpoint"):
                    problems.append("high_spin_reference_has_predecessor")
                expected_lineage = f"m{intended}:{row.get('checkpoint', '')}"
            else:
                predecessor = by_job.get(predecessor_id)
                if predecessor is None:
                    problems.append("predecessor_job_missing")
                    expected_lineage = ""
                else:
                    predecessor_stage = _manifest_int(predecessor.get("stage_index"), -1)
                    predecessor_multiplicity = _manifest_int(
                        predecessor.get("intended_multiplicity")
                    )
                    if predecessor.get("chain_id") != row.get("chain_id"):
                        problems.append("predecessor_chain_mismatch")
                    if predecessor_stage != stage - 1:
                        problems.append("predecessor_stage_not_immediate")
                    if predecessor_multiplicity - intended != 2:
                        problems.append("not_one_spin_flip")
                    if row.get("predecessor_multiplicity") != str(predecessor_multiplicity):
                        problems.append("predecessor_multiplicity_mismatch")
                    if row.get("predecessor_checkpoint") != predecessor.get("checkpoint"):
                        problems.append("predecessor_checkpoint_mismatch")
                    expected_lineage = (
                        f"{predecessor.get('checkpoint_lineage', '')}>"
                        f"m{intended}:{row.get('checkpoint', '')}"
                    )
                if row.get("initialization") != "checkpoint_spin_flip":
                    problems.append("spin_flip_initialization_unmarked")
                if row.get("audit_classification") != "sequential_checkpoint_spin_flip":
                    problems.append("spin_flip_audit_classification_unmarked")
            if row.get("checkpoint_lineage") != expected_lineage:
                problems.append("checkpoint_lineage_mismatch")
        else:
            problems.append("unknown_pathway")
        status_by_job[job_id] = "verified" if not problems else ";".join(problems)
    return rows, status_by_job, errors


def _locked_fragment_specifications(campaign: Path) -> dict[str, dict]:
    lock = campaign / "fragment_specifications.lock.json"
    if not lock.is_file():
        return {}
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    guesses = payload.get("guesses", payload) if isinstance(payload, dict) else payload
    if not isinstance(guesses, list):
        return {}
    return {
        _canonical_sha256(specification): specification
        for specification in guesses
        if isinstance(specification, dict)
    }


def _fragment_spin_alignment(diagnostic: SpinDiagnostics, specification: dict | None) -> str:
    """Compare converged net fragment spins with the requested alpha/beta orientations."""
    if specification is None:
        return "specification_unavailable"
    if not diagnostic.atomic_spins:
        return "spin_density_unavailable"
    spins = {index: spin for index, _, spin in diagnostic.atomic_spins}
    unresolved = False
    for fragment in specification.get("fragments", []):
        if int(fragment.get("multiplicity", 1)) <= 1:
            continue
        expected = _orientation_sign(fragment.get("orientation", "alpha"))
        indices = [int(index) for index in fragment.get("atoms", [])]
        if any(index not in spins for index in indices):
            unresolved = True
            continue
        net_spin = sum(spins[index] for index in indices)
        if abs(net_spin) < 0.10:
            unresolved = True
        elif (net_spin > 0) != (expected > 0):
            return "mismatch"
    return "unresolved" if unresolved else "matched"


def validate_spin_campaign(
    original: Path,
    new_outputs: Path,
    output: Path,
    geometry_tolerance: float = 0.05,
    spin_tolerance: float = 0.25,
    s2_tolerance: float = 0.25,
) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    manifest = new_outputs / "spin_jobs.csv"
    manifest_rows, lineage_by_job, manifest_errors = _spin_manifest_audit(manifest)
    fragment_specs_by_hash = _locked_fragment_specifications(new_outputs)
    manifest_lookup = {
        (Path(row.get("output", "")).name, _manifest_int(row.get("intended_charge")),
         _manifest_int(row.get("intended_multiplicity"))): row
        for row in manifest_rows
    }
    old, old_errors = _load_observations(original)
    new, new_errors = _load_observations(new_outputs, outputs_only=True)

    def provenance(observation: StateObservation) -> dict[str, str] | None:
        key = (
            Path(observation.record.source).name,
            observation.record.charge,
            observation.record.multiplicity,
        )
        return manifest_lookup.get(key)
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
        prepared = None if current is None else provenance(current)
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
            "new_pathway": "" if prepared is None else prepared.get("pathway", ""),
            "new_initialization": "untracked" if prepared is None else prepared.get("initialization", ""),
            "new_audit_classification": (
                "untracked_direct_or_external"
                if prepared is None else prepared.get("audit_classification", "")
            ),
            "new_chain_id": "" if prepared is None else prepared.get("chain_id", ""),
            "new_stage_index": "" if prepared is None else prepared.get("stage_index", ""),
            "new_predecessor_job_id": "" if prepared is None else prepared.get("predecessor_job_id", ""),
            "new_predecessor_checkpoint": (
                "" if prepared is None else prepared.get("predecessor_checkpoint", "")
            ),
            "new_checkpoint": "" if prepared is None else prepared.get("checkpoint", ""),
            "new_checkpoint_lineage": (
                "" if prepared is None else prepared.get("checkpoint_lineage", "")
            ),
            "new_lineage_status": (
                "untracked" if prepared is None else lineage_by_job.get(prepared.get("job_id", ""), "invalid")
            ),
        })
    with (output / "legacy_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]) if coverage_rows else ["status"])
        writer.writeheader()
        writer.writerows(coverage_rows)
    new_rows = []
    for index, observation in enumerate(new):
        diagnostic = observation.diagnostics
        prepared = provenance(observation)
        fragment_alignment = (
            "not_applicable"
            if prepared is None or prepared.get("pathway") != "fragment_guess"
            else _fragment_spin_alignment(
                diagnostic,
                fragment_specs_by_hash.get(prepared.get("fragment_spec_sha256", "")),
            )
        )
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
            "pathway": "untracked" if prepared is None else prepared.get("pathway", ""),
            "initialization": "untracked" if prepared is None else prepared.get("initialization", ""),
            "audit_classification": (
                "untracked_direct_or_external"
                if prepared is None else prepared.get("audit_classification", "")
            ),
            "chain_id": "" if prepared is None else prepared.get("chain_id", ""),
            "stage_index": "" if prepared is None else prepared.get("stage_index", ""),
            "predecessor_job_id": "" if prepared is None else prepared.get("predecessor_job_id", ""),
            "predecessor_checkpoint": (
                "" if prepared is None else prepared.get("predecessor_checkpoint", "")
            ),
            "checkpoint": "" if prepared is None else prepared.get("checkpoint", ""),
            "checkpoint_lineage": (
                "" if prepared is None else prepared.get("checkpoint_lineage", "")
            ),
            "lineage_status": (
                "untracked" if prepared is None else lineage_by_job.get(prepared.get("job_id", ""), "invalid")
            ),
            "fragment_spin_alignment": fragment_alignment,
        })
    with (output / "new_states.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(new_rows[0]) if new_rows else ["record_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(new_rows)
    planned_rows: list[dict[str, object]] = []
    if manifest_rows:
        observed = {
            (
                Path(observation.record.source).name,
                observation.record.charge,
                observation.record.multiplicity,
            ): observation
            for observation in new
        }
        rows_by_job = {row.get("job_id", ""): row for row in manifest_rows}
        observation_by_job: dict[str, StateObservation | None] = {}
        for row in manifest_rows:
            observation_by_job[row.get("job_id", "")] = observed.get((
                Path(row.get("output", "")).name,
                _manifest_int(row.get("intended_charge")),
                _manifest_int(row.get("intended_multiplicity")),
            ))
        complete_by_job: dict[str, bool] = {}

        def pathway_complete(job_id: str, visiting: set[str] | None = None) -> bool:
            if job_id in complete_by_job:
                return complete_by_job[job_id]
            visiting = set() if visiting is None else visiting
            if job_id in visiting:
                complete_by_job[job_id] = False
                return False
            visiting.add(job_id)
            row = rows_by_job.get(job_id)
            observation = observation_by_job.get(job_id)
            diagnostic = None if observation is None else observation.diagnostics
            predecessor_id = "" if row is None else row.get("predecessor_job_id", "")
            predecessor_complete = (
                True if not predecessor_id else pathway_complete(predecessor_id, visiting)
            )
            complete_by_job[job_id] = bool(
                row is not None
                and diagnostic is not None
                and diagnostic.normal_termination
                and diagnostic.optimized
                and predecessor_complete
                and lineage_by_job.get(job_id) == "verified"
            )
            visiting.remove(job_id)
            return complete_by_job[job_id]

        for row in manifest_rows:
            row_observation = observation_by_job.get(row.get("job_id", ""))
            row_diagnostic = None if row_observation is None else row_observation.diagnostics
            predecessor_id = row.get("predecessor_job_id", "")
            predecessor_complete = (
                True if not predecessor_id else pathway_complete(predecessor_id)
            )
            complete = pathway_complete(row.get("job_id", ""))
            fragment_alignment = (
                "not_applicable"
                if row.get("pathway") != "fragment_guess" or row_diagnostic is None
                else _fragment_spin_alignment(
                    row_diagnostic,
                    fragment_specs_by_hash.get(row.get("fragment_spec_sha256", "")),
                )
            )
            planned_rows.append({
                **{column: row.get(column, "") for column in SPIN_MANIFEST_COLUMNS},
                "lineage_status": lineage_by_job.get(row.get("job_id", ""), "invalid"),
                "observed": row_observation is not None,
                "normal_termination": False if row_diagnostic is None else row_diagnostic.normal_termination,
                "optimized": False if row_diagnostic is None else row_diagnostic.optimized,
                "stability": "" if row_diagnostic is None else row_diagnostic.stability,
                "spin_pattern": "" if row_diagnostic is None else row_diagnostic.spin_pattern,
                "root_signature": "" if row_diagnostic is None else row_diagnostic.root_signature,
                "s2_delta": "" if row_diagnostic is None else row_diagnostic.s2_delta,
                "electronic_root_characterized": bool(
                    row_diagnostic is not None
                    and row_diagnostic.root_signature
                    and (
                        row_diagnostic.s2_before is not None
                        or row_diagnostic.s2_after is not None
                    )
                ),
                "fragment_spin_alignment": fragment_alignment,
                "predecessor_complete": predecessor_complete,
                "pathway_complete_through_stage": complete,
            })
    with (output / "planned_state_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(planned_rows[0]) if planned_rows else ["job_id", "lineage_status", "observed"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(planned_rows)
    planned_missing = sum(not bool(row["observed"]) for row in planned_rows)
    lineage_errors = sum(status != "verified" for status in lineage_by_job.values()) + len(manifest_errors)
    untracked_new_states = sum(provenance(observation) is None for observation in new)
    planned_stages_incomplete = sum(
        bool(row["observed"]) and not bool(row["pathway_complete_through_stage"])
        for row in planned_rows
    )
    planned_states_uncharacterized = sum(
        bool(row["observed"]) and not bool(row["electronic_root_characterized"])
        for row in planned_rows
    )
    fragment_alignment_mismatches = sum(
        row["fragment_spin_alignment"] == "mismatch" for row in planned_rows
    )
    fragment_alignment_unresolved = sum(
        row["fragment_spin_alignment"]
        in {"unresolved", "spin_density_unavailable", "specification_unavailable"}
        for row in planned_rows
    )
    planned_states_without_stability = sum(
        bool(row["observed"]) and row["stability"] != "stable" for row in planned_rows
    )
    summary = {
        "legacy_records": len(old), "new_records": len(new), **counts,
        "novel_or_unmatched_new": len(new) - len(matched_new),
        "planned_stages": len(planned_rows), "planned_stages_missing": planned_missing,
        "lineage_errors": lineage_errors, "untracked_new_states": untracked_new_states,
        "planned_stages_incomplete": planned_stages_incomplete,
        "planned_states_uncharacterized": planned_states_uncharacterized,
        "fragment_alignment_mismatches": fragment_alignment_mismatches,
        "fragment_alignment_unresolved": fragment_alignment_unresolved,
        "planned_states_without_stability": planned_states_without_stability,
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
        f"- Spin-pathway lineage errors: {lineage_errors}",
        f"- New states not linked to the spin manifest: {untracked_new_states}",
        f"- Observed planned stages incomplete or blocked by a predecessor: {planned_stages_incomplete}",
        f"- Observed planned states without local-spin plus <S^2> characterization: {planned_states_uncharacterized}",
        f"- Fragment candidates with converged spin signs opposing the manual map: {fragment_alignment_mismatches}",
        f"- Fragment candidates whose requested orientation could not be resolved: {fragment_alignment_unresolved}",
        f"- Observed planned states without a stable-wavefunction result: {planned_states_without_stability}",
        f"- Novel or unmatched new candidates retained: {summary['novel_or_unmatched_new']}",
        "",
        "`alternative_root` is not discarded. It marks a geometrically matching charge/multiplicity state with a materially different local-spin distribution.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "errors.tsv").open("w", encoding="utf-8") as handle:
        for message in manifest_errors:
            handle.write(f"manifest\t{manifest}\t{message}\n")
        for job_id, status in lineage_by_job.items():
            if status != "verified":
                handle.write(f"lineage\t{job_id}\t{status}\n")
        for category, errors in (("legacy", old_errors), ("new", new_errors)):
            for name, message in errors:
                handle.write(f"{category}\t{name}\t{message}\n")
    return summary
