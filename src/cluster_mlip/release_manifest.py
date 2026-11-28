"""Qualify collected labels for a versioned training release.

`collect` decides which force frames can be parsed into labels. This module
decides which of those labels may be *released* for training, and records why,
one row per frame:

- **provenance**: the run (job) and frame, its Gaussian output and that file's
  sha256, the Link1 section the frame came from and the route Gaussian actually
  echoed for it, the parent warehouse record and its geometry hash, and the
  restart lineage the campaign manifest recorded;
- **state**: charge and multiplicity checked against the stage's declared
  (intended) state, <S^2> before and after annihilation against S(S+1), and the
  frame's own Mulliken spin sum;
- **convergence**: the frame's own SCF, whether its Link1 section reached a
  stationary point, and whether that section terminated normally;
- **decision**: include, hold or reject, with every reason that applies.

Rules (each is a reason; reject beats hold beats include):

- reject ``non_finite_label``: an energy or force component that is not finite;
- reject ``scf_not_converged``: the frame's own SCF printed a convergence
  failure (IOP(5/13=1) lets Gaussian continue past it; the label is not used);
- reject ``state_mismatch``: the frame's charge or multiplicity differs from
  the stage's intended charge/multiplicity;
- reject ``duplicate_label``: an identical label (state, geometry, energy,
  forces) already kept from another output;
- hold ``source_unavailable``: the Gaussian output cannot be read, so the label
  cannot be traced to a hashed file;
- hold ``no_s2``: no <S^2> printed for the frame;
- hold ``s2_nearer_other_state``: <S^2> after annihilation is nearer the
  S'(S'+1) of another spin state than of the declared one;
- hold ``s2_excess``: optional absolute cap (``max_s2_excess``);
- hold ``local_spin_sum_inconsistent``: the Mulliken spin sum differs from 2S;
- hold ``force_outlier``: a force component above ``max_force_ev_ang``;
- hold ``other_level``: a method/basis other than the release level (kept as a
  separate set, never mixed);
- hold ``route_level_mismatch``: the route echoed for the frame's section
  requests a different method/basis than the manifest's route;
- hold ``conflict_same_state``: another label for the same formula, charge,
  multiplicity, level and geometry has a different energy or forces (all of
  them are quarantined);
- hold ``ts_claim_unverified``: the frame is labelled as a saddle endpoint, but
  no frequency analysis of order 1 and no pathway evidence were supplied;
- hold ``split_leak``: frames of one parent geometry, split group or restart
  lineage sit in different splits.

A frame from an optimization that did not finish is not rejected for that: an
SCF-converged force frame with a clean state is a label of *that* geometry. It
is never a stationary point; ``stationary_claim`` says ``none`` for it.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .dataset import read_labeled_extxyz, write_labeled_extxyz
from .label_levels import route_label_level
from .models import LabeledFrame, geometry_signature

ENERGY_UNIT = "eV"
FORCE_UNIT = "eV/Angstrom"

MANIFEST_COLUMNS = [
    "frame_key", "run_id", "record_id", "chain_id", "batch", "gaussian_output",
    "source_sha256", "link1_section", "stage_index", "route", "method", "basis",
    "label_level", "formula", "chemistry", "charge", "multiplicity",
    "intended_charge", "intended_multiplicity", "s2_before", "s2_after", "s2_expected",
    "s2_nearest_multiplicity", "local_spin_consistent", "mulliken_spin_sum_error",
    "fe_moment_pattern", "energy", "energy_unit", "max_force", "rms_force", "force_unit",
    "scf_converged", "section_optimized", "section_normal_termination", "frame_role",
    "config_type", "stationary_claim", "parent_record_id", "source_geometry_sha256",
    "split", "split_group", "restart_of_job_id", "restart_attempt",
    "restart_source_output", "restart_seed_sha256", "label_sha256", "collect_dir",
    "decision", "reasons",
]

REJECT_REASONS = {"non_finite_label", "scf_not_converged", "state_mismatch", "duplicate_label"}

SADDLE_CONFIG_TYPES = {"transition_state", "first_order_saddle", "saddle", "higher_order_saddle"}


@dataclass
class ReleasePolicy:
    level: str | None = None
    max_force_ev_ang: float = 20.0
    max_s2_excess: float | None = None
    energy_tolerance_ev: float = 1e-5
    force_tolerance_ev_ang: float = 1e-4
    ts_evidence: set[str] = field(default_factory=set)
    require_families: tuple[str, ...] = ()
    min_frames_per_family: int = 1
    min_release_frames: int = 1


def chemistry_family(formula_symbols: Iterable[str]) -> str:
    """'Fe' for bare iron, else 'Fe' plus the other elements in alphabetical
    order: FeN, FeO, FeNO, FeHO ..."""
    symbols = set(formula_symbols)
    others = sorted(symbols - {"Fe"})
    return ("Fe" if "Fe" in symbols else "") + "".join(others)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _float_or_none(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _int_or_none(value: Any) -> int | None:
    text = str(value).strip() if value is not None else ""
    return int(text) if re.fullmatch(r"[+-]?\d+", text) else None


def nearest_multiplicity(s2: float, multiplicity: int) -> int:
    """The multiplicity (same electron-count parity) whose S(S+1) is nearest s2."""
    best = multiplicity
    best_gap = abs(s2 - ((multiplicity - 1) / 2) * ((multiplicity + 1) / 2))
    candidate = multiplicity % 2 or 2
    while True:
        spin = (candidate - 1) / 2
        gap = abs(s2 - spin * (spin + 1))
        if gap < best_gap - 1e-9:
            best, best_gap = candidate, gap
        if spin * (spin + 1) > s2 + 1 and candidate > multiplicity:
            break
        candidate += 2
    return best


def label_sha256(frame: LabeledFrame) -> str:
    record = frame.record
    parts = [f"{record.charge} {record.multiplicity} {frame.energy_ev:.10f}"]
    for atom, force in zip(record.atoms, frame.forces_ev_ang):
        parts.append(f"{atom.symbol} {atom.x:.8f} {atom.y:.8f} {atom.z:.8f} "
                     f"{force[0]:.8f} {force[1]:.8f} {force[2]:.8f}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _batch(path_text: str) -> str:
    match = re.search(r"batch_(\d+)", path_text)
    return match.group(1) if match else ""


def _method_basis(route: str) -> tuple[str, str, str]:
    if not route:
        return "", "", ""
    level = route_label_level(route)
    return level.functional, level.basis, level.key


def _frame_key(metadata: dict, record_id: str) -> str:
    job = str(metadata.get("job_id") or record_id)
    index = metadata.get("force_frame_index")
    section = metadata.get("gaussian_section_index")
    return f"{job}#s{section if section is not None else '?'}f{index if index is not None else '?'}"


def _stationary_claim(metadata: dict, config_type: str, ts_evidence: set[str],
                      keys: Iterable[str]) -> str:
    if metadata.get("frame_role") != "optimized_endpoint" or not metadata.get("section_optimized"):
        return "none"
    if config_type in SADDLE_CONFIG_TYPES:
        order = _int_or_none(metadata.get("imaginary_frequencies"))
        evidence = any(key in ts_evidence for key in keys if key)
        if order == 1 and evidence:
            return "saddle_order1_verified"
        return "saddle_unverified"
    if config_type == "minimum":
        return "minimum"
    return "optimized_unverified"


def _lineage_root(job: str, parents: dict[str, str]) -> str:
    seen = set()
    while job in parents and parents[job] and job not in seen:
        seen.add(job)
        job = parents[job]
    return job


def build_release(
    collect_dirs: list[Path],
    output: Path,
    policy: ReleasePolicy,
    campaign_root: Path | None = None,
) -> dict[str, Any]:
    """Write release_manifest.tsv, counts_by_chemistry.tsv, release_*.extxyz and
    release_summary.json under *output*; return the summary."""
    loaded: list[tuple[Path, str, LabeledFrame]] = []
    for directory in collect_dirs:
        # collect writes every frame to all.extxyz and again to exactly one of
        # its split files; the split files carry the split.
        split_files = [(name, directory / f"{name}.extxyz") for name in ("train", "valid", "test")]
        if any(path.exists() for _, path in split_files):
            for name, path in split_files:
                if path.exists():
                    loaded.extend((directory, name, frame) for frame in read_labeled_extxyz(path))
        else:
            loaded.extend((directory, "", frame)
                          for frame in read_labeled_extxyz(directory / "all.extxyz"))

    hashes: dict[Path, str | None] = {}

    def source_hash(metadata: dict) -> tuple[str, str]:
        relative = str(metadata.get("gaussian_output", ""))
        base = campaign_root or Path(str(metadata.get("collection_campaign", "")))
        path = base / relative
        if path not in hashes:
            hashes[path] = _file_sha256(path) if relative and path.is_file() else None
        return str(path), hashes[path] or ""

    rows: list[dict[str, Any]] = []
    for directory, split, frame in loaded:
        record = frame.record
        metadata = record.metadata
        reasons: list[str] = []
        route = str(metadata.get("force_route") or "")
        method, basis, level_key = _method_basis(route)
        manifest_level = str(metadata.get("label_level") or "")
        if not level_key:
            level_key = manifest_level
        elif manifest_level and manifest_level != level_key:
            manifest_method = manifest_level.split("|", 1)[0]
            if manifest_method.lower() != f"{method}/{basis}".lower():
                reasons.append("route_level_mismatch")
        energy_ok = _finite(frame.energy_ev)
        forces_ok = all(_finite(component) for force in frame.forces_ev_ang for component in force)
        if not (energy_ok and forces_ok):
            reasons.append("non_finite_label")
        components = [abs(c) for force in frame.forces_ev_ang for c in force if _finite(c)]
        max_force = max(components, default=float("nan"))
        rms = math.sqrt(sum(c * c for c in components) / len(components)) if components else float("nan")
        if components and max_force > policy.max_force_ev_ang:
            reasons.append("force_outlier")
        if metadata.get("scf_unconverged") or metadata.get("scf_convergence_warning"):
            reasons.append("scf_not_converged")
        intended_q = _int_or_none(metadata.get("intended_charge"))
        intended_m = _int_or_none(metadata.get("intended_multiplicity"))
        if (intended_q is not None and intended_q != record.charge) or (
                intended_m is not None and intended_m != record.multiplicity):
            reasons.append("state_mismatch")
        s2_after = _float_or_none(metadata.get("s2_after"))
        spin = (record.multiplicity - 1) / 2
        expected = spin * (spin + 1)
        nearest = ""
        if s2_after is None:
            reasons.append("no_s2")
        else:
            nearest_m = nearest_multiplicity(s2_after, record.multiplicity)
            nearest = str(nearest_m)
            if nearest_m != record.multiplicity:
                reasons.append("s2_nearer_other_state")
            if policy.max_s2_excess is not None and s2_after - expected > policy.max_s2_excess:
                reasons.append("s2_excess")
        if metadata.get("local_spin_consistent") is False:
            reasons.append("local_spin_sum_inconsistent")
        if policy.level and level_key and level_key != policy.level:
            reasons.append("other_level")
        source_path, sha = source_hash(metadata)
        if not sha:
            reasons.append("source_unavailable")
        config_type = record.config_type
        key = _frame_key(metadata, record.record_id)
        claim = _stationary_claim(
            metadata, config_type, policy.ts_evidence,
            (key, str(metadata.get("job_id", "")), str(metadata.get("parent_record_id", ""))),
        )
        if claim == "saddle_unverified":
            reasons.append("ts_claim_unverified")
        rows.append({
            "frame_key": key,
            "run_id": metadata.get("job_id", record.record_id),
            "record_id": record.record_id,
            "chain_id": metadata.get("chain_id", ""),
            "batch": _batch(str(metadata.get("gaussian_output", ""))),
            "gaussian_output": source_path,
            "source_sha256": sha,
            "link1_section": metadata.get("gaussian_section_index", ""),
            "stage_index": metadata.get("stage_index", ""),
            "route": route,
            "method": method,
            "basis": basis,
            "label_level": level_key,
            "formula": record.formula,
            "chemistry": chemistry_family(atom.symbol for atom in record.atoms),
            "charge": record.charge,
            "multiplicity": record.multiplicity,
            "intended_charge": "" if intended_q is None else intended_q,
            "intended_multiplicity": "" if intended_m is None else intended_m,
            "s2_before": metadata.get("s2_before", ""),
            "s2_after": "" if s2_after is None else s2_after,
            "s2_expected": round(expected, 6),
            "s2_nearest_multiplicity": nearest,
            "local_spin_consistent": metadata.get("local_spin_consistent", ""),
            "mulliken_spin_sum_error": metadata.get("mulliken_spin_sum_error", ""),
            "fe_moment_pattern": metadata.get("fe_moment_pattern", ""),
            "energy": f"{frame.energy_ev:.8f}" if energy_ok else str(frame.energy_ev),
            "energy_unit": ENERGY_UNIT,
            "max_force": f"{max_force:.6f}",
            "rms_force": f"{rms:.6f}",
            "force_unit": FORCE_UNIT,
            "scf_converged": "scf_not_converged" not in reasons,
            "section_optimized": bool(metadata.get("section_optimized")),
            "section_normal_termination": bool(metadata.get("section_normal_termination")),
            "frame_role": metadata.get("frame_role", ""),
            "config_type": config_type,
            "stationary_claim": claim,
            "parent_record_id": metadata.get("parent_record_id", ""),
            "source_geometry_sha256": metadata.get("source_geometry_sha256", ""),
            "split": split,
            "split_group": metadata.get("split_group", ""),
            "restart_of_job_id": metadata.get("restart_of_job_id", ""),
            "restart_attempt": metadata.get("restart_attempt", ""),
            "restart_source_output": metadata.get("restart_source_output", ""),
            "restart_seed_sha256": metadata.get("restart_seed_sha256", ""),
            "label_sha256": label_sha256(frame) if energy_ok and forces_ok else "",
            "collect_dir": str(directory),
            "_reasons": reasons,
            "_frame": frame,
            "_geometry": geometry_signature(record.atoms),
        })

    _mark_duplicates_and_conflicts(rows, policy)
    _mark_split_leaks(rows)

    for row in rows:
        reasons = list(dict.fromkeys(row["_reasons"]))
        if any(reason in REJECT_REASONS for reason in reasons):
            decision = "reject"
        elif reasons:
            decision = "hold"
        else:
            decision = "include"
        row["decision"] = decision
        row["reasons"] = ";".join(reasons)

    rows.sort(key=lambda row: (row["chemistry"], row["formula"], row["run_id"], row["frame_key"]))
    return _write_outputs(rows, output, policy, collect_dirs, campaign_root)


def _mark_duplicates_and_conflicts(rows: list[dict[str, Any]], policy: ReleasePolicy) -> None:
    by_state: dict[tuple, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if "non_finite_label" in row["_reasons"]:
            continue
        by_state[(row["formula"], row["charge"], row["multiplicity"], row["label_level"],
                  row["_geometry"])].append(row)
    for members in by_state.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda row: (str(row["run_id"]), row["frame_key"]))
        first = members[0]["_frame"]
        conflict = False
        for other in members[1:]:
            frame = other["_frame"]
            if abs(frame.energy_ev - first.energy_ev) > policy.energy_tolerance_ev or any(
                abs(a - b) > policy.force_tolerance_ev_ang
                for fa, fb in zip(frame.forces_ev_ang, first.forces_ev_ang)
                for a, b in zip(fa, fb)
            ):
                conflict = True
        if conflict:
            for member in members:
                member["_reasons"].append("conflict_same_state")
        else:
            for member in members[1:]:
                member["_reasons"].append("duplicate_label")


def _mark_split_leaks(rows: list[dict[str, Any]]) -> None:
    parents = {str(row["run_id"]): str(row["restart_of_job_id"] or "") for row in rows}
    groupings: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if "duplicate_label" in row["_reasons"]:
            continue
        for kind, value in (("parent", row["parent_record_id"]), ("group", row["split_group"]),
                            ("lineage", _lineage_root(str(row["run_id"]), parents))):
            if value:
                groupings[(kind, str(value))].append(row)
    for members in groupings.values():
        if len({member["split"] for member in members}) > 1:
            for member in members:
                if "split_leak" not in member["_reasons"]:
                    member["_reasons"].append("split_leak")


def _write_outputs(
    rows: list[dict[str, Any]], output: Path, policy: ReleasePolicy,
    collect_dirs: list[Path], campaign_root: Path | None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "release_manifest.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, delimiter="\t",
                                lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    included = [row for row in rows if row["decision"] == "include"]
    canonical = "\n".join(
        f"{row['frame_key']}\t{row['label_sha256']}\t{row['source_sha256']}\t{row['split']}"
        for row in sorted(included, key=lambda row: row["frame_key"])
    )
    release_sha = hashlib.sha256(canonical.encode()).hexdigest()

    split_files: dict[str, str] = {}
    by_split: dict[str, list[LabeledFrame]] = collections.defaultdict(list)
    for row in included:
        by_split[row["split"] or "unsplit"].append(row["_frame"])
    for split, frames in sorted(by_split.items()):
        path = output / f"release_{split}.extxyz"
        write_labeled_extxyz(frames, path)
        split_files[path.name] = _file_sha256(path)

    counts: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    states: dict[tuple[str, str], set] = collections.defaultdict(set)
    for row in rows:
        for key in ((row["chemistry"], "*"), (row["chemistry"], row["formula"])):
            counts[key][row["decision"]] += 1
            if row["decision"] == "include":
                states[key].add((row["formula"], row["charge"], row["multiplicity"]))
    with (output / "counts_by_chemistry.tsv").open("w", newline="", encoding="utf-8") as handle:
        table = csv.writer(handle, delimiter="\t", lineterminator="\n")
        table.writerow(["chemistry", "formula", "include", "hold", "reject",
                         "included_states", "included_parent_geometries"])
        for (family, formula), counter in sorted(counts.items()):
            parents = {row["parent_record_id"] for row in included
                       if row["chemistry"] == family and (formula == "*" or row["formula"] == formula)}
            table.writerow([family, formula, counter["include"], counter["hold"],
                             counter["reject"], len(states[(family, formula)]), len(parents)])

    reason_counts = collections.Counter(
        reason for row in rows for reason in row["reasons"].split(";") if reason)
    family_included = collections.Counter(row["chemistry"] for row in included)
    gaps = []
    for family in policy.require_families:
        have = family_included.get(family, 0)
        if have < policy.min_frames_per_family:
            gaps.append(f"{family}: {have} included frames < {policy.min_frames_per_family}")
    if len(included) < policy.min_release_frames:
        gaps.append(f"release: {len(included)} included frames < {policy.min_release_frames}")
    state_failures = sum(reason_counts.get(reason, 0) for reason in (
        "state_mismatch", "conflict_same_state", "split_leak"))
    if state_failures:
        gaps.append(f"state audit: {state_failures} frames with state_mismatch, "
                    "conflict_same_state or split_leak")
    levels = collections.Counter(row["label_level"] for row in rows)
    summary: dict[str, Any] = {
        "collect_dirs": [str(path) for path in collect_dirs],
        "campaign_root": str(campaign_root) if campaign_root else None,
        "policy": {
            "level": policy.level,
            "max_force_ev_ang": policy.max_force_ev_ang,
            "max_s2_excess": policy.max_s2_excess,
            "energy_tolerance_ev": policy.energy_tolerance_ev,
            "force_tolerance_ev_ang": policy.force_tolerance_ev_ang,
            "ts_evidence_keys": len(policy.ts_evidence),
            "require_families": list(policy.require_families),
            "min_frames_per_family": policy.min_frames_per_family,
            "min_release_frames": policy.min_release_frames,
        },
        "units": {"energy": ENERGY_UNIT, "forces": FORCE_UNIT},
        "frames": len(rows),
        "decisions": dict(collections.Counter(row["decision"] for row in rows)),
        "reasons": dict(sorted(reason_counts.items())),
        "levels_seen": dict(levels),
        "included_by_chemistry": dict(sorted(family_included.items())),
        "included_runs": len({row["run_id"] for row in included}),
        "included_parent_geometries": len({row["parent_record_id"] for row in included}),
        "source_files": len({row["gaussian_output"] for row in rows}),
        "source_files_unavailable": len({row["gaussian_output"] for row in rows
                                         if not row["source_sha256"]}),
        "release_sha256": release_sha,
        "release_files": split_files,
        "manifest_sha256": _file_sha256(manifest),
        "gate": "release" if not gaps else "hold_retraining",
        "gaps": gaps,
    }
    (output / "release_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
