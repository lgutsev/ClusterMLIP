from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import TypedDict

from .models import LabeledFrame, Record
from .spin import geometry_distance
from .stratify import (
    DEFAULT_BONDING_TOLERANCE,
    bonding_graph,
    connected_components,
    displacement_class,
    pes_region,
)

# (energy_ev, forces_ev_ang) aligned index-for-index with a list of frames --
# same shape as evaluate.Prediction, redefined here rather than imported so
# this module has no dependency on evaluate.py (which may in turn want to
# call into this module from the CLI layer without a cycle).
Prediction = tuple[float, list[tuple[float, float, float]]]


class CheckResult(TypedDict):
    name: str
    n_frames_considered: int
    metric_name: str
    metric_value: float | None
    threshold: float
    passed: bool | None
    notes: str


def _no_data_result(name: str, metric_name: str, threshold: float, notes: str) -> CheckResult:
    return {
        "name": name,
        "n_frames_considered": 0,
        "metric_name": metric_name,
        "metric_value": None,
        "threshold": threshold,
        "passed": None,
        "notes": f"No frames matched this check's criteria in the given dataset. {notes}",
    }


def _force_rms(forces: list[tuple[float, float, float]]) -> float:
    return math.sqrt(sum(x * x + y * y + z * z for x, y, z in forces) / (3 * len(forces)))


def _vector_error(ref: tuple[float, float, float], pred: tuple[float, float, float]) -> float:
    return math.sqrt(sum((r - p) ** 2 for r, p in zip(ref, pred)))


def _cosine_similarity(a: tuple[float, float, float], b: tuple[float, float, float]) -> float | None:
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    norm_a = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    norm_b = math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)
    if norm_a == 0 or norm_b == 0:
        return None  # a true zero-force vector has no direction; exclude, don't guess
    return dot / (norm_a * norm_b)


def _sum_forces(forces: list[tuple[float, float, float]], indices: set[int]) -> tuple[float, float, float]:
    return (
        sum(forces[i][0] for i in indices),
        sum(forces[i][1] for i in indices),
        sum(forces[i][2] for i in indices),
    )


def stationary_point_check(
    frames: list[LabeledFrame], predictions: list[Prediction], threshold_ev_ang: float = 1.0
) -> CheckResult:
    """pes_region in {minimum, saddle} are both true stationary points in the
    reference data (a saddle is a stationary point with different curvature
    than a minimum, not a non-stationary one) -- a model that has learned
    the surface should predict near-zero force at both.
    """
    values = [
        _force_rms(pred_forces)
        for frame, (_, pred_forces) in zip(frames, predictions)
        if pes_region(frame.record.config_type) in ("minimum", "saddle")
    ]
    if not values:
        return _no_data_result(
            "stationary_point_force", "predicted_force_rms_ev_ang_mean", threshold_ev_ang,
            "No frames were labeled minimum or saddle.",
        )
    mean_value = statistics.mean(values)
    return {
        "name": "stationary_point_force",
        "n_frames_considered": len(values),
        "metric_name": "predicted_force_rms_ev_ang_mean",
        "metric_value": mean_value,
        "threshold": threshold_ev_ang,
        "passed": mean_value <= threshold_ev_ang,
        "notes": (
            "Mean predicted force RMS at frames labeled minimum/saddle; should be "
            "small if the model recognizes these as (near-)stationary."
        ),
    }


def rattled_direction_check(
    frames: list[LabeledFrame], predictions: list[Prediction], threshold: float = 0.9
) -> CheckResult:
    """displacement_class=rattled: does the model point the force the right
    direction on off-equilibrium displacements, not just get the magnitude
    close (which force MAE/RMSE alone can hide)?
    """
    similarities: list[float] = []
    n_frames = 0
    for frame, (_, pred_forces) in zip(frames, predictions):
        if displacement_class(frame.record.config_type) != "rattled":
            continue
        frame_sims = [
            sim
            for ref, pred in zip(frame.forces_ev_ang, pred_forces)
            if (sim := _cosine_similarity(ref, pred)) is not None
        ]
        if frame_sims:
            similarities.extend(frame_sims)
            n_frames += 1
    if not similarities:
        return _no_data_result(
            "rattled_force_direction", "cosine_similarity_mean", threshold,
            "No rattled (off-equilibrium) frames in this dataset.",
        )
    mean_value = statistics.mean(similarities)
    return {
        "name": "rattled_force_direction",
        "n_frames_considered": n_frames,
        "metric_name": "cosine_similarity_mean",
        "metric_value": mean_value,
        "threshold": threshold,
        "passed": mean_value >= threshold,
        "notes": (
            "Mean cosine similarity between predicted and reference per-atom force "
            "vectors on rattled displacements, averaged over every atom in every "
            "rattled frame (n_frames_considered counts frames, not atoms)."
        ),
    }


def low_coordination_error_check(
    frames: list[LabeledFrame],
    predictions: list[Prediction],
    tolerance: float = DEFAULT_BONDING_TOLERANCE,
    ratio_threshold: float = 1.5,
) -> CheckResult:
    """coordination_class=low_coordination: force error should not
    concentrate disproportionately on the worst-bonded atom(s) -- the
    classic place an MLIP fit degrades first.
    """
    ratios: list[float] = []
    n_frames = 0
    for frame, (_, pred_forces) in zip(frames, predictions):
        graph = bonding_graph(frame.record.atoms, tolerance)
        if not graph:
            continue
        degrees = {i: len(neighbors) for i, neighbors in graph.items()}
        min_degree = min(degrees.values())
        if min_degree > 1:
            continue  # not a low-coordination frame
        flagged = [i for i, degree in degrees.items() if degree == min_degree]
        per_atom_errors = [_vector_error(ref, pred) for ref, pred in zip(frame.forces_ev_ang, pred_forces)]
        frame_mean_error = statistics.mean(per_atom_errors)
        if frame_mean_error == 0:
            continue
        flagged_mean_error = statistics.mean(per_atom_errors[i] for i in flagged)
        ratios.append(flagged_mean_error / frame_mean_error)
        n_frames += 1
    if not ratios:
        return _no_data_result(
            "low_coordination_error_concentration", "flagged_to_frame_mean_error_ratio", ratio_threshold,
            "No frames classified low_coordination (or all had zero force error).",
        )
    mean_ratio = statistics.mean(ratios)
    return {
        "name": "low_coordination_error_concentration",
        "n_frames_considered": n_frames,
        "metric_name": "flagged_to_frame_mean_error_ratio",
        "metric_value": mean_ratio,
        "threshold": ratio_threshold,
        "passed": mean_ratio <= ratio_threshold,
        "notes": (
            "Mean ratio of force-vector error on the lowest-bonded-degree atom(s) in "
            "each low-coordination frame vs. that frame's own mean per-atom error. "
            ">1 means error concentrates on the under-coordinated atom(s)."
        ),
    }


def fragmenting_force_check(
    frames: list[LabeledFrame],
    predictions: list[Prediction],
    tolerance: float = DEFAULT_BONDING_TOLERANCE,
    threshold: float = 0.7,
) -> CheckResult:
    """compactness_class=fragmenting: does the model agree with the
    reference on which way the separating pieces are being pushed/pulled?
    Threshold is lower than the rattled check's -- summed net force over a
    whole fragment is noisier (internal cancellation) than one atom's force.
    """
    similarities: list[float] = []
    n_frames = 0
    for frame, (_, pred_forces) in zip(frames, predictions):
        graph = bonding_graph(frame.record.atoms, tolerance)
        if not graph:
            continue
        components = connected_components(graph)
        if len(components) < 2:
            continue  # not fragmenting
        frame_sims = []
        for component in components:
            ref_net = _sum_forces(frame.forces_ev_ang, component)
            pred_net = _sum_forces(pred_forces, component)
            sim = _cosine_similarity(ref_net, pred_net)
            if sim is not None:
                frame_sims.append(sim)
        if frame_sims:
            similarities.extend(frame_sims)
            n_frames += 1
    if not similarities:
        return _no_data_result(
            "fragmenting_force_direction", "net_fragment_force_cosine_similarity_mean", threshold,
            "No frames classified fragmenting.",
        )
    mean_value = statistics.mean(similarities)
    return {
        "name": "fragmenting_force_direction",
        "n_frames_considered": n_frames,
        "metric_name": "net_fragment_force_cosine_similarity_mean",
        "metric_value": mean_value,
        "threshold": threshold,
        "passed": mean_value >= threshold,
        "notes": (
            "Mean cosine similarity between predicted and reference net force on "
            "each disconnected fragment of a fragmenting cluster -- catches a model "
            "inventing an unphysical attraction/repulsion between separating pieces."
        ),
    }


def _cluster_by_geometry(records: list[Record], tolerance: float) -> list[list[int]]:
    """Greedy single-pass clustering: each record joins the first existing
    cluster whose representative is within `tolerance` of it (via
    spin.geometry_distance -- already tested, translation/rotation/atom-
    order-invariant), else starts a new cluster. Good enough for grouping a
    handful of multiplicity-ladder siblings sharing one starting geometry;
    not a general-purpose clustering algorithm.
    """
    clusters: list[list[int]] = []
    representatives: list[Record] = []
    for index, record in enumerate(records):
        placed = False
        for cluster_index, representative in enumerate(representatives):
            if geometry_distance(representative, record) <= tolerance:
                clusters[cluster_index].append(index)
                placed = True
                break
        if not placed:
            clusters.append([index])
            representatives.append(record)
    return clusters


def spin_ordering_check(
    frames: list[LabeledFrame],
    predictions: list[Prediction],
    geometry_tolerance: float = 0.05,
    threshold: float = 0.8,
) -> CheckResult:
    """charge_spin_class: for frames sharing a formula/charge and a near-
    identical geometry (spin.geometry_distance) but different multiplicity
    -- typically siblings of a prepare-spins ladder -- does the model agree
    with DFT on which multiplicity is lowest-energy (the ground spin state)?
    A model that gets forces/energies right in aggregate can still invert
    the spin-state ordering, which validate-spins-style per-frame checks
    can't see since it needs a same-geometry, cross-multiplicity comparison.
    """
    by_formula_charge: dict[tuple[str, int], list[int]] = {}
    for index, frame in enumerate(frames):
        by_formula_charge.setdefault((frame.record.formula, frame.record.charge), []).append(index)

    agreements = 0
    total = 0
    for indices in by_formula_charge.values():
        records = [frames[i].record for i in indices]
        for cluster in _cluster_by_geometry(records, geometry_tolerance):
            if len({records[j].multiplicity for j in cluster}) < 2:
                continue  # need >=2 distinct spin states to compare an ordering
            group = [indices[j] for j in cluster]
            ref_ground = min((frames[k].energy_ev, frames[k].record.multiplicity) for k in group)[1]
            pred_ground = min((predictions[k][0], frames[k].record.multiplicity) for k in group)[1]
            total += 1
            agreements += ref_ground == pred_ground

    if total == 0:
        return _no_data_result(
            "spin_state_ordering", "ground_state_agreement_fraction", threshold,
            "No same-formula/charge, same-geometry, multi-multiplicity groups found "
            "(typically produced by prepare-spins ladders).",
        )
    fraction = agreements / total
    return {
        "name": "spin_state_ordering",
        "n_frames_considered": total,
        "metric_name": "ground_state_agreement_fraction",
        "metric_value": fraction,
        "threshold": threshold,
        "passed": fraction >= threshold,
        "notes": (
            f"Fraction of same-formula/charge, same-geometry (RMS <= {geometry_tolerance} "
            "Angstrom) multiplicity groups where the model agrees with DFT on the "
            "lowest-energy (ground) spin state. n_frames_considered counts groups, "
            "not individual frames."
        ),
    }


def run_all_checks(frames: list[LabeledFrame], predictions: list[Prediction]) -> list[CheckResult]:
    if len(frames) != len(predictions):
        raise ValueError("frames and predictions must be the same length")
    return [
        stationary_point_check(frames, predictions),
        rattled_direction_check(frames, predictions),
        low_coordination_error_check(frames, predictions),
        fragmenting_force_check(frames, predictions),
        spin_ordering_check(frames, predictions),
    ]


def write_physical_checks_report(
    frames: list[LabeledFrame], predictions: list[Prediction], output: Path
) -> list[CheckResult]:
    results = run_all_checks(frames, predictions)
    output.mkdir(parents=True, exist_ok=True)
    (output / "physical_checks.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Physical sanity checks",
        "",
        "One cheap, first-pass physical check per structural class. These are not a "
        "substitute for the full per-stratum error breakdown above, or for scientific "
        "review -- a passing check does not certify the model, and thresholds are "
        "first-pass defaults meant to be recalibrated once real error distributions "
        "are in hand.",
        "",
        "| Check | Frames/groups | Metric | Value | Threshold | Passed | Notes |",
        "|---|---:|---|---:|---:|---|---|",
    ]
    for result in results:
        value = "n/a" if result["metric_value"] is None else f"{result['metric_value']:.4f}"
        passed = "n/a" if result["passed"] is None else ("yes" if result["passed"] else "**NO**")
        lines.append(
            f"| {result['name']} | {result['n_frames_considered']} | {result['metric_name']} | "
            f"{value} | {result['threshold']} | {passed} | {result['notes']} |"
        )
    (output / "physical_checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results
