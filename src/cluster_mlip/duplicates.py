"""Find not-yet-run inputs that repeat another job of the same campaign.

A warehouse campaign is prepared one job per archived record, with no check
that two records are the same structure. Two inputs are the same job when they
have the same atoms, the same charge/multiplicity ladder, the same geometry and
the same *calculation that will run*. Running both costs twice and adds
nothing: ``collect`` drops the second copy's frames as exact duplicates anyway.

The calculation is what counts, not the label. A record labeled ``minimum``
and one labeled ``unknown`` both run a plain ``Opt`` ladder, so at the same
geometry they are one computation. Three kinds are kept apart, because holding
one in favour of another would lose something:

- ``opt``: a plain minimization (labels minimum, unknown, optimized_unverified,
  warehouse_structure, ...);
- ``saddle``: a transition-state search;
- ``fixed_geometry`` / ``higher_order``: path points and higher-order
  candidates, which the force-only policy labels where they are.

Geometry is compared by the sorted interatomic distances per element pair,
which does not change under rotation, translation or atom reordering (nor
under mirroring: enantiomers have the same energy). Two geometries match when
no distance differs by more than the tolerance.

Each group keeps one job: a finished one if there is one, else one that has
started, else one with a specific label over an unspecific one, else the first
by batch. Only *not-started* duplicates are listed -- those are the ones that
can still be held (``hold-inputs --list duplicate_inputs.tsv``). A restart
input, which reads its geometry from a checkpoint, is compared through the
root input it was built from.

The whole ladder is part of the identity: a ladder to M=5 and one to M=1 from
the same geometry share their first stages but are different jobs. Such
prefix-sharing inputs are counted in the summary, not listed.

Read-only: nothing in the campaign is modified.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .route_audit import audit_campaign_routes
from .routes import _CHARGE_MULT_LINE_RE, _LINK1_SPLIT_RE, input_coordinates

DUPLICATE_COLUMNS = [
    "input", "batch", "group", "kept_input", "kept_batch", "kept_state",
    "max_distance_difference_ang", "formula", "ladder", "calculation",
    "config_type", "kept_config_type", "intent", "geometry_role", "search_kinds",
]
_STATE_RANK = {"complete": 0, "incomplete": 1, "activity_unconfirmed": 1, "failed": 2,
               "not_started": 3}
# Labels that say nothing about what the geometry is; a specific label is the
# better copy to keep, because collect carries it into the dataset.
_UNSPECIFIC_LABELS = {"", "unknown", "warehouse_structure", "optimized_unverified",
                      "checkpoint_geometry"}
_OWN_CLASS_INTENTS = ("saddle", "fixed_geometry", "higher_order")


def calculation_kind(intent: str, search_kinds: str) -> str:
    """The kind of calculation a job runs, for deciding whether two are the same."""
    if intent in _OWN_CLASS_INTENTS:
        return intent
    # search_kinds lists one kind per optimizing stage ("minimum,minimum,...").
    kinds = sorted({kind for kind in search_kinds.split(",") if kind})
    return "opt" if kinds in ([], ["minimum"]) else "opt:" + "+".join(kinds)


def _ladder(text: str) -> str:
    """The charge/multiplicity line of every Link1 stage, e.g. ``0 9|0 7|0 5``."""
    stages = []
    for section in _LINK1_SPLIT_RE.split(text):
        line = next((candidate.strip() for candidate in section.splitlines()
                     if _CHARGE_MULT_LINE_RE.fullmatch(candidate)), "")
        stages.append(" ".join(line.split()))
    return "|".join(stages)


def _fingerprint(atoms: list[tuple[str, float, float, float]]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for i, (sym_a, ax, ay, az) in enumerate(atoms):
        for sym_b, bx, by, bz in atoms[i + 1:]:
            key = "-".join(sorted((sym_a, sym_b)))
            result.setdefault(key, []).append(
                math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2))
    for values in result.values():
        values.sort()
    return result


def _max_difference(left: dict[str, list[float]], right: dict[str, list[float]]) -> float:
    if left.keys() != right.keys() or any(len(left[k]) != len(right[k]) for k in left):
        return math.inf
    return max((abs(a - b) for key in left for a, b in zip(left[key], right[key])), default=0.0)


def _formula(atoms: list[tuple[str, float, float, float]]) -> str:
    counts = Counter(symbol for symbol, *_ in atoms)
    return "".join(f"{symbol}{counts[symbol] if counts[symbol] > 1 else ''}"
                   for symbol in sorted(counts))


def _batch_number(name: str) -> int:
    digits = name[6:] if name.startswith("batch_") else ""
    return int(digits) if digits.isdigit() else 10 ** 9


def find_duplicate_inputs(
    campaign: Path,
    destination: Path,
    *,
    tolerance: float = 0.02,
    seeds: Path | None = None,
) -> dict[str, Any]:
    """Write ``duplicate_inputs.tsv`` and ``duplicate_inputs.json``; see the module doc.

    ``seeds`` is the extxyz the campaign was prepared from; with it the audit
    classes path points and higher-order candidates from recorded evidence
    instead of file names.
    """
    campaign = campaign.resolve()
    audit = audit_campaign_routes(campaign, write_reports=False, seeds=seeds)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    not_comparable: Counter[str] = Counter()
    for row in audit["rows"]:
        source = campaign / row["input"]
        text = source.read_text(encoding="utf-8", errors="replace") if source.is_file() else ""
        atoms = input_coordinates(text) if text else None
        root = (row.get("restart_root_input") or "").strip()
        if atoms is None and root and (campaign / root).is_file():
            # A checkpoint-seeded restart is the same job as its root ladder.
            text = (campaign / root).read_text(encoding="utf-8", errors="replace")
            atoms = input_coordinates(text)
        if atoms is None:
            not_comparable[row["state"]] += 1
            continue
        kind = calculation_kind(str(row.get("intent", "")), str(row.get("search_kinds", "")))
        buckets.setdefault((_formula(atoms), kind, _ladder(text)), []).append(
            {"row": row, "fingerprint": _fingerprint(atoms)})

    duplicates: list[dict[str, Any]] = []
    group_sizes: list[int] = []
    label_pairs: Counter[str] = Counter()
    started_duplicates = 0
    group_id = 0
    keepers: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for bucket, items in sorted(buckets.items()):
        items.sort(key=lambda item: (
            _STATE_RANK.get(item["row"]["state"], 3),
            item["row"]["config_type"] in _UNSPECIFIC_LABELS,
            _batch_number(item["row"]["batch"]), item["row"]["input"],
        ))
        kept: list[dict[str, Any]] = []
        for item in items:
            match = None
            for candidate in kept:
                difference = _max_difference(candidate["fingerprint"], item["fingerprint"])
                if difference <= tolerance:
                    match = (candidate, difference)
                    break
            if match is None:
                item["members"] = 1
                item["group"] = None
                kept.append(item)
                continue
            keeper, difference = match
            keeper["members"] += 1
            if keeper["group"] is None:
                group_id += 1
                keeper["group"] = group_id
            row = item["row"]
            if row["state"] != "not_started":
                started_duplicates += 1
                continue
            label_pairs[f"{row['config_type']} -> {keeper['row']['config_type']}"] += 1
            duplicates.append({
                "input": Path(row["input"]).name, "batch": row["batch"],
                "group": keeper["group"], "kept_input": Path(keeper["row"]["input"]).name,
                "kept_batch": keeper["row"]["batch"], "kept_state": keeper["row"]["state"],
                "max_distance_difference_ang": f"{difference:.4f}",
                "formula": bucket[0], "ladder": bucket[2], "calculation": bucket[1],
                "config_type": row["config_type"],
                "kept_config_type": keeper["row"]["config_type"],
                "intent": row.get("intent", ""), "geometry_role": row.get("geometry_role", ""),
                "search_kinds": row.get("search_kinds", ""),
            })
        group_sizes += [item["members"] for item in kept if item["members"] > 1]
        keepers.setdefault((bucket[0], bucket[1]), []).extend(
            (bucket[2], item) for item in kept)

    # Not-started jobs whose whole ladder is the beginning of another job's
    # ladder at the same geometry: shared stages, but not the same job.
    prefix_shared = 0
    for members in keepers.values():
        for ladder, item in members:
            if item["row"]["state"] != "not_started":
                continue
            if any(
                other_ladder.startswith(ladder + "|")
                and _max_difference(other["fingerprint"], item["fingerprint"]) <= tolerance
                for other_ladder, other in members
            ):
                prefix_shared += 1

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "duplicate_inputs.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DUPLICATE_COLUMNS, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(duplicates)
    not_started = sum(1 for row in audit["rows"] if row["state"] == "not_started")
    summary = {
        "campaign": str(campaign), "tolerance_ang": tolerance,
        "seeds": str(seeds) if seeds else "",
        "inputs_audited": len(audit["rows"]), "not_started": not_started,
        "not_started_duplicates": len(duplicates),
        "kept_by_state": dict(Counter(row["kept_state"] for row in duplicates)),
        "duplicates_by_calculation": dict(Counter(row["calculation"] for row in duplicates)),
        "duplicates_by_formula": dict(Counter(row["formula"] for row in duplicates).most_common()),
        "label_of_duplicate_and_kept": dict(label_pairs.most_common()),
        "groups": len(group_sizes), "largest_group": max(group_sizes, default=0),
        "started_duplicates_not_listed": started_duplicates,
        "not_started_ladder_is_prefix_of_another_job": prefix_shared,
        "not_comparable_by_state": dict(not_comparable),
        "destination": str(destination),
    }
    (destination / "duplicate_inputs.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
