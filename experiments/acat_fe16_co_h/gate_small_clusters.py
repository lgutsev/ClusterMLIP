"""MLIP gate, part 1: existing UBPW91 labels on small Fe clusters with CO or H.

No Fe16 + CO/H reference label exists, so this panel uses what the warehouse has:
neutral Fe2/Fe4 with CO or 2 H, the bare clusters and the free molecules. Each
distinct geometry is represented by its lowest-energy multiplicity (the
"lowest-energy envelope"), because the foundation checkpoints take no spin input.
Energies only: the seeds file carries no forces.

    python gate_small_clusters.py SEEDS.extxyz --model name=path [--model ...] --out DIR
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase import Atoms

HARTREE = 27.211386245988
PANEL = ["CFe4O", "H2Fe4", "CFe2O", "H2Fe2", "Fe4", "Fe2", "CO", "H2", "Fe16"]  # ASE formula strings
ADSORPTION = [("CFe4O", "Fe4", "CO"), ("H2Fe4", "Fe4", "H2"), ("CFe2O", "Fe2", "CO"), ("H2Fe2", "Fe2", "H2")]


def info_value(comment: str, key: str) -> str | None:
    marker = f" {key}="
    at = (" " + comment).find(marker)
    if at < 0:
        return None
    rest = (" " + comment)[at + len(marker):]
    if rest.startswith('"'):
        return rest[1:rest.index('"', 1)]
    return rest.split()[0]


def read_seeds(path: Path, wanted: set[str]):
    with open(path, encoding="utf-8", errors="replace") as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.strip():
                continue
            n = int(line)
            comment = fh.readline()
            rows = [fh.readline().split() for _ in range(n)]
            atoms = Atoms([r[0] for r in rows], positions=[[float(x) for x in r[1:4]] for r in rows])
            formula = atoms.get_chemical_formula()
            if formula not in wanted:
                continue
            energy = info_value(comment, "legacy_energy_hartree")
            if energy is None or int(info_value(comment, "charge") or 0) != 0:
                continue
            yield formula, atoms, {"energy": float(energy) * HARTREE,
                                   "multiplicity": int(info_value(comment, "multiplicity") or 0),
                                   "record_id": info_value(comment, "record_id"),
                                   "config_type": info_value(comment, "config_type"),
                                   "route": info_value(comment, "legacy_route") or ""}


def distance_key(atoms: Atoms, step: float = 0.03) -> tuple:
    sym = atoms.get_chemical_symbols()
    d = atoms.get_all_distances()
    out = [(tuple(sorted((sym[i], sym[j]))), d[i, j]) for i, j in itertools.combinations(range(len(atoms)), 2)]
    out.sort()
    return tuple((p, round(x / step)) for p, x in out)


def ordering(ref: np.ndarray, pred: np.ndarray, min_gap: float) -> tuple[int, int]:
    good = total = 0
    for i, j in itertools.combinations(range(len(ref)), 2):
        if abs(ref[i] - ref[j]) < min_gap:
            continue
        total += 1
        good += int((ref[i] - ref[j]) * (pred[i] - pred[j]) > 0)
    return good, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", type=Path)
    ap.add_argument("--model", action="append", required=True, help="name=path")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--roles", default="minimum", help="comma list of config_type values, or all")
    args = ap.parse_args()
    from mace.calculators import MACECalculator

    groups: dict[str, dict[tuple, list]] = defaultdict(lambda: defaultdict(list))
    roles: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    methods: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for formula, atoms, info in read_seeds(args.seeds, set(PANEL)):
        roles[formula][info["config_type"] or "?"] += 1
        if args.roles != "all" and info["config_type"] not in args.roles.split(","):
            continue
        methods[formula][info["route"].split()[1] if len(info["route"].split()) > 1 else "?"] += 1
        groups[formula][distance_key(atoms)].append((info, atoms))

    reps: dict[str, list] = {}
    spin_spread = {}
    for formula, by_geom in groups.items():
        reps[formula] = [min(members, key=lambda m: m[0]["energy"]) for members in by_geom.values()]
        spreads = [max(m[0]["energy"] for m in ms) - min(m[0]["energy"] for m in ms)
                   for ms in by_geom.values() if len({m[0]["multiplicity"] for m in ms}) > 1]
        spin_spread[formula] = {"geometries_with_several_M": len(spreads),
                                "median_spread_ev": float(np.median(spreads)) if spreads else None,
                                "max_spread_ev": float(max(spreads)) if spreads else None}

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"roles_in_seeds": {f: dict(r) for f, r in roles.items()}, "roles_used": args.roles,
              "method_tokens_used": {f: dict(m) for f, m in methods.items()},
              "spin_spread_same_geometry": spin_spread, "models": {}}
    rows = []
    for spec in args.model:
        name, path = spec.split("=", 1)
        calc = MACECalculator(model_paths=path, device=args.device, default_dtype="float64")
        pred = {}
        for formula, chosen in reps.items():
            energies = []
            for _, atoms in chosen:
                a = atoms.copy()
                a.calc = calc
                energies.append(float(a.get_potential_energy()))
            pred[formula] = np.array(energies)
        per_formula = {}
        for formula, chosen in reps.items():
            ref = np.array([m[0]["energy"] for m in chosen])
            p = pred[formula]
            entry: dict = {"distinct_geometries": len(chosen)}
            i0 = int(np.argmin(ref))
            dref, dp = ref - ref[i0], p - p[i0]
            if len(chosen) >= 2:
                good, total = ordering(ref, p, 0.05)
                near = dref < 0.5
                near_good, near_total = ordering(ref[near], p[near], 0.02)
                entry.update(rel_energy_mae_ev=float(np.abs(dp - dref).mean()),
                             rel_energy_max_err_ev=float(np.abs(dp - dref).max()),
                             ref_range_ev=float(dref.max()), pairs_ordered=good, pairs_total=total,
                             pair_order_accuracy=good / total if total else None,
                             near_degenerate_pairs_ordered=near_good, near_degenerate_pairs_total=near_total,
                             model_minimum_is_ref_minimum=bool(int(np.argmin(p)) == i0),
                             ref_rank_of_model_minimum=int(np.argsort(ref).tolist().index(int(np.argmin(p))) + 1))
            # one magnetic branch at a time: pairs and errors within the same multiplicity only
            mults = np.array([m[0]["multiplicity"] for m in chosen])
            sm_good = sm_total = 0
            sm_err: list[float] = []
            for mult in sorted(set(mults.tolist())):
                sel = mults == mult
                if sel.sum() < 2:
                    continue
                g, t = ordering(ref[sel], p[sel], 0.05)
                sm_good, sm_total = sm_good + g, sm_total + t
                j = int(np.argmin(ref[sel]))
                sm_err.extend(np.abs((p[sel] - p[sel][j]) - (ref[sel] - ref[sel][j])).tolist())
            entry.update(same_M_pairs_ordered=sm_good, same_M_pairs_total=sm_total,
                         same_M_pair_order_accuracy=sm_good / sm_total if sm_total else None,
                         same_M_rel_energy_mae_ev=float(np.mean(sm_err)) if sm_err else None,
                         multiplicities=sorted(set(mults.tolist())))
            per_formula[formula] = entry
            for (info, _), a, b in zip(chosen, dref, dp):
                rows.append({"model": name, "formula": formula, "record_id": info["record_id"],
                             "config_type": info["config_type"], "multiplicity_of_lowest": info["multiplicity"],
                             "ref_rel_ev": round(float(a), 4), "model_rel_ev": round(float(b), 4)})
        ads = {}
        for ab, a, b in ADSORPTION:
            if all(f in reps for f in (ab, a, b)):
                def lowest(f: str) -> tuple[float, float]:
                    i = int(np.argmin([m[0]["energy"] for m in reps[f]]))
                    return reps[f][i][0]["energy"], float(pred[f][i])
                (rab, mab), (ra, ma), (rb, mb) = lowest(ab), lowest(a), lowest(b)
                ads[f"{a}+{b}->{ab}"] = {"ref_ev": round(rab - ra - rb, 3), "model_ev": round(mab - ma - mb, 3),
                                         "error_ev": round((mab - ma - mb) - (rab - ra - rb), 3)}
        report["models"][name] = {"path": path, "per_formula": per_formula, "adsorption_energy": ads}
    (args.out / "gate_small_clusters.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with open(args.out / "gate_small_clusters.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
