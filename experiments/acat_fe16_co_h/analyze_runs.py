"""Compare the search arms, pool their candidates and pick a DFT shortlist.

    python analyze_runs.py RUN_DIR [RUN_DIR ...] --out OUT_DIR [--second-model mace:<path>]

Every RUN_DIR holds ``<arm>_s<seed>/manifest.jsonl`` + ``structures.extxyz``.
All energies here are *model* energies of one fixed composition; nothing in this
script is a reference result.

Outputs: ``runs.csv`` (per arm and seed), ``arms.json`` (comparison),
``pool.csv`` + ``pool.extxyz`` (unique candidates over all runs),
``shortlist.csv`` + ``shortlist.extxyz`` (structures proposed for DFT), and
``progress.png`` / ``diversity.png``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ase.io import read, write

from common import load_catalyst, load_sites, make_calculator, read_manifest

from cluster_mlip.adsorbate_search import (Gene, analyze, build_candidate, fingerprint, fingerprint_distance,
                                           validate)

LOW_WINDOW = 0.30  # eV above the pooled model minimum: "low-energy" candidates
ANOMALY_FLAGS = ("co_stretched", "co_dissociated", "co_reacted", "h2_formed", "h_pooling", "co_o_bound",
                 "catalyst_rearranged")


def load_runs(run_dirs: list[Path]):
    runs = []
    for run_dir in run_dirs:
        for sub in sorted(p for p in run_dir.rglob("manifest.jsonl")):
            records = read_manifest(sub)
            frames = {a.info["candidate_id"]: a for a in read(sub.parent / "structures.extxyz", index=":")
                      if a.info.get("stage") == "relaxed"}
            summary = json.loads((sub.parent / "summary.json").read_text(encoding="utf-8"))
            runs.append({"dir": sub.parent, "arm": summary["arm"], "seed": summary["seed"], "records": records,
                         "frames": frames, "summary": summary})
    return runs


def usable(r: dict) -> bool:
    return r["status"] == "relaxed" and r["converged"] and not r["failure"] and r["energy"] is not None


def build_pool(runs, tol: float, etol: float):
    """Unique usable candidates over all runs; each pool entry lists who found it and when."""
    pool: list[dict] = []
    for run in runs:
        for r in run["records"]:
            if not usable(r):
                continue
            atoms = run["frames"][r["candidate_id"]]
            fp = fingerprint(atoms)
            hit = None
            for entry in pool:
                if abs(entry["energy"] - r["energy"]) < etol and fingerprint_distance(fp, entry["fp"]) < tol:
                    hit = entry
                    break
            if hit is None:
                hit = {"pool_id": f"P{len(pool):04d}", "energy": r["energy"], "fp": fp, "atoms": atoms,
                       "motif": r["motif"], "topology": r["topology"], "flags": r["flags"],
                       "first_candidate": r["candidate_id"], "found": {}}
                pool.append(hit)
            key = (run["arm"], run["seed"])
            hit["found"].setdefault(key, {"relaxation": r["cum_relaxations"], "calls": r["cum_calc_calls"],
                                          "wall_s": r["cum_wall_s"]})
    pool.sort(key=lambda e: e["energy"])
    e0 = pool[0]["energy"]
    for rank, entry in enumerate(pool, 1):
        entry["rank"], entry["e_rel"] = rank, entry["energy"] - e0
    return pool


def best_so_far(records, x_key):
    xs, ys, best = [], [], np.inf
    for r in records:
        if usable(r) and r["energy"] < best:
            best = r["energy"]
        if r["status"] == "relaxed" and np.isfinite(best):
            xs.append(r[x_key])
            ys.append(best)
    return np.array(xs), np.array(ys)


def at_or_before(xs, ys, x):
    idx = np.searchsorted(xs, x, side="right") - 1
    return float(ys[idx]) if idx >= 0 else None


def farthest_points(candidates, chosen_fps, k):
    picked = []
    fps = list(chosen_fps)
    remaining = list(candidates)
    while remaining and len(picked) < k:
        def score(entry):
            d = [fingerprint_distance(entry["fp"], f) for f in fps]
            return min(d) if d else 0.0
        best = max(remaining, key=score)
        picked.append(best)
        fps.append(best["fp"])
        remaining.remove(best)
    return picked


def constructed_controls(catalyst, sites):
    """Structures the search is unlikely to keep but the reference panel needs."""
    top = next(s for s in sites if s.kind == "top")
    far = [s for s in sites if s.kind == "hollow3" and not set(s.indices) & set(top.indices)]
    h1 = far[0]
    h2 = next(s for s in far if not set(s.indices) & set(h1.indices))
    rest = [s for s in far if not set(s.indices) & (set(h1.indices) | set(h2.indices))]
    out = []
    for name, genes in (
        ("control_CO_O_bound_top", [Gene("CO", top.id, "O"), Gene("H", h1.id), Gene("H", h2.id)]),
        ("control_CO_side_on_bridge", [Gene("CO", next(s.id for s in sites if s.kind == "bridge"
                                                          and not set(s.indices) & (set(h1.indices) | set(h2.indices))),
                                             "side", 75.0, 90.0), Gene("H", h1.id), Gene("H", h2.id)]),
    ):
        atoms = build_candidate(catalyst, sites, genes)
        out.append((name, atoms, validate(atoms, catalyst, {"CO": 1, "H": 2})))
    # dissociated C + O + 2 H: C and O on separate hollows (same elemental composition)
    atoms = build_candidate(catalyst, sites, [Gene("CO", rest[0].id, "C"), Gene("H", h1.id), Gene("H", h2.id)])
    o_index = len(catalyst) + 1
    atoms.positions[o_index] = rest[1].p + 1.25 * rest[1].n
    out.append(("control_C_plus_O_dissociated", atoms, []))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--second-model", help="mace:<path>: single points on the pool for a model-disagreement flag")
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--etol", type=float, default=0.02)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.run_dirs)
    catalyst = load_catalyst()
    sites = load_sites(catalyst).sites
    pool = build_pool(runs, args.tol, args.etol)
    e0 = pool[0]["energy"]

    # ---- per-run table
    run_rows = []
    for run in runs:
        s = run["summary"]
        recs = run["records"]
        flags = Counter(f for r in recs if r["status"] == "relaxed" for f in r["flags"])
        reject = Counter(reason for r in recs if r["status"] != "relaxed" for reason in r["reject_reasons"])
        low = [e for e in pool if e["e_rel"] < LOW_WINDOW and (run["arm"], run["seed"]) in e["found"]]
        run_rows.append({
            "arm": s["arm"], "seed": s["seed"], "proposals": s["proposals"], "invalid": s["invalid"],
            "duplicate_start": s["duplicate_start"], "relaxations": s["relaxations"], "converged": s["converged"],
            "duplicates_after_relax": s["duplicates_after_relax"], "unique": s["unique"], "failures": s["failures"],
            "unique_converged_valid": s["unique_converged_valid"], "motifs": s["motifs"],
            "calc_calls": s["calc_calls"], "wall_s": round(s["wall_s"], 1), "reference_dft_calls": 0,
            "best_e_rel_to_pool_min": round(s["best_energy"] - e0, 4),
            "found_pool_minimum": (s["arm"], s["seed"]) in pool[0]["found"],
            "low_energy_pool_entries_found": len(low),
            "flags": json.dumps(dict(flags), sort_keys=True), "reject_reasons": json.dumps(dict(reject), sort_keys=True),
            "operators": json.dumps(dict(Counter(r["operator"] for r in recs)), sort_keys=True),
        })
    with open(args.out / "runs.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)

    # ---- arm comparison at matched budgets
    arms = sorted({run["arm"] for run in runs})
    low_pool = [e for e in pool if e["e_rel"] < LOW_WINDOW]
    comparison: dict = {"pool_size": len(pool), "pool_low_energy_entries": len(low_pool), "low_window_ev": LOW_WINDOW,
                        "pool_min_energy": e0, "arms": {}}
    max_relax = min(run["summary"]["relaxations"] for run in runs)
    min_calls = min(run["summary"]["calc_calls"] for run in runs)
    min_wall = min(run["summary"]["wall_s"] for run in runs)
    for arm in arms:
        arm_runs = [run for run in runs if run["arm"] == arm]
        entry: dict = {"seeds": [run["seed"] for run in arm_runs]}
        for label, key, points in (("relaxations", "cum_relaxations", [50, 100, 150, max_relax]),
                                   ("calc_calls", "cum_calc_calls", [min_calls // 4, min_calls // 2, min_calls]),
                                   ("wall_s", "cum_wall_s", [min_wall / 4, min_wall / 2, min_wall])):
            table = {}
            for x in points:
                vals = []
                for run in arm_runs:
                    xs, ys = best_so_far(run["records"], key)
                    v = at_or_before(xs, ys, x)
                    vals.append(None if v is None else round(v - e0, 4))
                table[str(round(x, 1))] = vals
            entry[f"best_e_rel_vs_{label}"] = table
        entry["found_pool_minimum"] = [(arm, run["seed"]) in pool[0]["found"] for run in arm_runs]
        entry["relaxations_to_pool_minimum"] = [pool[0]["found"].get((arm, run["seed"]), {}).get("relaxation")
                                                for run in arm_runs]
        entry["low_energy_recall"] = [round(sum((arm, run["seed"]) in e["found"] for e in low_pool) / len(low_pool), 3)
                                      for run in arm_runs]
        entry["unique_converged_valid"] = [run["summary"]["unique_converged_valid"] for run in arm_runs]
        entry["motifs"] = [run["summary"]["motifs"] for run in arm_runs]
        entry["duplicate_rate_after_relax"] = [round(run["summary"]["duplicates_after_relax"] / run["summary"]["relaxations"], 3)
                                               for run in arm_runs]
        entry["rejected_before_relax"] = [run["summary"]["invalid"] + run["summary"]["duplicate_start"] for run in arm_runs]
        entry["calc_calls"] = [run["summary"]["calc_calls"] for run in arm_runs]
        entry["wall_s"] = [round(run["summary"]["wall_s"], 1) for run in arm_runs]
        # does the arm's own low-energy set contain more than one motif?
        entry["motifs_within_low_window"] = [len({e["motif"] for e in low_pool if (arm, run["seed"]) in e["found"]})
                                             for run in arm_runs]
        comparison["arms"][arm] = entry
    gaps = np.diff([e["e_rel"] for e in pool if e["e_rel"] < LOW_WINDOW])
    comparison["low_energy_gaps_ev"] = {"n": int(len(gaps)), "median": float(np.median(gaps)) if len(gaps) else None,
                                        "min": float(gaps.min()) if len(gaps) else None}
    comparison["pool_topologies"] = dict(Counter(e["topology"] for e in pool))
    comparison["pool_flags"] = dict(Counter(f for e in pool for f in e["flags"]))
    comparison["motifs_low_window"] = dict(Counter(e["motif"] for e in low_pool))

    # ---- second model: disagreement on relative energies (not an uncertainty estimate)
    if args.second_model:
        spec = args.second_model
        calc = make_calculator("mace:" + os.path.expanduser(spec[5:]), "cuda", "float32")
        e2 = []
        for entry in pool:
            a = entry["atoms"].copy()
            a.calc = calc
            e2.append(float(a.get_potential_energy()))
        e2 = np.array(e2) - e2[0]
        for entry, v in zip(pool, e2):
            entry["e_rel_second_model"] = float(v)
            entry["model_disagreement"] = float(v - entry["e_rel"])
        e1 = np.array([e["e_rel"] for e in pool])
        low = e1 < LOW_WINDOW
        good = total = 0
        idx = np.where(low)[0]
        for i_pos, i in enumerate(idx):
            for j in idx[i_pos + 1:]:
                if abs(e1[i] - e1[j]) < 0.02:
                    continue
                total += 1
                good += int((e1[i] - e1[j]) * (e2[i] - e2[j]) > 0)
        comparison["second_model"] = {"spec": spec, "rank_of_first_model_minimum": int(np.argsort(e2).tolist().index(0) + 1),
                                      "low_window_pairs_same_order": good, "low_window_pairs": total,
                                      "rel_energy_rms_difference_ev": float(np.sqrt(np.mean((e2 - e1) ** 2)))}

    # ---- shortlist for DFT
    shortlist: list[tuple[str, dict]] = []
    taken: set[str] = set()

    def take(reason: str, entry: dict) -> None:
        if entry["pool_id"] not in taken:
            taken.add(entry["pool_id"])
            shortlist.append((reason, entry))

    seen_motifs: set[str] = set()
    for entry in pool:  # A: lowest structure of each of the six lowest motifs
        if entry["motif"] not in seen_motifs and len(seen_motifs) < 6:
            seen_motifs.add(entry["motif"])
            take("lowest_of_motif", entry)
    for entry in pool[:12]:  # B: near-degenerate competitors of the model minimum
        if entry["e_rel"] < 0.10 and sum(r == "near_degenerate" for r, _ in shortlist) < 4:
            take("near_degenerate", entry)
    anomalies = [e for e in pool if set(e["flags"]) & set(ANOMALY_FLAGS)]
    for flag in ANOMALY_FLAGS:  # D: lowest example of each anomaly flag
        hit = next((e for e in anomalies if flag in e["flags"]), None)
        if hit is not None:
            take(f"anomaly:{flag}", hit)
    if args.second_model:
        for entry in sorted(pool, key=lambda e: -abs(e["model_disagreement"]))[:3]:
            take("largest_model_disagreement", entry)
    random_only = [e for e in pool if e["pool_id"] not in taken and any(arm == "random" for arm, _ in e["found"])]
    for entry in farthest_points(random_only, [e["fp"] for _, e in shortlist], 4):  # C: diverse random controls
        take("diverse_random_control", entry)

    frames, rows = [], []
    for reason, entry in shortlist:
        a = entry["atoms"].copy()
        a.info = {"shortlist_id": entry["pool_id"], "reason": reason, "motif": entry["motif"],
                  "model_e_rel": round(entry["e_rel"], 4), "first_candidate": entry["first_candidate"]}
        frames.append(a)
        rows.append({"shortlist_id": entry["pool_id"], "reason": reason, "model_rank": entry["rank"],
                     "model_e_rel_ev": round(entry["e_rel"], 4),
                     "second_model_e_rel_ev": round(entry.get("e_rel_second_model", float("nan")), 4),
                     "topology": entry["topology"], "motif": entry["motif"], "flags": ";".join(entry["flags"]),
                     "first_candidate": entry["first_candidate"], "relaxed_with_model": True,
                     "found_by": ";".join(sorted(f"{arm}-s{seed}" for arm, seed in entry["found"]))})
    for name, atoms, problems in constructed_controls(catalyst, sites):
        an = analyze(atoms, catalyst, sites)
        atoms.info = {"shortlist_id": name, "reason": "constructed_control", "motif": an.motif}
        frames.append(atoms)
        rows.append({"shortlist_id": name, "reason": "constructed_control", "model_rank": "", "model_e_rel_ev": "",
                     "second_model_e_rel_ev": "", "topology": an.topology, "motif": an.motif,
                     "flags": ";".join(an.flags + problems), "first_candidate": "", "relaxed_with_model": False,
                     "found_by": "constructed"})
    write(args.out / "shortlist.extxyz", frames)
    with open(args.out / "shortlist.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparison["shortlist"] = dict(Counter(r["reason"].split(":")[0] for r in rows))

    # ---- pool files
    pool_frames = []
    with open(args.out / "pool.csv", "w", newline="", encoding="utf-8") as fh:
        fields = ["pool_id", "rank", "e_rel_ev", "second_model_e_rel_ev", "topology", "motif", "flags", "n_runs_found",
                  "found_by", "first_candidate"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for entry in pool:
            writer.writerow({"pool_id": entry["pool_id"], "rank": entry["rank"], "e_rel_ev": round(entry["e_rel"], 4),
                             "second_model_e_rel_ev": round(entry.get("e_rel_second_model", float("nan")), 4),
                             "topology": entry["topology"], "motif": entry["motif"], "flags": ";".join(entry["flags"]),
                             "n_runs_found": len(entry["found"]),
                             "found_by": ";".join(sorted(f"{arm}-s{seed}@{v['relaxation']}" for (arm, seed), v in entry["found"].items())),
                             "first_candidate": entry["first_candidate"]})
            a = entry["atoms"].copy()
            a.info = {"pool_id": entry["pool_id"], "model_e_rel": round(entry["e_rel"], 4), "motif": entry["motif"]}
            pool_frames.append(a)
    write(args.out / "pool.extxyz", pool_frames)
    (args.out / "arms.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"random": "#4477aa", "evolutionary": "#cc6633"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (key, label) in zip(axes, (("cum_relaxations", "relaxations started"), ("cum_calc_calls", "energy/force calls"),
                                       ("cum_wall_s", "wall time (s)"))):
        for run in runs:
            xs, ys = best_so_far(run["records"], key)
            ax.step(xs, ys - e0, where="post", color=colors.get(run["arm"], "gray"), alpha=0.85, lw=1.4,
                    label=run["arm"] if run["seed"] == runs[0]["seed"] else None)
        ax.set_xlabel(label)
        ax.set_ylim(-0.02, 0.6)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("best model energy so far\nabove pooled minimum (eV)")
    axes[0].legend(frameon=False)
    fig.suptitle("Search progress on the model surface (not reference energies); one line per seed", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "progress.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for run in runs:
        xs, uniq, low_found = [], [], []
        seen_u = 0
        key = (run["arm"], run["seed"])
        when = sorted(e["found"][key]["relaxation"] for e in low_pool if key in e["found"])
        for r in run["records"]:
            if r["status"] != "relaxed":
                continue
            seen_u += int(r["unique"] and usable(r))
            xs.append(r["cum_relaxations"])
            uniq.append(seen_u)
            low_found.append(int(np.searchsorted(when, r["cum_relaxations"], side="right")))
        axes[0].plot(xs, uniq, color=colors.get(run["arm"], "gray"), alpha=0.85,
                     label=run["arm"] if run["seed"] == runs[0]["seed"] else None)
        axes[1].plot(xs, low_found, color=colors.get(run["arm"], "gray"), alpha=0.85)
    axes[0].set_xlabel("relaxations started")
    axes[0].set_ylabel("unique converged structures")
    axes[1].set_xlabel("relaxations started")
    axes[1].set_ylabel(f"pooled structures within {LOW_WINDOW} eV found")
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Diversity and low-energy coverage (model surface)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "diversity.png", dpi=150)
    print(json.dumps(comparison, indent=1))


if __name__ == "__main__":
    main()
