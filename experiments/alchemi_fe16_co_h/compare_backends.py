"""Compare relaxation outcomes of several backend runs on the same frozen pool.

    python compare_backends.py BENCH_DIR --reference native --out report.json

For every run (sub-folder with manifest.jsonl / final.extxyz / timings.json): terminal
status counts, bookkeeping (every input present once, composition and roles kept,
frozen atoms unmoved, converged candidates really below fmax when the forces are
recomputed natively), and against the reference run per candidate: same structure
(permutation-aware distance test), same motif, energy difference. Rankings are
compared on candidates that end in the same structure; pairs closer than the
numerical tolerance are counted as unresolved rather than ordered.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from cluster_mlip.adsorbate_search import same_structure  # noqa: E402

ENERGY_TOL = 1e-4  # eV: float32 repeatability is ~3e-5 eV (parity check)


def load(run: Path):
    rows = [json.loads(line) for line in open(run / "manifest.jsonl", encoding="utf-8")]
    frames = read(run / "final.extxyz", index=":")
    timings = json.loads((run / "timings.json").read_text(encoding="utf-8"))
    return rows, frames, timings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", type=Path)
    ap.add_argument("--reference", default="native")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fmax", type=float, default=0.05)
    args = ap.parse_args()
    runs = {p.name: load(p) for p in sorted(args.bench.iterdir()) if (p / "manifest.jsonl").exists()}
    ref_rows, ref_frames, _ = runs[args.reference]
    report: dict = {"reference": args.reference, "energy_tolerance_eV": ENERGY_TOL, "runs": {}}
    for name, (rows, frames, timings) in runs.items():
        n = len(rows)
        entry: dict = {
            "inputs": timings["n"], "outputs": n,
            "every_input_once": [r["pool_index"] for r in rows] == list(range(timings["n"])),
            "ids_match_reference": [r["candidate_id"] for r in rows] == [r["candidate_id"] for r in ref_rows][:n],
            "status": dict(Counter(r["status"] for r in rows)),
            "composition_and_roles_kept": all(r["composition_ok"] and r["roles_ok"] for r in rows),
            "max_catalyst_move_A": max(r["catalyst_max_move"] for r in rows),
            "converged_above_fmax_on_native_recompute": sum(r["status"] == "converged"
                                                            and r["fmax_free_native_recompute"] > args.fmax * 1.02 for r in rows),
            "max_abs_reported_minus_native_energy_eV": max(abs(r["energy"] - r["energy_native_recompute"]) for r in rows),
            "steps_per_candidate_mean": float(np.mean([r["steps"] for r in rows])),
            "topologies": dict(Counter(r["topology"] for r in rows)),
            "flags": dict(Counter(f for r in rows for f in r["flags"])),
            "motifs": dict(Counter(r["motif"] for r in rows)),
            "timings": timings["runs"],
        }
        if name != args.reference:
            both = [i for i in range(n) if rows[i]["status"] == "converged" and ref_rows[i]["status"] == "converged"]
            same = [i for i in both if same_structure(frames[i], ref_frames[i], tol=0.15)]
            de = np.array([rows[i]["energy_native_recompute"] - ref_rows[i]["energy_native_recompute"] for i in same])
            de_all = np.array([rows[i]["energy_native_recompute"] - ref_rows[i]["energy_native_recompute"] for i in both])
            entry["vs_reference"] = {
                "both_converged": len(both), "same_structure": len(same),
                "same_motif": sum(rows[i]["motif"] == ref_rows[i]["motif"] for i in both),
                "different_minimum": len(both) - len(same),
                "lower_energy_than_reference": int((de_all < -ENERGY_TOL).sum()),
                "higher_energy_than_reference": int((de_all > ENERGY_TOL).sum()),
                "same_structure_abs_dE_eV": {"max": float(np.abs(de).max()) if len(de) else None,
                                              "median": float(np.median(np.abs(de))) if len(de) else None},
                "all_abs_dE_eV_median": float(np.median(np.abs(de_all))) if len(de_all) else None,
            }
            e_a = np.array([rows[i]["energy_native_recompute"] for i in same])
            e_b = np.array([ref_rows[i]["energy_native_recompute"] for i in same])
            agree = disagree = unresolved = 0
            tol = max(ENERGY_TOL, 2 * float(np.abs(de).max())) if len(de) else ENERGY_TOL
            for x in range(len(same)):
                for y in range(x + 1, len(same)):
                    if abs(e_b[x] - e_b[y]) < tol:
                        unresolved += 1
                    elif (e_a[x] - e_a[y]) * (e_b[x] - e_b[y]) > 0:
                        agree += 1
                    else:
                        disagree += 1
            entry["vs_reference"]["ranking_pairs"] = {"resolution_eV": tol, "agree": agree, "disagree": disagree,
                                                       "unresolved_within_tolerance": unresolved}
            best = lambda rs: min((r["energy_native_recompute"] for r in rs if r["status"] == "converged"))  # noqa: E731
            entry["vs_reference"]["best_energy_minus_reference_best_eV"] = float(best(rows) - best(ref_rows[:n]))
        report["runs"][name] = entry
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, e in report["runs"].items():
        t = e["timings"][-1]
        print(name, e["status"], "steps/cand", round(e["steps_per_candidate_mean"], 1), "run_s", round(t["run_s"], 1),
              "cand/h", round(t["candidates_per_hour_warm"]), "vram", round(t["peak_vram_mb"]),
              "| " + json.dumps(e.get("vs_reference", {})))


if __name__ == "__main__":
    main()
