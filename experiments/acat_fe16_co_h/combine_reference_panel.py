"""Combine reviewed reference batches into one panel and score the models on it.

    python combine_reference_panel.py REVIEW.json [REVIEW.json ...] --out panel.json [--csv panel.csv]

Inputs are the outputs of review_reference_batch.py. All structures have the same
composition and charge, so total energies are compared directly, per multiplicity.
Only states whose own Force-link SCF converged are used; a flagged state is listed
but takes no part in any statistic.

Per model and multiplicity: mean absolute error of relative energies (reference = lowest
usable UBPW91 structure at that multiplicity), and pair ordering on pairs that UBPW91
separates by more than a threshold. Also per structure: the lowest multiplicity and the
49 -> 47 gap.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

PAIR_THRESHOLD_EV = 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reviews", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    rows, model_e = [], {}
    for path in args.reviews:
        rev = json.loads(path.read_text(encoding="utf-8"))
        for i, fr in enumerate(rev["frames"]):
            row = {"package": rev["package"].split("_")[0], "frame": fr["frame"], "draft_job": fr["draft_job"],
                   "why": fr["why"], "motif": fr["motif"], "hours": round(fr["seconds"] / 3600, 2),
                   "lowest_multiplicity": fr.get("lowest_multiplicity"), "states": {}}
            for m, st in fr["states"].items():
                row["states"][m] = {"usable": st["usable"], "energy_ev": st.get("energy_ev"),
                                    "energy_ev_scf_not_converged": st.get("energy_ev_scf_not_converged"),
                                    "scf_flag_stable_opt": st["stable_opt"]["scf_not_converged"],
                                    "scf_flag_force": st["force"]["scf_not_converged"],
                                    "force_rms": st.get("force_rms_ev_a")}
            rows.append(row)
            for name, entry in rev["models"].items():
                model_e.setdefault(name, []).append(entry["energies_ev"][i])
    mults = sorted({m for r in rows for m in r["states"]}, key=int, reverse=True)

    report: dict = {"structures": len(rows), "pair_threshold_ev": PAIR_THRESHOLD_EV, "usable_states": {}, "models": {}}
    ref = {}
    for m in mults:
        e = np.array([r["states"][m]["energy_ev"] if r["states"][m]["usable"] else np.nan for r in rows])
        ref[m] = e - np.nanmin(e)
        report["usable_states"][m] = int(np.isfinite(e).sum())
    for r, *vals in zip(rows, *[ref[m] for m in mults]):
        r["ubpw91_rel_ev"] = {m: (None if np.isnan(v) else float(v)) for m, v in zip(mults, vals)}
        e49, e47 = r["states"]["49"], r["states"]["47"]
        r["gap_49_to_47_ev"] = (e47["energy_ev"] - e49["energy_ev"]) if e49["usable"] and e47["usable"] else None
    for name, energies in model_e.items():
        e = np.array(energies)
        entry = {}
        for m in mults:
            ok = np.isfinite(ref[m])
            # same reference structure as UBPW91's lowest usable one
            anchor = int(np.nanargmin(ref[m]))
            pred = e - e[anchor]
            err = pred[ok] - ref[m][ok]
            agree = disagree = 0
            for a, b in combinations(np.where(ok)[0], 2):
                d_ref = ref[m][a] - ref[m][b]
                if abs(d_ref) < PAIR_THRESHOLD_EV:
                    continue
                if d_ref * (pred[a] - pred[b]) > 0:
                    agree += 1
                else:
                    disagree += 1
            entry[m] = {"mae_ev": float(np.abs(err).mean()), "max_abs_error_ev": float(np.abs(err).max()),
                        "pairs_agree": agree, "pairs_disagree": disagree,
                        "pair_ordering_percent": round(100 * agree / max(1, agree + disagree), 1),
                        "model_lowest_structure": rows[int(np.argmin(np.where(ok, pred, np.inf)))]["draft_job"],
                        "ubpw91_lowest_structure": rows[anchor]["draft_job"],
                        "ubpw91_rel_of_model_lowest_ev": float(ref[m][int(np.argmin(np.where(ok, pred, np.inf)))])}
        report["models"][name] = entry
        for r, v in zip(rows, e - e.min()):
            r.setdefault("model_rel_ev", {})[name] = float(v)
    gaps = [r["gap_49_to_47_ev"] for r in rows if r["gap_49_to_47_ev"] is not None]
    report["spin"] = {"lowest_multiplicity_counts": {str(k): sum(r["lowest_multiplicity"] == k for r in rows)
                                                     for k in sorted({r["lowest_multiplicity"] for r in rows}, reverse=True)},
                      "gap_49_to_47_ev": {"n": len(gaps), "min": min(gaps), "median": float(np.median(gaps)), "max": max(gaps),
                                          "below_0p10": sum(g < 0.10 for g in gaps)}}
    report["hours_per_task"] = {"min": min(r["hours"] for r in rows), "max": max(r["hours"] for r in rows),
                                "total_core_hours_12_cores": round(12 * sum(r["hours"] for r in rows))}
    report["rows"] = rows
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            names = list(model_e)
            w.writerow(["package", "frame", "draft_job", "why", "motif", "lowest_M"] + [f"ubpw91_rel_M{m}_eV" for m in mults]
                       + ["gap_49_47_eV"] + [f"{n}_rel_eV" for n in names] + ["flagged_states"])
            for r in rows:
                flagged = ";".join(f"M{m}" for m in mults if not r["states"][m]["usable"])
                w.writerow([r["package"], r["frame"], r["draft_job"], r["why"], r["motif"], r["lowest_multiplicity"]]
                           + ["" if r["ubpw91_rel_ev"][m] is None else f"{r['ubpw91_rel_ev'][m]:.4f}" for m in mults]
                           + ["" if r["gap_49_to_47_ev"] is None else f"{r['gap_49_to_47_ev']:.4f}"]
                           + [f"{r['model_rel_ev'][n]:.4f}" for n in names] + [flagged])
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    for r in sorted(rows, key=lambda r: (r["ubpw91_rel_ev"]["49"] is None, r["ubpw91_rel_ev"]["49"] or 0)):
        print(r["package"], r["frame"], r["draft_job"], r["motif"][:44].ljust(44),
              "49:", "  flag " if r["ubpw91_rel_ev"]["49"] is None else f"{r['ubpw91_rel_ev']['49']:6.3f}",
              "gap:", "  n/a" if r["gap_49_to_47_ev"] is None else f"{r['gap_49_to_47_ev']:6.3f}", "lowM", r["lowest_multiplicity"],
              " ".join(f"{v:6.3f}" for v in r["model_rel_ev"].values()))


if __name__ == "__main__":
    main()
