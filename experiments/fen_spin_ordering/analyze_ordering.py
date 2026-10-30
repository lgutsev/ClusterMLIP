"""Analyse the Fe_n MagPT ordering runs: total vs structure-only vs magnetic ordering, and same-geometry pairs.

    python experiments/fen_spin_ordering/analyze_ordering.py \
        --states D:/MLIP_Work_Folder/Cluster_MLIP/fen_oct9_states/states.extxyz \
        --report experiments/fen_spin_ordering/report

Per (n, q) group with >= 2 multiplicities, each predictor is compared with the UBPW91 energies by Spearman rank
correlation, by the fraction of state pairs with |dE_Gaussian| >= 0.1 eV that it orders the same way, and by whether
it picks the same ground-state multiplicity:
  magpt_true      MagPT, Mulliken moments
  magpt_uniform   MagPT, same total moment spread evenly
  magpt_zero      MagPT, all moments 0: structure only
  magnetic        magpt_true - magpt_zero: the part of MagPT's energy that comes from the moments
  M_ascending     "lower multiplicity is lower" (no model)
Same-geometry pairs (sorted-distance RMS < 0.05 A, different M, same n and q) isolate the spin effect: there the
structure term nearly cancels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PREDICTORS = ("magpt_true", "magpt_uniform", "magpt_zero", "magnetic", "M_ascending")
SAME_GEOMETRY_RMS = 0.05


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return math.nan
    ra = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(float)
    rb = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort").astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def fingerprint(pos: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    return np.sort(d[np.triu_indices(len(d), 1)])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args(argv)
    from ase.io import read

    rep = Path(args.report)
    rows = list(csv.DictReader(open(rep / "magpt_states.csv", newline="", encoding="utf-8")))
    zero = {r["state_id"]: float(r["magpt_zero_ev"]) for r in csv.DictReader(open(rep / "magpt_zero_moment.csv", encoding="utf-8"))}
    pos = {a.info["state_id"]: a.positions.copy() for a in read(args.states, ":") if a.info["clean"]}
    for r in rows:
        for k in ("n", "charge", "multiplicity", "n_imaginary"):
            r[k] = int(r[k])
        for k in ("gaussian_energy_ev", "magpt_true_ev", "magpt_uniform_ev", "magpt_scale0.9_ev", "magpt_scale1.1_ev"):
            r[k] = float(r[k])
        r["artifact"] = r["artifact"] == "True"
        r["magpt_true"], r["magpt_uniform"] = r["magpt_true_ev"], r["magpt_uniform_ev"]
        r["magpt_zero"] = zero[r["state_id"]]
        r["magnetic"] = r["magpt_true"] - r["magpt_zero"]
        r["M_ascending"] = float(r["multiplicity"])

    groups = defaultdict(list)
    for r in rows:
        groups[(r["n"], r["charge"])].append(r)

    def evaluate(keep) -> dict:
        per_group, pooled = [], {p: {"conc": 0, "pairs": 0, "gs": 0, "rho": []} for p in PREDICTORS}
        for (n, q), members in sorted(groups.items()):
            rs = [r for r in members if keep(r)]
            if len({r["multiplicity"] for r in rs}) < 2:
                continue
            g = np.array([r["gaussian_energy_ev"] for r in rs])
            entry = {"n": n, "charge": q, "states": len(rs), "M_min": min(r["multiplicity"] for r in rs),
                     "M_max": max(r["multiplicity"] for r in rs), "gaussian_ground_M": rs[int(np.argmin(g))]["multiplicity"],
                     "gaussian_spread_ev": float(g.max() - g.min())}
            for p in PREDICTORS:
                v = np.array([r[p] for r in rs])
                big = [(i, j) for i in range(len(g)) for j in range(i + 1, len(g)) if abs(g[j] - g[i]) >= 0.1]
                conc = sum(np.sign(g[j] - g[i]) == np.sign(v[j] - v[i]) for i, j in big)
                rho = spearman(g, v)
                gs = rs[int(np.argmin(v))]["multiplicity"] == entry["gaussian_ground_M"]
                entry[f"{p}_spearman"] = rho
                entry[f"{p}_ground_M"] = rs[int(np.argmin(v))]["multiplicity"]
                entry[f"{p}_pairs_concordant"] = f"{conc}/{len(big)}"
                pooled[p]["conc"] += int(conc)
                pooled[p]["pairs"] += len(big)
                pooled[p]["gs"] += int(gs)
                if not math.isnan(rho):
                    pooled[p]["rho"].append(rho)
            per_group.append(entry)
        summary = {p: {"groups": len(per_group), "median_spearman": float(np.median(s["rho"])) if s["rho"] else math.nan,
                       "groups_spearman_positive": int(sum(x > 0 for x in s["rho"])), "groups_with_spearman": len(s["rho"]),
                       "pairs_dE_ge_0.1eV": s["pairs"],
                       "pairs_concordant_fraction": s["conc"] / s["pairs"] if s["pairs"] else math.nan,
                       "ground_state_M_matches": s["gs"]} for p, s in pooled.items()}
        return {"summary": summary, "groups": per_group}

    subsets = {"all_clean": lambda r: True, "minima_only": lambda r: r["n_imaginary"] <= 0,
               "no_artifact_minima": lambda r: not r["artifact"] and r["n_imaginary"] <= 0}
    result = {k: evaluate(f) for k, f in subsets.items()}

    # same-geometry, different-M pairs: the structure term cancels, so only the spin part is tested
    pairs = []
    for (n, q), rs in groups.items():
        fps = [fingerprint(pos[r["state_id"]]) for r in rs]
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if rs[i]["multiplicity"] == rs[j]["multiplicity"] or len(fps[i]) != len(fps[j]):
                    continue
                d = float(np.sqrt(np.mean((fps[i] - fps[j]) ** 2)))
                if d >= SAME_GEOMETRY_RMS:
                    continue
                a, b = sorted((rs[i], rs[j]), key=lambda r: r["multiplicity"])
                dg = b["gaussian_energy_ev"] - a["gaussian_energy_ev"]
                pairs.append({"n": n, "charge": q, "M_low": a["multiplicity"], "M_high": b["multiplicity"],
                              "fingerprint_rms_ang": d, "gaussian_dE_ev": dg,
                              **{f"{p}_dE_ev": b[p] - a[p] for p in ("magpt_true", "magpt_uniform", "magpt_zero", "magnetic")},
                              "magpt_true_sign_agrees": bool(np.sign(dg) == np.sign(b["magpt_true"] - a["magpt_true"])),
                              "artifact_in_pair": a["artifact"] or b["artifact"]})
    big = [p for p in pairs if abs(p["gaussian_dE_ev"]) >= 0.05]
    result["same_geometry_pairs"] = {
        "pairs": len(pairs), "pairs_dE_ge_0.05eV": len(big),
        "sign_agree_dE_ge_0.05eV": sum(p["magpt_true_sign_agrees"] for p in big),
        "magpt_says_higher_M_lower": sum(p["magpt_true_dE_ev"] < 0 for p in pairs),
        "gaussian_says_higher_M_lower": sum(p["gaussian_dE_ev"] < 0 for p in pairs),
        "spearman_dE_true": spearman([p["gaussian_dE_ev"] for p in pairs], [p["magpt_true_dE_ev"] for p in pairs]),
    }
    with open(rep / "same_geometry_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        if pairs:
            w = csv.DictWriter(fh, fieldnames=list(pairs[0]))
            w.writeheader()
            w.writerows(pairs)
    with open(rep / "groups.csv", "w", newline="", encoding="utf-8") as fh:
        g = result["minima_only"]["groups"]
        w = csv.DictWriter(fh, fieldnames=list(g[0]))
        w.writeheader()
        w.writerows(g)
    (rep / "ordering_summary.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    for k in subsets:
        print(k)
        for p, s in result[k]["summary"].items():
            print(f"  {p:14s} median rho {s['median_spearman']:+.2f}  rho>0 {s['groups_spearman_positive']}/{s['groups_with_spearman']}"
                  f"  pairs(|dE|>=0.1) {s['pairs_concordant_fraction']:.2f} of {s['pairs_dE_ge_0.1eV']}  GS-M {s['ground_state_M_matches']}/{s['groups']}")
    print("same-geometry pairs", json.dumps(result["same_geometry_pairs"], default=float))


if __name__ == "__main__":
    main()
