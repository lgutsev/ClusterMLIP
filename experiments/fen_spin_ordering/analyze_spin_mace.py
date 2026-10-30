"""Compare the spin MACE models with MagPT on the Fe_n ordering test (run after analyze_ordering.py).

    python experiments/fen_spin_ordering/analyze_spin_mace.py \
        --states D:/MLIP_Work_Folder/Cluster_MLIP/fen_oct9_states/states.extxyz \
        --report experiments/fen_spin_ordering/report

Same metrics as analyze_ordering.py, on the same clean states:
  whole (n, q) groups      median Spearman, pairs with |dE_Gaussian| >= 0.1 eV ordered correctly, ground-state M
  adjacent same-geometry   Delta M = 2, |dE_Gaussian| >= 0.05 eV, one pair per (n, q, M_low) (closest geometry):
                           sign agreement overall and by size, and for C whether the sign survives x0.9 / x1.1
Everything is also reported for neutral Fe16 alone, the only size the models were trained on.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_ordering import SAME_GEOMETRY_RMS, fingerprint, spearman

RUNS = ("long_B_s1", "long_B_s2", "long_C_s1", "long_C_s2")
SIZE_BINS = ((2, 6), (7, 12), (13, 18), (19, 25))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args(argv)
    from ase.io import read

    rep = Path(args.report)
    mace = {r["state_id"]: r for r in csv.DictReader(open(rep / "spin_mace_states.csv", newline="", encoding="utf-8"))}
    zero = {r["state_id"]: float(r["magpt_zero_ev"]) for r in csv.DictReader(open(rep / "magpt_zero_moment.csv", encoding="utf-8"))}
    pos = {a.info["state_id"]: a.positions.copy() for a in read(args.states, ":") if a.info["clean"]}
    rows = []
    for r in csv.DictReader(open(rep / "magpt_states.csv", newline="", encoding="utf-8")):
        m = mace[r["state_id"]]
        row = {"state_id": r["state_id"], "n": int(r["n"]), "charge": int(r["charge"]), "M": int(r["multiplicity"]),
               "g": float(r["gaussian_energy_ev"]), "magpt_true": float(r["magpt_true_ev"]), "magpt_zero": zero[r["state_id"]]}
        for k, v in m.items():
            if k.endswith("_ev"):
                row[k.removesuffix("_ev")] = float(v)
        rows.append(row)
    predictors = ["magpt_true", "magpt_zero"] + [f"{run}_true" for run in RUNS] + [f"{run}_uniform" for run in RUNS if "_C_" in run]

    groups = defaultdict(list)
    for r in rows:
        groups[(r["n"], r["charge"])].append(r)

    def whole_groups(keep_group) -> dict:
        out = {p: {"rho": [], "conc": 0, "pairs": 0, "gs": 0, "groups": 0} for p in predictors}
        for key, rs in groups.items():
            if not keep_group(key) or len({r["M"] for r in rs}) < 2:
                continue
            g = np.array([r["g"] for r in rs])
            big = [(i, j) for i in range(len(g)) for j in range(i + 1, len(g)) if abs(g[j] - g[i]) >= 0.1]
            for p in predictors:
                v = np.array([r[p] for r in rs])
                rho = spearman(g, v)
                if not math.isnan(rho):
                    out[p]["rho"].append(rho)
                out[p]["conc"] += int(sum(np.sign(g[j] - g[i]) == np.sign(v[j] - v[i]) for i, j in big))
                out[p]["pairs"] += len(big)
                out[p]["gs"] += int(rs[int(np.argmin(v))]["M"] == rs[int(np.argmin(g))]["M"])
                out[p]["groups"] += 1
        return {p: {"groups": s["groups"], "median_spearman": float(np.median(s["rho"])) if s["rho"] else math.nan,
                    "pairs_dE_ge_0.1eV": s["pairs"], "pairs_concordant_fraction": s["conc"] / s["pairs"] if s["pairs"] else math.nan,
                    "ground_state_M_matches": s["gs"]} for p, s in out.items()}

    # adjacent same-geometry pairs, the selection used for the MagPT numbers in the README
    best = {}
    for (n, q), rs in groups.items():
        for i in range(len(rs)):
            for j in range(len(rs)):
                a, b = rs[i], rs[j]
                if b["M"] - a["M"] != 2 or abs(b["g"] - a["g"]) < 0.05 or len(pos[a["state_id"]]) != len(pos[b["state_id"]]):
                    continue
                d = float(np.sqrt(np.mean((fingerprint(pos[a["state_id"]]) - fingerprint(pos[b["state_id"]])) ** 2)))
                if d < SAME_GEOMETRY_RMS and ((n, q, a["M"]) not in best or d < best[(n, q, a["M"])][0]):
                    best[(n, q, a["M"])] = (d, a, b)
    adjacent = [(k, a, b) for k, (_, a, b) in sorted(best.items())]

    def sign_ok(p, a, b) -> bool:
        return bool(np.sign(b["g"] - a["g"]) == np.sign(b[p] - a[p]))

    def adjacent_summary(sel) -> dict:
        res = {}
        for p in predictors:
            ok = [sign_ok(p, a, b) for _, a, b in sel]
            res[p] = {"pairs": len(ok), "sign_agree": int(sum(ok)),
                      "says_higher_M_lower": int(sum(b[p] < a[p] for _, a, b in sel)),
                      "by_size": {f"{lo}-{hi}": f"{sum(sign_ok(p, a, b) for k, a, b in sel if lo <= k[0] <= hi)}/"
                                                f"{sum(1 for k, *_ in sel if lo <= k[0] <= hi)}" for lo, hi in SIZE_BINS}}
        res["gaussian_says_higher_M_lower"] = int(sum(b["g"] < a["g"] for _, a, b in sel))
        for run in (r for r in RUNS if "_C_" in r):
            res[f"{run}_sign_stable_x0.9_x1.1"] = int(sum(
                len({np.sign(b[f"{run}_{c}"] - a[f"{run}_{c}"]) for c in ("scale0.9", "true", "scale1.1")}) == 1 for _, a, b in sel))
            res[f"{run}_median_dE_change_x0.9_x1.1_ev"] = float(np.median(
                [abs((b[f"{run}_scale1.1"] - a[f"{run}_scale1.1"]) - (b[f"{run}_scale0.9"] - a[f"{run}_scale0.9"])) for _, a, b in sel] or [math.nan]))
        return res

    trained = {r["state_id"]: r for r in csv.DictReader(open(rep / "spin_mace_states.csv", encoding="utf-8"))}
    result = {
        "whole_groups_all": whole_groups(lambda k: True),
        "whole_groups_fe16_neutral": whole_groups(lambda k: k == (16, 0)),
        "adjacent_pairs_all": adjacent_summary(adjacent),
        "adjacent_pairs_fe16_neutral": adjacent_summary([x for x in adjacent if x[0][:2] == (16, 0)]),
        "states_in_training_range": sum(r["in_training_range"] == "True" for r in trained.values()),
        "states_geometry_in_training": sum(r["geometry_in_training"] == "True" for r in trained.values()),
    }
    (rep / "spin_mace_ordering_summary.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    for key in ("whole_groups_all", "whole_groups_fe16_neutral"):
        print(key)
        for p, s in result[key].items():
            print(f"  {p:22s} rho {s['median_spearman']:+.2f}  pairs {s['pairs_concordant_fraction']:.2f} of {s['pairs_dE_ge_0.1eV']}"
                  f"  GS-M {s['ground_state_M_matches']}/{s['groups']}")
    for key in ("adjacent_pairs_all", "adjacent_pairs_fe16_neutral"):
        print(key, json.dumps(result[key], default=float))
    print("in training range", result["states_in_training_range"], "geometry in training", result["states_geometry_in_training"])


if __name__ == "__main__":
    main()
