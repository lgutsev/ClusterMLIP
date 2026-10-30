"""MagPT spin-state ordering test on legacy UBPW91/6-311+G* Fe_n (n = 2-25), per (n, charge).

    D:/MLIP_Work_Folder/envs/magpt_dp/python.exe experiments/fen_spin_ordering/run_magpt_ordering.py \
        --states D:/MLIP_Work_Folder/Cluster_MLIP/fen_oct9_states/states.extxyz \
        --checkpoint D:/MLIP_Work_Folder/external/magpt/MagPT/model/MagPT_399168.pth \
        --out experiments/fen_spin_ordering/report

Unlike the Fe16 fixed-geometry test, each state here sits at its own optimized geometry (adiabatic ordering): the
question is whether MagPT, given each state's geometry and Mulliken moments m_i = (0, 0, s_i), orders the states of
one (n, q) the way UBPW91 does. Only relative energies within a group are compared. Isolated clusters, no cell
(nopbc agreed with 20-30 A boxes to 0.04 meV in the Fe16 test). Conditions per state:
  true      Mulliken moments as extracted
  uniform   (M - 1)/n on every atom: same total moment, no per-atom pattern (does the pattern matter?)
  scale0.9 / scale1.1   magnitude sensitivity, signs kept
Baselines that need no model: rank by M ascending ("lower M lower") and descending.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cluster_mlip.magpt_diagnostic import DeepSpinEvaluator, moment_vectors, pairwise_ordering, spearman  # noqa: E402

CONDITIONS = ("true", "uniform", "scale0.9", "scale1.1")


def moments_for(cond: str, spins: np.ndarray, multiplicity: int) -> np.ndarray:
    if cond == "true":
        return moment_vectors(spins)
    if cond == "uniform":
        return moment_vectors(np.full(len(spins), (multiplicity - 1) / len(spins)))
    return moment_vectors(spins, scale=float(cond.removeprefix("scale")))


def group_metrics(rows: list[dict], key: str) -> dict:
    g = np.array([r["gaussian_energy_ev"] for r in rows])
    p = np.array([r[key] for r in rows])
    order = pairwise_ordering(g, p, tie_ev=0.0)
    big = [(i, j) for i in range(len(g)) for j in range(i + 1, len(g)) if abs(g[j] - g[i]) >= 0.1]
    agree_big = sum(np.sign(g[j] - g[i]) == np.sign(p[j] - p[i]) for i, j in big)
    return {"spearman": order["spearman_rho"], "kendall": order["kendall_tau_a"],
            "pairs": order["n_pairs"], "pairs_concordant": order["concordant_pairs"],
            "pairs_dE_ge_0.1eV": len(big), "pairs_dE_ge_0.1eV_concordant": int(agree_big),
            "ground_state_M_match": bool(rows[int(np.argmin(g))]["multiplicity"] == rows[int(np.argmin(p))]["multiplicity"])}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    import torch
    from ase.io import read

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = [a for a in read(args.states, ":") if a.info["clean"]]
    groups: dict[tuple[int, int], list] = defaultdict(list)
    for a in frames:
        groups[(len(a), int(a.info["charge"]))].append(a)
    groups = {k: v for k, v in groups.items() if len({int(a.info["multiplicity"]) for a in v}) >= 2}
    model = DeepSpinEvaluator(args.checkpoint)
    t0 = time.time()
    # Resumable: every finished state is appended to magpt_states.csv at once, and a rerun skips those states
    # (the first full run was killed at Fe22 when the external drive holding the environment dropped out).
    csv_path = out / "magpt_states.csv"
    rows, done = [], set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add(r["state_id"])
                rows.append(r)
    writer_fh = open(csv_path, "a", newline="", encoding="utf-8")
    writer = None
    for (n, q), members in sorted(groups.items()):
        for a in members:
            if a.info["state_id"] in done:
                continue
            spins = np.array(a.arrays["mulliken_spin"], dtype=float)
            M = int(a.info["multiplicity"])
            r = {"n": n, "charge": q, "multiplicity": M, "state_id": a.info["state_id"], "source": a.info["source"],
                 "gaussian_energy_ev": float(a.info["REF_energy"]), "artifact": bool(a.info["artifact"]),
                 "genuine_antiparallel": bool(np.any(spins < -0.5) and not np.any(np.abs(spins[spins < -0.5]) > 5.0)),
                 "n_imaginary": int(a.info["n_imaginary"]), "scf_flags": a.info["scf_flags"],
                 "total_moment": round(float(spins.sum()), 4), "mean_abs_moment": round(float(np.abs(spins).mean()), 4),
                 "max_abs_moment": round(float(np.abs(spins).max()), 4)}
            for cond in CONDITIONS:
                e, _f, fm = model(a.positions, None, moments_for(cond, spins, M), a.get_chemical_symbols())
                r[f"magpt_{cond}_ev"] = e
                if cond == "true":
                    r["magpt_mag_force_rms"] = float(np.sqrt(np.mean(fm[:, 2] ** 2)))
            if writer is None:
                writer = csv.DictWriter(writer_fh, fieldnames=list(r))
                if not done:
                    writer.writeheader()
            writer.writerow(r)
            writer_fh.flush()
            rows.append(r)
        print(f"Fe{n} q={q:+d}: {len(members)} states, {time.time() - t0:.0f} s", flush=True)
    writer_fh.close()

    # rows read back from the CSV are strings: restore the types the analysis needs
    for r in rows:
        for k in ("n", "charge", "multiplicity", "n_imaginary"):
            r[k] = int(r[k])
        for k in ["gaussian_energy_ev"] + [f"magpt_{c}_ev" for c in CONDITIONS]:
            r[k] = float(r[k])
        for k in ("artifact", "genuine_antiparallel"):
            r[k] = r[k] in (True, "True")

    def evaluate(subset_name: str, keep) -> dict:
        per_group, pooled = [], defaultdict(lambda: [0, 0])
        for (n, q) in sorted(groups):
            rs = [r for r in rows if r["n"] == n and r["charge"] == q and keep(r)]
            if len({r["multiplicity"] for r in rs}) < 2:
                continue
            for r in rs:
                r["rank_M_asc"] = r["multiplicity"]
                r["rank_M_desc"] = -r["multiplicity"]
            entry = {"n": n, "charge": q, "states": len(rs), "multiplicities": sorted({r["multiplicity"] for r in rs}),
                     "gaussian_ground_M": rs[int(np.argmin([r["gaussian_energy_ev"] for r in rs]))]["multiplicity"]}
            for key in [f"magpt_{c}_ev" for c in CONDITIONS] + ["rank_M_asc", "rank_M_desc"]:
                m = group_metrics(rs, key)
                entry[key] = m
                pooled[key][0] += m["pairs_dE_ge_0.1eV_concordant"]
                pooled[key][1] += m["pairs_dE_ge_0.1eV"]
                pooled[key + "|gs"] = pooled.get(key + "|gs", [0, 0])
                pooled[key + "|gs"][0] += int(m["ground_state_M_match"])
                pooled[key + "|gs"][1] += 1
            per_group.append(entry)
        summary = {}
        for key in [f"magpt_{c}_ev" for c in CONDITIONS] + ["rank_M_asc", "rank_M_desc"]:
            sp = [g[key]["spearman"] for g in per_group if not math.isnan(g[key]["spearman"])]
            conc, tot = pooled[key]
            gs, ng = pooled[key + "|gs"]
            summary[key] = {"groups": len(per_group), "median_spearman": float(np.median(sp)) if sp else math.nan,
                            "groups_positive_spearman": int(sum(x > 0 for x in sp)), "groups_with_spearman": len(sp),
                            "pairs_dE_ge_0.1eV": tot, "pairs_concordant_fraction": conc / tot if tot else math.nan,
                            "ground_state_M_matches": gs, "ground_state_M_groups": ng}
        return {"subset": subset_name, "summary": summary, "groups": per_group}

    subsets = {
        "all_clean": lambda r: True,
        "minima_only": lambda r: r["n_imaginary"] <= 0,
        "no_artifact_site": lambda r: not r["artifact"],
        "no_artifact_minima": lambda r: not r["artifact"] and r["n_imaginary"] <= 0,
    }
    result = {name: evaluate(name, keep) for name, keep in subsets.items()}
    import deepmd

    result["provenance"] = {"checkpoint": Path(args.checkpoint).name, "deepmd_version": deepmd.__version__,
                            "torch": torch.__version__, "threads": args.threads, "cell": "none (nopbc)",
                            "states_file": args.states, "runtime_s": round(time.time() - t0)}
    (out / "magpt_ordering.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    for name in subsets:
        print(name, json.dumps(result[name]["summary"], default=float))


if __name__ == "__main__":
    main()
