"""Evaluate A/B models on the population-labelled dataset.

usage: python ab_eval.py DATASET_DIR OUT.json NAME=MODEL[:local_moment] [NAME=MODEL...]

For every model and split: energy (meV/atom) and force (meV/A) RMSE, overall and by split
group, multiplicity and spin pattern (number of Fe with moment < -1), next to geometry-blind
baselines (per-M training mean energy; zero force). Also the vertical spin-state gap
E(M) - E(M') on same-geometry pairs (DFT vs model). The calculator is reset before every
evaluation: ASE's result cache ignores custom per-atom arrays.
"""
import collections, hashlib, json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

dataset, out_path, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
splits = {s: read(f"{dataset}/{s}.extxyz", ":") for s in ("train", "valid", "test")}
mean_e = collections.defaultdict(list)
for a in splits["train"]:
    mean_e[int(a.info["spin"])].append(a.info["REF_energy"] / len(a))
mean_e = {k: float(np.mean(v)) for k, v in mean_e.items()}


def meta(a):
    md = a.info.get("metadata")
    return json.loads(md) if isinstance(md, str) else (md or {})


def pattern(a):
    if "local_moment" not in a.arrays:
        return "n/a"
    return f"{int((a.arrays['local_moment'] < -1).sum())} antiparallel"


def origin(a):
    return "campaign" if "collection_campaign" in meta(a) else "legacy"


def summarize(rows):
    return {"frames": len(rows),
            "rmse_e_meV_atom": round(1000 * float(np.sqrt(np.mean([r["de"] ** 2 for r in rows]))), 2),
            "baseline_e_meV_atom": round(1000 * float(np.sqrt(np.mean([r["de0"] ** 2 for r in rows]))), 2),
            "rmse_f_meV_A": round(1000 * float(np.sqrt(np.mean([r["df2"] for r in rows]))), 2),
            "zero_force_meV_A": round(1000 * float(np.sqrt(np.mean([r["f2"] for r in rows]))), 2)}


report = {}
for spec in specs:
    name, _, rest = spec.partition("=")
    # a trailing ":<array name>" marks a model with a per-atom input (paths may contain "D:")
    path, key = (rest.rsplit(":", 1) if rest.endswith(":local_moment") else (rest, ""))
    calc = MACECalculator(model_paths=path, device="cuda", default_dtype="float64",
                          arrays_keys={"local_moment": key} if key else None)
    res = {"model": path, "local_moment_input": bool(key), "splits": {}}
    for split, frames in splits.items():
        rows, energies = [], {}
        for i, a in enumerate(frames):
            calc.reset()
            b = a.copy(); b.calc = calc
            e, f = b.get_potential_energy(), b.get_forces()
            energies[i] = e
            n, M, ref_f = len(a), int(a.info["spin"]), a.arrays["REF_forces"]
            rows.append({"group": meta(a).get("split_group"), "M": M, "pattern": pattern(a), "origin": origin(a),
                         "de": (e - a.info["REF_energy"]) / n,
                         "de0": mean_e.get(M, np.mean(list(mean_e.values()))) - a.info["REF_energy"] / n,
                         "df2": float(((f - ref_f) ** 2).mean()), "f2": float((ref_f ** 2).mean())})
        by = lambda k: {str(v): summarize([r for r in rows if r[k] == v]) for v in sorted({r[k] for r in rows}, key=str)}
        # vertical gaps on identical geometries
        geo = collections.defaultdict(list)
        for i, a in enumerate(frames):
            geo[hashlib.sha1(np.round(a.positions, 4).tobytes()).hexdigest()].append(i)
        gaps = []
        for idx in geo.values():
            for x in range(len(idx)):
                for y in range(x + 1, len(idx)):
                    i, j = idx[x], idx[y]
                    if frames[i].info["spin"] == frames[j].info["spin"]:
                        continue
                    if frames[i].info["spin"] < frames[j].info["spin"]:
                        i, j = j, i
                    ref = frames[i].info["REF_energy"] - frames[j].info["REF_energy"]
                    gaps.append({"M_pair": f"{frames[i].info['spin']}-{frames[j].info['spin']}",
                                 "dft_gap_eV": ref, "model_gap_eV": energies[i] - energies[j]})
        gap_err = [g["model_gap_eV"] - g["dft_gap_eV"] for g in gaps]
        res["splits"][split] = {"all": summarize(rows), "by_group": by("group"), "by_M": by("M"),
                                "by_pattern": by("pattern"), "by_origin": by("origin"),
                                "vertical_gaps": {"pairs": len(gaps),
                                                  "mae_meV": round(1000 * float(np.mean(np.abs(gap_err))), 1) if gaps else None,
                                                  "max_abs_meV": round(1000 * float(np.max(np.abs(gap_err))), 1) if gaps else None,
                                                  "dft_gap_range_meV": [round(1000 * min(g["dft_gap_eV"] for g in gaps), 1),
                                                                        round(1000 * max(g["dft_gap_eV"] for g in gaps), 1)] if gaps else None}}
    report[name] = res
    for split, s in res["splits"].items():
        a = s["all"]; g = s["vertical_gaps"]
        print(f"{name:6} {split:5} E {a['rmse_e_meV_atom']:7.2f} ({a['baseline_e_meV_atom']:6.2f}) "
              f"F {a['rmse_f_meV_A']:7.2f} ({a['zero_force_meV_A']:6.2f})  gaps n={g['pairs']} MAE {g['mae_meV']} meV")
json.dump(report, open(out_path, "w"), indent=2)
