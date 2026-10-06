"""Per-example errors and spin-state ordering for A/B models.

usage: python ordering_report.py DATASET_DIR OUT.json NAME=MODEL[:local_moment] ...

* The central-negative-spin example: every frame whose most negative moment is < -5 (e.g. the
  Fe16 M=53 frame with atom 12 = -5.61): energy and force error per model.
* Spin-state ordering on matched geometries: for every pair of frames with identical positions
  and different multiplicity, does the model put the two states in the same order as DFT?
  Reported as the fraction of pairs ordered correctly, per split, and the gap error.
Moments go through cluster_mlip.local_moments.set_local_moments (canonical sign, cache guard).
"""
import collections, hashlib, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from cluster_mlip.local_moments import set_local_moments  # noqa: E402

dataset, out_path, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
splits = {s: read(f"{dataset}/{s}.extxyz", ":") for s in ("train", "valid", "test")}
report = {}
for spec in specs:
    name, _, rest = spec.partition("=")
    path, key = (rest.rsplit(":", 1) if rest.endswith(":local_moment") else (rest, ""))
    calc = MACECalculator(model_paths=path, device="cuda", default_dtype="float64",
                          arrays_keys={"local_moment": "local_moment"} if key else None)
    res = {"example_central_negative": [], "ordering": {}}
    for split, frames in splits.items():
        E = {}
        for i, a in enumerate(frames):
            b = a.copy()
            if key:
                set_local_moments(b, a.arrays["local_moment"])
            calc.reset(); b.calc = calc
            E[i] = b.get_potential_energy()
            if a.arrays["local_moment"].min() < -5:
                f = b.get_forces()
                res["example_central_negative"].append({
                    "split": split, "record_id": a.info.get("record_id"), "M": int(a.info["spin"]),
                    "min_moment": float(a.arrays["local_moment"].min()),
                    "de_meV_atom": 1000 * (E[i] - a.info["REF_energy"]) / len(a),
                    "rmse_f_meV_A": 1000 * float(np.sqrt(((f - a.arrays["REF_forces"]) ** 2).mean()))})
        geo = collections.defaultdict(list)
        for i, a in enumerate(frames):
            geo[hashlib.sha1(np.round(a.positions, 4).tobytes()).hexdigest()].append(i)
        ok = n = 0; err = []
        for idx in geo.values():
            for x in range(len(idx)):
                for y in range(x + 1, len(idx)):
                    i, j = idx[x], idx[y]
                    if frames[i].info["spin"] == frames[j].info["spin"]:
                        continue
                    dft = frames[i].info["REF_energy"] - frames[j].info["REF_energy"]
                    mod = E[i] - E[j]
                    n += 1; ok += np.sign(dft) == np.sign(mod); err.append(1000 * abs(mod - dft))
        res["ordering"][split] = {"pairs": n, "correct_order_fraction": (ok / n) if n else None,
                                  "gap_mae_meV": float(np.mean(err)) if err else None}
    ex = res["example_central_negative"]
    res["example_summary"] = {"frames": len(ex),
                              "mean_abs_de_meV_atom": float(np.mean([abs(e["de_meV_atom"]) for e in ex])) if ex else None,
                              "mean_rmse_f_meV_A": float(np.mean([e["rmse_f_meV_A"] for e in ex])) if ex else None}
    report[name] = res
    o = res["ordering"]
    print(f"{name}: central-negative frames {len(ex)}, |dE| {res['example_summary']['mean_abs_de_meV_atom']:.1f} meV/atom, "
          f"F {res['example_summary']['mean_rmse_f_meV_A']:.1f} meV/A | order correct "
          + " ".join(f"{s} {v['correct_order_fraction']:.2f} (n={v['pairs']})" for s, v in o.items() if v["pairs"]))
json.dump(report, open(out_path, "w"), indent=1)
