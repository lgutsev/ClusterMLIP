"""Relax DFT minima with model A (global charge/M) and model B (FIXED DFT local moments).

usage: python relax_test.py DATASET_DIR OUT.json A=MODEL B=MODEL:local_moment [...]

Starts from every converged DFT frame (stage_frame_position == "final") that has a full moment
table, relaxes with BFGS (fmax 0.02 eV/A, <= 300 steps) and reports the maximum and RMS atomic
displacement from the DFT minimum (after optimal rotation) and the energy drop. A good model
keeps a DFT minimum near where it is. Model B relaxations hold the supplied moments fixed: this
is a fixed-local-moment relaxation, not a self-consistent magnetic one.
"""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
from ase.io import read
from ase.optimize import BFGS
from mace.calculators import MACECalculator
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from cluster_mlip.local_moments import set_local_moments  # noqa: E402

dataset, out_path, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
frames = []
for split in ("train", "valid", "test"):
    for a in read(f"{dataset}/{split}.extxyz", ":"):
        md = json.loads(a.info["metadata"]) if isinstance(a.info.get("metadata"), str) else {}
        if md.get("stage_frame_position") == "final":
            frames.append((split, a))


def aligned_disp(x, y):
    x0, y0 = x - x.mean(0), y - y.mean(0)
    R, _ = Rotation.align_vectors(x0, y0)
    d = np.linalg.norm(x0 - y0 @ R.as_matrix().T, axis=1)
    return float(d.max()), float(np.sqrt((d ** 2).mean()))


report = {}
for spec in specs:
    name, _, rest = spec.partition("=")
    path, key = (rest.rsplit(":", 1) if rest.endswith(":local_moment") else (rest, ""))
    calc = MACECalculator(model_paths=path, device="cuda", default_dtype="float64",
                          arrays_keys={"local_moment": "local_moment"} if key else None)
    rows = []
    for split, a in frames:
        b = a.copy()
        if key:
            set_local_moments(b, a.arrays["local_moment"])
        calc.reset(); b.calc = calc
        e0 = b.get_potential_energy()
        opt = BFGS(b, logfile=None); ok = opt.run(fmax=0.02, steps=300)
        dmax, drms = aligned_disp(b.positions, a.positions)
        rows.append({"split": split, "M": int(a.info["spin"]), "converged": bool(ok), "steps": opt.nsteps,
                     "max_disp_A": dmax, "rms_disp_A": drms, "energy_drop_meV_atom": 1000 * (e0 - b.get_potential_energy()) / len(a)})
    summ = {s: {"n": len([r for r in rows if r["split"] == s]),
                "median_max_disp_A": float(np.median([r["max_disp_A"] for r in rows if r["split"] == s])),
                "median_drop_meV_atom": float(np.median([r["energy_drop_meV_atom"] for r in rows if r["split"] == s])),
                "converged": sum(r["converged"] for r in rows if r["split"] == s)}
            for s in ("train", "valid", "test") if any(r["split"] == s for r in rows)}
    report[name] = {"model": path, "fixed_local_moments": bool(key), "summary": summ, "rows": rows}
    print(name, json.dumps(summ))
json.dump(report, open(out_path, "w"), indent=1)
