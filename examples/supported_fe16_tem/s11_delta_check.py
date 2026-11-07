"""Step 11: how good is the interaction term?  Compare it with VASP dE_int labels.

Reads interaction.extxyz from `cluster-mlip vasp-collect` and evaluates the configured
--interaction term (common.py) on the same frames:

  * the default, 'subtractive' MACE-MP-0 + D3, quantifies how wrong the foundation
    model's Fe/support interaction is (the MgO question in the README), and
  * a model trained on the interaction labels is validated on held-out frames.

Also prints, per support, the range of the VASP interaction energies and the model error.

usage: s11_delta_check.py <labels_dir> [--interaction SPEC] [--support-model SPEC]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import read

from common import add_model_args, build_models
from cluster_mlip.delta import reset_calculator

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("labels", help="output directory of cluster-mlip vasp-collect")
add_model_args(parser)
args = parser.parse_args()
models = build_models(args)
calc = models.interaction

frames = read(Path(args.labels) / "interaction.extxyz", ":")
rows = []
for a in frames:
    ref_e = float(a.info["REF_energy"])
    ref_f = np.asarray(a.arrays["REF_forces"])
    x = a.copy()
    x.info = {"charge": int(a.info.get("charge", 0)), "spin": int(a.info.get("spin", 1))}
    reset_calculator(calc)
    x.calc = calc
    e = float(x.get_potential_energy())
    f = np.asarray(x.get_forces(apply_constraint=False))
    rows.append(dict(structure_id=a.info["structure_id"], support=a.info.get("support", "?"),
                     multiplicity=int(a.info.get("multiplicity", 1)), E_int_vasp=ref_e, E_int_model=e,
                     dE=e - ref_e, force_mae=float(np.abs(f - ref_f).mean()),
                     force_max_err=float(np.linalg.norm(f - ref_f, axis=1).max())))
    print(f"{rows[-1]['structure_id']:32s} M={rows[-1]['multiplicity']:3d}  "
          f"E_int VASP {ref_e:+8.3f}  model {e:+8.3f}  dF_MAE {rows[-1]['force_mae']:.3f} eV/A", flush=True)

by_support = defaultdict(list)
for r in rows:
    by_support[r["support"]].append(r)
summary = {}
for s, rs in by_support.items():
    de = np.array([r["dE"] for r in rs])
    summary[s] = dict(n=len(rs), E_int_vasp_mean=float(np.mean([r["E_int_vasp"] for r in rs])),
                      E_int_vasp_range=[float(min(r["E_int_vasp"] for r in rs)),
                                        float(max(r["E_int_vasp"] for r in rs))],
                      energy_mae=float(np.abs(de).mean()), energy_bias=float(de.mean()),
                      force_mae=float(np.mean([r["force_mae"] for r in rs])))
    print(f"{s}: n={len(rs)}  VASP E_int {summary[s]['E_int_vasp_range'][0]:+.2f}..{summary[s]['E_int_vasp_range'][1]:+.2f} eV"
          f"  model MAE {summary[s]['energy_mae']:.3f} eV (bias {summary[s]['energy_bias']:+.3f})"
          f"  force MAE {summary[s]['force_mae']:.3f} eV/A")
path = models.out / "delta_check.json"
path.write_text(json.dumps({"models": models.describe(), "summary": summary, "rows": rows}, indent=1))
print("wrote", path)
