"""Smoke checks for a model trained with a per-atom `local_moment` input (model B).

usage: python smoke_local_moment.py MODEL.model DATASET.extxyz [OUT.json]

Uses ASE's MACECalculator exactly as a user would (arrays_keys must map the input).
Checks, on a frame that has one antiparallel atom:
  1. the input enters the energy: moving the antiparallel moment to another atom
     (same geometry, same multiplicity, same moment sum) changes E;
  2. a missing input fails loudly, both without arrays_keys and with the array absent;
  3. rotation invariance (E unchanged, forces rotate with the frame);
  4. permutation invariance (atoms and their moments permuted together);
  5. global spin reversal (all moments negated): reported, NOT enforced by the model;
  6. reload: a second load of the saved model gives identical results;
  7. ASE result caching: ASE reuses the last result when positions/numbers/cell/pbc/
     magmoms are unchanged and does NOT compare custom arrays, so a fixed-geometry
     scan over moments returns stale energies. Checked with and without the guard
     (moments mirrored into initial_magmoms, which ASE does compare).
Every check except 7 resets the calculator before each evaluation.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator
from scipy.spatial.transform import Rotation

model_path, data_path = sys.argv[1:3]
out_path = sys.argv[3] if len(sys.argv) > 3 else None
KEYS = {"local_moment": "local_moment"}


def calc(arrays_keys=KEYS):
    return MACECalculator(model_paths=model_path, device="cuda", default_dtype="float64",
                          arrays_keys=dict(arrays_keys) if arrays_keys is not None else None)


def ef(atoms, c, reset=True):
    if reset:
        c.reset()
    a = atoms.copy(); a.calc = c
    return a.get_potential_energy(), a.get_forces()


frames = read(data_path, ":")
base = next(a for a in frames if (a.arrays["local_moment"] < -1).sum() == 1)
lm = base.arrays["local_moment"].copy()
neg = int(np.argmin(lm))
c = calc()
E0, F0 = ef(base, c)
res = {"frame": base.info.get("record_id"), "M": int(base.info["spin"]), "moment_sum": float(lm.sum()),
       "antiparallel_atom": neg + 1, "E0": E0}

# 1. same geometry/M/sum, antiparallel moment placed on each other atom in turn
dEs = []
for k in range(len(base)):
    if k == neg:
        continue
    b = base.copy(); m = lm.copy(); m[[neg, k]] = m[[k, neg]]
    b.arrays["local_moment"] = m
    dEs.append(ef(b, c)[0] - E0)
dEs = np.array(dEs)
res["move_antiparallel_dE_eV"] = {"min": float(dEs.min()), "max": float(dEs.max()),
                                  "max_abs": float(np.abs(dEs).max())}
res["input_enters_energy"] = bool(np.abs(dEs).max() > 1e-6)

# 2. missing input must fail, never fall back to zeros
failures = {}
try:
    ef(base, calc(arrays_keys=None)); failures["no_arrays_keys"] = "NO ERROR (silent)"
except Exception as exc:  # noqa: BLE001
    failures["no_arrays_keys"] = type(exc).__name__
b = base.copy(); del b.arrays["local_moment"]
try:
    ef(b, c); failures["array_absent"] = "NO ERROR (silent)"
except Exception as exc:  # noqa: BLE001
    failures["array_absent"] = type(exc).__name__
res["missing_input"] = failures
res["missing_input_fails_loudly"] = all(v != "NO ERROR (silent)" for v in failures.values())

# 3. rotation
R = Rotation.random(random_state=0).as_matrix()
b = base.copy(); b.positions = (base.positions - base.positions.mean(0)) @ R.T
E, F = ef(b, c)
res["rotation_dE_eV"] = float(E - E0)
res["rotation_max_dF_eVA"] = float(np.abs(F - F0 @ R.T).max())

# 4. permutation of atoms together with their moments
p = np.random.default_rng(1).permutation(len(base))
b = base[p]; b.arrays["local_moment"] = lm[p]
E, F = ef(b, c)
res["permutation_dE_eV"] = float(E - E0)
res["permutation_max_dF_eVA"] = float(np.abs(F - F0[p]).max())

# 5. global reversal (not a symmetry of this model)
b = base.copy(); b.arrays["local_moment"] = -lm
res["reversal_dE_eV"] = float(ef(b, c)[0] - E0)

# 6. reload
E2, F2 = ef(base, calc())
res["reload_dE_eV"] = float(E2 - E0)
res["reload_max_dF_eVA"] = float(np.abs(F2 - F0).max())

# 7. caching pitfall: same geometry, different moments, no reset
moved = base.copy(); m = lm.copy(); m[[neg, (neg + 1) % len(base)]] = m[[(neg + 1) % len(base), neg]]
moved.arrays["local_moment"] = m
c7 = calc(); e_a = ef(base, c7, reset=False)[0]; e_b = ef(moved, c7, reset=False)[0]
e_true = ef(moved, calc())[0]
guard_a, guard_b = base.copy(), moved.copy()
guard_a.set_initial_magnetic_moments(guard_a.arrays["local_moment"])
guard_b.set_initial_magnetic_moments(guard_b.arrays["local_moment"])
c8 = calc(); g_a = ef(guard_a, c8, reset=False)[0]; g_b = ef(guard_b, c8, reset=False)[0]
res["cache_without_reset_dE_eV"] = float(e_b - e_a)
res["cache_true_dE_eV"] = float(e_true - e_a)
res["cache_with_magmom_guard_dE_eV"] = float(g_b - g_a)

print(json.dumps(res, indent=2))
if out_path:
    json.dump(res, open(out_path, "w"), indent=2)
