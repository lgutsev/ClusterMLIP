"""Step 7: can a spin-aware foundation model (MACE-POLAR-1) reproduce the UBPW91 Fe16 spin ladder?

Sections (each saved to out/polar_spin.json as soon as it finishes):
  sanity   O2, Fe2, Fe4 ladders -- confirms charge/spin are passed correctly
  states   float64 single points of POLAR-1 and spin-blind MACE-MP-0 on every
           adiabatic UBPW91 state (lowest record per geometry group and multiplicity)
  ladder   E(M), M = 1..65, at the 6 lowest DFT geometries
  relax    (only with --relax) relax each DFT minimum with POLAR-1 at its ground M

Every energy is a float64 single point: POLAR energies are ~ -5.5e5 eV for Fe16 and
float32 rounds them to 0.0625 eV steps, the size of the gaps being tested.
The predicted atomic spins are kept as a validity check: an Fe atom carries at most
~4 unpaired electrons, so |s_i| far above that means the spin equilibration diverged.

usage: s7_polar_spin.py <seeds.extxyz> [polar-1-s|m|l] [--relax]
Needs graph_electrostatics v0.4.0 (the API mace-torch 0.3.16 calls):
  pip install --no-deps "git+https://github.com/WillBaldwin0/graph_electrostatics@v0.4.0"
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import LBFGS

warnings.filterwarnings("ignore")
import torch  # noqa: E402
from mace.calculators import mace_mp, mace_polar  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SEEDS = args[0] if args else "seeds.extxyz"
MODEL = args[1] if len(args) > 1 else "polar-1-m"
RELAX = "--relax" in sys.argv
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
RES = OUT / "polar_spin.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HARTREE = 27.211386245988
LADDER = list(range(1, 67, 2))
FE_SPIN_BOUND = 4.0

polar64 = mace_polar(MODEL, device=DEVICE, default_dtype="float64")
mp64 = mace_mp(model="medium", device=DEVICE, default_dtype="float64")
res: dict = {"model": MODEL}
t0 = time.time()


def save():
    RES.write_text(json.dumps(res, indent=1, default=float))


def sp(calc, atoms, mult):
    a = atoms.copy()
    a.info = {"charge": 0, "spin": int(mult)}
    a.pbc = False
    a.calc = calc
    e = float(a.get_potential_energy())
    s = calc.results.get("spins")
    s = None if s is None else np.asarray(s, float)
    return e, s


# ------------------------------------------------------------------ sanity
fe4 = Atoms("Fe4", positions=[[0, 0, 0], [2.3, 0, 0], [1.15, 1.99, 0], [1.15, 0.66, 1.88]])
sanity = {}
for name, atoms, mults in (("O2", Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.21]]), [1, 3, 5]),
                           ("Fe2", Atoms("Fe2", positions=[[0, 0, 0], [0, 0, 2.02]]), [1, 3, 5, 7, 9]),
                           ("Fe4", fe4, [1, 3, 5, 7, 9, 11, 13, 15, 17])):
    rows = []
    for m in mults:
        e, s = sp(polar64, atoms, m)
        rows.append(dict(mult=m, E=e, spin_sum=float(s.sum()), spin_max=float(np.abs(s).max())))
    e0 = min(r["E"] for r in rows)
    for r in rows:
        r["E_rel"] = r["E"] - e0
    sanity[name] = rows
    print(name, "ground M =", min(rows, key=lambda r: r["E"])["mult"], flush=True)
res["sanity"] = sanity
save()


# ------------------------------------------------------------------ warehouse
def fingerprint(a):
    d = a.get_all_distances()
    return np.sort(d[np.triu_indices(len(a), 1)])


records = [a for a in read(SEEDS, ":")
           if a.get_chemical_formula() == "Fe16"
           and a.info.get("config_type") in ("minimum", "optimized_unverified")
           and a.info.get("legacy_energy_hartree") is not None]
groups: list[dict] = []
for a in records:
    fp = fingerprint(a)
    for g in groups:
        if np.abs(fp - g["fp"]).max() < 0.10:
            g["members"].append(a)
            break
    else:
        groups.append(dict(fp=fp, members=[a]))
for g in groups:
    best = {}
    for a in g["members"]:
        m = int(a.info["multiplicity"])
        e = float(a.info["legacy_energy_hartree"]) * HARTREE
        if m not in best or e < best[m][0]:
            best[m] = (e, a)
    g["by_mult"] = best
    g["e_min"] = min(v[0] for v in best.values())
    g["m0"] = min(best, key=lambda m: best[m][0])
groups.sort(key=lambda g: g["e_min"])
e_ref = groups[0]["e_min"]
res.update(n_records=len(records), n_groups=len(groups))
print(f"{len(records)} Fe16 records, {len(groups)} geometry groups", flush=True)

# ------------------------------------------------------------------ states
states = []
for gi, g in enumerate(groups):
    for m, (e_dft, a) in sorted(g["by_mult"].items()):
        e_pol, s = sp(polar64, a, m)
        e_mp, _ = sp(mp64, a, m)
        states.append(dict(group=gi, mult=m, e_dft=e_dft - e_ref, e_polar=e_pol, e_mp=e_mp,
                           spin_sum=float(s.sum()), spin_max=float(np.abs(s).max())))
res["states"] = states
save()
bad = np.mean([s["spin_max"] > FE_SPIN_BOUND for s in states])
print(f"{len(states)} adiabatic states; {100 * bad:.0f}% have max|s_i| > {FE_SPIN_BOUND} "
      f"[{time.time() - t0:.0f}s]", flush=True)

# ------------------------------------------------------------------ ladder
curves = []
for gi, g in enumerate(groups[:6]):
    geom = g["by_mult"][g["m0"]][1]
    e, smax = [], []
    for m in LADDER:
        x, s = sp(polar64, geom, m)
        e.append(x)
        smax.append(float(np.abs(s).max()))
    e = np.array(e) - e[LADDER.index(g["m0"])]
    curves.append(dict(group=gi, dft_ground_mult=g["m0"],
                       dft={int(m): v[0] - g["by_mult"][g["m0"]][0] for m, v in g["by_mult"].items()},
                       polar=e.tolist(), spin_max=smax))
    print(f"ladder group {gi}: DFT M={g['m0']}, POLAR argmin M={LADDER[int(np.argmin(e))]}, "
          f"E range {e.min():.0f}..{e.max():.0f} eV", flush=True)
res["ladder"] = dict(mults=LADDER, curves=curves)
save()

# ------------------------------------------------------------------ relax (opt-in)
if RELAX:
    polar32 = mace_polar(MODEL, device=DEVICE, default_dtype="float32")
    relaxed = []
    for gi, g in enumerate(groups):
        a = g["by_mult"][g["m0"]][1].copy()
        a.info = {"charge": 0, "spin": g["m0"]}
        a.calc = polar32
        opt = LBFGS(a, maxstep=0.15, logfile=None)
        ok = opt.run(fmax=0.03, steps=1500)
        e64, s = sp(polar64, a, g["m0"])
        a.calc = None  # shared calculator: its cached energy belongs to another structure
        relaxed.append(dict(group=gi, mult=g["m0"], E=e64, steps=opt.nsteps, ok=bool(ok),
                            dft_rel=g["e_min"] - e_ref, spin_max=float(np.abs(s).max()), atoms=a))
    write(OUT / "polar_relaxed.extxyz", [r.pop("atoms") for r in relaxed])
    res["relaxed"] = relaxed
    save()
print(f"done in {time.time() - t0:.0f}s")
