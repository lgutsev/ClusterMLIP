"""Step 1: collect the distinct Fe16 minima from the warehouse seeds.

Per geometry we keep the lowest UBPW91 energy over all multiplicities (MACE-MP-0
has no spin, so the comparison is against the spin-optimised DFT surface).
"""
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

SEEDS = sys.argv[1] if len(sys.argv) > 1 else "seeds.extxyz"  # FenOm warehouse seeds
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
HARTREE = 27.211386245988

frames = [a for a in read(SEEDS, ":")
          if a.get_chemical_formula() == "Fe16"
          and a.info.get("config_type") in ("minimum", "optimized_unverified")
          and a.info.get("legacy_energy_hartree") is not None]


def fingerprint(a):
    d = a.get_all_distances()
    return np.sort(d[np.triu_indices(len(a), 1)])


groups: list[dict] = []
for a in frames:
    fp = fingerprint(a)
    e = float(a.info["legacy_energy_hartree"]) * HARTREE
    for g in groups:
        if np.abs(fp - g["fp"]).max() < 0.10:
            g["members"].append(a)
            if e < g["e"]:
                g.update(e=e, best=a)
            break
    else:
        groups.append(dict(fp=fp, e=e, best=a, members=[a]))

groups.sort(key=lambda g: g["e"])
e0 = groups[0]["e"]
print(f"{len(frames)} Fe16 optimised records -> {len(groups)} distinct geometries")
for i, g in enumerate(groups[:15]):
    mults = sorted({int(m.info["multiplicity"]) for m in g["members"]})
    print(f"  {i:2d} dE={g['e'] - e0:6.3f} eV  best M={int(g['best'].info['multiplicity'])}"
          f"  n={len(g['members'])}  mults={mults}")

# every distinct DFT geometry, for mapping DFT minima onto MACE-MP-0 basins
allg = []
for k, g in enumerate(groups):
    a = g["best"].copy()
    a.info = dict(isomer=f"D{k:02d}", dft_rel_eV=g["e"] - e0,
                  dft_mult=int(g["best"].info["multiplicity"]),
                  record_id=g["best"].info["record_id"], n_records=len(g["members"]))
    a.positions -= a.positions.mean(0)
    allg.append(a)
write(OUT / "all_isomers.extxyz", allg)
print("wrote", len(allg), "distinct DFT geometries")
