"""Step 3: Langevin MD of the lowest-energy supported Fe16 on each support (MACE-MP-0 + D3).

Same conditions as the abTEM tutorial (700 K, 10 ps) except a 1 fs step
(graphene C-C modes) and a 50 fs sampling interval -> 200 frames.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, thermalize_momenta

warnings.filterwarnings("ignore")
import torch  # noqa: E402
from mace.calculators import mace_mp  # noqa: E402

OUT = Path(__file__).parent / "out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_K, DT_FS, TOTAL_FS, EVERY = 700, 1.0, float(sys.argv[1]) if len(sys.argv) > 1 else 10_000, 50

calc = mace_mp(model="medium", dispersion=True, default_dtype="float32", device=DEVICE)

SUPPORTS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["graphene", "MgO100"]
for sname in SUPPORTS:
    frames = read(OUT / f"supported_{sname}.extxyz", ":")
    block = json.loads((OUT / "results.json").read_text())[sname]["isomers"]
    iso = min(block, key=lambda n: block[n]["E_tot"])
    atoms = next(a for a in frames if a.info["isomer"] == iso).copy()
    # extxyz drops constraints; restore them (MgO bottom layer / one pinned C)
    from ase.constraints import FixAtoms
    is_cluster = np.array([s == "Fe" for s in atoms.get_chemical_symbols()])
    zs = atoms.positions[~is_cluster, 2]
    if sname == "MgO100":
        fix = np.where(~is_cluster & (atoms.positions[:, 2] < zs.min() + 0.5))[0]
    else:
        sup = np.where(~is_cluster)[0]
        fix = [int(sup[np.argmin(np.linalg.norm(atoms.positions[sup, :2], axis=1))])]
    atoms.set_constraint(FixAtoms(indices=fix))
    atoms.calc = calc
    rng = np.random.default_rng(13)
    thermalize_momenta(atoms, T_K, rng=rng)
    Stationary(atoms)
    traj = []
    dyn = Langevin(atoms, timestep=DT_FS * units.fs, temperature_K=T_K, friction=0.002,
                   fixcm=False, rng=rng)
    dyn.attach(lambda: traj.append(atoms.copy()), interval=EVERY)
    t0 = time.time()
    dyn.run(int(TOTAL_FS / DT_FS))
    for f in traj:
        f.info.update(support=sname, isomer=iso)
        f.calc = None
    write(OUT / f"md_{sname}.extxyz", traj)
    print(f"{sname} {iso}: {len(traj)} frames, {time.time() - t0:.0f}s, "
          f"T_end={atoms.get_temperature():.0f} K", flush=True)
