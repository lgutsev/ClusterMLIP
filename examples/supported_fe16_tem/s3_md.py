"""Step 3: Langevin MD of the lowest-energy supported Fe16 on each support.

Same conditions as the abTEM tutorial (700 K, 10 ps) except a 1 fs step
(graphene C-C modes) and a 50 fs sampling interval -> 200 frames.

Uses the same model options as s2 (see common.py) and reads s2's results from the
same output directory; the multiplicity is the one s2 found for the supported state.
With a Delta-model every saved frame also carries its three energy terms
(E_cluster, E_support, E_interaction), which s10 uses to pick VASP snapshots.

usage: s3_md.py [total_fs] [supports] [--model SPEC ...]
"""
import argparse
import json
import time
import warnings

import numpy as np
from ase import units
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, thermalize_momenta

warnings.filterwarnings("ignore")
from common import add_model_args, build_models  # noqa: E402
from cluster_mlip.delta import reset_calculator  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("total_fs", nargs="?", type=float, default=10_000)
parser.add_argument("supports", nargs="?", default="graphene,MgO100")
add_model_args(parser)
args = parser.parse_args()
models = build_models(args)
OUT = models.out
T_K, DT_FS, TOTAL_FS, EVERY = 700, 1.0, args.total_fs, 50
calc = models.total

for sname in args.supports.split(","):
    frames = read(OUT / f"supported_{sname}.extxyz", ":")
    block = json.loads((OUT / "results.json").read_text())[sname]["isomers"]
    iso = min(block, key=lambda n: block[n]["E_tot"])
    atoms = next(a for a in frames if a.info["isomer"] == iso).copy()
    # extxyz drops constraints; restore them (MgO bottom layer / one pinned C)
    if "cluster" in atoms.arrays:
        is_cluster = atoms.arrays["cluster"].astype(bool)
    else:  # baseline files written before the 'cluster' column existed
        is_cluster = np.array([s == "Fe" for s in atoms.get_chemical_symbols()])
        atoms.arrays["cluster"] = is_cluster.astype(int)
    zs = atoms.positions[~is_cluster, 2]
    if sname == "MgO100":
        fix = np.where(~is_cluster & (atoms.positions[:, 2] < zs.min() + 0.5))[0]
    else:
        sup = np.where(~is_cluster)[0]
        fix = [int(sup[np.argmin(np.linalg.norm(atoms.positions[sup, :2], axis=1))])]
    atoms.set_constraint(FixAtoms(indices=fix))
    mult = int(block[iso].get("M_supported") or atoms.info.get("spin", 1))
    atoms.info = {"charge": 0, "spin": mult}
    reset_calculator(calc)
    atoms.calc = calc
    rng = np.random.default_rng(13)
    thermalize_momenta(atoms, T_K, rng=rng)
    Stationary(atoms)
    traj = []

    def snapshot():
        f = atoms.copy()
        f.info.update(support=sname, isomer=iso, charge=0, spin=mult, multiplicity=mult,
                      E_pot=float(atoms.get_potential_energy()))
        if models.is_delta:
            f.info.update({f"E_{k}": v for k, v in calc.terms.items()})
        f.calc = None
        traj.append(f)

    dyn = Langevin(atoms, timestep=DT_FS * units.fs, temperature_K=T_K, friction=0.002,
                   fixcm=False, rng=rng)
    dyn.attach(snapshot, interval=EVERY)
    t0 = time.time()
    dyn.run(int(TOTAL_FS / DT_FS))
    write(OUT / f"md_{sname}.extxyz", traj)
    print(f"{sname} {iso} (M={mult}): {len(traj)} frames, {time.time() - t0:.0f}s, "
          f"T_end={atoms.get_temperature():.0f} K", flush=True)
