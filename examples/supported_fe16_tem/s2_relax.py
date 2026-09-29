"""Step 2: is the support a perturbation?  Fe16 isomers, gas phase vs graphene vs MgO(100).

One model for everything (MACE-MP-0 medium + D3(BJ), float32 on GPU), so every comparison is
at a single level of theory.  For each isomer x support we try N_ORIENT random
landing orientations, relax, keep the lowest, then decompose

    E_ads = E_int + E_strain(cluster) + E_strain(support)

with E_int evaluated at the frozen supported geometry.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, graphene, make_supercell
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.optimize import FIRE, LBFGS
from scipy.spatial.transform import Rotation

warnings.filterwarnings("ignore")
import torch  # noqa: E402
from mace.calculators import mace_mp  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "out"
SMOKE = bool(os.environ.get("SMOKE"))
RES_NAME = "results_smoke.json" if SMOKE else "results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_ORIENT = int(sys.argv[1]) if len(sys.argv) > 1 else 6
N_BASINS = 8
LAND_HEIGHT = 2.0  # lowest Fe above the top support layer, Angstrom
VACUUM = 12.0      # above the cluster top

calc = mace_mp(model="medium", dispersion=True, default_dtype="float32", device=DEVICE)
print("device", DEVICE, flush=True)


def energy(atoms: Atoms) -> float:
    atoms = atoms.copy()
    atoms.calc = calc
    return float(atoms.get_potential_energy())


def relax(atoms: Atoms, fmax: float, steps: int, log: str | None = None) -> tuple[Atoms, int, bool]:
    atoms.calc = calc
    if SMOKE:
        steps = 3
    opt = LBFGS(atoms, maxstep=0.15, logfile=log)
    ok = opt.run(fmax=fmax, steps=steps)
    if not ok:  # LBFGS occasionally stalls on the flat physisorption surface
        opt2 = FIRE(atoms, logfile=log)
        ok = opt2.run(fmax=fmax, steps=steps)
        return atoms, opt.nsteps + opt2.nsteps, bool(ok)
    return atoms, opt.nsteps, bool(ok)


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    p = p - p.mean(0)
    q = q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((p @ r - q) ** 2).sum(1).mean()))


# ---------------------------------------------------------------- supports
def build_graphene() -> Atoms:
    g = graphene(a=2.46, vacuum=0.0)
    g = make_supercell(g, [[1, 0, 0], [1, 2, 0], [0, 0, 1]])  # orthogonal a x sqrt3 a
    g = g.repeat((8, 5, 1))
    g.cell[2, 2] = 1.0
    g.pbc = True
    g.info["name"] = "graphene"
    g.info["n_fixed_layers"] = 0
    return g


def build_mgo() -> Atoms:
    b = bulk("MgO", "rocksalt", a=4.21, cubic=True)
    b.calc = calc
    FIRE(FrechetCellFilter(b), logfile=None).run(fmax=0.005, steps=300)
    a = float(b.cell[0, 0])
    print(f"MgO lattice (MACE-MP-0+D3): {a:.3f} A", flush=True)
    s = bulk("MgO", "rocksalt", a=a, cubic=True).repeat((4, 4, 2))
    z = np.round(s.positions[:, 2], 3)
    layers = np.unique(z)
    s = s[z < layers[-1] - 0.1]  # 3 atomic layers
    s.info["name"] = "MgO100"
    s.info["n_fixed_layers"] = 1
    return s


def finish_support(s: Atoms) -> Atoms:
    s = s.copy()
    s.positions[:, 2] -= s.positions[:, 2].min()
    s.cell[2, 2] = s.positions[:, 2].max() + LAND_HEIGHT + 7.0 + VACUUM
    s.positions[:, 2] += 1.0
    s.pbc = True
    zmin = s.positions[:, 2].min()
    if s.info.get("n_fixed_layers", 0):
        s.set_constraint(FixAtoms(indices=np.where(s.positions[:, 2] < zmin + 0.5)[0]))
    else:
        # freestanding graphene: pin one far atom so the sheet cannot drift
        far = int(np.argmin(np.linalg.norm(s.positions[:, :2], axis=1)))
        s.set_constraint(FixAtoms(indices=[far]))
    return s


def land(cluster: Atoms, support: Atoms, rot: Rotation) -> Atoms:
    c = cluster.copy()
    c.positions = rot.apply(c.positions - c.positions.mean(0))
    top = support.positions[:, 2].max()
    c.positions[:, :2] += support.cell.diagonal()[:2] / 2
    c.positions[:, 2] += top + LAND_HEIGHT - c.positions[:, 2].min()
    sys_ = support.copy() + c
    sys_.set_constraint(support.constraints)
    return sys_


# ---------------------------------------------------------------- run
all_dft = read(OUT / "all_isomers.extxyz", ":")
if SMOKE:
    all_dft = all_dft[:3]
res_path = OUT / RES_NAME
res = json.loads(res_path.read_text()) if res_path.exists() else {}
t0 = time.time()


def fingerprint(a):
    d = a.get_all_distances()
    return np.sort(d[np.triu_indices(len(a), 1)])


def spe(a: Atoms, e: float) -> Atoms:
    """attach the right energy: the shared calculator's cached results belong to the last system."""
    a = a.copy()
    a.calc = SinglePointCalculator(a, energy=e)
    return a


# gas phase: relax every distinct DFT geometry, then group into MACE-MP-0 basins
relaxed = []
for iso in all_dft:
    a = iso.copy()
    a.pbc = False
    a.cell = None
    a, n, ok = relax(a, fmax=0.02, steps=3000)
    e = energy(a)
    relaxed.append(dict(dft=iso, mace=a, E=e, fp=fingerprint(a), steps=n, ok=ok,
                        rmsd=kabsch_rmsd(iso.positions, a.positions)))
    print(f"gas {iso.info['isomer']}: dft_rel={iso.info['dft_rel_eV']:.3f} E={e:.4f} "
          f"rmsd(DFT->MACE)={relaxed[-1]['rmsd']:.3f} steps={n} ok={ok}", flush=True)

basins: list[dict] = []
for r in sorted(relaxed, key=lambda r: r["dft"].info["dft_rel_eV"]):
    for b in basins:
        if np.abs(r["fp"] - b["fp"]).max() < 0.05 and abs(r["E"] - b["E"]) < 0.005:
            b["members"].append(r)
            break
    else:
        basins.append(dict(fp=r["fp"], E=r["E"], rep=r, members=[r]))
basins.sort(key=lambda b: b["E"])
res["gas_all"] = {r["dft"].info["isomer"]: dict(dft_rel_eV=r["dft"].info["dft_rel_eV"], E=r["E"],
                                                  rmsd_vs_dft=r["rmsd"], converged=r["ok"])
                  for r in relaxed}
print(f"{len(relaxed)} DFT geometries -> {len(basins)} MACE-MP-0 basins", flush=True)
if len(basins) > N_BASINS:
    idx = np.unique(np.round(np.linspace(0, len(basins) - 1, N_BASINS)).astype(int))
    chosen = [basins[i] for i in idx]
else:
    chosen = basins
gas = {}
res["gas"] = {}
for k, b in enumerate(chosen):
    name = f"B{k}"
    rep = b["rep"]
    gas[name] = rep["mace"]
    mem = [m["dft"].info["isomer"] for m in b["members"]]
    res["gas"][name] = dict(E=b["E"], rmsd_vs_dft=rep["rmsd"],
                            dft_rel_eV=rep["dft"].info["dft_rel_eV"],
                            dft_mult=rep["dft"].info["dft_mult"], members=mem)
    for m in mem:
        res["gas_all"][m]["basin"] = name
    print(f"basin {name}: E_rel={b['E'] - basins[0]['E']:.3f} eV, {len(mem)} DFT members "
          f"(best DFT rel {rep['dft'].info['dft_rel_eV']:.3f})", flush=True)
res["n_basins_total"] = len(basins)
res_path.write_text(json.dumps(res, indent=1, default=lambda o: o.item()))
write(OUT / (("smoke_" if SMOKE else "") + "gas_relaxed.extxyz"),
      [spe(a, res["gas"][n]["E"]) for n, a in gas.items()])
for n, a in gas.items():
    a.info["isomer"] = n

rng = np.random.default_rng(7)
rotations = [Rotation.identity()] + [Rotation.random(random_state=int(rng.integers(1e9)))
                                     for _ in range(N_ORIENT - 1)]

for builder in (build_graphene, build_mgo):
    support = finish_support(builder())
    sname = support.info["name"]
    ns = len(support)
    slab, n, ok = relax(support.copy(), fmax=0.01, steps=1000)
    e_slab = energy(slab)
    print(f"{sname}: {ns} atoms, E_slab={e_slab:.4f} ok={ok}", flush=True)
    block = res.setdefault(sname, {"E_slab": e_slab, "n_support": ns, "isomers": {}})
    block["E_slab"] = e_slab
    best_frames = []
    for name, g in gas.items():
        trials = []
        for k, rot in enumerate(rotations):
            sys_ = land(g, slab, rot)
            sys_, n, ok = relax(sys_, fmax=0.03, steps=1500)
            e = energy(sys_)
            trials.append((e, k, n, ok, sys_))
            print(f"  {sname} {name} orient {k}: E_ads={e - e_slab - res['gas'][name]['E']:+.3f} "
                  f"steps={n} ok={ok}  [{time.time() - t0:.0f}s]", flush=True)
        trials.sort(key=lambda t: t[0])
        e_tot, k, n, ok, best = trials[0]
        cl = best[ns:].copy()
        cl.pbc = False
        cl.cell = None
        sub = best[:ns].copy()
        sub.set_constraint()
        e_cl = energy(cl)
        e_sub = energy(sub)
        e_int = e_tot - e_cl - e_sub
        e_gas = res["gas"][name]["E"]
        zc = cl.positions[:, 2]
        top = sub.positions[:, 2].max()
        block["isomers"][name] = dict(
            E_tot=e_tot, best_orient=k, converged=ok,
            E_ads=e_tot - e_slab - e_gas,
            E_int=e_int,
            strain_cluster=e_cl - e_gas,
            strain_support=e_sub - e_slab,
            rmsd_vs_gas=kabsch_rmsd(g.positions, cl.positions),
            orient_E_ads=[t[0] - e_slab - e_gas for t in trials],
            n_contact=int((zc < top + 2.8).sum()),
            height=float(zc.min() - top),
        )
        best_frames.append(spe(best, e_tot))
        best_frames[-1].info.update(isomer=name, support=sname)
        d = block["isomers"][name]
        print(f"{sname} {name}: E_ads={d['E_ads']:+.3f}  E_int={e_int:+.3f}  "
              f"strain_cl={d['strain_cluster']:+.3f}  rmsd={d['rmsd_vs_gas']:.3f}", flush=True)
        res_path.write_text(json.dumps(res, indent=1, default=lambda o: o.item()))
    write(OUT / (("smoke_" if SMOKE else "") + f"supported_{sname}.extxyz"), best_frames)

res_path.write_text(json.dumps(res, indent=1, default=lambda o: o.item()))
print(f"done in {time.time() - t0:.0f}s")
