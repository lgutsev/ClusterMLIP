"""Step 2: is the support a perturbation?  Fe16 isomers, gas phase vs graphene vs MgO(100).

For each isomer x support we try N_ORIENT random landing orientations, relax, keep the
lowest, then decompose

    E_ads = E_int + E_strain(cluster) + E_strain(support)

with E_int evaluated at the frozen supported geometry.

Models (see common.py): by default everything is MACE-MP-0 medium + D3(BJ), float32 --
the baseline in the README, written to out/. With a spin-aware gas-phase model
(``--model path/to/fe16.model``) every isomer is relaxed at each of --multiplicities and
the lowest (adiabatic) state is kept, and the supported system is the Delta-model

    E = E_gas(cluster; q=0, M) + E_support + dE_int

(dE_int from --interaction). ``--rescan-support-m`` also relaxes the best landing at the
other multiplicities, in case the support changes the preferred M.

usage: s2_relax.py [n_orient] [--model SPEC] [--support-model SPEC] [--interaction SPEC]
                   [--multiplicities 49,51,53] [--rescan-support-m] [--isomers PATH]
"""
import argparse
import json
import os
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
from common import HERE, add_model_args, build_models  # noqa: E402
from cluster_mlip.delta import mark_cluster, relax_over_multiplicities, reset_calculator  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("n_orient", nargs="?", type=int, default=6)
parser.add_argument("--isomers", default=str(HERE / "out" / "all_isomers.extxyz"),
                    help="distinct DFT geometries from s1_select.py")
parser.add_argument("--rescan-support-m", action="store_true")
add_model_args(parser)
args = parser.parse_args()

models = build_models(args)
OUT = models.out
SMOKE = bool(os.environ.get("SMOKE"))
RES_NAME = "results_smoke.json" if SMOKE else "results.json"
N_ORIENT = args.n_orient
N_BASINS = 8
LAND_HEIGHT = 2.0  # lowest Fe above the top support layer, Angstrom
VACUUM = 12.0      # above the cluster top

gas_calc = models.gas.calculator
support_calc = models.support.calculator
total_calc = models.total


def energy(atoms: Atoms, calc) -> float:
    atoms = atoms.copy()
    reset_calculator(calc)  # the cache ignores atoms.info["spin"]
    atoms.calc = calc
    return float(atoms.get_potential_energy())


def relax(atoms: Atoms, calc, fmax: float, steps: int, log: str | None = None) -> tuple[Atoms, int, bool]:
    reset_calculator(calc)
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
    b.calc = support_calc
    FIRE(FrechetCellFilter(b), logfile=None).run(fmax=0.005, steps=300)
    a = float(b.cell[0, 0])
    print(f"MgO lattice ({models.support.spec}): {a:.3f} A", flush=True)
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
    s.info = {"name": s.info["name"], "charge": 0, "spin": 1}
    return s


def land(cluster: Atoms, support: Atoms, rot: Rotation, mult: int | None) -> Atoms:
    c = cluster.copy()
    c.positions = rot.apply(c.positions - c.positions.mean(0))
    top = support.positions[:, 2].max()
    c.positions[:, :2] += support.cell.diagonal()[:2] / 2
    c.positions[:, 2] += top + LAND_HEIGHT - c.positions[:, 2].min()
    sup = support.copy()
    for a in (sup, c):
        for key in list(a.arrays):
            if key not in ("numbers", "positions"):
                del a.arrays[key]
    sys_ = sup + c
    sys_.set_constraint(support.constraints)
    mark_cluster(sys_, np.r_[np.zeros(len(support)), np.ones(len(c))])
    sys_.info = {"charge": 0, "spin": int(mult) if mult else 1}
    return sys_


def fingerprint(a):
    d = a.get_all_distances()
    return np.sort(d[np.triu_indices(len(a), 1)])


def spe(a: Atoms, e: float) -> Atoms:
    """attach the right energy: the shared calculator's cached results belong to the last system."""
    a = a.copy()
    a.calc = SinglePointCalculator(a, energy=e)
    return a


# ---------------------------------------------------------------- run
all_dft = read(args.isomers, ":")
if SMOKE:
    all_dft = all_dft[:3]
res_path = OUT / RES_NAME
res = json.loads(res_path.read_text()) if res_path.exists() else {}
res["models"] = models.describe()
t0 = time.time()

# gas phase: relax every distinct DFT geometry (at each M for a spin-aware model), then
# group into basins of the model's surface
relaxed = []
for iso in all_dft:
    a = iso.copy()
    a.pbc = False
    a.cell = None
    a.info = {"charge": 0, "spin": int(iso.info.get("dft_mult", 1))}
    steps = [0]

    def _relax(x, steps=steps):
        x, n, ok = relax(x, gas_calc, fmax=0.02, steps=3000)
        steps[0] += n
        return ok

    scan = relax_over_multiplicities(a, gas_calc, models.multiplicities, _relax,
                                     energy_of=lambda x: energy(x, gas_calc))
    a = scan.atoms
    # a spin-blind model has no multiplicity of its own: carry the DFT one (VASP needs a real M)
    m_state = scan.multiplicity if models.multiplicities else int(iso.info.get("dft_mult", 1))
    relaxed.append(dict(dft=iso, mace=a, E=scan.energy, M=m_state, E_by_M=scan.by_multiplicity,
                        fp=fingerprint(a), steps=steps[0], ok=scan.converged,
                        rmsd=kabsch_rmsd(iso.positions, a.positions)))
    print(f"gas {iso.info['isomer']}: dft_rel={iso.info['dft_rel_eV']:.3f} (M={iso.info.get('dft_mult')}) "
          f"E={scan.energy:.4f} M={m_state} rmsd(DFT->model)={relaxed[-1]['rmsd']:.3f} "
          f"steps={steps[0]} ok={scan.converged}", flush=True)

basins: list[dict] = []
for r in sorted(relaxed, key=lambda r: r["dft"].info["dft_rel_eV"]):
    for b in basins:
        # spin-blind: the DFT M does not split a basin of the model's surface
        if ((not models.multiplicities or r["M"] == b["M"]) and np.abs(r["fp"] - b["fp"]).max() < 0.05
                and abs(r["E"] - b["E"]) < 0.005):
            b["members"].append(r)
            break
    else:
        basins.append(dict(fp=r["fp"], E=r["E"], M=r["M"], rep=r, members=[r]))
basins.sort(key=lambda b: b["E"])
res["M_source"] = "model spin scan" if models.multiplicities else "DFT multiplicity of the basin representative"
res["gas_all"] = {r["dft"].info["isomer"]: dict(dft_rel_eV=r["dft"].info["dft_rel_eV"],
                                                  dft_mult=r["dft"].info.get("dft_mult"), E=r["E"],
                                                  M=r["M"], E_by_M=r["E_by_M"],
                                                  rmsd_vs_dft=r["rmsd"], converged=r["ok"])
                  for r in relaxed}
print(f"{len(relaxed)} DFT geometries -> {len(basins)} basins of {models.gas.spec}", flush=True)
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
    res["gas"][name] = dict(E=b["E"], M=b["M"], rmsd_vs_dft=rep["rmsd"],
                            dft_rel_eV=rep["dft"].info["dft_rel_eV"],
                            dft_mult=rep["dft"].info["dft_mult"], members=mem)
    for m in mem:
        res["gas_all"][m]["basin"] = name
    print(f"basin {name}: E_rel={b['E'] - basins[0]['E']:.3f} eV, M={b['M']}, {len(mem)} DFT members "
          f"(best DFT rel {rep['dft'].info['dft_rel_eV']:.3f})", flush=True)
res["n_basins_total"] = len(basins)
res_path.write_text(json.dumps(res, indent=1, default=lambda o: o.item()))
gas_frames = []
for n, a in gas.items():
    f = spe(a, res["gas"][n]["E"])
    f.info.update(isomer=n, charge=0, spin=int(res["gas"][n]["M"] or 1))
    gas_frames.append(f)
write(OUT / (("smoke_" if SMOKE else "") + "gas_relaxed.extxyz"), gas_frames)
for n, a in gas.items():
    a.info["isomer"] = n

rng = np.random.default_rng(7)
rotations = [Rotation.identity()] + [Rotation.random(random_state=int(rng.integers(1e9)))
                                     for _ in range(N_ORIENT - 1)]


def decompose(best: Atoms, ns: int, e_tot: float) -> tuple[float, float, float]:
    """(E_cluster, E_support, E_int) at the frozen supported geometry."""
    if models.is_delta:
        energy(best, total_calc)  # refresh the calculator's terms for exactly this system
        t = total_calc.terms
        return t["cluster"], t["support"], t["interaction"]
    cl = best[ns:].copy()
    cl.pbc = False
    cl.cell = None
    cl.info = dict(best.info)
    sub = best[:ns].copy()
    sub.set_constraint()
    sub.info = {"charge": 0, "spin": 1}
    e_cl, e_sub = energy(cl, gas_calc), energy(sub, support_calc)
    return e_cl, e_sub, e_tot - e_cl - e_sub


for builder in (build_graphene, build_mgo):
    support = finish_support(builder())
    sname = support.info["name"]
    ns = len(support)
    slab, n, ok = relax(support.copy(), support_calc, fmax=0.01, steps=1000)
    e_slab = energy(slab, support_calc)
    print(f"{sname}: {ns} atoms, E_slab={e_slab:.4f} ok={ok}", flush=True)
    block = res.setdefault(sname, {"E_slab": e_slab, "n_support": ns, "isomers": {}})
    block["E_slab"] = e_slab
    best_frames = []
    for name, g in gas.items():
        m_gas = res["gas"][name]["M"]
        e_gas = res["gas"][name]["E"]
        trials = []
        for k, rot in enumerate(rotations):
            sys_ = land(g, slab, rot, m_gas)
            sys_, n, ok = relax(sys_, total_calc, fmax=0.03, steps=1500)
            e = energy(sys_, total_calc)
            trials.append((e, k, n, ok, sys_, m_gas))
            print(f"  {sname} {name} orient {k}: E_ads={e - e_slab - e_gas:+.3f} "
                  f"steps={n} ok={ok}  [{time.time() - t0:.0f}s]", flush=True)
        trials.sort(key=lambda t: t[0])
        rescan = {}
        if args.rescan_support_m and models.multiplicities:
            base = trials[0][4]
            for m in models.multiplicities:
                if m == m_gas:
                    continue
                x = base.copy()
                x.set_constraint(base.constraints)
                x.info["spin"] = int(m)
                x, n, ok = relax(x, total_calc, fmax=0.03, steps=1500)
                e = energy(x, total_calc)
                # the gas reference stays the gas-phase ground state, so E_ads includes the spin change
                rescan[int(m)] = e - e_slab - e_gas
                trials.append((e, trials[0][1], n, ok, x, m))
            trials.sort(key=lambda t: t[0])
        e_tot, k, n, ok, best, m_sup = trials[0]
        e_cl, e_sub, e_int = decompose(best, ns, e_tot)
        zc = best.positions[ns:, 2]
        top = best.positions[:ns, 2].max()
        block["isomers"][name] = dict(
            E_tot=e_tot, best_orient=k, converged=ok, M_gas=m_gas, M_supported=m_sup,
            E_ads=e_tot - e_slab - e_gas,
            E_int=e_int,
            strain_cluster=e_cl - e_gas,
            strain_support=e_sub - e_slab,
            rmsd_vs_gas=kabsch_rmsd(g.positions, best.positions[ns:]),
            orient_E_ads=[t[0] - e_slab - e_gas for t in trials[:N_ORIENT]],
            E_ads_by_M=rescan,
            n_contact=int((zc < top + 2.8).sum()),
            height=float(zc.min() - top),
        )
        frame = spe(best, e_tot)
        frame.info.update(isomer=name, support=sname, charge=0, spin=int(m_sup or 1),
                          multiplicity=int(m_sup or 1), structure_id=f"{sname}_{name}")
        best_frames.append(frame)
        d = block["isomers"][name]
        print(f"{sname} {name}: E_ads={d['E_ads']:+.3f}  E_int={e_int:+.3f}  "
              f"strain_cl={d['strain_cluster']:+.3f}  rmsd={d['rmsd_vs_gas']:.3f}  M={m_sup}", flush=True)
        res_path.write_text(json.dumps(res, indent=1, default=lambda o: o.item()))
    write(OUT / (("smoke_" if SMOKE else "") + f"supported_{sname}.extxyz"), best_frames)

res_path.write_text(json.dumps(res, indent=1, default=lambda o: o.item()))
print(f"done in {time.time() - t0:.0f}s")
