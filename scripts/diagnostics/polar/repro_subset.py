"""Reproduce a representative subset of the saved MACE-POLAR-1 Fe16 run (s7_polar_spin.py).

usage: python repro_subset.py SEEDS.extxyz SAVED_polar_spin.json OUT.json [polar-1-m]

Needs graph_longrange v0.4.0 (repo graph_electrostatics@v0.4.0) in an isolated env.
Same state selection as s7 (lowest record per fingerprint group and multiplicity), float64.
Recomputes: the O2/Fe2 sanity ladders; every state of group 0; and the saved states with the
lowest / 25th / median / 75th / highest max|atomic spin|. Also records:
  * checkpoint identity: sha256 of every cached POLAR checkpoint file;
  * determinism: one state evaluated twice;
  * caching: the same geometry at two multiplicities WITHOUT a reset (stale result expected)
    vs with a reset.
POLAR's predicted per-atom spins are its own internal populations, not Mulliken populations;
no fixed magnitude bound is assumed -- the spin sum vs 2S and the energy are what is checked.
"""
import hashlib, json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import torch
from ase import Atoms
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from cluster_mlip.delta import load_model, reset_calculator  # noqa: E402

seeds, saved_path, out_path = sys.argv[1:4]
name = sys.argv[4] if len(sys.argv) > 4 else "polar-1-m"
HARTREE = 27.211386245988
saved = json.load(open(saved_path))
calc = load_model(f"mace-polar:{name}", device="cuda" if torch.cuda.is_available() else "cpu",
                  dtype="float64").calculator
res = {"model": name, "torch": torch.__version__, "checkpoints": {}}
for d in (Path.home() / ".cache" / "mace", Path.home() / ".cache" / "mace.old"):
    for f in sorted(d.glob("MACEPOLAR*")):
        res["checkpoints"][str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()


def sp(atoms, mult, reset=True):
    a = atoms.copy(); a.info = {"charge": 0, "spin": int(mult)}; a.pbc = False
    if reset:
        reset_calculator(calc)
    a.calc = calc
    e = float(a.get_potential_energy())
    s = calc.results.get("spins")
    s = None if s is None else np.asarray(s, float)
    return e, (None if s is None else float(s.sum())), (None if s is None else float(np.abs(s).max()))


# sanity, compared with the saved values
res["sanity"] = []
for nm, atoms in (("O2", Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.21]])),
                  ("Fe2", Atoms("Fe2", positions=[[0, 0, 0], [0, 0, 2.02]]))):
    for row in saved["sanity"][nm]:
        e, ss, sm = sp(atoms, row["mult"])
        res["sanity"].append({"system": nm, "mult": row["mult"], "dE_vs_saved_eV": e - row["E"],
                              "spin_sum": ss, "saved_spin_sum": row["spin_sum"]})

# same grouping as s7
def fingerprint(a):
    d = a.get_all_distances(); return np.sort(d[np.triu_indices(len(a), 1)])
records = [a for a in read(seeds, ":") if a.get_chemical_formula() == "Fe16"
           and a.info.get("config_type") in ("minimum", "optimized_unverified")
           and a.info.get("legacy_energy_hartree") is not None]
groups = []
for a in records:
    fp = fingerprint(a)
    for g in groups:
        if np.abs(fp - g["fp"]).max() < 0.10:
            g["members"].append(a); break
    else:
        groups.append({"fp": fp, "members": [a]})
for g in groups:
    best = {}
    for a in g["members"]:
        m = int(a.info["multiplicity"]); e = float(a.info["legacy_energy_hartree"]) * HARTREE
        if m not in best or e < best[m][0]:
            best[m] = (e, a)
    g["by_mult"] = best; g["e_min"] = min(v[0] for v in best.values())
groups.sort(key=lambda g: g["e_min"])
res["n_groups"] = len(groups)

states = saved["states"]
order = sorted(range(len(states)), key=lambda i: states[i]["spin_max"])
picks = sorted({*[i for i, s in enumerate(states) if s["group"] == 0],
                *[order[int(q * (len(order) - 1))] for q in (0, 0.25, 0.5, 0.75, 1.0)]})
res["states"] = []
t0 = time.time()
for i in picks:
    s = states[i]
    e_dft, a = groups[s["group"]]["by_mult"][s["mult"]]
    e, ss, sm = sp(a, s["mult"])
    res["states"].append({"index": i, "group": s["group"], "mult": s["mult"],
                          "dft_total_eV": e_dft, "polar_eV": e, "polar_minus_dft_eV": e - e_dft,
                          "dE_vs_saved_eV": e - s["e_polar"], "spin_sum": ss, "two_S": s["mult"] - 1,
                          "spin_max": sm, "saved_spin_max": s["spin_max"]})
res["seconds"] = time.time() - t0

# determinism and caching on group 0's ground state
g0 = groups[0]; m0 = min(g0["by_mult"], key=lambda m: g0["by_mult"][m][0]); a0 = g0["by_mult"][m0][1]
e1 = sp(a0, m0)[0]; e2 = sp(a0, m0)[0]
other = m0 - 2 if m0 > 1 else m0 + 2
e_no_reset = sp(a0, other, reset=False)[0]; e_reset = sp(a0, other)[0]
res["determinism_dE_eV"] = e2 - e1
res["cache"] = {"mult_a": m0, "mult_b": other, "E_b_without_reset_minus_E_a": e_no_reset - e2,
                "E_b_with_reset_minus_E_a": e_reset - e2}
json.dump(res, open(out_path, "w"), indent=1)
for r in res["states"]:
    print(f"state {r['index']:2d} g{r['group']} M={r['mult']}: POLAR-DFT {r['polar_minus_dft_eV']:+.1f} eV, "
          f"vs saved {r['dE_vs_saved_eV']:+.2e}, spin sum {r['spin_sum']:.2f} (2S={r['two_S']}), max|s| {r['spin_max']:.1f}")
print("sanity max |dE vs saved|:", max(abs(r["dE_vs_saved_eV"]) for r in res["sanity"]))
print("determinism:", res["determinism_dE_eV"], "| cache:", res["cache"])
print("checkpoints:", {Path(k).parent.name + "/" + Path(k).name: v[:12] for k, v in res["checkpoints"].items()})
