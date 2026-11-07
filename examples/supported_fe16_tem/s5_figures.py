"""Step 5: figures."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402
from ase.visualize.plot import plot_atoms  # noqa: E402

# $EXAMPLE_OUT selects a non-baseline run (e.g. out_<model>/); its figures go next to it
OUT = Path(os.environ.get("EXAMPLE_OUT") or Path(__file__).parent / "out")
FIG = Path(__file__).parent / "figures" if "EXAMPLE_OUT" not in os.environ else OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C = {"graphene": "#2a78d6", "MgO100": "#eb6834", "gas": "#1baf7a"}
LABEL = {"graphene": "on graphene", "MgO100": "on MgO(100)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
names = sorted(res["gas"])
gas_E = np.array([res["gas"][n]["E"] for n in names])
gas_rel = gas_E - gas_E.min()
dft_rel = np.array([res["gas"][n]["dft_rel_eV"] for n in names])

# ------------------------------------------------------------------ Fig 1: ranking
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
a = ax[0]
ga = res["gas_all"]
dx = np.array([v["dft_rel_eV"] for v in ga.values()])
ex = np.array([v["E"] for v in ga.values()])
ex = ex - ex.min()
top = max(dx.max(), ex.max()) + 0.1
a.plot([0, top], [0, top], color=INK2, lw=1, ls="--", zorder=1)
a.scatter(dx, ex, s=26, color=C["gas"], edgecolor="white", lw=1, zorder=3)
a.text(0.03, 0.97, f"{len(ga)} UBPW91 minima\n-> {res['n_basins_total']} MACE-MP-0 basins",
       transform=a.transAxes, ha="left", va="top", fontsize=8, color=INK)
a.set_xlabel("UBPW91 ΔE (best multiplicity), eV")
a.set_ylabel("MACE-MP-0 ΔE after relaxing, eV")
a.set_title(f"(a) Foundation model vs our DFT, isolated Fe16\nSpearman ρ = {spearman(dx, ex):.2f}",
            loc="left", fontsize=9.5)

a = ax[1]
lim = [min(-0.3, 0), 0]
for s in ("graphene", "MgO100"):
    iso = res[s]["isomers"]
    e = np.array([iso[n]["E_tot"] for n in names])
    rel = e - e.min()
    lim[1] = max(lim[1], rel.max(), gas_rel.max())
    rho = spearman(gas_rel, rel)
    a.scatter(gas_rel, rel, s=46, color=C[s], edgecolor="white", lw=1.5, zorder=3,
              label=f"{LABEL[s]}  (ρ = {rho:.2f})")
    for n, x, y in zip(names, gas_rel, rel):
        a.annotate(n, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
a.plot([0, lim[1] + 0.2], [0, lim[1] + 0.2], color=INK2, lw=1, ls="--", zorder=1)
a.text(lim[1] * 0.62, lim[1] * 0.62 + 0.12, "ranking unchanged", rotation=0, fontsize=7.5,
       color=INK2)
a.set_xlabel("gas-phase ΔE (MACE-MP-0), eV")
a.set_ylabel("supported ΔE (same model), eV")
a.set_title("(b) Does the support reorder the isomers?", loc="left", fontsize=9.5)
a.legend(loc="upper left", fontsize=7.5)

a = ax[2]
x = np.arange(len(names))
for j, s in enumerate(("graphene", "MgO100")):
    iso = res[s]["isomers"]
    off = (j - 0.5) * 0.3
    for i, n in enumerate(names):
        orient = np.array(iso[n]["orient_E_ads"])
        a.scatter(np.full(len(orient), i + off), orient, s=10, color=C[s], alpha=0.35, lw=0)
    best = np.array([iso[n]["E_ads"] for n in names])
    a.scatter(x + off, best, s=40, color=C[s], edgecolor="white", lw=1.5, zorder=3,
              label=LABEL[s])
a.set_xticks(x, names)
a.set_ylabel("adsorption energy E_ads, eV")
a.set_title("(c) E_ads per isomer (faint = other landing orientations)", loc="left",
            fontsize=9.5)
a.legend(loc="center left", fontsize=7.5)
fig.tight_layout()
fig.savefig(FIG / "fig1_isomer_ranking.png")

# ------------------------------------------------------------------ Fig 2: perturbation size
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
gaps = np.diff(np.sort(gas_rel))
for j, s in enumerate(("graphene", "MgO100")):
    iso = res[s]["isomers"]
    off = (j - 0.5) * 0.36
    eint = np.array([iso[n]["E_int"] for n in names])
    scl = np.array([iso[n]["strain_cluster"] for n in names])
    rmsd = np.array([iso[n]["rmsd_vs_gas"] for n in names])
    eads = np.array([iso[n]["E_ads"] for n in names])
    ax[0].bar(x + off, eint, width=0.34, color=C[s], label=LABEL[s])
    ax[1].bar(x + off, scl, width=0.34, color=C[s], label=LABEL[s])
    ax[2].bar(x + off, rmsd, width=0.34, color=C[s], label=LABEL[s])
    # spread of E_ads across isomers = how much the support can reshuffle the ranking
    res[s]["E_ads_spread"] = float(eads.max() - eads.min())
for a, t, yl in zip(ax, ["(a) Interaction energy at the supported geometry",
                         "(b) Cluster strain (deformation) energy",
                         "(c) Cluster distortion vs gas-phase minimum"],
                    ["E_int, eV", "E_strain(cluster), eV", "Kabsch RMSD, Å"]):
    a.set_xticks(x, names)
    a.set_title(t, loc="left", fontsize=9.5)
    a.set_ylabel(yl)
    a.axhline(0, color=INK2, lw=0.8)
ax[2].legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "fig2_perturbation_size.png")

# ------------------------------------------------------------------ Fig 3: structures
gas = dict(zip(res["gas"], read(OUT / "gas_relaxed.extxyz", ":")))  # file is in basin order
sup = {s: {a.info["isomer"]: a for a in read(OUT / f"supported_{s}.extxyz", ":")}
       for s in ("graphene", "MgO100")}
low = {s: min(res[s]["isomers"], key=lambda n: res[s]["isomers"][n]["E_tot"]) for s in sup}
fig, ax = plt.subplots(1, 5, figsize=(14, 3.4), gridspec_kw=dict(width_ratios=[0.7, 1, 1, 1, 1]))
plot_atoms(gas["B0"], ax[0], radii=0.9, rotation="10x,10y")
ax[0].set_title("B0 (global min), gas phase", fontsize=9)
for k, s in enumerate(("graphene", "MgO100")):
    a = sup[s][low[s]].copy()
    cen = a.positions[[i for i, c in enumerate(a.get_chemical_symbols()) if c == "Fe"]].mean(0)
    keep = np.linalg.norm(a.positions[:, :2] - cen[:2], axis=1) < 9
    a = a[keep]
    plot_atoms(a, ax[1 + 2 * k], radii=0.8, rotation="-90x")
    ax[1 + 2 * k].set_title(f"{low[s]} {LABEL[s]}: side", fontsize=9)
    iso = res[s]["isomers"][low[s]]
    note = (f"lowest Fe {iso['height']:.1f} Å above top layer, "
            f"{iso['n_contact']} Fe within 2.8 Å")
    ax[1 + 2 * k].text(0.5, -0.04, note, transform=ax[1 + 2 * k].transAxes, ha="center",
                       va="top", fontsize=8, color=INK2)
    plot_atoms(a, ax[2 + 2 * k], radii=0.8)
    ax[2 + 2 * k].set_title(f"{low[s]} {LABEL[s]}: top", fontsize=9)
for a in ax:
    a.set_axis_off()
fig.tight_layout()
fig.savefig(FIG / "fig3_structures.png")

# ------------------------------------------------------------------ Fig 4: MD
def kabsch_rmsd(p, q):
    p, q = p - p.mean(0), q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt(((p @ (u @ np.diag([1, 1, d]) @ vt) - q) ** 2).sum(1).mean()))


md = {}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
for s in ("graphene", "MgO100"):
    p = OUT / f"md_{s}.extxyz"
    if not p.exists():
        continue
    tr = read(p, ":")
    fe = np.array([c == "Fe" for c in tr[0].get_chemical_symbols()])
    t = np.arange(len(tr)) * 0.05
    pos = np.array([a.positions[fe] for a in tr])
    rm = [kabsch_rmsd(pos[0], p_) for p_ in pos]
    com = pos.mean(1)
    lat = np.linalg.norm(com[:, :2] - com[0, :2], axis=1)
    top = np.array([a.positions[~fe, 2].max() for a in tr])
    h = pos[:, :, 2].min(1) - top
    md[s] = dict(rmsd_final=rm[-1], com_drift_final=float(lat[-1]))
    ax[0].plot(t, rm, color=C[s], lw=2, label=LABEL[s])
    ax[1].plot(t, lat, color=C[s], lw=2, label=LABEL[s])
    ax[2].plot(t, h, color=C[s], lw=2, label=LABEL[s])
for a, yl, tt in zip(ax, ["Fe16 RMSD vs t=0, Å", "cluster COM lateral drift, Å",
                          "lowest Fe − highest support atom, Å"],
                     ["(a) Shape fluctuation", "(b) Diffusion on the support",
                      "(c) Anchoring height (sheet buckles up)"]):
    a.set_xlabel("time, ps")
    a.set_ylabel(yl)
    a.set_title(tt, loc="left", fontsize=9.5)
ax[0].legend(fontsize=7.5)
fig.suptitle("MACE-MP-0 + D3 Langevin MD, 700 K", x=0.01, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig4_md.png")

summary = {
    "spearman_dft_vs_mace_gas_all": spearman(dx, ex),
    "n_dft": len(ga), "n_basins": res["n_basins_total"],
    **{f"spearman_gas_vs_{s}": spearman(gas_rel, np.array([res[s]["isomers"][n]["E_tot"]
                                                           for n in names]))
       for s in ("graphene", "MgO100")},
    "gas_isomer_gaps_min_median": [float(gaps.min()), float(np.median(gaps))],
    **{f"E_ads_spread_{s}": res[s]["E_ads_spread"] for s in ("graphene", "MgO100")},
    "lowest_supported": low, "md": md,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
