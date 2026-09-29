"""Step 8: figure for the POLAR-1 spin-ladder check."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).parent / "out"
FIG = Path(__file__).parent / "figures"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})
BOUND = 4.0

res = json.loads((OUT / "polar_spin.json").read_text())
fig, ax = plt.subplots(1, 4, figsize=(16, 3.9))

# (a) sanity: small systems behave
a = ax[0]
for (name, rows), col in zip(res["sanity"].items(), (C1, C2, C3)):
    m = [r["mult"] for r in rows]
    a.plot(m, [r["E_rel"] for r in rows], "-o", color=col, lw=2, ms=5, mec="white", mew=1,
           label=f"{name}: ground M = {min(rows, key=lambda r: r['E'])['mult']}")
a.set_xlabel("spin multiplicity M")
a.set_ylabel("E(M) − E(ground), eV")
a.set_title("(a) Sanity: small systems behave", loc="left", fontsize=9.5)
a.legend(fontsize=7.5, loc="upper center")

# (b) validity: predicted atomic spins
a = ax[1]
st = res["states"]
a.scatter([s["mult"] for s in st], [s["spin_max"] for s in st], s=24, color=C1, edgecolor="white",
          lw=1, zorder=3, label=f"Fe16 warehouse states (n={len(st)})")
for (name, rows), col in zip(res["sanity"].items(), (C1, C2, C3)):
    if name == "O2":
        continue
    a.scatter([r["mult"] for r in rows], [max(r["spin_max"], 1e-2) for r in rows], s=24,
              marker="s", color=col if name != "Fe2" else C2, edgecolor="white", lw=1, zorder=3,
              label=f"{name} (sanity)")
a.axhspan(1e-2, BOUND, color=C3, alpha=0.12, lw=0)
a.axhline(BOUND, color=INK2, lw=0.8, ls="--")
a.text(2, BOUND * 1.25, "physical: |s_i| ≤ 4 for Fe", fontsize=7.5, color=INK2)
a.set_yscale("log")
a.set_xlabel("spin multiplicity M")
a.set_ylabel("max |predicted atomic spin|")
a.set_title("(b) Spin equilibration diverges for Fe16", loc="left", fontsize=9.5)
a.legend(fontsize=7, loc="lower right")

# (c) the Fe16 ladder
a = ax[2]
M = np.array(res["ladder"]["mults"])
for k, c in enumerate(res["ladder"]["curves"]):
    a.plot(M, c["polar"], color=C1, lw=1.4, alpha=0.8,
           label="POLAR-1, 6 lowest DFT geometries" if k == 0 else None)
    dm = sorted((int(m), v) for m, v in c["dft"].items())
    a.plot([m for m, _ in dm], [v for _, v in dm], "o", color=C3, ms=6, mec="white", mew=1.2,
           label="UBPW91 (adiabatic)" if k == 0 else None, zorder=3)
a.axvspan(1, 11, color=INK2, alpha=0.08, lw=0)
a.text(2, 3e5, "OMol25\nM ≤ 11", fontsize=7.5, color=INK2)
a.set_yscale("symlog", linthresh=1.0)
a.set_xlabel("spin multiplicity M")
a.set_ylabel("E(M) − E(DFT ground M), eV (symlog)")
a.set_title("(c) Fe16 spin ladder", loc="left", fontsize=9.5)
a.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.3, 0.97))

# (d) adiabatic spin gaps
a = ax[3]
by_group: dict = {}
for s in st:
    by_group.setdefault(s["group"], []).append(s)
xd, yp = [], []
for g in by_group.values():
    if len(g) < 2:
        continue
    ref = min(g, key=lambda s: s["e_dft"])
    for s in g:
        if s is not ref:
            xd.append(s["e_dft"] - ref["e_dft"])
            yp.append(s["e_polar"] - ref["e_polar"])
xd, yp = np.array(xd), np.array(yp)
a.scatter(xd, yp, s=26, color=C1, edgecolor="white", lw=1, zorder=3)
a.axhline(0, color=INK2, lw=0.8)
a.set_yscale("symlog", linthresh=1.0)
a.set_xlabel("UBPW91 gap to the group's ground M, eV")
a.set_ylabel("POLAR-1 gap, same geometries, eV (symlog)")
a.set_title(f"(d) Adiabatic spin gaps ({len(xd)} pairs)\nDFT: {xd.min():.2f}…{xd.max():.2f} eV, "
            f"POLAR-1: {yp.min():.0f}…{yp.max():.0f} eV", loc="left", fontsize=9.5)

fig.suptitle(f"MACE-POLAR-1 ({res['model']}) on the FenOm Fe16 spin ladder — float64 single points. "
             "Works for O2/Fe2/Fe4, diverges for Fe16 at M = 49–53 (52 unpaired e⁻, far outside "
             "OMol25's M ≤ 11)", x=0.01, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig6_polar_spin_ladder.png")
print("saved", FIG / "fig6_polar_spin_ladder.png")
