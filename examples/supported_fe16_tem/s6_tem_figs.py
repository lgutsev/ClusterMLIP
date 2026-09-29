"""Step 6: abTEM figures."""
import os
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402
from ase.visualize.plot import plot_atoms  # noqa: E402

OUT = Path(os.environ.get("EXAMPLE_OUT") or os.environ.get("TEM_OUT") or Path(__file__).parent / "out")
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


# ------------------------------------------------------------------ Fig 5/6: TEM
tem_p = OUT / "tem_images.npz"
if tem_p.exists():
    T = np.load(tem_p)
    for view in ("profile", "plan"):
        sups = [s for s in ("graphene", "MgO100") if f"{s}_{view}_short" in T]
        fig, ax = plt.subplots(len(sups), 3, figsize=(11, 3.3 * len(sups)), squeeze=False)
        for r, s in enumerate(sups):
            k = f"{s}_{view}"
            ext = T[k + "_extent"]
            fe = T[k + "_fe"]
            if view == "profile":
                y0 = T[k + "_sup_top"].mean() - 4.0
                y1 = fe[:, :, 1].max() + 3.0
                xc = fe[:, :, 0].mean()
                x0, x1 = xc - 9, xc + 9
            else:
                c = fe.reshape(-1, 3).mean(0)
                x0, x1, y0, y1 = c[0] - 8, c[0] + 8, c[1] - 8, c[1] + 8
            imgs = [T[k + "_short"], T[k + "_avg"], T[k + "_noisy"]]
            ny, nx = imgs[0].shape[1], imgs[0].shape[0]
            sl = (slice(max(0, int(x0 / ext[0] * nx)), int(x1 / ext[0] * nx)),
                  slice(max(0, int(y0 / ext[1] * ny)), int(y1 / ext[1] * ny)))
            crops = [im[sl] for im in imgs]
            vmin = min(c_.min() for c_ in crops[:2])
            vmax = max(c_.max() for c_ in crops[:2])
            for col, (im, tt) in enumerate(zip(crops, ["short exposure (1 frame)",
                                                       "10 ps time average",
                                                       "10 ps average + shot noise (2e4 e⁻/Å²)"])):
                a = ax[r, col]
                extent = [sl[0].start * ext[0] / nx, sl[0].stop * ext[0] / nx,
                          sl[1].start * ext[1] / ny, sl[1].stop * ext[1] / ny]
                kw = dict(vmin=vmin, vmax=vmax) if col < 2 else {}
                a.imshow(im.T, origin="lower", cmap="gray", extent=extent, **kw)
                a.set_title(f"{LABEL[s]}: {tt}", fontsize=8.5, loc="left")
                a.set_xticks([])
                a.set_yticks([])
                a.grid(False)
                # 5 A scale bar on a dark backing, top-right (vacuum side in profile)
                from matplotlib.patches import Rectangle
                bx0, by = extent[1] - 7.0, extent[3] - 2.4
                a.add_patch(Rectangle((bx0, by - 0.6), 6.4, 2.6, color="black", alpha=0.55,
                                      lw=0))
                a.plot([bx0 + 0.7, bx0 + 5.7], [by, by], color="white", lw=3,
                       solid_capstyle="butt")
                a.text(bx0 + 3.2, by + 0.5, "5 Å", color="white", ha="center", fontsize=8)
        fig.suptitle(f"abTEM HRTEM, {view} view (200 kV, Cs = −8 µm, Scherzer), "
                     "Fe16 from MACE-MP-0 MD at 700 K", x=0.01, ha="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / f"fig5_tem_{view}.png")

