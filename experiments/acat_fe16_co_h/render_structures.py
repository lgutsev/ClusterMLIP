"""Small structure pictures (PNG, transparent background) for the report.

    python render_structures.py ANALYSIS_DIR --out PICS_DIR

Draws the site map of the catalyst and every shortlist structure as depth-sorted
circles: Fe, C, O, H, with lines for C-O and adsorbate-Fe contacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.io import read

from common import load_catalyst, load_sites

COLOR = {"Fe": "#c9773c", "C": "#4a4a4a", "O": "#d8392b", "H": "#f4f4f4"}
EDGE = "#6b6b6b"
SITE_COLOR = {"top": "#1f6fb5", "bridge": "#2a9d6f", "hollow3": "#8a4fbf", "hollow4": "#b5179e"}


def view_matrix(toward: np.ndarray, tilt_deg: float = 35.0) -> np.ndarray:
    """Rows = screen x, screen y, depth (toward the viewer). The viewer looks at the side
    `toward` points to, tilted so the adsorbates are seen at an angle, not head-on."""
    z = toward / np.linalg.norm(toward)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.radians(tilt_deg)
    z2 = np.cos(t) * z + np.sin(t) * x
    x2 = np.cos(t) * x - np.sin(t) * z
    return np.array([x2, y, z2])


def draw(ax, atoms, rot, centre, bonds=True, scale=0.88):
    p = (atoms.positions - centre) @ rot.T
    sym = atoms.get_chemical_symbols()
    rad = np.maximum(np.array([covalent_radii[atomic_numbers[s]] for s in sym]) * scale, 0.36)
    role = atoms.arrays["role"] if "role" in atoms.arrays else np.ones(len(atoms), dtype=int)
    lines = []
    if bonds:
        ads = np.where(role == 2)[0]
        for i in ads:
            for j in range(len(atoms)):
                if j == i or (role[j] == 2 and j < i):
                    continue
                d = np.linalg.norm(atoms.positions[i] - atoms.positions[j])
                limit = 1.25 * (covalent_radii[atomic_numbers[sym[i]]] + covalent_radii[atomic_numbers[sym[j]]])
                if d < limit:
                    lines.append((i, j, role[j] == 2))
    order = np.argsort(p[:, 2])
    for i, j, strong in lines:
        ax.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], color="#3b3b3b" if strong else "#9a9a9a",
                lw=3.0 if strong else 1.1, zorder=float(min(p[i, 2], p[j, 2])) + 0.01, solid_capstyle="round")
    for i in order:
        ax.add_patch(plt.Circle((p[i, 0], p[i, 1]), rad[i], facecolor=COLOR.get(sym[i], "#999999"), edgecolor=EDGE,
                                lw=0.9, zorder=float(p[i, 2]) + 0.02))
    return p


def finish(fig, ax, path, extent=5.2):
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=110, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    catalyst = load_catalyst()
    report = load_sites(catalyst)
    centre = catalyst.positions.mean(axis=0)

    # site map: sites on the half of the cluster facing the viewer
    rot = view_matrix(np.array([0.3, -0.5, 0.8]), tilt_deg=0.0)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    draw(ax, catalyst, rot, centre, bonds=False)
    for s in report.sites:
        if s.n @ rot[2] < 0.25:
            continue
        q = (s.p + 0.15 * s.n - centre) @ rot.T
        ax.scatter([q[0]], [q[1]], s=46 if s.kind == "top" else 30, color=SITE_COLOR[s.kind], edgecolors="white",
                   linewidths=0.6, zorder=100, marker={"top": "o", "bridge": "s", "hollow3": "^", "hollow4": "D"}[s.kind])
    for i in report.surface_ids:
        q = (catalyst.positions[i] - centre) @ rot.T
        if q[2] > 0.3:
            ax.text(q[0] - 0.45, q[1] + 0.45, str(i), ha="center", va="center", fontsize=7.5, color="#2b1a0c", zorder=101)
    finish(fig, ax, args.out / "sites.png", extent=4.2)

    frames = read(args.analysis_dir / "shortlist.extxyz", index=":")
    for k, atoms in enumerate(frames):
        ads = atoms.positions[atoms.arrays["role"] == 2]
        near = ads[np.linalg.norm(ads - centre, axis=1) < 6.0]
        toward = (near if len(near) else ads).mean(axis=0) - centre
        rot = view_matrix(toward)
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        p = draw(ax, atoms, rot, centre)
        extent = max(4.6, float(np.abs(p[:, :2]).max()) + 0.8)
        finish(fig, ax, args.out / f"shortlist_{k:03d}.png", extent=extent)
    print(f"wrote {len(frames) + 1} pictures to {args.out}")


if __name__ == "__main__":
    main()
