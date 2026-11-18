"""Shared geometry helpers for the TM13 benchmark (numpy + scipy only)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

BOND_CUTOFF = {"Fe": 3.10, "Co": 3.00, "Ni": 3.00, "Cu": 3.10}  # Angstrom; sits in the first/second-shell gap


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def central_atom(pos: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(pos - pos.mean(0), axis=1)))


def bonds(pos: np.ndarray, cutoff: float) -> list[tuple[int, int, float]]:
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    n = len(pos)
    return [(i, j, float(d[i, j])) for i in range(n) for j in range(i + 1, n) if d[i, j] < cutoff]


def describe(pos: np.ndarray, cutoff: float) -> dict:
    """Motif summary: bond count, coordination of the most central atom, mean nearest-neighbour distance."""
    b = bonds(pos, cutoff)
    cn = np.zeros(len(pos), int)
    for i, j, _ in b:
        cn[i] += 1
        cn[j] += 1
    c = central_atom(pos)
    r = np.linalg.norm(pos - pos.mean(0), axis=1)
    centred = bool(cn[c] == 12 and r[c] < 0.5)
    if len(pos) == 13 and centred and len(b) == 42:
        motif = "icosahedron-like (centre CN 12, 42 bonds)"
    elif len(pos) == 13 and centred and len(b) == 36:
        motif = "cuboctahedron-like (centre CN 12, 36 bonds)"
    elif centred:
        motif = f"centred, {len(b)} bonds"
    else:
        motif = f"no 12-coordinate centre, {len(b)} bonds, max CN {int(cn.max())}"
    dist = np.array([x[2] for x in b])
    return {"motif": motif, "n_bonds": len(b), "centre_cn": int(cn[c]), "centre_offset_A": float(r[c]),
            "mean_nn_A": float(dist.mean()) if len(dist) else float("nan"),
            "min_nn_A": float(dist.min()) if len(dist) else float("nan"),
            "max_nn_A": float(dist.max()) if len(dist) else float("nan"), "bond_cutoff_A": cutoff}


def kabsch(p: np.ndarray, q: np.ndarray, allow_reflection: bool = False) -> tuple[np.ndarray, float]:
    """Rotation R minimising |p R - q| for centred p, q; returns (R, rmsd)."""
    u, _, vt = np.linalg.svd(p.T @ q)
    s = np.eye(3)
    if not allow_reflection and np.linalg.det(u @ vt) < 0:
        s[2, 2] = -1
    r = u @ s @ vt
    return r, float(np.sqrt(np.mean(np.sum((p @ r - q) ** 2, axis=1))))


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def permutation_rmsd(a: np.ndarray, b: np.ndarray, starts: int = 300, seed: int = 0,
                     allow_reflection: bool = False) -> float:
    """Aligned RMSD between two same-element clusters, minimised over atom permutation and rotation.

    Alternates Hungarian assignment and Kabsch rotation from `starts` seeded random orientations and keeps
    the lowest value. With allow_reflection the mirror image counts as the same structure.
    """
    a = a - a.mean(0)
    b = b - b.mean(0)
    rng = np.random.default_rng(seed)
    best = np.inf
    mirrors = (1.0, -1.0) if allow_reflection else (1.0,)
    for k in range(starts):
        for m in mirrors:
            p = (a * m) @ (np.eye(3) if k == 0 else _random_rotation(rng))
            last = np.inf
            for _ in range(50):
                cost = ((p[:, None] - b[None]) ** 2).sum(2)
                _, col = linear_sum_assignment(cost)
                rot, val = kabsch(p, b[col])
                p = p @ rot
                if last - val < 1e-10:
                    break
                last = val
            best = min(best, val)
    return float(best)
