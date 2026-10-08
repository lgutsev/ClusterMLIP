"""Shared helpers for the Fe16 local-spin experiment (see README.md)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(os.environ.get("FE16_ROOT", "D:/MLIP_Work_Folder/Cluster_MLIP/fe16_v1"))
# Full collected Fe16 set (1,251 frames) and its population-labelled subset (150 frames,
# verified against the raw G09 logs in 07_crosswalk).
COLLECT_ALL = ROOT / "02d_fe16_tol05" / "all.extxyz"
SPIN_ALL = ROOT / "06_ab_provisional" / "all.extxyz"
OUT = Path(os.environ.get("FE16_LS_OUT", str(ROOT / "12_local_spin_mp")))
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def meta(atoms) -> dict:
    m = atoms.info.get("metadata", {})
    return json.loads(m) if isinstance(m, str) else dict(m)


def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rotation R minimising |P R - Q| for centred P, Q (proper rotations only)."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    return U @ D @ Vt


def _start_rotations(P: np.ndarray, Q: np.ndarray) -> list[np.ndarray]:
    """Principal-axis frame matches (24 proper sign/axis choices) as starting guesses."""
    def axes(X):
        w, v = np.linalg.eigh(X.T @ X)
        return v[:, np.argsort(w)]

    Ap, Aq = axes(P), axes(Q)
    rots = []
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                S = np.diag([sx, sy, sz])
                R = Ap @ S @ Aq.T
                if np.linalg.det(R) > 0:
                    rots.append(R)
    rots.append(np.eye(3))
    return rots


def aligned_match(PA: np.ndarray, PB: np.ndarray, ZA=None, ZB=None, iters: int = 20):
    """Permutation- and rotation-aware match of B onto A.

    Returns (rmsd, perm, R, cA, cB) with  (PB[perm] - cB) @ R  ~=  PA - cA.
    Alternates Hungarian assignment (within element) and Kabsch from several
    principal-axis starts; keeps the best.
    """
    cA, cB = PA.mean(0), PB.mean(0)
    P, Q = PA - cA, PB - cB
    n = len(P)
    if ZA is None:
        ZA = ZB = np.zeros(n, int)
    big = 1e6 * (ZA[:, None] != ZB[None, :])
    best = (np.inf, None, None)
    for R0 in _start_rotations(Q, P):
        R = R0
        perm = None
        for _ in range(iters):
            QR = Q @ R
            cost = ((P[:, None, :] - QR[None, :, :]) ** 2).sum(-1) + big
            _, new = linear_sum_assignment(cost)
            if perm is not None and np.array_equal(new, perm):
                break
            perm = new
            R = kabsch(Q[perm], P)
        rmsd = float(np.sqrt(((Q[perm] @ R - P) ** 2).sum(1).mean()))
        if rmsd < best[0]:
            best = (rmsd, perm, R)
    return best[0], best[1], best[2], cA, cB
