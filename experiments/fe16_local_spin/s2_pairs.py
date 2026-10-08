"""Step 2: do near-identical geometries at the same (q, M) carry different forces
when their local-spin patterns differ?

For every frame pair: permutation/rotation-aware aligned RMSD, the matched local-spin
distance D_s, and the matched, rotated force difference D_F (both on the same atom
correspondence). Writes results/pairs.csv, results/pairs_summary.json and figures.
"""
from __future__ import annotations

import csv
import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

from common import RESULTS, SPIN_ALL, aligned_match, meta

NEAR = 0.10      # A, "near-identical geometry"
DS_SAME = 0.25   # mu_B, matched local spins effectively equal
DS_DIFF = 1.0    # mu_B, a different local-spin pattern (e.g. the flipped site moved)


def main() -> None:
    fr = read(SPIN_ALL, ":")
    n = len(fr)
    X = [a.positions.copy() for a in fr]
    F = [a.arrays["REF_forces"].copy() for a in fr]
    S = [a.arrays["local_moment"].copy() for a in fr]
    M = [int(a.info["multiplicity"]) for a in fr]
    Q = [int(a.info["charge"]) for a in fr]
    E = [a.info["REF_energy"] for a in fr]
    md = [meta(a) for a in fr]

    rows = []
    for i, j in itertools.combinations(range(n), 2):
        rmsd, perm, R, cA, cB = aligned_match(X[i], X[j])
        sB = S[j][perm]
        FB = F[j][perm] @ R
        d_s = float(np.sqrt(((S[i] - sB) ** 2).mean()))
        d_f = float(np.sqrt(((F[i] - FB) ** 2).mean()))
        # Permutation-free spin descriptor check: sorted moments differ?
        d_s_sorted = float(np.sqrt(((np.sort(S[i]) - np.sort(S[j])) ** 2).mean()))
        neg_i = int(np.argmin(S[i])) if S[i].min() < 0 else -1
        neg_j = int(np.where(perm == int(np.argmin(S[j])))[0][0]) if S[j].min() < 0 else -1
        rows.append({
            "i": i, "j": j, "id_i": fr[i].info["record_id"], "id_j": fr[j].info["record_id"],
            "job_i": md[i].get("job_id"), "job_j": md[j].get("job_id"),
            "M_i": M[i], "M_j": M[j], "q_i": Q[i], "q_j": Q[j],
            "same_state": M[i] == M[j] and Q[i] == Q[j],
            "rmsd": rmsd, "D_s": d_s, "D_s_sorted": d_s_sorted, "D_F": d_f,
            "dE_meV_atom": 1000 * abs(E[i] - E[j]) / len(fr[i]),
            "neg_site_i": neg_i, "neg_site_j_mapped": neg_j,
            "Frms_i": float(np.sqrt((F[i] ** 2).mean())), "Frms_j": float(np.sqrt((F[j] ** 2).mean())),
        })

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    same = [r for r in rows if r["same_state"]]
    near = [r for r in same if r["rmsd"] < NEAR]

    def stats(sel):
        if not sel:
            return {"n": 0}
        df = np.array([r["D_F"] for r in sel])
        return {"n": len(sel), "D_F_median_meV_A": round(1000 * float(np.median(df)), 2),
                "D_F_max_meV_A": round(1000 * float(df.max()), 2),
                "rmsd_median_A": round(float(np.median([r["rmsd"] for r in sel])), 4),
                "dE_median_meV_atom": round(float(np.median([r["dE_meV_atom"] for r in sel])), 2)}

    # Bin by RMSD so the geometry contribution to D_F is compared like-for-like.
    bins = [0, 0.01, 0.03, 0.1, 0.3, 1.0]
    binned = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = [r for r in same if lo <= r["rmsd"] < hi]
        binned.append({"rmsd_bin_A": f"[{lo},{hi})",
                       "same_spin": stats([r for r in b if r["D_s"] < DS_SAME]),
                       "different_spin": stats([r for r in b if r["D_s"] >= DS_DIFF])})
    ds = np.array([r["D_s"] for r in near])
    dfn = np.array([r["D_F"] for r in near])
    rm = np.array([r["rmsd"] for r in near])
    corr = {}
    if len(near) > 3:
        from scipy.stats import spearmanr
        corr = {"spearman_DF_vs_Ds": round(float(spearmanr(dfn, ds)[0]), 3),
                "spearman_DF_vs_rmsd": round(float(spearmanr(dfn, rm)[0]), 3)}
    summary = {
        "frames": n, "pairs": len(rows), "same_q_M_pairs": len(same),
        "thresholds": {"near_rmsd_A": NEAR, "same_spin_Ds": DS_SAME, "different_spin_Ds": DS_DIFF},
        "near_same_state_pairs": len(near),
        "near_same_state_same_spin": stats([r for r in near if r["D_s"] < DS_SAME]),
        "near_same_state_different_spin": stats([r for r in near if r["D_s"] >= DS_DIFF]),
        "near_same_state_intermediate_spin": stats([r for r in near if DS_SAME <= r["D_s"] < DS_DIFF]),
        "near_correlations": corr,
        "binned_same_state": binned,
        "near_pairs_identical_sorted_moments_but_permuted": sum(
            1 for r in near if r["D_s"] >= DS_DIFF and r["D_s_sorted"] < DS_SAME),
    }
    (RESULTS / "pairs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.array([r["rmsd"] for r in same])
    y = 1000 * np.array([r["D_F"] for r in same])
    c = np.array([r["D_s"] for r in same])
    sc = ax[0].scatter(np.maximum(x, 1e-4), y, c=c, s=8, cmap="viridis", vmin=0, vmax=max(2.5, c.max()))
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].axvline(NEAR, ls=":", c="grey")
    ax[0].set_xlabel("aligned, permutation-matched RMSD (Å)")
    ax[0].set_ylabel("force difference $D_F$ (meV/Å)")
    ax[0].set_title("Fe16 frame pairs at the same q, M")
    fig.colorbar(sc, ax=ax[0], label="local-spin distance $D_s$ ($\\mu_B$)")
    if near:
        ax[1].scatter(ds, 1000 * dfn, c=rm, s=14, cmap="magma_r")
        ax[1].set_xlabel("local-spin distance $D_s$ ($\\mu_B$)")
        ax[1].set_ylabel("$D_F$ (meV/Å)")
        ax[1].set_title(f"near-identical geometries (RMSD < {NEAR} Å), n={len(near)}")
    fig.tight_layout()
    fig.savefig(RESULTS / "pairs_diagnostic.png", dpi=130)


if __name__ == "__main__":
    main()
