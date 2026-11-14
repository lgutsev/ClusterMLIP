"""Review follow-up (a): which held-out frames are actually unseen?

s4 split by frame, so many held-out frames have a near-duplicate in training. Each
held-out frame is classed by its nearest training frame (permutation/rotation-aligned
RMSD):

  same_state_twin  a training frame at the same (q, M) within TWIN Å   -> memorization, not a test
  other_M_twin     same geometry in training only at another M          -> tests the M dependence
  unseen           no training frame within TWIN Å at any M             -> the real held-out test

Errors are then reported per class from results/predictions.csv. Also counts the unique
geometries / (geometry, M) states in the 150 labelled frames.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ase.io import read

from common import OUT, RESULTS, SPIN_ALL, aligned_match

TWIN = 0.01  # Å


def clusters(frames, key):
    """Greedy single-linkage clusters of frames within TWIN Å (optionally within key)."""
    labels = [-1] * len(frames)
    reps: list[int] = []
    for i, a in enumerate(frames):
        for c, r in enumerate(reps):
            if key(frames[r]) == key(a) and aligned_match(frames[r].positions, a.positions)[0] < TWIN:
                labels[i] = c
                break
        else:
            reps.append(i)
            labels[i] = len(reps) - 1
    return len(reps)


def main() -> None:
    train = read(OUT / "data" / "train.extxyz", ":")
    held = read(OUT / "data" / "heldout.extxyz", ":")
    rows = []
    for k, a in enumerate(held):
        same, other = np.inf, np.inf
        for b in train:
            r = aligned_match(a.positions, b.positions)[0]
            if (a.info["multiplicity"], a.info["charge"]) == (b.info["multiplicity"], b.info["charge"]):
                same = min(same, r)
            else:
                other = min(other, r)
        cls = "same_state_twin" if same < TWIN else ("other_M_twin" if other < TWIN else "unseen")
        rows.append({"k": k, "record_id": a.info["record_id"], "M": int(a.info["multiplicity"]),
                     "nearest_same_state_A": same, "nearest_other_M_A": other, "class": cls})
    cl = pd.DataFrame(rows)
    cl.to_csv(RESULTS / "heldout_classes.csv", index=False)

    allf = read(SPIN_ALL, ":")
    uniq = {"labelled_frames": len(allf),
            "unique_geometries_0.01A": clusters(allf, lambda a: 0),
            "unique_geometry_M_states_0.01A": clusters(allf, lambda a: (a.info["multiplicity"], a.info["charge"])),
            "train_unique_geometry_M_states_0.01A": clusters(train, lambda a: (a.info["multiplicity"], a.info["charge"]))}

    p = pd.read_csv(RESULTS / "predictions.csv")
    p = p[(p.split == "heldout")].merge(cl[["k", "class"]], on="k")
    out = []
    for (regime, model, variant, c), g in p.groupby(["regime", "model", "variant", "class"]):
        out.append({"regime": regime, "model": model, "variant": variant, "class": c,
                    "frames": int(g.k.nunique()),
                    "E_MAE_meV_atom": round(float(g.dE_meV_atom.abs().mean()), 2),
                    "F_MAE_meV_A": round(float(g.F_mae_meV_A.mean()), 1),
                    "F_RMSE_meV_A": round(float(np.sqrt((g.F_rmse_meV_A ** 2).mean())), 1),
                    "zero_force_RMSE_meV_A": round(float(np.sqrt((g.F_ref_rms_meV_A ** 2).mean())), 1),
                    "seeds": int(g.seed.nunique())})
    summary = {"twin_threshold_A": TWIN, "class_counts": cl["class"].value_counts().to_dict(),
               "unseen_M": cl[cl["class"] == "unseen"].M.value_counts().sort_index().to_dict(),
               "uniqueness": uniq, "errors": out}
    (RESULTS / "heldout_unseen.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, indent=2))
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
