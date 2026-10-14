"""Step 4/6: the shared 60-frame memorization subset (+ the other 90 frames held out).

Seeded with the closest same-(q, M) pairs whose local-spin patterns differ (step 6) and
with same-geometry M -> M-2 hand-offs (including those that create a new antiparallel
site), then filled by a seeded draw stratified over (source geometry, M). Every model
sees exactly these files; model C additionally reads the `local_moment` column, which
the others ignore.
"""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd
from ase.io import read, write

from common import OUT, RESULTS, SPIN_ALL, meta

N_TRAIN = 60
SEED = 7


def main() -> None:
    fr = read(SPIN_ALL, ":")
    p = pd.read_csv(RESULTS / "pairs.csv")
    chosen: list[int] = []
    tags: dict[int, str] = {}

    def add(i, tag):
        i = int(i)
        if i not in chosen:
            chosen.append(i)
            tags[i] = tag

    same = p[p.same_state]
    diff = same[same.D_s >= 1.0].sort_values("rmsd")
    # distinct frames among the closest different-pattern pairs
    for _, r in diff.head(40).iterrows():
        if len(chosen) >= 16:
            break
        add(r.i, "diff_spin_pair"); add(r.j, "diff_spin_pair")
    cross = p[(~p.same_state) & (p.rmsd < 0.01)].copy()
    cross["newneg"] = (cross.neg_site_i != cross.neg_site_j_mapped)
    for _, r in cross.sort_values(["newneg", "D_s"], ascending=False).head(20).iterrows():
        if len(chosen) >= 28:
            break
        add(r.i, "same_geom_other_M"); add(r.j, "same_geom_other_M")

    rng = random.Random(SEED)
    strata: dict[tuple, list[int]] = {}
    for k, a in enumerate(fr):
        strata.setdefault((meta(a).get("split_group"), int(a.info["multiplicity"])), []).append(k)
    keys = sorted(strata, key=str)
    for k in keys:
        rng.shuffle(strata[k])
    while len(chosen) < N_TRAIN:
        progressed = False
        for k in keys:
            while strata[k] and strata[k][0] in chosen:
                strata[k].pop(0)
            if strata[k] and len(chosen) < N_TRAIN:
                add(strata[k].pop(0), "stratified_fill"); progressed = True
        if not progressed:
            break

    held = [k for k in range(len(fr)) if k not in chosen]
    OUT.mkdir(parents=True, exist_ok=True)
    data = OUT / "data"
    data.mkdir(exist_ok=True)
    train = [fr[k] for k in chosen]
    for k, a in zip(chosen, train):
        a.info["subset_tag"] = tags[k]
        a.info["frame_index_150"] = k
    for k in held:
        fr[k].info["frame_index_150"] = k
    write(data / "train.extxyz", train, format="extxyz")
    write(data / "heldout.extxyz", [fr[k] for k in held], format="extxyz")
    s = np.concatenate([a.arrays["local_moment"] for a in train])
    summary = {
        "train_frames": len(chosen), "heldout_frames": len(held),
        "tags": pd.Series(list(tags.values())).value_counts().to_dict(),
        "train_M": pd.Series([int(a.info["multiplicity"]) for a in train]).value_counts().sort_index().to_dict(),
        "train_groups": int(len({meta(a).get("split_group") for a in train})),
        "train_frames_with_negative_site": int(sum((a.arrays["local_moment"] < 0).any() for a in train)),
        "train_local_moment_range": [float(s.min()), float(s.max())],
        "train_F_rms_eV_A": float(np.sqrt(np.mean([(a.arrays["REF_forces"] ** 2).mean() for a in train]))),
        "train_indices_150": chosen,
    }
    (RESULTS / "subset.json").write_text(json.dumps({k: v for k, v in summary.items()}, indent=2, default=int), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=int))


if __name__ == "__main__":
    main()
