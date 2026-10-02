"""Leave-one-family-out folds over campaign + same-level legacy Fe16 frames.

usage: python build_lofo.py CAMPAIGN_DATASET LEGACY_same_level.extxyz OUT_DIR [TOL_A] [MIN_FAMILY_FRAMES]

Families = trajectory-level groups (any frame of one record within TOL, default 0.05 A, of any
frame of another; campaign split_group merges kept). Every family with >= MIN_FAMILY_FRAMES
(default 100) is held out once as `test` (all of its frames, both origins). For each fold two
training sets are written: `campaign` (campaign frames of the other families) and `combined`
(campaign + legacy frames of the other families). `valid` = a seeded 10% frame-level slice of the
training frames, for early stopping only; it never contains test-family frames. Small families
always stay in training.
"""
import collections, hashlib, json, sys
from pathlib import Path
import numpy as np
from ase.io import read, write

camp_dir, legacy_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
tol = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
min_family = int(sys.argv[5]) if len(sys.argv) > 5 else 100
if out_dir.exists() and any(out_dir.iterdir()):
    sys.exit(f"{out_dir} is not empty")


def meta(a):
    md = a.info.get("metadata")
    return json.loads(md) if isinstance(md, str) else dict(md or {})


camp = read(camp_dir / "all.extxyz", ":")
(level,) = {meta(a).get("label_level") for a in camp}
legacy = [a for a in read(legacy_path, ":")
          if (meta(a).get("geometry_role") or a.info.get("config_type")) == "legacy_minimum_optimization"]
assert all(meta(a).get("label_level") == level for a in legacy), "legacy level differs"
frames = [(a, "campaign") for a in camp] + [(a, "legacy") for a in legacy]
rec = [f"{s}:{a.info.get('parent_record_id') or meta(a).get('parent_record_id')}" for a, s in frames]
recs = sorted(set(rec)); idx = {r: i for i, r in enumerate(recs)}; par = list(range(len(recs)))
def find(i):
    while par[i] != i:
        par[i] = par[par[i]]; i = par[i]
    return i
def union(a, b):
    a, b = find(a), find(b)
    if a != b: par[b] = a
old = collections.defaultdict(list)
for (a, s), r in zip(frames, rec):
    if s == "campaign": old[meta(a).get("split_group")].append(idx[r])
for m in old.values():
    for x in m[1:]: union(m[0], x)
fp = np.array([np.sort(a.get_all_distances()[np.triu_indices(len(a), 1)]) for a, _ in frames])
sq = (fp ** 2).sum(1); rid = np.array([idx[r] for r in rec])
for st in range(0, len(fp), 256):
    d2 = (sq[st:st + 256, None] + sq[None, :] - 2 * fp[st:st + 256] @ fp.T) / fp.shape[1]
    for i, j in zip(*np.nonzero(d2 <= tol * tol)):
        union(rid[i + st], rid[j])
fam = [f"fam{find(idx[r]):03d}" for r in rec]
sizes = collections.Counter(fam)
held = [f for f, n in sizes.most_common() if n >= min_family]
summary = {"label_level": level, "tolerance_A": tol, "families": dict(sizes.most_common()), "folds": {}}
for k, f in enumerate(held):
    fold = out_dir / f"fold{k}_{f}"
    test = [(a, s) for (a, s), g in zip(frames, fam) if g == f]
    rest = [(a, s) for (a, s), g in zip(frames, fam) if g != f]
    info = {"held_out_family": f, "test_frames": len(test),
            "test_origin": dict(collections.Counter(s for _, s in test)),
            "test_M": dict(sorted(collections.Counter(int(a.info["spin"]) for a, _ in test).items()))}
    for arm, keep in (("campaign", {"campaign"}), ("combined", {"campaign", "legacy"})):
        pool = [(a, s) for a, s in rest if s in keep]
        valid, train = [], []
        for a, s in pool:
            h = hashlib.sha256(f"lofo|{a.info.get('record_id')}|{s}".encode()).digest()[0]
            (valid if h < 26 else train).append(a)          # ~10% frame-level, for early stopping
        d = fold / arm; d.mkdir(parents=True, exist_ok=True)
        write(d / "train.extxyz", train, format="extxyz"); write(d / "valid.extxyz", valid, format="extxyz")
        write(d / "test.extxyz", [a for a, _ in test], format="extxyz")
        write(d / "all.extxyz", train + valid + [a for a, _ in test], format="extxyz")
        info[arm] = {"train": len(train), "valid": len(valid),
                     "train_M": sorted({int(a.info["spin"]) for a in train})}
    summary["folds"][fold.name] = info
(out_dir / "lofo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=1))
