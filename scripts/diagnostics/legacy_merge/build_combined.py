"""Merge campaign and same-level legacy Fe16 frames with leakage-safe grouping, then split.

usage: python build_combined.py CAMPAIGN_DATASET LEGACY_same_level.extxyz OUT_DIR [TOL_A]

* Legacy frames: only geometry_role/config_type "legacy_minimum_optimization", and only if their
  label_level equals the campaign's single label_level (refused otherwise).
* Grouping at the TRAJECTORY level: two records share a group when ANY frame of one lies within
  TOL (default 0.05 A) sorted-pair-distance RMS of any frame of the other (catches different
  starts that converge to the same minimum), plus the campaign's existing split_group merges.
* Split: groups ranked by a seeded hash; seeds scanned with a rule fixed in advance (valid and
  test each >= 10% of frames and >= 2 groups, train >= 60% and covering every multiplicity);
  the most balanced passing seed is used. Sizes and coverage only -- never model results.
"""
import collections, hashlib, json, sys
from pathlib import Path
import numpy as np
from ase.io import read, write

camp_dir, legacy_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
tol = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
if out_dir.exists() and any(out_dir.iterdir()):
    sys.exit(f"{out_dir} is not empty")
out_dir.mkdir(parents=True, exist_ok=True)


def meta(a):
    md = a.info.get("metadata")
    return json.loads(md) if isinstance(md, str) else dict(md or {})


camp = read(camp_dir / "all.extxyz", ":")
levels = {meta(a).get("label_level") for a in camp}
if len(levels) != 1:
    sys.exit(f"campaign has {len(levels)} label levels")
level = levels.pop()
legacy_all = read(legacy_path, ":")
legacy = [a for a in legacy_all if (meta(a).get("geometry_role") or a.info.get("config_type")) == "legacy_minimum_optimization"]
bad = [a for a in legacy if meta(a).get("label_level") != level]
if bad:
    sys.exit(f"{len(bad)} legacy frames are not at {level}")
frames = [(a, "campaign") for a in camp] + [(a, "legacy") for a in legacy]
rec = [f"{src}:{a.info.get('parent_record_id') or meta(a).get('parent_record_id')}" for a, src in frames]

# union-find over records
recs = sorted(set(rec)); idx = {r: i for i, r in enumerate(recs)}; parent = list(range(len(recs)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[rb] = ra
old = collections.defaultdict(list)
for (a, src), r in zip(frames, rec):
    if src == "campaign": old[meta(a).get("split_group")].append(idx[r])
for members in old.values():
    for m in members[1:]: union(members[0], m)
fp = np.array([np.sort(a.get_all_distances()[np.triu_indices(len(a), 1)]) for a, _ in frames])
sq = (fp ** 2).sum(1)
rid = np.array([idx[r] for r in rec])
links = 0
for start in range(0, len(fp), 256):
    blk = fp[start:start + 256]
    d2 = (sq[start:start + 256, None] + sq[None, :] - 2 * blk @ fp.T) / fp.shape[1]
    ii, jj = np.nonzero(d2 <= tol * tol)
    for i, j in zip(ii + start, jj):
        if rid[i] != rid[j] and find(rid[i]) != find(rid[j]):
            union(rid[i], rid[j]); links += 1
group_of = {r: f"grp:{find(idx[r]):04d}" for r in recs}

frames_per_group = collections.Counter(group_of[r] for r in rec)
mults_per_group = collections.defaultdict(set)
for (a, _), r in zip(frames, rec):
    mults_per_group[group_of[r]].add(int(a.info["spin"]))
groups = sorted(frames_per_group); total = len(frames); all_m = set().union(*mults_per_group.values())


def assign(seed):
    order = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}|{g}".encode()).digest())
    n = len(order); n_test = max(1, round(0.15 * n)); n_valid = max(1, round(0.15 * n))
    return {g: ("test" if k < n_test else "valid" if k < n_test + n_valid else "train") for k, g in enumerate(order)}


best = None
for seed in range(5000):
    sp = assign(seed); size = collections.Counter(); ng = collections.Counter(); cover = collections.defaultdict(set)
    for g, s in sp.items():
        size[s] += frames_per_group[g]; ng[s] += 1; cover[s] |= mults_per_group[g]
    if (size["valid"] >= 0.10 * total and size["test"] >= 0.10 * total and size["train"] >= 0.60 * total
            and ng["valid"] >= 2 and ng["test"] >= 2 and cover["train"] == all_m):
        score = (abs(size["valid"] - size["test"]), -size["train"])
        if best is None or score < best[0]:
            best = (score, seed, sp, dict(size), dict(ng))
if best is None:
    print("groups:", len(groups), "| frames per group (largest first):", sorted(frames_per_group.values(), reverse=True)[:15],
          "| cross-record links:", links)
    sys.exit("no seed satisfies the split rule")
_, seed, split, size, ng = best

out = collections.defaultdict(list)
for (a, src), r in zip(frames, rec):
    md = meta(a); md["split_group"] = group_of[r]; md["dataset_origin"] = src
    b = a.copy(); b.info["metadata"] = json.dumps(md, sort_keys=True)
    out[split[group_of[r]]].append(b)
for name in ("train", "valid", "test"):
    write(out_dir / f"{name}.extxyz", out[name], format="extxyz")
write(out_dir / "all.extxyz", out["train"] + out["valid"] + out["test"], format="extxyz")
origin = {s: dict(collections.Counter(json.loads(b.info["metadata"])["dataset_origin"] for b in out[s])) for s in out}
summary = {"label_level": level, "tolerance_A": tol, "frames": total, "campaign_frames": len(camp),
           "legacy_frames": len(legacy), "records": len(recs), "groups": len(groups),
           "cross_record_links": links, "split_seed": seed, "split_frames": size, "split_groups": ng,
           "split_origin": origin}
(out_dir / "combined_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
