"""Freeze the inputs of the ALCHEMI pilot from the ACAT pilot's real candidates.

    python build_pool.py ACAT_RUN_DIR ACAT_ANALYSIS_DIR --out pool/

Writes two files (extxyz with per-atom ``role``/``mol_id`` and per-structure IDs):

* ``pool128.extxyz``: the first 128 starting structures the random arm (seed 0)
  relaxed, in their original order. Real, distinct candidates; nothing replicated.
* ``fixture12.extxyz``: single-point parity panel: clean Fe16, relaxed structures of
  distinct motifs from the shortlist (CO bound), two unrelaxed starts, and three
  moderately distorted copies (adsorbate atoms displaced by a fixed 0.05 A pattern).
  Desorbed-CO structures are left out on purpose (quarantined: out of domain).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "acat_fe16_co_h"))
from common import load_catalyst  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("analysis_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    run = args.run_dir / "random_s0"
    relaxed_ids = [json.loads(line)["candidate_id"] for line in open(run / "manifest.jsonl", encoding="utf-8")
                   if json.loads(line)["status"] == "relaxed"][:args.size]
    starts = {a.info["candidate_id"]: a for a in read(run / "structures.extxyz", index=":") if a.info["stage"] == "start"}
    pool = []
    for k, cid in enumerate(relaxed_ids):
        a = starts[cid].copy()
        a.info = {"pool_index": k, "candidate_id": cid, "family": "acat_random_s0_start", "charge": 0}
        pool.append(a)
    write(args.out / "pool128.extxyz", pool)

    catalyst = load_catalyst()
    catalyst.set_array("mol_id", np.full(len(catalyst), -1))
    catalyst.set_array("cluster", np.ones(len(catalyst), dtype=int))
    catalyst.info = {"fixture_id": "F00_clean_fe16", "family": "clean", "charge": 0}
    fixture = [catalyst]
    shortlist = read(args.analysis_dir / "shortlist.extxyz", index=":")
    rows = list(csv.DictReader(open(args.analysis_dir / "shortlist.csv", encoding="utf-8")))
    bound = [(r, a) for r, a in zip(rows, shortlist) if "desorbed" not in r["flags"] and r["relaxed_with_model"] == "True"]
    seen: set[str] = set()
    for r, a in bound:
        if r["motif"] in seen or len(seen) == 6:
            continue
        seen.add(r["motif"])
        b = a.copy()
        b.info = {"fixture_id": f"F{len(fixture):02d}_{r['shortlist_id']}", "family": "relaxed_motif", "motif": r["motif"],
                  "charge": 0}
        fixture.append(b)
    for k in (0, 1):
        b = pool[k].copy()
        b.info = {"fixture_id": f"F{len(fixture):02d}_start_{pool[k].info['candidate_id']}", "family": "unrelaxed_start",
                  "charge": 0}
        fixture.append(b)
    rng = np.random.default_rng(20261109)
    for k in (1, 2, 3):
        b = fixture[k].copy()
        ads = np.where(b.arrays["role"] == 2)[0]
        step = rng.normal(size=(len(ads), 3))
        step *= 0.05 / np.linalg.norm(step, axis=1, keepdims=True)
        b.positions[ads] += step
        b.info = {"fixture_id": f"F{len(fixture):02d}_distorted_{fixture[k].info['fixture_id']}", "family": "distorted_0.05A",
                  "charge": 0}
        fixture.append(b)
    write(args.out / "fixture12.extxyz", fixture)
    meta = {"pool": {"file": "pool128.extxyz", "n": len(pool), "sha256": sha(args.out / "pool128.extxyz"),
                     "source": str(run), "candidate_ids": relaxed_ids},
            "fixture": {"file": "fixture12.extxyz", "n": len(fixture), "sha256": sha(args.out / "fixture12.extxyz"),
                        "ids": [a.info["fixture_id"] for a in fixture]},
            "quarantined": "desorbed-CO structures (out of domain for the model; see ACAT audit)"}
    (args.out / "pool.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(len(pool), "pool structures;", len(fixture), "fixture structures:", meta["fixture"]["ids"])


if __name__ == "__main__":
    main()
