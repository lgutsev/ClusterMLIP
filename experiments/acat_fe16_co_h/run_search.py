"""Run one or more search arms from a JSON config.

    python run_search.py configs/smoke_toy.json --out OUT_DIR

Config keys: ``calculator`` (``toy`` or ``mace:<path>``), ``device``, ``dtype``,
``arms`` (list of ``random`` / ``evolutionary``), ``seeds`` and ``search``
(fields of ``SearchConfig``). Each (arm, seed) writes ``<out>/<arm>_s<seed>/``
with ``manifest.jsonl``, ``structures.extxyz`` and ``summary.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from common import CATALYST_FILE, HERE, load_catalyst, load_sites, make_calculator

from cluster_mlip.adsorbate_search import AdsorbateSearch, SearchConfig, site_table


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    spec = cfg["calculator"]
    if spec.startswith("mace:"):
        spec = "mace:" + str(Path(os.path.expanduser(spec[5:])).resolve())
    catalyst = load_catalyst()
    report = load_sites(catalyst)
    calc = make_calculator(spec, cfg.get("device", "cuda"), cfg.get("dtype", "float32"))
    args.out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=HERE, capture_output=True, text=True).stdout.strip())
    provenance = {
        "config": cfg, "config_file": str(args.config), "git_commit": commit, "git_dirty": dirty,
        "catalyst_file": CATALYST_FILE.name, "catalyst_sha256": sha256(CATALYST_FILE),
        "catalyst_info": {k: (v.item() if hasattr(v, "item") else v) for k, v in catalyst.info.items()},
        "catalyst_atom_ids": list(range(len(catalyst))), "surface_ids": report.surface_ids,
        "interior_ids": report.interior_ids, "site_counts": report.counts(), "site_notes": report.notes,
        "calculator": spec,
        "calculator_sha256": sha256(Path(spec[5:])) if spec.startswith("mace:") else None,
        "calculator_takes_charge_or_spin": False,
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (args.out / "sites.json").write_text(json.dumps(site_table(report), indent=1) + "\n", encoding="utf-8")
    for arm in args.arms or cfg["arms"]:
        for seed in args.seeds if args.seeds is not None else cfg["seeds"]:
            search = AdsorbateSearch(catalyst, report.sites, calc, SearchConfig(arm=arm, seed=seed, **cfg.get("search", {})))
            search.run()
            search.write(args.out / f"{arm}_s{seed}")
            print(json.dumps(search.summary()), flush=True)


if __name__ == "__main__":
    main()
