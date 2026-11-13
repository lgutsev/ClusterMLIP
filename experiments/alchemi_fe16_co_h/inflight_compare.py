"""Compare inflight-batching results with the fixed-batch runs of relax_bench.py.

    python inflight_compare.py INFLIGHT_DIR BENCH_DIR --out report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from cluster_mlip.adsorbate_search import same_structure  # noqa: E402


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path / "manifest.jsonl", encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inflight", type=Path)
    ap.add_argument("bench", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report: dict = {}
    for run in sorted(p for p in args.inflight.iterdir() if (p / "manifest.jsonl").exists()):
        a, fa = rows(run), read(run / "final.extxyz", index=":")
        entry = {"timings": json.loads((run / "timings.json").read_text(encoding="utf-8")),
                 "every_input_once": [r["pool_index"] for r in a] == list(range(len(a)))}
        for ref in ("native", "alchemi_b1"):
            b, fb = rows(args.bench / ref), read(args.bench / ref / "final.extxyz", index=":")
            both = [i for i in range(len(a)) if a[i]["status"] == "converged" and b[i]["status"] == "converged"]
            same = [i for i in both if same_structure(fa[i], fb[i], tol=0.15)]
            de = np.array([a[i]["energy"] - b[i]["energy_native_recompute"] for i in both])
            entry["vs_" + ref] = {
                "both_converged": len(both), "same_structure": len(same),
                "median_abs_dE_eV": float(np.median(np.abs(de))),
                "best_energy_minus_reference_best_eV": float(min(a[i]["energy"] for i in both)
                                                             - min(b[i]["energy_native_recompute"] for i in both))}
        report[run.name] = entry
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
