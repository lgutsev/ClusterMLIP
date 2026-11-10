"""Export the per-run progress curves as one small JSON (for the report page and figures).

    python export_curves.py RUN_DIR ANALYSIS_DIR --out curves.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import read_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("analysis_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--low-window", type=float, default=0.30)
    args = ap.parse_args()
    arms = json.loads((args.analysis_dir / "arms.json").read_text(encoding="utf-8"))
    e0 = arms["pool_min_energy"]
    low_found: dict[str, list[int]] = {}
    with open(args.analysis_dir / "pool.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if float(row["e_rel_ev"]) >= args.low_window:
                continue
            for item in row["found_by"].split(";"):
                run, at = item.split("@")
                low_found.setdefault(run, []).append(int(at))
    runs = []
    for manifest in sorted(args.run_dir.rglob("manifest.jsonl")):
        records = read_manifest(manifest)
        best, n_unique = None, 0
        relax, calls, best_curve, unique_curve = [], [], [], []
        for r in records:
            if r["status"] != "relaxed":
                continue
            ok = r["converged"] and not r["failure"] and r["energy"] is not None
            if ok and (best is None or r["energy"] < best):
                best = r["energy"]
            n_unique += int(ok and r["unique"])
            relax.append(r["cum_relaxations"])
            calls.append(r["cum_calc_calls"])
            best_curve.append(None if best is None else round(best - e0, 4))
            unique_curve.append(n_unique)
        name = f"{records[0]['arm']}-s{records[0]['seed']}"
        when = sorted(low_found.get(name, []))
        low_curve = [sum(1 for w in when if w <= x) for x in relax]
        runs.append({"arm": records[0]["arm"], "seed": records[0]["seed"], "relaxations": relax, "calls": calls,
                     "best": best_curve, "unique": unique_curve, "low_found": low_curve})
    args.out.write_text(json.dumps({"low_window_ev": args.low_window, "runs": runs}) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
