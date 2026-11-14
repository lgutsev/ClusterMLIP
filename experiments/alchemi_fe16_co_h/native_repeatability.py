"""Run-to-run repeatability of the native workflow (ASE BFGS + MACECalculator, float32, GPU).

    python native_repeatability.py POOL.extxyz MODEL --out report.json

The same 8 candidates used by reliability_checks.py are relaxed twice in one process.
This is the yardstick for the batched backend: a difference between batch sizes only
counts as a batching effect if it is larger than what the native workflow shows
against itself.
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
from cluster_mlip.adsorbate_search import native_evaluator, same_structure  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", type=Path)
    ap.add_argument("model")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from mace.calculators import MACECalculator

    eight = read(args.pool, index=":")[16:24]
    ev = native_evaluator(MACECalculator(model_paths=args.model, device="cuda", default_dtype="float32"),
                          fmax=0.05, max_steps=200)
    a, b = ev(eight), ev(eight)
    report = {
        "n": len(eight),
        "converged": [[r.converged for _, r in run] for run in (a, b)],
        "steps": [[r.steps for _, r in run] for run in (a, b)],
        "same_structure": sum(same_structure(x, y, tol=0.15) for (x, _), (y, _) in zip(a, b)),
        "max_abs_dE_eV": max(abs(x.energy - y.energy) for (_, x), (_, y) in zip(a, b)),
        "max_position_diff_A": max(float(np.abs(x.positions - y.positions).max()) for (x, _), (y, _) in zip(a, b)),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
