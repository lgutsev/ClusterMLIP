"""MLIP gate, part 2: forces by element on existing UBPW91 force labels.

Uses the collected Warehouse-2 force frames (UBPW91/6-311++G*): neutral Fe clusters
carrying CO or H, and bare Fe16. Reports the force RMSE of each model per element,
next to the RMS of the reference forces themselves ("zero" = a model that predicts
no force). Adsorbate elements (C, O, H) and Fe are kept apart.

    python gate_forces.py DATASET.extxyz --model name=path [...] --out DIR [--max-per-formula 300]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import iread


def wanted(formula_counts: dict[str, int]) -> str | None:
    els = set(formula_counts)
    if els == {"Fe"} and formula_counts["Fe"] == 16:
        return "Fe16 (bare)"
    if "Fe" not in els or not els <= {"Fe", "C", "O", "H"}:
        return None
    if els == {"Fe", "C", "O"} and formula_counts["C"] == 1 and formula_counts["O"] == 1:
        return f"Fe{formula_counts['Fe']}(CO)"
    if els == {"Fe", "H"}:
        return f"Fe{formula_counts['Fe']}H{formula_counts['H']}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-per-formula", type=int, default=300)
    ap.add_argument("--max-force", type=float, default=50.0, help="drop frames with a reference force above this (eV/A)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from mace.calculators import MACECalculator

    groups: dict[str, list] = defaultdict(list)
    dropped = 0
    for atoms in iread(args.dataset, index=":"):
        if int(atoms.info.get("charge", 0)) != 0 or "REF_forces" not in atoms.arrays:
            continue
        counts: dict[str, int] = {}
        for s in atoms.get_chemical_symbols():
            counts[s] = counts.get(s, 0) + 1
        name = wanted(counts)
        if name is None:
            continue
        if np.abs(atoms.arrays["REF_forces"]).max() > args.max_force:
            dropped += 1
            continue
        groups[name].append(atoms)
    rng = np.random.default_rng(0)
    for name, frames in groups.items():
        if len(frames) > args.max_per_formula:
            keep = sorted(rng.choice(len(frames), size=args.max_per_formula, replace=False).tolist())
            groups[name] = [frames[i] for i in keep]

    report: dict = {"dataset": str(args.dataset), "dropped_force_outliers": dropped,
                    "frames_used": {k: len(v) for k, v in groups.items()}, "models": {}}
    for spec in args.model:
        label, path = spec.split("=", 1)
        calc = MACECalculator(model_paths=path, device=args.device, default_dtype="float32")
        out = {}
        for name, frames in sorted(groups.items()):
            err: dict[str, list] = defaultdict(list)
            ref: dict[str, list] = defaultdict(list)
            for atoms in frames:
                a = atoms.copy()
                a.calc = calc
                f = a.get_forces()
                for el in set(a.get_chemical_symbols()):
                    sel = np.array(a.get_chemical_symbols()) == el
                    err[el].append((f[sel] - atoms.arrays["REF_forces"][sel]).ravel())
                    ref[el].append(atoms.arrays["REF_forces"][sel].ravel())
            out[name] = {el: {"model_rmse_ev_per_A": round(float(np.sqrt(np.mean(np.concatenate(err[el]) ** 2))), 3),
                              "zero_model_rmse_ev_per_A": round(float(np.sqrt(np.mean(np.concatenate(ref[el]) ** 2))), 3),
                              "components": int(len(np.concatenate(err[el])))} for el in sorted(err)}
        report["models"][label] = out
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gate_forces.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
