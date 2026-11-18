"""Evaluate models on a TM13 set, relax the source geometry, and score against DFT labels when they exist.

    python experiments/tm13_benchmark/score_models.py --set experiments/tm13_benchmark/sets/fe13_legacy_ubpw91_m45 \
        --models experiments/tm13_benchmark/models.json --out experiments/tm13_benchmark/report \
        [--dft LABELS.extxyz] [--reference-m 45]

CPU only (2 threads): the laptop GPU is shared with other sessions.

models.json: {"name": {"path": ..., "spin_input": true|false, "trained_on": ..., "note": ...}}. Spin-aware
models get info["spin"] = multiplicity and info["charge"]; they are evaluated at every M in --multiplicities.
Spin-blind models have one answer per geometry (multiplicity column empty). "zero_force" is the baseline.

--dft: extxyz from collect_dft.py (info: config, multiplicity, branch, usable; REF_energy eV; REF_forces eV/A).
Without it the DFT columns stay empty and status = "dft_pending".

Outputs: results.csv (one row per configuration x model x multiplicity), relaxations.csv, forces.json
(raw model forces, eV/A), relaxed/<model>_M<m>.xyz, summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tm13_common import BOND_CUTOFF, central_atom, describe, permutation_rmsd, sha256  # noqa: E402


def stats(f: np.ndarray, centre: int) -> dict:
    shell = np.delete(f, centre, axis=0)
    return {"rmse": float(np.sqrt(np.mean(f**2))), "mae": float(np.mean(np.abs(f))),
            "max": float(np.abs(f).max()), "centre": float(np.sqrt(np.mean(f[centre] ** 2))),
            "surface": float(np.sqrt(np.mean(shell**2)))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dft")
    ap.add_argument("--multiplicities", default="43,45,47")
    ap.add_argument("--reference-m", type=int, default=45)
    ap.add_argument("--fmax", type=float, default=0.005)
    ap.add_argument("--no-relax", action="store_true")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    import torch
    from ase.io import read, write
    from ase.optimize import BFGS
    from mace.calculators import MACECalculator

    torch.set_num_threads(2)
    setdir, out = Path(args.set), Path(args.out)
    (out / "relaxed").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((setdir / "manifest.json").read_text())
    configs = {c["name"]: read(setdir / c["file"]) for c in manifest["configurations"]}
    centre = manifest["central_atom_1based"] - 1
    cutoff = BOND_CUTOFF[manifest["element"]]
    mults = [int(m) for m in args.multiplicities.split(",")]
    models = json.loads(Path(args.models).read_text())

    dft: dict[tuple[str, int], dict] = {}
    if args.dft:
        for a in read(args.dft, ":"):
            if a.info.get("usable", True) and a.info.get("branch", "selected") == "selected":
                dft[(str(a.info["config"]), int(a.info["multiplicity"]))] = {
                    "e": float(a.info["REF_energy"]), "f": np.array(a.arrays["REF_forces"], float)}

    rows, relax_rows, raw = [], [], {}

    def add(model: str, m, name: str, e: float | None, f: np.ndarray, e0: float | None) -> None:
        ref = dft.get((name, m if m is not None else args.reference_m))
        ref0 = dft.get(("c0_source", m if m is not None else args.reference_m))
        s = stats(f, centre)
        row = {"set": manifest["set"], "config": name, "model": model, "multiplicity": "" if m is None else m,
               "dft_multiplicity_compared": (m if m is not None else args.reference_m),
               "model_energy_eV": "" if e is None else f"{e:.6f}",
               "model_dE_vs_c0_eV": "" if e is None or e0 is None else f"{e - e0:.6f}",
               "model_force_rms_eV_A": f"{s['rmse']:.5f}", "model_force_max_eV_A": f"{s['max']:.5f}",
               "model_force_centre_rms_eV_A": f"{s['centre']:.5f}", "model_force_surface_rms_eV_A": f"{s['surface']:.5f}"}
        if ref is None:
            row.update(status="dft_pending", dft_dE_vs_c0_eV="", dE_error_eV="", force_rmse_eV_A="", force_mae_eV_A="",
                       force_rmse_centre_eV_A="", force_rmse_surface_eV_A="", dft_force_rms_eV_A="")
        else:
            err = stats(f - ref["f"], centre)
            dde = "" if ref0 is None else ref["e"] - ref0["e"]
            row.update(status="scored", dft_dE_vs_c0_eV="" if dde == "" else f"{dde:.6f}",
                       dE_error_eV="" if dde == "" or e is None or e0 is None else f"{(e - e0) - dde:.6f}",
                       force_rmse_eV_A=f"{err['rmse']:.5f}", force_mae_eV_A=f"{err['mae']:.5f}",
                       force_rmse_centre_eV_A=f"{err['centre']:.5f}", force_rmse_surface_eV_A=f"{err['surface']:.5f}",
                       dft_force_rms_eV_A=f"{stats(ref['f'], centre)['rmse']:.5f}")
        rows.append(row)
        raw.setdefault(model, {}).setdefault(str(m), {})[name] = f.tolist()

    for name, a in configs.items():
        add("zero_force", None, name, None, np.zeros((len(a), 3)), None)

    ident = {}
    for model, spec in models.items():
        path = Path(spec["path"])
        ident[model] = {**spec, "sha256": sha256(path)}
        calc = MACECalculator(model_paths=str(path), device="cpu", default_dtype="float64")
        for m in mults if spec["spin_input"] else [None]:
            def attach(a):
                b = a.copy()
                b.info = {"spin": m, "charge": 0} if m is not None else {}
                calc.reset()
                b.calc = calc
                return b
            e0 = None
            for name, a in configs.items():
                b = attach(a)
                e, f = float(b.get_potential_energy()), np.array(b.get_forces())
                e0 = e if name == "c0_source" else e0
                add(model, m, name, e, f, e0)
            if args.no_relax:
                continue
            b = attach(configs["c0_source"])
            opt = BFGS(b, logfile=None)
            converged = opt.run(fmax=args.fmax, steps=500)
            start, end = configs["c0_source"].positions, b.positions
            d0, d1 = describe(start, cutoff), describe(end, cutoff)
            tag = f"{model}_M{m}" if m is not None else model
            write(out / "relaxed" / f"{tag}.xyz", b, comment=f"{manifest['set']} relaxed with {tag}")
            relax_rows.append({"set": manifest["set"], "model": model, "multiplicity": "" if m is None else m,
                               "converged": bool(converged), "steps": opt.get_number_of_steps(), "fmax_eV_A": args.fmax,
                               "energy_gain_eV": f"{float(b.get_potential_energy()) - e0:.6f}",
                               "start_motif": d0["motif"], "final_motif": d1["motif"],
                               "motif_changed": d0["motif"] != d1["motif"],
                               "start_mean_nn_A": f"{d0['mean_nn_A']:.4f}", "final_mean_nn_A": f"{d1['mean_nn_A']:.4f}",
                               "mean_nn_change_A": f"{d1['mean_nn_A'] - d0['mean_nn_A']:.4f}",
                               "rmsd_perm_aligned_A": f"{permutation_rmsd(end, start):.4f}",
                               "dft_endpoint_check": "needed" if d0["motif"] != d1["motif"] else "not_needed"})
        print("done", model, flush=True)
        del calc

    def dump(path: Path, data: list[dict]) -> None:
        if data:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)

    dump(out / "results.csv", rows)
    dump(out / "relaxations.csv", relax_rows)
    (out / "forces.json").write_text(json.dumps(raw), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"set": manifest["set"], "units": {"energy": "eV", "force": "eV/A", "length": "A"},
                                                  "dft_labels": args.dft, "dft_states_used": len(dft),
                                                  "reference_multiplicity_for_spin_blind_models": args.reference_m,
                                                  "models": ident}, indent=1) + "\n", encoding="utf-8")
    print("rows", len(rows), "relaxations", len(relax_rows), "dft states", len(dft))


if __name__ == "__main__":
    main()
