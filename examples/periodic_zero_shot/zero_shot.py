"""Zero-shot check of foundation MLIPs against `vasp-ingest` frames.

Two steps, so each model runs in the environment it needs:

    python zero_shot.py predict frames.extxyz --backend mace --model PATH --name mp0-medium -o preds/
    python zero_shot.py predict frames.extxyz --backend uma  --model PATH --task omat --name uma-s-1p1 -o preds/
    python zero_shot.py report  frames.extxyz preds/*.npz -o report/ [--reactions reactions.json]

`predict` is a single point on every DFT geometry (no relaxation) and writes
<name>.npz. `report` needs only NumPy + ASE and compares every model with the
DFT labels:

* forces on mobile atoms (selective-dynamics ``fixed`` atoms excluded), overall
  and per element, against the predict-zero baseline;
* relative energies along each job (E_i - E_last), which removes each model's
  constant per-composition offset;
* optional reaction energies between final frames of jobs (``--reactions``:
  {"name": {"job_a": +1, "job_b": -1, ...}}), at the DFT geometries.

The models take no charge or spin input. Frames that differ only in the
electronic state get identical predictions, and that is reported, not hidden.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read


def _frames(path: Path):
    return read(str(path), index=":", format="extxyz")


def command_predict(args: argparse.Namespace) -> int:
    frames = _frames(Path(args.frames))
    if args.backend == "mace":
        from mace.calculators import MACECalculator

        calc = MACECalculator(model_paths=args.model, device=args.device, default_dtype="float64")
    else:
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

        # "general" avoids the Triton GPU kernels (Triton is not available on Windows).
        settings = InferenceSettings(execution_mode=args.uma_execution_mode)
        unit = load_predict_unit(args.model, inference_settings=settings, device=args.device)
        calc = FAIRChemCalculator(unit, task_name=args.task)
    energies, forces, ids = [], [], []
    start = time.time()
    for k, atoms in enumerate(frames):
        atoms = atoms.copy()  # drop the reference SinglePointCalculator
        if args.backend == "uma" and args.task != "omol":
            # FAIRChemCalculator feeds info["charge"]/["spin"] to every task head; only omol
            # was trained on them (the others saw 0), and our frames carry the DFT multiplicity.
            for key in ("charge", "spin"):
                atoms.info.pop(key, None)
        atoms.calc = calc
        energies.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(), dtype=float))
        ids.append(atoms.info["structure_id"])
        if k % 100 == 0:
            print(f"{args.name}: {k + 1}/{len(frames)} frames, {time.time() - start:.0f} s", flush=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{args.name}.npz",
        energy=np.array(energies), forces=np.concatenate(forces), n_atoms=np.array([len(f) for f in forces]),
        structure_id=np.array(ids), model=args.model, backend=args.backend, task=args.task or "",
        seconds=time.time() - start,
    )
    print(f"{args.name}: {len(frames)} frames in {time.time() - start:.0f} s")
    return 0


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def command_report(args: argparse.Namespace) -> int:
    frames = _frames(Path(args.frames))
    ids = [a.info["structure_id"] for a in frames]
    jobs = [a.info["job"] for a in frames]
    e_ref = np.array([a.info["REF_energy"] for a in frames])
    f_ref = [np.asarray(a.arrays["REF_forces"]) for a in frames]
    mobile = [~np.asarray(a.arrays.get("fixed", np.zeros(len(a), bool)), dtype=bool) for a in frames]
    symbols = [np.array(a.get_chemical_symbols()) for a in frames]
    minimum = np.array([a.info["config_type"] == "relaxed_minimum" for a in frames])
    reactions = json.loads(Path(args.reactions).read_text()) if args.reactions else {}
    last_index = {job: max(i for i, j in enumerate(jobs) if j == job) for job in set(jobs)}

    def reaction_energy(energy: np.ndarray, terms: dict[str, float]) -> float:
        return float(sum(c * energy[last_index[job]] for job, c in terms.items()))

    elements = sorted({s for sym in symbols for s in sym})
    report: dict = {"frames": len(frames), "jobs": sorted(set(jobs)), "models": {}}
    f_all_ref = np.concatenate([f[m] for f, m in zip(f_ref, mobile)])
    report["zero_force_baseline"] = {"rmse": _rmse(f_all_ref), "mae": float(np.mean(np.abs(f_all_ref)))}
    report["dft_reactions"] = {name: reaction_energy(e_ref, t) for name, t in reactions.items()}

    rows = []
    for path in args.predictions:
        data = np.load(path, allow_pickle=False)
        name = Path(path).stem
        if list(data["structure_id"]) != ids:
            raise SystemExit(f"{path}: frames differ from {args.frames}")
        splits = np.cumsum(data["n_atoms"])[:-1]
        f_pred = np.split(data["forces"], splits)
        e_pred = np.asarray(data["energy"])
        diff = [p - r for p, r in zip(f_pred, f_ref)]
        d_mobile = np.concatenate([d[m] for d, m in zip(diff, mobile)])
        per_element = {}
        for el in elements:
            sel = np.concatenate([d[m & (s == el)] for d, m, s in zip(diff, mobile, symbols)])
            ref = np.concatenate([r[m & (s == el)] for r, m, s in zip(f_ref, mobile, symbols)])
            per_element[el] = {"rmse": _rmse(sel), "zero_baseline_rmse": _rmse(ref), "n_atoms": int(len(sel))}
        d_min = np.concatenate([d[m] for d, m, k in zip(diff, mobile, minimum) if k])
        cosines = []
        for p, r, m in zip(f_pred, f_ref, mobile):
            norm = np.linalg.norm(p[m], axis=1) * np.linalg.norm(r[m], axis=1)
            big = norm > 0.1 ** 2  # both |F| > ~0.1 eV/A: direction is meaningful
            cosines.extend((np.sum(p[m] * r[m], axis=1)[big] / norm[big]).tolist())
        rel_err = []
        per_job = {}
        for job in sorted(set(jobs)):
            idx = [i for i, j in enumerate(jobs) if j == job]
            last = idx[-1]
            err = (e_pred[idx] - e_pred[last]) - (e_ref[idx] - e_ref[last])
            n = len(frames[last])
            per_job[job] = {
                "frames": len(idx),
                "relative_energy_mae_meV_atom": float(np.mean(np.abs(err)) / n * 1000),
                "relaxation_energy_dft_eV": float(e_ref[idx[0]] - e_ref[last]),
                "relaxation_energy_model_eV": float(e_pred[idx[0]] - e_pred[last]),
                "final_offset_eV": float(e_pred[last] - e_ref[last]),
                "final_offset_meV_atom": float((e_pred[last] - e_ref[last]) / n * 1000),
            }
            rel_err.extend((err / n).tolist())
        model = {
            "model": str(data["model"]), "backend": str(data["backend"]), "task": str(data["task"]),
            "seconds": float(data["seconds"]),
            "force_rmse_mobile": _rmse(d_mobile),
            "force_mae_mobile": float(np.mean(np.abs(d_mobile))),
            "force_rmse_at_minima": _rmse(d_min),
            "force_cosine_median": float(np.median(cosines)) if cosines else float("nan"),
            "force_per_element": per_element,
            "relative_energy_mae_meV_atom": float(np.mean(np.abs(rel_err)) * 1000),
            "jobs": per_job,
            "reactions": {name: reaction_energy(e_pred, t) for name, t in reactions.items()},
        }
        report["models"][name] = model
        rows.append(name)

    # Frames whose positions are identical (same geometry, different electronic state)
    # get identical predictions from a spin-blind model; list them.
    same_geometry = []
    finals = {job: frames[i] for job, i in last_index.items()}
    names = sorted(finals)
    for a_i, a in enumerate(names):
        for b in names[a_i + 1:]:
            fa, fb = finals[a], finals[b]
            if len(fa) == len(fb) and fa.get_chemical_formula() == fb.get_chemical_formula():
                dmax = float(np.max(np.abs(fa.positions - fb.positions)))
                if dmax < 0.01:
                    same_geometry.append({
                        "jobs": [a, b], "max_displacement_A": dmax,
                        "dft_dE_meV": float((e_ref[last_index[a]] - e_ref[last_index[b]]) * 1000),
                        "dft_states": [f"{finals[j].info.get('spin_constraint')} M={finals[j].info.get('multiplicity')}"
                                       for j in (a, b)],
                    })
    report["same_geometry_final_frames"] = same_geometry

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "zero_shot_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    lines = [
        "| model | F RMSE mobile (eV/Å) | F RMSE at minima | median cos | rel. E MAE (meV/atom) | "
        + " | ".join(f"{r} (eV)" for r in reactions) + " |",
        "|---" * (5 + len(reactions)) + "|",
        f"| DFT / zero-force baseline | {report['zero_force_baseline']['rmse']:.3f} | | | | "
        + " | ".join(f"{v:+.3f}" for v in report["dft_reactions"].values()) + " |",
    ]
    for name in rows:
        m = report["models"][name]
        lines.append(
            f"| {name} | {m['force_rmse_mobile']:.3f} | {m['force_rmse_at_minima']:.3f} | "
            f"{m['force_cosine_median']:.3f} | {m['relative_energy_mae_meV_atom']:.2f} | "
            + " | ".join(f"{v:+.3f}" for v in m["reactions"].values()) + " |"
        )
    (out / "zero_shot_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("frames")
    p.add_argument("--backend", choices=["mace", "uma"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--task", help="UMA task head, e.g. omat")
    p.add_argument("--name", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--uma-execution-mode", default="general")
    p.add_argument("-o", "--output", default="preds")
    p.set_defaults(func=command_predict)
    r = sub.add_parser("report")
    r.add_argument("frames")
    r.add_argument("predictions", nargs="+")
    r.add_argument("--reactions", help="JSON {name: {job: coefficient}} evaluated on final frames")
    r.add_argument("-o", "--output", default="report")
    r.set_defaults(func=command_report)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
