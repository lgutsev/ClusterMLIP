"""Relax the frozen candidate pool with one backend and write a per-candidate manifest.

    python relax_bench.py POOL.extxyz MODEL --backend native|alchemi --out DIR
                          [--batch-size 16] [--limit 128] [--tag NAME] [--repeat 1]

Backends (same checkpoint file, float32, GPU, all non-adsorbate atoms fixed,
convergence = largest force norm on a free atom <= fmax, at most --max-steps steps):

  native   ASE BFGS (maxstep 0.1) + MACECalculator, one candidate at a time. This is
           the ACAT pilot's workflow.
  alchemi  nvalchemi LBFGS (maxstep 0.1) on static batches of --batch-size. Each
           candidate is recorded at the first step it meets the criterion; the batch
           keeps stepping until every member has, or the step cap is reached.

Every input gets exactly one terminal status: converged, step_cap, numerical_failure.
After the run, the raw model forces at every final geometry are recomputed with the
native calculator, independently of the backend that produced the geometry.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read, write

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent / "acat_fe16_co_h"))

from cluster_mlip.adsorbate_search import CallCounter, analyze, relax  # noqa: E402
from common import load_catalyst, load_sites  # noqa: E402

ROLE_ADSORBATE = 2


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_native(pool, model_path, fmax, max_steps, timers):
    from mace.calculators import MACECalculator

    t = time.perf_counter()
    calc = MACECalculator(model_paths=model_path, device="cuda", default_dtype="float32")
    counter = CallCounter(calc)
    warm = pool[0].copy()
    warm.calc = calc
    warm.get_forces()
    sync()
    timers["setup_s"] = time.perf_counter() - t
    results = []
    t0 = time.perf_counter()
    for atoms in pool:
        a = atoms.copy()
        res = relax(a, counter, fmax=fmax, max_steps=max_steps)
        status = "numerical_failure" if res.error else ("converged" if res.converged else "step_cap")
        results.append({"status": status, "steps": res.steps, "calls": res.calls, "energy": res.energy, "atoms": a,
                        "batch": None})
    sync()
    timers["run_s"] = time.perf_counter() - t0
    timers["model_calls"] = counter.calls
    return results


def run_alchemi(pool, model_path, fmax, max_steps, batch_size, timers):
    from alchemi_backend import load_wrapper, to_batch
    from nvalchemi.dynamics import LBFGS
    from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
    from nvalchemi.dynamics.hooks import FreezeAtomsHook
    from nvalchemi.hooks import NeighborListHook

    t = time.perf_counter()
    model = load_wrapper(model_path, device="cuda", dtype=torch.float32)

    def make_optimizer():
        hooks = [NeighborListHook(model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE),
                 FreezeAtomsHook(mask_key="frozen", zero_velocities=False)]
        return LBFGS(model, maxstep=0.1, hooks=hooks, convergence_hook=ConvergenceHook.from_forces(threshold=fmax))

    warm = to_batch(pool[:1])
    make_optimizer().step(warm)
    sync()
    timers["setup_s"] = time.perf_counter() - t
    results: list = [None] * len(pool)
    timers.update(convert_s=0.0, step_s=0.0, extract_s=0.0, batch_steps=0, system_steps=0, active_system_steps=0)
    t0 = time.perf_counter()
    for start in range(0, len(pool), batch_size):
        chunk = list(range(start, min(start + batch_size, len(pool))))
        t = time.perf_counter()
        batch = to_batch([pool[i] for i in chunk], slots=chunk)
        frozen0 = batch.positions[batch["frozen"]].clone()
        sync()
        timers["convert_s"] += time.perf_counter() - t
        opt = make_optimizer()
        done: dict[int, dict] = {}
        failed: set[int] = set()
        step = 0
        while step < max_steps and len(done) + len(failed) < len(chunk):
            t = time.perf_counter()
            batch, converged = opt.step(batch)
            sync()
            timers["step_s"] += time.perf_counter() - t
            step += 1
            timers["batch_steps"] += 1
            timers["system_steps"] += len(chunk)
            timers["active_system_steps"] += len(chunk) - len(done) - len(failed)
            t = time.perf_counter()
            energy = batch.energy.reshape(-1)
            finite = torch.isfinite(energy)
            ptr = batch.batch_ptr
            for g in range(len(chunk)):
                if g in done or g in failed:
                    continue
                if not bool(finite[g]) or not bool(torch.isfinite(batch.positions[ptr[g]:ptr[g + 1]]).all()):
                    failed.add(g)
            if converged is not None:
                for g in converged.tolist():
                    if g not in done and g not in failed:
                        done[g] = {"steps": step, "energy": float(energy[g]),
                                   "positions": batch.positions[ptr[g]:ptr[g + 1]].detach().cpu().numpy().astype(float)}
            timers["extract_s"] += time.perf_counter() - t
        t = time.perf_counter()
        slots = batch["slot"].reshape(-1).tolist()
        assert slots == chunk, "batch order changed"
        moved = float((batch.positions[batch["frozen"]] - frozen0).abs().max())
        ptr = batch.batch_ptr
        for g, i in enumerate(chunk):
            a = pool[i].copy()
            if g in done:
                a.positions = done[g]["positions"]
                rec = {"status": "converged", "steps": done[g]["steps"], "energy": done[g]["energy"]}
            else:
                a.positions = batch.positions[ptr[g]:ptr[g + 1]].detach().cpu().numpy().astype(float)
                rec = {"status": "numerical_failure" if g in failed else "step_cap", "steps": step,
                       "energy": float(batch.energy.reshape(-1)[g])}
            rec.update(calls=rec["steps"], atoms=a, batch=start // batch_size, frozen_max_move=moved)
            results[i] = rec
        timers["extract_s"] += time.perf_counter() - t
    sync()
    timers["run_s"] = time.perf_counter() - t0
    timers["model_calls"] = timers["batch_steps"]
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", type=Path)
    ap.add_argument("model")
    ap.add_argument("--backend", required=True, choices=["native", "alchemi"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    pool = read(args.pool, index=":")[:args.limit]
    tag = args.tag or (args.backend if args.backend == "native" else f"alchemi_b{args.batch_size}")
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)
    catalyst = load_catalyst()
    sites = load_sites(catalyst).sites

    timings = []
    for rep in range(args.repeat):
        torch.cuda.reset_peak_memory_stats()
        timers: dict = {}
        t_all = time.perf_counter()
        if args.backend == "native":
            results = run_native(pool, args.model, args.fmax, args.max_steps, timers)
        else:
            results = run_alchemi(pool, args.model, args.fmax, args.max_steps, args.batch_size, timers)
        timers["total_s"] = time.perf_counter() - t_all
        timers["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        timers["converged"] = sum(r["status"] == "converged" for r in results)
        timers["candidates_per_hour_warm"] = 3600 * timers["converged"] / timers["run_s"]
        timings.append(timers)
        print(json.dumps({"tag": tag, "rep": rep, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in timers.items()}}),
              flush=True)

    # independent recompute of raw forces at the final geometries (native calculator), then chemistry audit
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=args.model, device="cuda", default_dtype="float32")
    rows, frames = [], []
    for i, (src, r) in enumerate(zip(pool, results)):
        a = r["atoms"]
        b = a.copy()
        b.set_constraint()
        b.calc = calc
        forces = b.get_forces()
        e_native = float(b.get_potential_energy())
        free = a.arrays["role"] == ROLE_ADSORBATE
        an = analyze(a, catalyst, sites)
        cat_move = float(np.abs(a.positions[~free] - src.positions[~free]).max())
        rows.append({"pool_index": i, "candidate_id": src.info["candidate_id"], "backend": tag, "batch": r["batch"],
                     "status": r["status"], "steps": r["steps"], "calls": r["calls"], "energy": r["energy"],
                     "energy_native_recompute": e_native,
                     "fmax_free_native_recompute": float(np.linalg.norm(forces[free], axis=1).max()),
                     "catalyst_max_move": cat_move, "topology": an.topology, "motif": an.motif, "flags": an.flags,
                     "failure": an.failure,
                     "composition_ok": sorted(a.get_chemical_symbols()) == sorted(src.get_chemical_symbols()),
                     "roles_ok": bool((a.arrays["role"] == src.arrays["role"]).all()
                                      and (a.arrays["mol_id"] == src.arrays["mol_id"]).all())})
        c = a.copy()
        c.set_constraint()
        c.calc = None
        c.info = {"pool_index": i, "candidate_id": src.info["candidate_id"], "backend": tag, "status": r["status"]}
        frames.append(c)
    with open(out / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    write(out / "final.extxyz", frames)
    (out / "timings.json").write_text(json.dumps({"tag": tag, "backend": args.backend, "batch_size": args.batch_size,
                                                  "n": len(pool), "fmax": args.fmax, "max_steps": args.max_steps,
                                                  "model": args.model, "runs": timings}, indent=2) + "\n", encoding="utf-8")
    status = {s: sum(r["status"] == s for r in rows) for s in ("converged", "step_cap", "numerical_failure")}
    print(json.dumps({"tag": tag, "status": status, "inputs": len(pool), "outputs": len(rows),
                      "max_fmax_recomputed_converged": max((r["fmax_free_native_recompute"] for r in rows
                                                            if r["status"] == "converged"), default=None),
                      "max_catalyst_move": max(r["catalyst_max_move"] for r in rows),
                      "max_abs_energy_vs_native_recompute": max(abs(r["energy"] - r["energy_native_recompute"]) for r in rows)}))


if __name__ == "__main__":
    main()
