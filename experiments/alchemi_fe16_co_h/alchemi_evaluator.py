"""Batch evaluator for `AdsorbateSearch.run_batched` backed by nvalchemi-toolkit.

``AlchemiEvaluator`` implements the ``BatchEvaluator`` contract of
`cluster_mlip.adsorbate_search`: a list of start structures in, one
``(relaxed structure, RelaxResult)`` per structure out, in the same order.

Rules it keeps:

* every non-adsorbate atom is frozen (same as the native Stage A constraint) and the
  convergence test is the largest force norm on a free atom;
* a structure is recorded at the first step it meets the criterion, even though the
  static batch keeps stepping for the others;
* a structure whose energy or positions become non-finite is reported as an error for
  that structure only; a step cap is reported as not converged;
* no force clamping, no stabilisation hooks.
"""
from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
import torch
from ase import Atoms

from alchemi_backend import load_wrapper, to_batch

ROLE_ADSORBATE = 2


def _fmax_free(forces: torch.Tensor, free: torch.Tensor) -> float:
    """Largest force norm on a free atom; 0 when nothing is free (a bare frozen catalyst)."""
    return float(torch.linalg.norm(forces[free], dim=-1).max()) if bool(free.any()) else 0.0


def relax_batch(model: Any, structures: Sequence[Atoms], *, fmax: float, max_steps: int,
                stats: dict | None = None) -> list[dict[str, Any]]:
    from nvalchemi.dynamics import LBFGS
    from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
    from nvalchemi.dynamics.hooks import FreezeAtomsHook
    from nvalchemi.hooks import NeighborListHook

    n = len(structures)
    batch = to_batch(structures, slots=list(range(n)))
    frozen0 = batch.positions[batch["frozen"]].clone()
    hooks = [NeighborListHook(model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE),
             FreezeAtomsHook(mask_key="frozen", zero_velocities=False)]
    opt = LBFGS(model, maxstep=0.1, hooks=hooks, convergence_hook=ConvergenceHook.from_forces(threshold=fmax))
    done: dict[int, dict] = {}
    failed: dict[int, int] = {}
    step = 0
    while step < max_steps and len(done) + len(failed) < n:
        batch, converged = opt.step(batch)
        step += 1
        energy = batch.energy.reshape(-1)
        ptr = batch.batch_ptr
        finite_e = torch.isfinite(energy)
        finite_x = torch.isfinite(batch.positions).all(dim=-1)
        finite_f = torch.isfinite(batch.forces).all(dim=-1)
        for g in range(n):
            if g in done or g in failed:
                continue
            sl = slice(int(ptr[g]), int(ptr[g + 1]))
            if not (bool(finite_e[g]) and bool(finite_x[sl].all()) and bool(finite_f[sl].all())):
                failed[g] = step
        if converged is not None:
            for g in converged.tolist():
                if g not in done and g not in failed:
                    sl = slice(int(ptr[g]), int(ptr[g + 1]))
                    free = ~batch["frozen"][sl]
                    done[g] = {"steps": step, "energy": float(energy[g]),
                               "fmax": _fmax_free(batch.forces[sl], free),
                               "positions": batch.positions[sl].detach().cpu().numpy().astype(float)}
    if batch["slot"].reshape(-1).tolist() != list(range(n)):
        raise RuntimeError("the toolkit reordered the batch; results cannot be matched to inputs")
    frozen_move = float((batch.positions[batch["frozen"]] - frozen0).abs().nan_to_num(nan=float("inf")).max())
    ptr = batch.batch_ptr
    out = []
    for g, atoms in enumerate(structures):
        final = atoms.copy()
        sl = slice(int(ptr[g]), int(ptr[g + 1]))
        if g in done:
            final.positions = done[g]["positions"]
            rec = {"status": "converged", **{k: done[g][k] for k in ("steps", "energy", "fmax")}}
        elif g in failed:
            rec = {"status": "numerical_failure", "steps": failed[g], "energy": float("nan"), "fmax": float("nan")}
        else:
            final.positions = batch.positions[sl].detach().cpu().numpy().astype(float)
            free = ~batch["frozen"][sl]
            rec = {"status": "step_cap", "steps": step, "energy": float(batch.energy.reshape(-1)[g]),
                   "fmax": _fmax_free(batch.forces[sl], free)}
        rec.update(atoms=final, frozen_max_move=frozen_move)
        out.append(rec)
    if stats is not None:
        stats["batch_steps"] = stats.get("batch_steps", 0) + step
        stats["system_steps"] = stats.get("system_steps", 0) + step * n
        stats["useful_system_steps"] = stats.get("useful_system_steps", 0) + sum(r["steps"] for r in out)
    return out


class AlchemiEvaluator:
    """``BatchEvaluator`` for `AdsorbateSearch.run_batched`."""

    def __init__(self, model_path: str, *, fmax: float, max_steps: int, max_batch: int = 64,
                 device: str = "cuda", dtype: torch.dtype = torch.float32):
        self.model = load_wrapper(model_path, device=device, dtype=dtype)
        self.fmax, self.max_steps, self.max_batch = fmax, max_steps, max_batch
        self.stats: dict = {}

    def __call__(self, structures: list[Atoms]):
        from cluster_mlip.adsorbate_search import RelaxResult

        results = []
        for start in range(0, len(structures), self.max_batch):
            chunk = structures[start:start + self.max_batch]
            t0 = time.perf_counter()
            recs = relax_batch(self.model, chunk, fmax=self.fmax, max_steps=self.max_steps, stats=self.stats)
            wall = (time.perf_counter() - t0) / len(chunk)
            for rec in recs:
                error = "non-finite energy, forces or positions" if rec["status"] == "numerical_failure" else None
                results.append((rec["atoms"], RelaxResult(rec["status"] == "converged", rec["steps"], rec["steps"],
                                                          rec["energy"], rec["fmax"], wall, error)))
        return results


def corrupted(atoms: Atoms) -> Atoms:
    """A deliberately broken input for the failure-isolation check: two adsorbate atoms on one point."""
    bad = atoms.copy()
    ads = np.where(bad.arrays["role"] == ROLE_ADSORBATE)[0]
    bad.positions[ads[1]] = bad.positions[ads[0]]
    return bad
