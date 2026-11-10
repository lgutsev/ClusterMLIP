"""Inflight batching: converged structures leave the batch and new ones take their slots.

    python inflight_bench.py POOL.extxyz MODEL --slots 16 --out DIR [--n 128]

Uses the toolkit's own machinery (FusedStage + SizeAwareSampler + HostMemory sink).
A structure leaves the batch at the step it meets the force criterion, so the stored
geometry is the converged one. Results are matched back to inputs by the ``slot``
property, never by position in the sink.

Limit of this mode in toolkit 0.3.0: there is no per-structure step cap, only a cap on
the total number of batch steps; structures still in the batch when it is reached are
reported as step_cap, structures never admitted as not_started.
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

from alchemi_backend import load_wrapper, to_data  # noqa: E402


class PoolDataset:
    def __init__(self, structures, model, device, dtype):
        from nvalchemi.data import Batch
        from nvalchemi.neighbors import compute_neighbors

        self.items = [to_data(a, device=device, dtype=dtype, slot=i) for i, a in enumerate(structures)]
        self.meta = []
        for a, d in zip(structures, self.items):
            b = Batch.from_data_list([d], device=device)
            compute_neighbors(b, config=model.model_config.neighbor_config)
            self.meta.append((len(a), int(b.num_edges_per_graph.sum()) if b.num_edges_per_graph.numel() else 0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i], {}

    def get_metadata(self, i):
        return self.meta[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", type=Path)
    ap.add_argument("model")
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-batch-steps", type=int, default=4000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from nvalchemi.dynamics import LBFGS, FusedStage, SizeAwareSampler
    from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
    from nvalchemi.dynamics.hooks import FreezeAtomsHook
    from nvalchemi.dynamics.sinks import HostMemory
    from nvalchemi.hooks import NeighborListHook

    pool = read(args.pool, index=":")[: args.n]
    t0 = time.perf_counter()
    model = load_wrapper(args.model, device="cuda", dtype=torch.float32)
    dataset = PoolDataset(pool, model, "cuda", torch.float32)
    setup_s = time.perf_counter() - t0
    sampler = SizeAwareSampler(dataset, max_batch_size=args.slots, max_atoms=args.slots * max(len(a) for a in pool))
    sink = HostMemory(capacity=len(pool) + args.slots)
    hooks = [NeighborListHook(model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE),
             FreezeAtomsHook(mask_key="frozen", zero_velocities=False)]
    opt = LBFGS(model, maxstep=0.1, hooks=hooks, convergence_hook=ConvergenceHook.from_forces(threshold=args.fmax))
    fused = FusedStage(sub_stages=[(0, opt)], sampler=sampler, sinks=[sink], refill_frequency=1)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    left = fused.run(None, n_steps=args.max_batch_steps)
    torch.cuda.synchronize()
    run_s = time.perf_counter() - t1

    rows: dict[int, dict] = {}
    finals = {}

    def take(batch, status):
        ptr = batch.batch_ptr.cpu()
        slots = batch["slot"].reshape(-1).cpu().tolist()
        pos = batch.positions.detach().cpu().numpy().astype(float)
        forces = batch.forces.detach().cpu().numpy()
        frozen = batch["frozen"].cpu().numpy()
        energy = batch.energy.reshape(-1).detach().cpu().numpy()
        for g, s in enumerate(slots):
            sl = slice(int(ptr[g]), int(ptr[g + 1]))
            if s in rows:
                raise RuntimeError(f"input {s} came back twice")
            atoms = pool[s].copy()
            moved = float(np.abs(pos[sl][frozen[sl]] - atoms.positions[frozen[sl]]).max())
            atoms.positions = pos[sl]
            finals[s] = atoms
            rows[s] = {"pool_index": s, "status": status, "energy": float(energy[g]),
                       "fmax_free": float(np.linalg.norm(forces[sl][~frozen[sl]], axis=1).max()),
                       "catalyst_max_move": moved}

    if len(sink):
        take(sink.read(), "converged")
    if left is not None:
        take(left, "step_cap")
    for s in range(len(pool)):
        rows.setdefault(s, {"pool_index": s, "status": "not_started"})
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "manifest.jsonl", "w", encoding="utf-8") as fh:
        for s in range(len(pool)):
            fh.write(json.dumps(rows[s]) + "\n")
    write(args.out / "final.extxyz", [finals.get(s, pool[s]) for s in range(len(pool))])
    conv = [r for r in rows.values() if r["status"] == "converged"]
    summary = {"n": len(pool), "slots": args.slots, "setup_s": round(setup_s, 2), "run_s": round(run_s, 2),
               "batch_steps": int(fused.step_count), "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
               "status": {k: sum(r["status"] == k for r in rows.values()) for k in ("converged", "step_cap", "not_started")},
               "candidates_per_hour": 3600 * len(conv) / run_s,
               "max_fmax_reported_converged": max((r["fmax_free"] for r in conv), default=None),
               "max_catalyst_move": max((r["catalyst_max_move"] for r in conv), default=None)}
    (args.out / "timings.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
