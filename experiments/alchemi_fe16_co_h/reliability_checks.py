"""Reliability checks for batched relaxation (not timing).

    python reliability_checks.py POOL.extxyz MODEL --out DIR

1. Batch independence of a relaxation: the same 8 candidates relaxed alone (batch 1),
   together (batch 8), and together in reversed order.
2. Failure isolation: the same 8 plus one deliberately corrupted input (two atoms on
   one point). The bad input must end as a failure; its companions must be unchanged.
3. Step cap: the same 8 with a cap of 5 steps must all report step_cap, not converged.
4. Frozen atoms: catalyst coordinates checked after every step of one batch, not only at the end.
5. Mixed sizes: Fe16, Fe16 + CO, Fe16 + 2H and Fe16 + CO + 2H in one batch against each alone.
6. Search integration: `AdsorbateSearch.run_batched` with the toolkit evaluator; ledger
   accounting and composition/role checks on every relaxed structure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent / "acat_fe16_co_h"))

from alchemi_backend import to_batch  # noqa: E402
from alchemi_evaluator import AlchemiEvaluator, corrupted, relax_batch  # noqa: E402
from cluster_mlip.adsorbate_search import AdsorbateSearch, SearchConfig, same_structure  # noqa: E402
from common import load_catalyst, load_sites  # noqa: E402


def compare(a, b):
    """Per-structure agreement between two result lists (same inputs, same order)."""
    out = {"status_equal": [x["status"] == y["status"] for x, y in zip(a, b)],
           "same_structure": [x["status"] != "converged" or same_structure(x["atoms"], y["atoms"], tol=0.15) for x, y in zip(a, b)]}
    de = [abs(x["energy"] - y["energy"]) for x, y in zip(a, b) if x["status"] == y["status"] == "converged"]
    dx = [float(np.abs(x["atoms"].positions - y["atoms"].positions).max()) for x, y in zip(a, b)
          if x["status"] == y["status"] == "converged"]
    return {"all_status_equal": all(out["status_equal"]), "same_structure": int(sum(out["same_structure"])), "n": len(a),
            "max_abs_dE_eV": max(de) if de else None, "max_position_diff_A": max(dx) if dx else None,
            "steps_a": [x["steps"] for x in a], "steps_b": [y["steps"] for y in b]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", type=Path)
    ap.add_argument("model")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pool = read(args.pool, index=":")
    eight = pool[16:24]
    ev = AlchemiEvaluator(args.model, fmax=0.05, max_steps=200)
    model = ev.model
    report: dict = {}

    alone = [relax_batch(model, [a], fmax=0.05, max_steps=200)[0] for a in eight]
    together = relax_batch(model, eight, fmax=0.05, max_steps=200)
    rev = relax_batch(model, eight[::-1], fmax=0.05, max_steps=200)[::-1]
    report["batch_independence"] = {"batch1_vs_batch8": compare(alone, together), "batch8_vs_reversed": compare(together, rev),
                                    "batch8_repeat": compare(together, relax_batch(model, eight, fmax=0.05, max_steps=200))}

    bad = corrupted(pool[30])
    with_bad = relax_batch(model, eight[:4] + [bad] + eight[4:], fmax=0.05, max_steps=200)
    companions = with_bad[:4] + with_bad[5:]
    report["failure_isolation"] = {"bad_input_status": with_bad[4]["status"], "bad_input_steps": with_bad[4]["steps"],
                                   "companions_vs_clean_batch": compare(together, companions),
                                   "outputs": len(with_bad), "inputs": 9}

    capped = relax_batch(model, eight, fmax=0.05, max_steps=5)
    report["step_cap"] = {"statuses": [r["status"] for r in capped], "none_reported_converged_falsely":
                          all(r["status"] == "step_cap" or r["steps"] <= 5 for r in capped)}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reliability.json").write_text(json.dumps(report, indent=2) + chr(10), encoding="utf-8")
    # frozen atoms checked at every step
    from nvalchemi.dynamics import LBFGS
    from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
    from nvalchemi.dynamics.hooks import FreezeAtomsHook
    from nvalchemi.hooks import NeighborListHook
    batch = to_batch(eight)
    x0 = batch.positions[batch["frozen"]].clone()
    opt = LBFGS(model, maxstep=0.1, hooks=[NeighborListHook(model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE),
                                          FreezeAtomsHook(mask_key="frozen", zero_velocities=False)],
                convergence_hook=ConvergenceHook.from_forces(threshold=0.05))
    worst_move, worst_force = 0.0, 0.0
    for _ in range(40):
        batch, _c = opt.step(batch)
        worst_move = max(worst_move, float((batch.positions[batch["frozen"]] - x0).abs().max()))
        worst_force = max(worst_force, float(batch.forces[batch["frozen"]].abs().max()))
    report["frozen_atoms_every_step"] = {"steps_checked": 40, "max_move_A": worst_move,
                                         "max_stored_force_on_frozen": worst_force}

    # mixed sizes: strip adsorbates from one candidate
    full = pool[40]
    role, mol = full.arrays["role"], full.arrays["mol_id"]
    sym = np.array(full.get_chemical_symbols())
    variants = {"Fe16": full[role != 2], "Fe16+CO": full[(role != 2) | np.isin(sym, ["C", "O"])],
                "Fe16+2H": full[(role != 2) | (sym == "H")], "Fe16+CO+2H": full}
    solo = [relax_batch(model, [a], fmax=0.05, max_steps=200)[0] for a in variants.values()]
    mixed = relax_batch(model, list(variants.values()), fmax=0.05, max_steps=200)
    report["mixed_sizes"] = {"sizes": [len(a) for a in variants.values()], "names": list(variants),
                             "statuses": [r["status"] for r in mixed], "solo_vs_mixed": compare(solo, mixed)}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reliability.json").write_text(json.dumps(report, indent=2) + chr(10), encoding="utf-8")
    # search integration
    catalyst = load_catalyst()
    sites = load_sites(catalyst).sites
    cfg = SearchConfig(arm="random", seed=0, budget=24)
    search = AdsorbateSearch(catalyst, sites, None, cfg)
    search.run_batched(ev, batch_size=12)
    recs = search.records
    relaxed = [r for r in recs if r["status"] == "relaxed"]
    ok = True
    for r in relaxed:
        start, final = search.structures[r["candidate_id"]]
        ok &= final.get_chemical_symbols() == start.get_chemical_symbols()
        ok &= bool((final.arrays["role"] == start.arrays["role"]).all() and (final.arrays["mol_id"] == start.arrays["mol_id"]).all())
        ok &= bool(np.allclose(final.positions[:16], catalyst.positions))
    report["search_integration"] = {
        "proposals": len(recs), "relaxed": len(relaxed), "budget": cfg.budget,
        "ledger_in_proposal_order": [r["proposal_index"] for r in recs] == list(range(len(recs))),
        "statuses": {s: sum(r["status"] == s for r in recs) for s in ("relaxed", "invalid", "duplicate_start")},
        "converged": sum(r["converged"] for r in relaxed), "composition_roles_catalyst_kept": bool(ok),
        "same_start_structures_as_native_pilot": [r["start_sha"] for r in relaxed][:8],
        "evaluator_stats": ev.stats, "unique": sum(r["unique"] for r in relaxed),
        "motifs": sorted({r["motif"] for r in relaxed})}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reliability.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
