"""Propose structures for a first active-learning round. A proposal only: nothing is submitted.

    python al_propose.py prepare-all LABELS.extxyz WORK --multiplicity 49
    python finetune_test.py run WORK --fold 0 --model NAME=FOUNDATION ...        (trains the final model)
    python al_propose.py propose WORK --name NAME --labels LABELS.extxyz --pool POOL.extxyz --out DIR [--n 8]

Two-stage model, as found usable in section 17 of docs/acat_cluster_feasibility.md: the fine-tuned
model gives forces (relaxation); its energy plus a Gaussian-process correction on its own descriptors
gives the ranking energy and an uncertainty. Here both are trained on ALL labelled frames at one
multiplicity (two structure groups are kept out of the fine-tune for early stopping only).

propose:
1. Single points of every pool structure (the search's unique candidates, at the search model's
   geometries): corrected energy and uncertainty. The process's standard deviation is doubled, because
   the calibration check found it overconfident by about that factor.
2. The --shortlist best by acquisition (corrected energy minus the widened standard deviation) are
   relaxed with the fine-tuned model, Fe frozen, and scored again.
3. Structures equal to an already labelled one, or to each other, are removed.
4. Picks: most by lowest acquisition, the rest by largest uncertainty among structures predicted within
   --explore-window of the lowest. Every pick is checked with the search's own structure analysis
   (composition, fragments, flags).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read, write

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from delta_gp_test import SAME_MINIMUM, Featurizer  # noqa: E402  (imports mace before e3nn)
from delta_gp_diagnostics import gp  # noqa: E402
from cluster_mlip.adsorbate_search import CallCounter, analyze, relax, same_structure  # noqa: E402
from common import load_catalyst, load_sites  # noqa: E402

WIDEN = 2.0


def labelled(labels: Path, multiplicity: int) -> list[Atoms]:
    out = []
    for a in read(labels, index=":"):
        if int(a.info["multiplicity"]) != multiplicity:
            continue
        meta = a.info["metadata"]
        meta = json.loads(meta) if isinstance(meta, str) else meta
        clean = Atoms(a.get_chemical_symbols(), positions=a.positions)
        clean.info = {"REF_energy": float(a.info["REF_energy"]), "config_type": str(a.info["config_type"]),
                      "structure": meta["structure"], "group": SAME_MINIMUM.get(meta["structure"], meta["structure"]),
                      "record_id": str(a.info["record_id"]), "multiplicity": multiplicity}
        clean.arrays["REF_forces"] = np.array(a.arrays["REF_forces"])
        out.append(clean)
    return out


def prepare_all(args) -> None:
    frames = labelled(args.labels, args.multiplicity)
    rank: dict[str, float] = {}
    for a in frames:
        if a.info["config_type"] == "panel_fixed":
            rank[a.info["group"]] = min(rank.get(a.info["group"], np.inf), a.info["REF_energy"])
    groups = sorted(rank, key=rank.get)
    valid_groups = [groups[len(groups) // 3], groups[2 * len(groups) // 3]]
    d = args.work / "fold_0"
    d.mkdir(parents=True, exist_ok=True)
    train = [a for a in frames if a.info["group"] not in valid_groups]
    valid = [a for a in frames if a.info["group"] in valid_groups]
    write(d / "train.extxyz", train, format="extxyz")
    write(d / "valid.extxyz", valid, format="extxyz")
    write(d / "test.extxyz", valid, format="extxyz")
    plan = {"multiplicity": args.multiplicity, "labels": args.labels.name, "frames": len(frames), "groups": len(groups),
            "purpose": "final model on all labels; the validation groups are used for early stopping only",
            "folds": [{"fold": 0, "held_out_groups": [], "validation_groups": valid_groups,
                       "frames": {"train": len(train), "valid": len(valid), "test": len(valid)}, "held_out_panel_structures": []}]}
    (args.work / "folds.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=1))


def propose(args) -> None:
    torch.set_default_dtype(torch.float64)
    plan = json.loads((args.work / "folds.json").read_text(encoding="utf-8"))
    tuned = args.work / "fold_0" / args.name / f"{args.name}_fold0.model"
    feat = Featurizer(str(tuned), device="cpu")
    counts = torch.tensor([16.0, 1.0, 1.0, 2.0])
    lab = labelled(args.labels, plan["multiplicity"])
    xs, ys = [], []
    for a in lab:
        e, means, _ = feat(a)
        xs.append(means.detach())
        ys.append(a.info["REF_energy"] - float(e))
    x, y = torch.stack(xs), torch.tensor(ys)
    mu, sd = x.mean(0), x.std(0) + 1e-12
    z = (x - mu) / sd
    group = np.array([a.info["group"] for a in lab])
    panel = np.array([a.info["config_type"] == "panel_fixed" for a in lab])
    dist = torch.sqrt(((z[:, None] - z[None]) ** 2).sum(-1)).reshape(-1)
    length0 = float(dist[dist > 0].median())
    best = None
    for mult in (0.5, 1.0, 2.0, 4.0, 8.0):
        for noise in (1e-6, 1e-4, 1e-2):
            errs = []
            for g in sorted(set(group[panel])):
                tr = torch.as_tensor(np.where(group != g)[0])
                te = torch.as_tensor(np.where((group == g) & panel)[0])
                mean, _ = gp(z[tr], y[tr], z[te], counts, length0 * mult, noise)
                errs += (mean - y[te]).abs().tolist()
            if best is None or np.mean(errs) < best[0]:
                best = (float(np.mean(errs)), length0 * mult, noise)
    loo_mae, length, noise = best
    shift = float(y.mean())

    def score(atoms_list):
        e_ft, zq = [], []
        for a in atoms_list:
            e, means, _ = feat(a)
            e_ft.append(float(e))
            zq.append((means.detach() - mu) / sd)
        mean, s = gp(z, y, torch.stack(zq), counts, length, noise)
        e_corr = np.array(e_ft) + mean.numpy() - shift  # on the fine-tuned model's scale
        return np.array(e_ft), e_corr, s.numpy() * WIDEN

    pool = read(args.pool, index=":")
    _, e1, s1 = score(pool)
    acq1 = e1 - s1
    short = list(np.argsort(acq1)[: args.shortlist])
    # optional: also relax the most uncertain pool structures, so the list is not only low-energy look-alikes
    short += [i for i in np.argsort(-s1) if i not in set(short)][: args.explore_shortlist]

    from mace.calculators import MACECalculator

    calc = CallCounter(MACECalculator(model_paths=str(tuned), device="cpu", default_dtype="float64"))
    catalyst = load_catalyst()
    sites = load_sites(catalyst).sites
    relaxed, info = [], []
    for i in short:
        a = pool[i].copy()
        res = relax(a, calc, fmax=0.05, max_steps=args.max_steps)
        an = analyze(a, catalyst, sites)
        relaxed.append(a)
        info.append({"pool_id": pool[i].info["pool_id"], "pool_motif": pool[i].info.get("motif"), "converged": bool(res.converged),
                     "steps": int(res.steps), "motif": an.motif, "topology": an.topology, "flags": list(an.flags),
                     "failure": bool(an.failure), "start_acquisition_ev": float(acq1[i] - acq1.min())})
    _, e2, s2 = score(relaxed)
    e_low = float(e2.min())
    known = [a for a in lab if a.info["config_type"] in ("panel_fixed", "stationary_fe_frozen", "stationary_all_free")]
    for a in known + relaxed:
        if "role" not in a.arrays:
            a.set_array("role", np.array([1] * 16 + [2] * 4))
            a.set_array("mol_id", np.array([-1] * 16 + [0, 0, 1, 2]))
    rows = []
    for k, (a, meta) in enumerate(zip(relaxed, info)):
        match = next((b.info["structure"] for b in known if same_structure(a, b, tol=0.15)), None)
        rows.append({**meta, "index": k, "e_corrected_rel_ev": float(e2[k] - e_low), "sd_widened_ev": float(s2[k]),
                     "acquisition_rel_ev": float(e2[k] - s2[k] - (e2 - s2).min()), "already_labelled_as": match})
    candidates = [r for r in rows if r["already_labelled_as"] is None and not r["failure"] and r["converged"]]
    picks: list[dict] = []

    def take(r: dict, why: str) -> bool:
        if any(same_structure(relaxed[r["index"]], relaxed[p["index"]], tol=0.15) for p in picks):
            return False
        if args.max_per_motif and sum(p["motif"] == r["motif"] for p in picks) >= args.max_per_motif:
            return False
        picks.append({**r, "picked_for": why})
        return True

    n_exploit = args.n - (args.n // 4 if args.n_explore is None else args.n_explore)
    for r in sorted(candidates, key=lambda r: r["acquisition_rel_ev"]):
        if len(picks) >= n_exploit:
            break
        take(r, "lowest corrected energy minus uncertainty")
    for r in sorted(candidates, key=lambda r: -r["sd_widened_ev"]):
        if len(picks) >= args.n:
            break
        if r["e_corrected_rel_ev"] <= args.explore_window:
            take(r, f"largest uncertainty within {args.explore_window} eV of the lowest")

    args.out.mkdir(parents=True, exist_ok=True)
    frames = []
    for j, p in enumerate(picks):
        a = relaxed[p["index"]].copy()
        a.info = {"al_id": f"AL1_{j:02d}", "pool_id": p["pool_id"], "motif": p["motif"], "picked_for": p["picked_for"],
                  "e_corrected_rel_ev": p["e_corrected_rel_ev"], "sd_widened_ev": p["sd_widened_ev"]}
        frames.append(a)
    write(args.out / "al_round1_proposal.extxyz", frames, format="extxyz")
    with open(args.out / "al_round1_proposal.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["al_id", "pool_id", "picked_for", "motif", "flags", "e_corrected_rel_ev", "sd_widened_ev", "acquisition_rel_ev",
                    "relax_steps"])
        for j, p in enumerate(picks):
            w.writerow([f"AL1_{j:02d}", p["pool_id"], p["picked_for"], p["motif"], ";".join(p["flags"]),
                        f"{p['e_corrected_rel_ev']:.3f}", f"{p['sd_widened_ev']:.3f}", f"{p['acquisition_rel_ev']:.3f}", p["steps"]])
    summary = {"model": tuned.name, "multiplicity": plan["multiplicity"], "labelled_frames": len(lab), "pool": len(pool),
               "gp": {"length": length, "noise": noise, "leave_one_group_out_mae_ev": loo_mae, "uncertainty_widening": WIDEN},
               "pool_single_point": {"sd_widened_median_ev": float(np.median(s1)), "sd_widened_max_ev": float(s1.max())},
               "shortlist": len(short), "shortlist_converged": int(sum(r["converged"] for r in rows)),
               "shortlist_already_labelled": int(sum(r["already_labelled_as"] is not None for r in rows)),
               "shortlist_flagged_failure": int(sum(r["failure"] for r in rows)),
               "candidates_after_filters": len(candidates), "picks": picks, "all_relaxed": rows,
               "calculator_calls_for_relaxation": int(calc.calls)}
    (args.out / "al_round1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("picks", "all_relaxed")}, indent=1))
    for j, p in enumerate(picks):
        print(f"  AL1_{j:02d} {p['pool_id']}  E {p['e_corrected_rel_ev']:6.3f}  sd {p['sd_widened_ev']:.3f}  {p['motif']}  [{p['picked_for']}] {p['flags']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare-all")
    p.add_argument("labels", type=Path)
    p.add_argument("work", type=Path)
    p.add_argument("--multiplicity", type=int, default=49)
    p.set_defaults(fn=prepare_all)
    p = sub.add_parser("propose")
    p.add_argument("work", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--shortlist", type=int, default=120)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--explore-window", type=float, default=0.5)
    p.add_argument("--explore-shortlist", type=int, default=0, help="also relax this many of the most uncertain pool structures")
    p.add_argument("--n-explore", type=int, default=None, help="picks reserved for the largest uncertainty (default n // 4)")
    p.add_argument("--max-per-motif", type=int, default=0, help="at most this many picks with the same site motif (0 = no limit)")
    p.set_defaults(fn=propose)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
