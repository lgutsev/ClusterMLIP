"""Held-out test of the spin input of a from-scratch MACE on Fe16 and Fe16 + CO + 2H labels.

    python spin_heldout_test.py prepare COH_LABELS.extxyz LADDER_LABELS.extxyz WORK [--folds 3]
    python spin_heldout_test.py run     WORK --fold K --variant spin|blind [--epochs 300] [--device cuda]
    python spin_heldout_test.py score   WORK --out report.json [--device cuda]

Question: does the total-spin input generalise to geometries the model never saw?

Data. The CO/H panel has 20 fixed geometries each labelled at M = 49, 47 and 45 (same atoms, same
positions, different energy and forces), plus optimisation frames at M = 49 and 47. The bare Fe16 ladder
has M = 49 to 39. Folds hold out whole CO/H structure groups (every frame derived from a structure, at
every M); the bare Fe16 ladder is always in training. Structures 000, 008 and the relaxed minimum A are
one group.

Variants. "spin": ScaleShiftMACE with the categorical total-spin (and total-charge) embedding and the
embedding readout, as `cluster-mlip train` writes it. "blind": the same model without the embedding;
for one geometry it can only predict one energy, so its spin gaps are zero by construction.

Scores, on held-out panel geometries only:
  spin gaps   E(M-2) - E(M) at fixed geometry against UBPW91: MAE, sign agreement, rank correlation,
              compared with a constant-gap baseline (the mean gap of the fold's training structures);
  energies    relative energies within one M across held-out structures (within a fold);
  forces      RMSE on Fe and on the adsorbate atoms per M against a zero-force baseline.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

SAME_MINIMUM = {"fe16coh2_000": "A", "fe16coh2_008": "A", "fe16coh2_minA_m49": "A", "fe16coh2_minA_m47": "A"}
MULTS = (49, 47, 45)
EMBEDDING = {  # as src/cluster_mlip/training.py: the explicit "key" is what makes the model read info["spin"]
    "total_spin": {"type": "categorical", "per": "graph", "key": "spin", "in_dim": 1, "emb_dim": 128,
                   "num_classes": 101, "offset": 0},
    "total_charge": {"type": "categorical", "per": "graph", "key": "charge", "in_dim": 1, "emb_dim": 128,
                     "num_classes": 201, "offset": 100},
}


def clean(a: Atoms, structure: str, group: str) -> Atoms:
    out = Atoms(a.get_chemical_symbols(), positions=a.positions)
    m = int(a.info["multiplicity"])
    out.info = {"REF_energy": float(a.info["REF_energy"]), "spin": m, "charge": int(a.info["charge"]), "multiplicity": m,
                "config_type": str(a.info["config_type"]), "structure": structure, "group": group,
                "record_id": str(a.info["record_id"])}
    out.arrays["REF_forces"] = np.array(a.arrays["REF_forces"])
    return out


def load(coh: Path, ladder: Path) -> tuple[list[Atoms], list[Atoms]]:
    rows = []
    for a in read(coh, index=":"):
        meta = a.info["metadata"]
        meta = json.loads(meta) if isinstance(meta, str) else meta
        s = meta["structure"]
        rows.append(clean(a, s, SAME_MINIMUM.get(s, s)))
    bare = [clean(a, "fe16_bare", "fe16_bare") for a in read(ladder, index=":")]
    return rows, bare


def prepare(args) -> None:
    rows, bare = load(args.coh, args.ladder)
    rank: dict[str, float] = {}
    for a in rows:
        if a.info["config_type"] == "panel_fixed" and a.info["multiplicity"] == 49:
            rank[a.info["group"]] = min(rank.get(a.info["group"], np.inf), a.info["REF_energy"])
    groups = sorted({a.info["group"] for a in rows}, key=lambda g: rank.get(g, np.inf))
    folds = [groups[k::args.folds] for k in range(args.folds)]
    plan = {"coh_frames": len(rows), "bare_frames": len(bare), "groups": len(groups), "folds": []}
    for k, held in enumerate(folds):
        rest = [g for g in groups if g not in held]
        valid_groups = rest[1::max(2, len(rest) // 2)][:2]
        split = {"train": bare + [a for a in rows if a.info["group"] not in held and a.info["group"] not in valid_groups],
                 "valid": [a for a in rows if a.info["group"] in valid_groups],
                 "test": [a for a in rows if a.info["group"] in held]}
        d = args.work / f"fold_{k}"
        d.mkdir(parents=True, exist_ok=True)
        for name, frames in split.items():
            write(d / f"{name}.extxyz", frames, format="extxyz")
        plan["folds"].append({"fold": k, "held_out_groups": held, "validation_groups": valid_groups,
                              "frames": {n: len(f) for n, f in split.items()},
                              "train_by_multiplicity": {str(m): sum(a.info["multiplicity"] == m for a in split["train"])
                                                        for m in sorted({a.info["multiplicity"] for a in split["train"]})}})
    (args.work / "folds.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=1))


def run(args) -> None:
    d = args.work / f"fold_{args.fold}"
    out = d / args.variant
    out.mkdir(parents=True, exist_ok=True)
    argv = ["mace_run_train", f"--name={args.variant}_fold{args.fold}", "--model=ScaleShiftMACE",
            f"--train_file={d / 'train.extxyz'}", f"--valid_file={d / 'valid.extxyz'}", f"--test_file={d / 'test.extxyz'}",
            "--energy_key=REF_energy", "--forces_key=REF_forces", "--total_charge_key=charge", "--total_spin_key=spin"]
    if args.variant == "spin":
        argv += ["--embedding_specs=" + json.dumps(EMBEDDING, separators=(",", ":")), "--use_embedding_readout=True"]
    argv += ["--E0s=average",
             "--interaction_first=RealAgnosticResidualNonLinearInteractionBlock",
             "--interaction=RealAgnosticResidualNonLinearInteractionBlock",
             "--num_interactions=2", "--correlation=3", "--max_ell=3", "--r_max=6.0", "--num_radial_basis=8",
             f"--hidden_irreps={args.hidden}", "--MLP_irreps=16x0e", "--scaling=rms_forces_scaling",
             # lr and force weight from the Fe16 baseline diagnosis (0.005 / 100 stalls on a zero-force plateau)
             "--loss=weighted", "--energy_weight=1.0", "--forces_weight=10.0", "--stress_weight=0", "--lr=0.001",
             f"--batch_size={args.batch_size}", f"--valid_batch_size={args.batch_size}", f"--max_num_epochs={args.epochs}", "--patience=60",
             "--ema", "--ema_decay=0.999", "--default_dtype=float64", f"--device={args.device}", f"--seed={args.seed}",
             "--eval_interval=5"]
    resumed = any((out / "checkpoints").glob("*.pt"))  # an interrupted run continues from its last checkpoint
    if resumed:
        argv.append("--restart_latest")
    (out / "argv.json").write_text(json.dumps(argv, indent=1) + "\n", encoding="utf-8")
    with open(out / "train_stdout.log", "a" if resumed else "w", encoding="utf-8") as log:
        rc = subprocess.run(argv, cwd=out, stdout=log, stderr=subprocess.STDOUT).returncode
    text = (out / "train_stdout.log").read_text(encoding="utf-8", errors="replace")
    if args.variant == "spin" and ("total_spin: 0," in text or "total_spin: 0]" in text):
        raise SystemExit("MACE loaded zero total_spin labels: the model would be spin-blind")
    print(f"{args.variant} fold {args.fold}: rc={rc}")
    sys.exit(rc)


def rmse(x) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def spearman(x, y) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def panel(frames: list[Atoms]) -> dict[str, dict[int, Atoms]]:
    out: dict[str, dict[int, Atoms]] = {}
    for a in frames:
        if a.info["config_type"] == "panel_fixed":
            out.setdefault(a.info["structure"], {})[int(a.info["multiplicity"])] = a
    return out


def score(args) -> None:
    import torch  # noqa: F401
    from mace.calculators import MACECalculator

    plan = json.loads((args.work / "folds.json").read_text(encoding="utf-8"))
    report = {"folds": plan["folds"], "variants": {}}
    gap_names = [(49, 47), (47, 45)]
    # reference gaps and the constant-gap baseline (mean gap over the fold's training structures)
    ref_rows = []
    for f in plan["folds"]:
        d = args.work / f"fold_{f['fold']}"
        train_panel, test_panel = panel(read(d / "train.extxyz", index=":")), panel(read(d / "test.extxyz", index=":"))
        for hi, lo in gap_names:
            base = [p[lo].info["REF_energy"] - p[hi].info["REF_energy"] for p in train_panel.values() if hi in p and lo in p]
            for s, p in sorted(test_panel.items()):
                if hi in p and lo in p:
                    ref_rows.append({"fold": f["fold"], "structure": s, "gap": f"{lo}-{hi}",
                                     "ref": p[lo].info["REF_energy"] - p[hi].info["REF_energy"], "constant": float(np.mean(base))})
    for variant in ("spin", "blind"):
        gaps, energies, forces, sens = [], [], {m: {"Fe": [], "ads": [], "Fe0": [], "ads0": []} for m in MULTS}, []
        missing = False
        for f in plan["folds"]:
            d = args.work / f"fold_{f['fold']}"
            models = sorted((d / variant).glob(f"{variant}_fold{f['fold']}*.model"))
            models = [m for m in models if "compiled" not in m.name and "stagetwo" not in m.name]
            if not models:
                missing = True
                break
            calc = MACECalculator(model_paths=str(models[0]), device=args.device, default_dtype="float64")
            pred: dict[tuple[str, int], tuple[float, np.ndarray]] = {}
            for s, p in sorted(panel(read(d / "test.extxyz", index=":")).items()):
                for m, a in p.items():
                    b = a.copy()
                    b.info = {"spin": m, "charge": 0}
                    b.calc = calc
                    pred[(s, m)] = (float(b.get_potential_energy()), np.array(b.get_forces()))
                    df = pred[(s, m)][1] - a.arrays["REF_forces"]
                    forces[m]["Fe"].append(df[:16]); forces[m]["ads"].append(df[16:])
                    forces[m]["Fe0"].append(a.arrays["REF_forces"][:16]); forces[m]["ads0"].append(a.arrays["REF_forces"][16:])
                for hi, lo in gap_names:
                    if hi in p and lo in p:
                        gaps.append({"fold": f["fold"], "structure": s, "gap": f"{lo}-{hi}", "pred": pred[(s, lo)][0] - pred[(s, hi)][0]})
                if 49 in p and 47 in p:
                    sens.append(float(np.abs(pred[(s, 47)][1] - pred[(s, 49)][1]).max()))
            for m in MULTS:
                have = sorted(s for (s, mm) in pred if mm == m)
                if len(have) < 2:
                    continue
                e_ref = np.array([panel(read(d / "test.extxyz", index=":"))[s][m].info["REF_energy"] for s in have])
                e_pred = np.array([pred[(s, m)][0] for s in have])
                energies.append({"fold": f["fold"], "M": m, "structures": have, "ref_rel": (e_ref - e_ref.min()).tolist(),
                                 "pred_rel": (e_pred - e_pred[int(np.argmin(e_ref))]).tolist()})
            del calc
        if missing:
            continue
        entry: dict = {"spin_gaps": {}, "relative_energies": {}, "force_rmse_ev_per_A": {},
                       "max_force_change_49_to_47_ev_per_A_median": float(np.median(sens)) if sens else None}
        for hi, lo in gap_names:
            name = f"{lo}-{hi}"
            ref = {(r["fold"], r["structure"]): r for r in ref_rows if r["gap"] == name}
            got = {(g["fold"], g["structure"]): g["pred"] for g in gaps if g["gap"] == name}
            keys = sorted(ref)
            r = np.array([ref[k]["ref"] for k in keys]); p = np.array([got[k] for k in keys]); c = np.array([ref[k]["constant"] for k in keys])
            entry["spin_gaps"][f"E(M={lo}) - E(M={hi})"] = {
                "n": len(keys), "ref_mean_ev": float(r.mean()), "ref_sd_ev": float(r.std()), "pred_mean_ev": float(p.mean()),
                "mae_ev": float(np.abs(p - r).mean()), "mae_constant_gap_baseline_ev": float(np.abs(c - r).mean()),
                "mae_zero_gap_ev": float(np.abs(r).mean()),
                "sign_agreement": int((np.sign(p) == np.sign(r)).sum()), "sign_agreement_constant_baseline": int((np.sign(c) == np.sign(r)).sum()),
                "spearman": spearman(p, r),
                "rows": [{"fold": k[0], "structure": k[1], "ref_ev": float(a), "pred_ev": float(b), "constant_ev": float(cc)}
                         for k, a, b, cc in zip(keys, r, p, c)]}
        for m in MULTS:
            rows = [e for e in energies if e["M"] == m]
            if not rows:
                continue
            err = np.concatenate([np.array(e["pred_rel"]) - np.array(e["ref_rel"]) for e in rows])
            agree = total = 0
            for e in rows:
                a, b = np.array(e["ref_rel"]), np.array(e["pred_rel"])
                for i in range(len(a)):
                    for j in range(i + 1, len(a)):
                        total += 1
                        agree += int(np.sign(a[i] - a[j]) == np.sign(b[i] - b[j]))
            entry["relative_energies"][f"M={m}"] = {"structures": int(sum(len(e["structures"]) for e in rows)),
                                                    "mae_within_fold_ev": float(np.abs(err).mean()),
                                                    "pairs_ordered_within_fold_percent": 100.0 * agree / max(total, 1), "pairs": total}
            entry["force_rmse_ev_per_A"][f"M={m}"] = {
                "Fe": rmse(np.concatenate(forces[m]["Fe"])), "adsorbate": rmse(np.concatenate(forces[m]["ads"])),
                "zero_baseline_Fe": rmse(np.concatenate(forces[m]["Fe0"])), "zero_baseline_adsorbate": rmse(np.concatenate(forces[m]["ads0"]))}
        report["variants"][variant] = entry
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for v, e in report["variants"].items():
        print(f"== {v}  (median max force change 49->47 at fixed geometry: {e['max_force_change_49_to_47_ev_per_A_median']})")
        for g, x in e["spin_gaps"].items():
            print(f"  {g}: n={x['n']} ref {x['ref_mean_ev']:+.3f}+-{x['ref_sd_ev']:.3f}  pred mean {x['pred_mean_ev']:+.3f}  "
                  f"MAE {x['mae_ev']:.3f} (constant {x['mae_constant_gap_baseline_ev']:.3f}, zero {x['mae_zero_gap_ev']:.3f})  "
                  f"sign {x['sign_agreement']}/{x['n']} (constant {x['sign_agreement_constant_baseline']})  rho {x['spearman']:.2f}")
        for m, x in e["relative_energies"].items():
            fr = e["force_rmse_ev_per_A"][m]
            print(f"  {m}: rel-E MAE {x['mae_within_fold_ev']:.3f} eV, pairs {x['pairs_ordered_within_fold_percent']:.0f}% of {x['pairs']};  "
                  f"forces Fe {fr['Fe']:.3f} (zero {fr['zero_baseline_Fe']:.3f}) ads {fr['adsorbate']:.3f} (zero {fr['zero_baseline_adsorbate']:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("coh", type=Path); p.add_argument("ladder", type=Path); p.add_argument("work", type=Path)
    p.add_argument("--folds", type=int, default=3)
    p = sub.add_parser("run")
    p.add_argument("work", type=Path); p.add_argument("--fold", type=int, required=True)
    p.add_argument("--variant", choices=["spin", "blind"], required=True)
    p.add_argument("--epochs", type=int, default=300); p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=11); p.add_argument("--hidden", default="128x0e + 128x1o + 128x2e")
    p.add_argument("--batch-size", type=int, default=8)
    p = sub.add_parser("score")
    p.add_argument("work", type=Path); p.add_argument("--out", type=Path, required=True); p.add_argument("--device", default="cuda")
    args = ap.parse_args()
    {"prepare": prepare, "run": run, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
