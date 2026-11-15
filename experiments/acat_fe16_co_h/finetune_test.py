"""Naive fine-tune of a foundation MACE model on the Fe16 + CO + 2H labels, scored on held-out structures.

    python finetune_test.py prepare LABELS.extxyz WORK --multiplicity 49 [--folds 3]
    python finetune_test.py run     WORK --fold 0 --model NAME=PATH [--device cuda] [--epochs 100]
    python finetune_test.py score   WORK --model NAME=PATH --out report.json [--device cpu]

The second route of the roadmap, the counterpart of delta_gp_test.py: here the model is trained on
energies and forces. One multiplicity at a time (the foundation models take no spin input).

prepare  Grouped folds. A group is one starting structure with every frame derived from it; structures
         000, 008 and the relaxed minimum are one group (same minimum). Groups are dealt to folds in
         order of their UBPW91 panel energy, so every fold spans the energy range. Within a fold's
         training groups, two are set aside for validation (early stopping). Writes
         WORK/fold_K/{train,valid,test}.extxyz with REF_energy / REF_forces, and WORK/folds.json.
run      One fold: mace_run_train with the project's fine-tune defaults (lr 1e-4, forces weight 100,
         EMA 0.99, AMSGrad, float64, at most 100 epochs, patience 20), keeping the foundation model's E0s
         and shifting the labels by one constant onto its energy scale.
score    For every fold whose model exists: the fine-tuned model and the unmodified foundation model on
         that fold's held-out panel frames. Same scores as delta_gp_test.py: relative-energy error and
         pair ordering pooled over folds, force error per element next to the zero-force baseline.
         Fine-tuned energies are compared after removing each fold's mean error on its own training
         frames (the constant that E0s absorb).

Run `run` only when the GPU is free; `prepare` and `score` are CPU work.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

SAME_MINIMUM = {"fe16coh2_000": "A", "fe16coh2_008": "A", "fe16coh2_minA_m49": "A", "fe16coh2_minA_m47": "A"}
PAIR_THRESHOLD_EV = 0.05


def load(labels: Path, multiplicity: int) -> list[dict]:
    rows = []
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
        rows.append({"atoms": clean, **clean.info})
    return rows


def prepare(args) -> None:
    rows = load(args.labels, args.multiplicity)
    panel = [r for r in rows if r["config_type"] == "panel_fixed"]
    rank = {}
    for r in panel:
        rank[r["group"]] = min(rank.get(r["group"], np.inf), r["REF_energy"])
    groups = sorted(rank, key=rank.get)
    folds = [groups[k::args.folds] for k in range(args.folds)]
    plan = {"multiplicity": args.multiplicity, "labels": args.labels.name, "frames": len(rows), "groups": len(groups), "folds": []}
    for k, held in enumerate(folds):
        rest = [g for g in groups if g not in held]
        valid_groups = rest[1::max(2, len(rest) // 2)][:2]  # two groups, spread over the energy range, never the lowest
        d = args.work / f"fold_{k}"
        d.mkdir(parents=True, exist_ok=True)
        split = {"train": [r for r in rows if r["group"] not in held and r["group"] not in valid_groups],
                 "valid": [r for r in rows if r["group"] in valid_groups],
                 "test": [r for r in rows if r["group"] in held]}
        for name, rs in split.items():
            write(d / f"{name}.extxyz", [r["atoms"] for r in rs], format="extxyz")
        plan["folds"].append({"fold": k, "held_out_groups": held, "validation_groups": valid_groups,
                              "frames": {n: len(rs) for n, rs in split.items()},
                              "held_out_panel_structures": sorted(r["structure"] for r in split["test"] if r["config_type"] == "panel_fixed")})
    (args.work / "folds.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=1))


def run(args) -> None:
    name, path = args.model.split("=", 1)
    d = args.work / f"fold_{args.fold}"
    out = d / name
    out.mkdir(parents=True, exist_ok=True)
    # Put the labels on the foundation model's energy scale. Gaussian total energies and the model's
    # energies differ by a large constant (one composition, so exactly one constant). "average" E0s
    # do not remove it: they fit the totals and the model's interaction energy then comes on top
    # (a dry run started 3.4 eV/atom off). So: keep the foundation model's own E0s and subtract from
    # every label the mean of E(reference) - E(foundation model) over this fold's training frames.
    import torch

    foundation = torch.load(path, map_location="cpu", weights_only=False)
    e0 = foundation.atomic_energies_fn.atomic_energies.detach().double().reshape(-1).tolist()
    zs = [int(z) for z in foundation.atomic_numbers.tolist()]
    e0s = "{" + ", ".join(f"{z}: {e0[zs.index(z)]}" for z in (1, 6, 8, 26)) + "}"
    del foundation
    train = read(d / "train.extxyz", index=":")
    base = evaluate(path, train, "cpu")
    constant = float(np.mean([a.info["REF_energy"] - e for a, (e, _) in zip(train, base)]))
    for split in ("train", "valid", "test"):
        frames = read(d / f"{split}.extxyz", index=":")
        for a in frames:
            a.info["REF_energy"] = float(a.info["REF_energy"]) - constant
        write(out / f"{split}_shifted.extxyz", frames, format="extxyz")
    (out / "energy_shift.json").write_text(json.dumps({"constant_ev": constant, "E0s": e0s}, indent=1) + "\n", encoding="utf-8")
    argv = ["mace_run_train", f"--name={name}_fold{args.fold}", "--model=ScaleShiftMACE",
            f"--train_file={out / 'train_shifted.extxyz'}", f"--valid_file={out / 'valid_shifted.extxyz'}",
            f"--test_file={out / 'test_shifted.extxyz'}",
            "--energy_key=REF_energy", "--forces_key=REF_forces", f"--foundation_model={path}",
            "--multiheads_finetuning=False", f"--E0s={e0s}", "--loss=weighted", "--energy_weight=1.0",
            "--forces_weight=100.0", "--stress_weight=0", "--lr=0.0001", f"--batch_size={args.batch_size}",
            f"--valid_batch_size={args.batch_size}", f"--max_num_epochs={args.epochs}", "--patience=20", "--ema",
            "--ema_decay=0.99", "--default_dtype=float64", f"--device={args.device}", f"--seed={args.seed}", "--amsgrad",
            f"--work_dir={out}"]
    (out / "command.txt").write_text(" ".join(argv) + "\n", encoding="utf-8")
    print(" ".join(argv))
    with open(out / "train_stdout.log", "w", encoding="utf-8") as log:
        code = subprocess.run(argv, cwd=out, stdout=log, stderr=subprocess.STDOUT).returncode
    print("exit", code)
    raise SystemExit(code)


def evaluate(model_path: str, frames: list[Atoms], device: str) -> list[tuple[float, np.ndarray]]:
    from mace.calculators import MACECalculator

    calc = MACECalculator(model_paths=str(model_path), device=device, default_dtype="float64")
    out = []
    for a in frames:
        b = a.copy()
        b.calc = calc
        out.append((float(b.get_potential_energy()), np.array(b.get_forces())))
    return out


def score(args) -> None:
    name, base_path = args.model.split("=", 1)
    plan = json.loads((args.work / "folds.json").read_text(encoding="utf-8"))
    results, used = [], []
    for fold in plan["folds"]:
        d = args.work / f"fold_{fold['fold']}"
        models = sorted((d / name).glob(f"{name}_fold{fold['fold']}*.model"))
        models = [m for m in models if "compiled" not in m.name and "stagetwo" not in m.name]
        if not models:
            continue
        tuned = models[0]
        train = read(d / "train.extxyz", index=":")
        test = [a for a in read(d / "test.extxyz", index=":") if a.info["config_type"] == "panel_fixed"]
        shift = float(np.mean([a.info["REF_energy"] - e for a, (e, _) in zip(train, evaluate(tuned, train, args.device))]))
        ft = evaluate(tuned, test, args.device)
        base = evaluate(base_path, test, args.device)
        used.append({"fold": fold["fold"], "model_file": tuned.name, "train_shift_ev": shift, "held_out_panel": len(test)})
        for a, (e_ft, f_ft), (e_b, f_b) in zip(test, ft, base):
            results.append({"structure": a.info["structure"], "fold": fold["fold"], "e_ref": a.info["REF_energy"],
                            "e_base": e_b, "e_ft": e_ft + shift, "f_ref": a.arrays["REF_forces"], "f_base": f_b, "f_ft": f_ft,
                            "symbols": a.get_chemical_symbols()})
    if not results:
        raise SystemExit("no fine-tuned model found; run the folds first")

    def energy_scores(key: str) -> dict:
        err = np.array([r[key] - r["e_ref"] for r in results])
        rel = err - err.mean()
        pred = np.array([r[key] for r in results])
        ref = np.array([r["e_ref"] for r in results])
        agree = disagree = 0
        for i, j in combinations(range(len(results)), 2):
            if abs(ref[i] - ref[j]) < PAIR_THRESHOLD_EV:
                continue
            if (ref[i] - ref[j]) * (pred[i] - pred[j]) > 0:
                agree += 1
            else:
                disagree += 1
        low = int(np.argmin(pred))
        return {"relative_energy_mae_ev": float(np.abs(rel).mean()), "max_abs_relative_error_ev": float(np.abs(rel).max()),
                "pairs_agree": agree, "pairs_disagree": disagree,
                "pair_ordering_percent": round(100 * agree / max(1, agree + disagree), 1),
                "predicted_lowest": results[low]["structure"], "ubpw91_rel_of_predicted_lowest_ev": float(ref[low] - ref.min())}

    def force_scores(key: str | None) -> dict:
        out = {}
        for label, pick in (("Fe", lambda s: s == "Fe"), ("adsorbate", lambda s: s != "Fe")):
            diffs = []
            for r in results:
                mask = np.array([pick(s) for s in r["symbols"]])
                pred = np.zeros_like(r["f_ref"]) if key is None else r[key]
                diffs.append((pred - r["f_ref"])[mask])
            out[label] = float(np.sqrt((np.concatenate(diffs) ** 2).mean()))
        return out

    report = {"model": name, "multiplicity": plan["multiplicity"], "folds_scored": used, "panel_structures_scored": len(results),
              "folds_missing": [f["fold"] for f in plan["folds"] if f["fold"] not in {u["fold"] for u in used}],
              "energy": {"uncorrected": energy_scores("e_base"), "fine_tuned": energy_scores("e_ft")},
              "force_rmse_ev_per_A": {"zero_baseline": force_scores(None), "uncorrected": force_scores("f_base"),
                                      "fine_tuned": force_scores("f_ft")},
              "per_structure": [{"structure": r["structure"], "fold": r["fold"],
                                 "ubpw91_rel_ev": r["e_ref"] - min(x["e_ref"] for x in results),
                                 "fine_tuned_error_ev": r["e_ft"] - r["e_ref"]} for r in results]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_structure"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("labels", type=Path)
    p.add_argument("work", type=Path)
    p.add_argument("--multiplicity", type=int, default=49)
    p.add_argument("--folds", type=int, default=3)
    p.set_defaults(fn=prepare)
    p = sub.add_parser("run")
    p.add_argument("work", type=Path)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--model", required=True, help="NAME=PATH of the foundation model")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    p.set_defaults(fn=run)
    p = sub.add_parser("score")
    p.add_argument("work", type=Path)
    p.add_argument("--model", required=True, help="NAME=PATH of the foundation model")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
