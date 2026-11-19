"""Combined test: the Delta-model energy correction on top of each fold's fine-tuned model.

    python delta_on_finetune.py WORK --name mpa0 --out report.json [--device cpu]

WORK is the folder of finetune_test.py (folds.json, fold_K/{train,valid,test}.extxyz and the fine-tuned
model of each fold). For fold K the base model is that fold's fine-tuned model; the Gaussian process of
delta_gp_test.py (per-element mean invariant descriptors, energies only) is trained on the fold's
training and validation frames, which are the frames the fine-tune itself saw, and applied to the
held-out panel frames. Length scale and noise are chosen by leave-one-group-out error inside those
frames. Forces of the combined model come from automatic differentiation.

So a held-out structure is unseen by both stages. Scores are the same as in the other two tests, pooled
over folds, plus pair ordering within folds.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from delta_gp_test import ELEMENTS, PAIR_THRESHOLD_EV, Featurizer, fit, kernel, predict  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    plan = json.loads((args.work / "folds.json").read_text(encoding="utf-8"))
    results, chosen = [], []
    for fold in plan["folds"]:
        k = fold["fold"]
        d = args.work / f"fold_{k}"
        tuned = d / args.name / f"{args.name}_fold{k}.model"
        feat = Featurizer(str(tuned), device=args.device)
        seen = read(d / "train.extxyz", index=":") + read(d / "valid.extxyz", index=":")
        test = [a for a in read(d / "test.extxyz", index=":") if a.info["config_type"] == "panel_fixed"]
        counts = torch.tensor([float(sum(s == el for s in seen[0].get_chemical_symbols())) for el in ELEMENTS], device=args.device)
        rows = []
        for a in seen:
            e, means, _ = feat(a)
            rows.append({"group": a.info["group"], "config": a.info["config_type"], "x": means.detach(),
                         "resid": float(a.info["REF_energy"]) - float(e)})
        allx = torch.stack([r["x"] for r in rows])
        mu, sd = allx.mean(0), allx.std(0) + 1e-12
        z = (allx - mu) / sd
        y = torch.tensor([r["resid"] for r in rows], device=args.device)
        dist = torch.sqrt(((z[:, None] - z[None]) ** 2).sum(-1)).reshape(-1)
        length0 = float(dist[dist > 0].median())
        groups = sorted({r["group"] for r in rows})
        gidx = {g: torch.tensor([i for i, r in enumerate(rows) if r["group"] == g], device=args.device) for g in groups}
        panel_idx = {g: torch.tensor([i for i, r in enumerate(rows) if r["group"] == g and r["config"] == "panel_fixed"],
                                     device=args.device, dtype=torch.long) for g in groups}
        best = None
        for length in (length0 * m for m in (0.5, 1.0, 2.0, 4.0, 8.0)):
            for noise in (1e-6, 1e-4, 1e-2):
                errs = []
                for g in groups:
                    va = panel_idx[g]
                    if len(va) == 0:
                        continue
                    keep = torch.tensor([i for i in range(len(rows)) if rows[i]["group"] != g], device=args.device)
                    m = fit(z[keep], y[keep], counts, length, noise)
                    errs += (predict(z[va], z[keep], counts, length, m) - y[va]).abs().tolist()
                score = float(np.mean(errs))
                if best is None or score < best[0]:
                    best = (score, length, noise)
        _, length, noise = best
        model = fit(z, y, counts, length, noise)
        chosen.append({"fold": k, "frames_seen": len(rows), "length": length / length0, "noise": noise, "inner_mae_ev": best[0],
                       "mean_residual_ev": float(y.mean()), "residual_sd_ev": float(y.std())})
        for a in test:
            e_base, means, pos = feat(a)
            zq = ((means - mu) / sd)[None]
            corr = predict(zq, z, counts, length, model)[0]
            f_base = -torch.autograd.grad(e_base, pos, retain_graph=True)[0]
            f_comb = f_base - torch.autograd.grad(corr, pos)[0]
            results.append({"structure": a.info["structure"], "fold": k, "e_ref": float(a.info["REF_energy"]),
                            "e_ft": float(e_base) + float(y.mean()), "e_comb": float(e_base) + float(corr),
                            "f_ref": np.array(a.arrays["REF_forces"]), "f_ft": f_base.detach().cpu().numpy(),
                            "f_comb": f_comb.detach().cpu().numpy(), "symbols": a.get_chemical_symbols()})

    def energy_scores(key: str) -> dict:
        err = np.array([r[key] - r["e_ref"] for r in results])
        rel = err - err.mean()
        pred = np.array([r[key] for r in results])
        ref = np.array([r["e_ref"] for r in results])
        agree = disagree = w_agree = w_disagree = 0
        for i, j in combinations(range(len(results)), 2):
            if abs(ref[i] - ref[j]) < PAIR_THRESHOLD_EV:
                continue
            ok = (ref[i] - ref[j]) * (pred[i] - pred[j]) > 0
            agree, disagree = agree + ok, disagree + (not ok)
            if results[i]["fold"] == results[j]["fold"]:
                w_agree, w_disagree = w_agree + ok, w_disagree + (not ok)
        low = int(np.argmin(pred))
        window = ref - ref.min() < 0.5
        return {"relative_energy_mae_ev": float(np.abs(rel).mean()), "max_abs_relative_error_ev": float(np.abs(rel).max()),
                "pair_ordering_percent": round(100 * agree / (agree + disagree), 1),
                "pair_ordering_within_folds_percent": round(100 * w_agree / (w_agree + w_disagree), 1),
                "low_window_error_sd_ev": float(err[window].std()), "low_window_n": int(window.sum()),
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

    ref_min = min(r["e_ref"] for r in results)
    report = {"name": args.name, "multiplicity": plan["multiplicity"], "panel_structures_scored": len(results),
              "gp_per_fold": chosen,
              "energy": {"fine_tuned": energy_scores("e_ft"), "fine_tuned_plus_delta": energy_scores("e_comb")},
              "force_rmse_ev_per_A": {"zero_baseline": force_scores(None), "fine_tuned": force_scores("f_ft"),
                                      "fine_tuned_plus_delta": force_scores("f_comb")},
              "per_structure": [{"structure": r["structure"], "fold": r["fold"], "ubpw91_rel_ev": r["e_ref"] - ref_min,
                                 "fine_tuned_error_ev": r["e_ft"] - r["e_ref"], "combined_error_ev": r["e_comb"] - r["e_ref"]}
                                for r in results]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_structure"}, indent=1))
    for r in sorted(report["per_structure"], key=lambda r: r["ubpw91_rel_ev"]):
        print(f"  {r['structure'][-3:]} fold {r['fold']}  ref {r['ubpw91_rel_ev']:6.3f}  fine-tuned err {r['fine_tuned_error_ev']:+7.3f}  combined err {r['combined_error_ev']:+7.3f}")


if __name__ == "__main__":
    main()
