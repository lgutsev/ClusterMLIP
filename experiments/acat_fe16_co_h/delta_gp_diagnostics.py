"""Two checks before any active learning: is the Delta-model data-limited, and is its uncertainty usable?

    python delta_gp_diagnostics.py LABELS.extxyz MODEL --multiplicity 49 --out report.json [--device cpu]

Same model as delta_gp_test.py (Gaussian process on per-element mean invariant descriptors of the
foundation model, energies only). Energies only, so no automatic differentiation is needed here.

1. Learning curve. For n = 2 ... 17 training structure groups: many random choices of n groups; train on
   every frame of those groups, predict the panel frames of all the other groups. Reported: mean absolute
   error of relative energies on the unseen structures (errors centred within each test set), with its
   spread over the random choices. Still falling at the largest n means more structures would help;
   flat means a floor for this model and these descriptors.

2. Uncertainty calibration. Leave one group out; the process's predictive standard deviation for each
   held-out panel structure against its actual error. Reported: how many errors fall within one and two
   standard deviations, the rank correlation between predicted uncertainty and absolute error, and how
   many of the five largest errors are among the five most uncertain structures. An active learner picks
   by this uncertainty, so if it does not track the error the learner picks the wrong structures.

Hyperparameters (length scale, noise) are chosen once, by leave-one-group-out energy error on all groups,
and then held fixed in both checks, so the curve shows the effect of data alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from delta_gp_test import ELEMENTS, SAME_MINIMUM, Featurizer, kernel


def gp(xtr, ytr, xq, counts, length, noise):
    """Mean and predictive standard deviation (signal variance from the training residuals)."""
    shift = ytr.mean()
    k = kernel(xtr, xtr, counts, length)
    scale = k.diagonal().mean()
    kt = k / scale + noise * torch.eye(len(ytr), dtype=k.dtype)
    alpha = torch.linalg.solve(kt, ytr - shift)
    sigma2 = float((ytr - shift) @ alpha / len(ytr))
    kq = kernel(xq, xtr, counts, length) / scale
    mean = shift + kq @ alpha
    kqq = (counts ** 2).sum() / scale  # k(x, x) for the additive kernel: every element term is 1
    var = kqq + noise - (kq * torch.linalg.solve(kt, kq.T).T).sum(-1)
    return mean, torch.sqrt(torch.clamp(sigma2 * var, min=0.0))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path)
    ap.add_argument("model")
    ap.add_argument("--multiplicity", type=int, default=49)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    feat = Featurizer(args.model, device=args.device)
    rows = []
    for a in read(args.labels, index=":"):
        if int(a.info["multiplicity"]) != args.multiplicity:
            continue
        meta = a.info["metadata"]
        meta = json.loads(meta) if isinstance(meta, str) else meta
        e, means, _ = feat(a)
        rows.append({"structure": meta["structure"], "group": SAME_MINIMUM.get(meta["structure"], meta["structure"]),
                     "panel": a.info["config_type"] == "panel_fixed", "x": means.detach().cpu(),
                     "resid": float(a.info["REF_energy"]) - float(e), "e_ref": float(a.info["REF_energy"])})
    counts = torch.tensor([16.0, 1.0, 1.0, 2.0])
    assert ELEMENTS == ["Fe", "C", "O", "H"]
    x = torch.stack([r["x"] for r in rows])
    z = (x - x.mean(0)) / (x.std(0) + 1e-12)
    y = torch.tensor([r["resid"] for r in rows])
    group = np.array([r["group"] for r in rows])
    panel = np.array([r["panel"] for r in rows])
    groups = sorted({g for g, p in zip(group, panel) if p})
    dist = torch.sqrt(((z[:, None] - z[None]) ** 2).sum(-1)).reshape(-1)
    length0 = float(dist[dist > 0].median())

    def idx(mask):
        return torch.as_tensor(np.where(mask)[0])

    # hyperparameters once, by leave-one-group-out on all groups
    best = None
    for mult in (0.5, 1.0, 2.0, 4.0, 8.0):
        for noise in (1e-6, 1e-4, 1e-2):
            errs = []
            for g in groups:
                tr, te = idx(group != g), idx((group == g) & panel)
                mean, _ = gp(z[tr], y[tr], z[te], counts, length0 * mult, noise)
                errs += (mean - y[te]).abs().tolist()
            score = float(np.mean(errs))
            if best is None or score < best[0]:
                best = (score, mult, noise)
    _, mult, noise = best
    length = length0 * mult

    # 1. learning curve
    rng = np.random.default_rng(0)
    curve = []
    base_err = np.array([r["resid"] for r in rows])
    for n in (2, 4, 6, 8, 10, 12, 14, 16, 17):
        maes, base = [], []
        for _ in range(args.repeats if n < 17 else len(groups)):
            chosen = rng.choice(groups, size=n, replace=False) if n < 17 else np.array([g for g in groups if g != groups[len(maes)]])
            tr = idx(np.isin(group, chosen))
            te_mask = ~np.isin(group, chosen) & panel
            te = idx(te_mask)
            mean, _ = gp(z[tr], y[tr], z[te], counts, length, noise)
            err = (mean - y[te]).numpy()
            if len(err) > 1:
                maes.append(float(np.abs(err - err.mean()).mean()))
                b = base_err[te_mask]
                base.append(float(np.abs(b - b.mean()).mean()))
            else:  # one held-out structure: absolute error after the training shift
                maes.append(float(np.abs(err).mean()))
        curve.append({"training_groups": n, "choices": len(maes), "mae_ev_mean": float(np.mean(maes)),
                      "mae_ev_sd": float(np.std(maes)), "mae_ev_median": float(np.median(maes)),
                      "uncorrected_mae_ev_mean": float(np.mean(base)) if base and n < 17 else None,
                      "note": "leave-one-group-out, absolute error" if n == 17 else "errors centred within each test set"})

    # 2. calibration, leave one group out
    cal = []
    for g in groups:
        tr, te_mask = idx(group != g), (group == g) & panel
        mean, sd = gp(z[tr], y[tr], z[idx(te_mask)], counts, length, noise)
        for i, m, s in zip(np.where(te_mask)[0], mean.tolist(), sd.tolist()):
            cal.append({"structure": rows[i]["structure"], "error_ev": m - rows[i]["resid"], "predicted_sd_ev": s,
                        "e_ref": rows[i]["e_ref"]})
    err = np.abs(np.array([c["error_ev"] for c in cal]))
    sd = np.array([c["predicted_sd_ev"] for c in cal])
    top_err = set(np.argsort(-err)[:5])
    top_sd = set(np.argsort(-sd)[:5])
    e_min = min(c["e_ref"] for c in cal)
    report = {"model": Path(args.model).name, "multiplicity": args.multiplicity, "frames": len(rows), "groups": len(groups),
              "hyperparameters": {"length_scale_multiple": mult, "noise": noise, "leave_one_group_out_mae_ev": best[0]},
              "learning_curve": curve,
              "calibration": {"structures": len(cal), "within_1_sd": int((err <= sd).sum()), "within_2_sd": int((err <= 2 * sd).sum()),
                              "expected_within_1_sd": round(0.683 * len(cal), 1), "expected_within_2_sd": round(0.954 * len(cal), 1),
                              "spearman_uncertainty_vs_abs_error": spearman(sd, err),
                              "top5_errors_among_top5_uncertain": len(top_err & top_sd),
                              "median_predicted_sd_ev": float(np.median(sd)), "median_abs_error_ev": float(np.median(err)),
                              "per_structure": [{"structure": c["structure"], "ubpw91_rel_ev": c["e_ref"] - e_min,
                                                 "error_ev": c["error_ev"], "predicted_sd_ev": c["predicted_sd_ev"]} for c in cal]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("hyperparameters", report["hyperparameters"])
    for c in curve:
        print(f"  n={c['training_groups']:2d}  MAE {c['mae_ev_mean']:.3f} +/- {c['mae_ev_sd']:.3f} (median {c['mae_ev_median']:.3f})  "
              f"uncorrected {c['uncorrected_mae_ev_mean'] if c['uncorrected_mae_ev_mean'] is None else round(c['uncorrected_mae_ev_mean'], 3)}  [{c['choices']} choices]")
    print({k: v for k, v in report["calibration"].items() if k != "per_structure"})
    for c in sorted(report["calibration"]["per_structure"], key=lambda c: -abs(c["error_ev"])):
        print(f"  {c['structure'][-3:]}  ref {c['ubpw91_rel_ev']:6.3f}  error {c['error_ev']:+7.3f}  predicted sd {c['predicted_sd_ev']:.3f}")


if __name__ == "__main__":
    main()
