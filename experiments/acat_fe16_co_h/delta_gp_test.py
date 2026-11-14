"""Test a Delta-model on a foundation MACE model against the Fe16 + CO + 2H UBPW91 labels.

    python delta_gp_test.py LABELS.extxyz MODEL --multiplicity 49 --out report.json

Method (after Christiansen and Hammer, arXiv:2502.21179, there with CHGNet): a Gaussian process on the
foundation model's own invariant atomic descriptors, aggregated to one mean descriptor per element,
trained on total-energy differences E(UBPW91) - E(model). Forces of the corrected model come from
automatic differentiation through the model and the kernel term. Choices made here that the paper does
not fix: RBF kernel between descriptors of the same element only, descriptors standardised on the
training set, a constant energy shift fitted on the training set (one composition, so one constant),
length scale and noise chosen by leave-one-group-out error inside the training set.

Evaluation: leave-one-structure-group-out. A group is one starting structure with every frame derived
from it (its optimization steps); structures 000 and 008 and the fully relaxed minimum are one group
because they relax to the same minimum. The model for a held-out structure never saw any frame of it.
Scored on the panel frames (search-model geometries) at one multiplicity: relative-energy error, pair
ordering, and force error per element next to the uncorrected model and to predicting zero force.

The models take no spin input, so one multiplicity is fitted at a time.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator  # isort: skip  (must come before e3nn: mace sets the torch.load mode e3nn needs)
from mace.modules.utils import extract_invariant  # isort: skip
from e3nn import o3  # isort: skip

ELEMENTS = ["Fe", "C", "O", "H"]
SAME_MINIMUM = {"fe16coh2_000": "A", "fe16coh2_008": "A", "fe16coh2_minA_m49": "A", "fe16coh2_minA_m47": "A"}
PAIR_THRESHOLD_EV = 0.05


class Featurizer:
    def __init__(self, model_path: str):
        self.calc = MACECalculator(model_paths=model_path, device="cuda", default_dtype="float64")
        self.model = self.calc.models[0]
        irreps = o3.Irreps(str(self.model.products[0].linear.irreps_out))
        self.l_max = irreps.lmax
        self.n_inv = irreps.dim // (self.l_max + 1) ** 2
        self.n_layers = int(self.model.num_interactions)
        per_layer = [irreps.dim] * self.n_layers
        per_layer[-1] = self.n_inv
        self.keep = int(sum(per_layer))

    def __call__(self, atoms):
        """Base energy (tensor), per-element mean invariant descriptors (4 x D tensor), positions (leaf)."""
        batch = self.calc._atoms_to_batch(atoms).to_dict()
        batch["positions"].requires_grad_(True)
        out = self.model(batch, compute_force=False)
        feats = extract_invariant(out["node_feats"], num_layers=self.n_layers, num_features=self.n_inv, l_max=self.l_max)
        feats = feats[:, : self.keep]
        symbols = np.array(atoms.get_chemical_symbols())
        means = torch.stack([feats[torch.as_tensor(symbols == el, device=feats.device)].mean(dim=0) for el in ELEMENTS])
        return out["energy"].reshape(()), means, batch["positions"]


def kernel(a: torch.Tensor, b: torch.Tensor, counts: torch.Tensor, length: float) -> torch.Tensor:
    """Additive kernel between structures: sum over elements of n_el^2 * RBF(mean descriptors)."""
    d2 = ((a[:, None] - b[None]) ** 2).sum(-1)  # (Na, Nb, n_elements)
    return (counts ** 2 * torch.exp(-d2 / (2 * length ** 2))).sum(-1)


def fit(x: torch.Tensor, y: torch.Tensor, counts: torch.Tensor, length: float, noise: float):
    shift = y.mean()
    k = kernel(x, x, counts, length)
    scale = k.diagonal().mean()
    alpha = torch.linalg.solve(k / scale + noise * torch.eye(len(y), dtype=k.dtype, device=k.device), y - shift)
    return shift, alpha, scale


def predict(xq: torch.Tensor, x: torch.Tensor, counts, length, model) -> torch.Tensor:
    shift, alpha, scale = model
    return shift + (kernel(xq, x, counts, length) / scale) @ alpha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path)
    ap.add_argument("model")
    ap.add_argument("--multiplicity", type=int, default=49)
    ap.add_argument("--train", choices=["panel", "all"], default="all", help="panel frames only, or also optimization frames")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)

    frames = [a for a in read(args.labels, index=":") if int(a.info["multiplicity"]) == args.multiplicity]
    feat = Featurizer(args.model)
    rows = []
    for a in frames:
        meta = a.info["metadata"]
        meta = json.loads(meta) if isinstance(meta, str) else meta
        structure = meta["structure"]
        e, means, _ = feat(a)
        rows.append({"atoms": a, "structure": structure, "group": SAME_MINIMUM.get(structure, structure),
                     "config": a.info["config_type"], "e_ref": float(a.info["REF_energy"]),
                     "f_ref": np.array(a.arrays["REF_forces"]), "e_base": float(e), "x": means.detach()})
    counts = torch.tensor([float(sum(s == el for s in frames[0].get_chemical_symbols())) for el in ELEMENTS], device="cuda")
    panel = [r for r in rows if r["config"] == "panel_fixed"]
    pool = panel if args.train == "panel" else rows
    groups = sorted({r["group"] for r in panel})

    # descriptor scale and a reference length from the pool
    allx = torch.stack([r["x"] for r in pool])
    mu, sd = allx.mean(0), allx.std(0) + 1e-12
    for r in rows:
        r["z"] = (r["x"] - mu) / sd
    zpool = torch.stack([r["z"] for r in pool])
    d = torch.sqrt(((zpool[:, None] - zpool[None]) ** 2).sum(-1)).reshape(-1)
    length0 = float(d[d > 0].median())
    grid = [(length0 * m, n) for m in (0.5, 1.0, 2.0, 4.0, 8.0) for n in (1e-6, 1e-4, 1e-2)]

    def resid(rs):
        return torch.tensor([r["e_ref"] - r["e_base"] for r in rs], device="cuda")

    def stack(rs):
        return torch.stack([r["z"] for r in rs])

    results, chosen = [], []
    for held in groups:
        train = [r for r in pool if r["group"] != held]
        test = [r for r in panel if r["group"] == held]
        tg = sorted({r["group"] for r in train})
        best = None
        for length, noise in grid:  # inner leave-one-group-out on the training set, energies only
            errs = []
            for g in tg:
                tr = [r for r in train if r["group"] != g]
                va = [r for r in train if r["group"] == g and r["config"] == "panel_fixed"]
                if not va:
                    continue
                m = fit(stack(tr), resid(tr), counts, length, noise)
                errs += (predict(stack(va), stack(tr), counts, length, m) - resid(va)).abs().tolist()
            score = float(np.mean(errs))
            if best is None or score < best[0]:
                best = (score, length, noise)
        _, length, noise = best
        chosen.append({"held_out": held, "length": length / length0, "noise": noise, "inner_mae_ev": best[0]})
        xtr, model = stack(train), fit(stack(train), resid(train), counts, length, noise)
        for r in test:
            e_base, means, pos = feat(r["atoms"])
            z = ((means - mu) / sd)[None]
            corr = predict(z, xtr, counts, length, model)[0]
            f_base = -torch.autograd.grad(e_base, pos, retain_graph=True)[0]
            f_corr = f_base - torch.autograd.grad(corr, pos)[0]
            results.append({"structure": r["structure"], "group": held, "e_ref": r["e_ref"],
                            "e_base_shifted": float(e_base), "e_corrected": float(e_base + corr),
                            "f_ref": r["f_ref"], "f_base": f_base.detach().cpu().numpy(), "f_corr": f_corr.detach().cpu().numpy(),
                            "symbols": r["atoms"].get_chemical_symbols()})

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
                "pairs_agree": agree, "pairs_disagree": disagree, "pair_ordering_percent": round(100 * agree / (agree + disagree), 1),
                "predicted_lowest": results[low]["structure"], "ubpw91_rel_of_predicted_lowest_ev": float(ref[low] - ref.min())}

    def force_scores(key: str | None) -> dict:
        out = {}
        for name, pick in (("Fe", lambda s: s == "Fe"), ("adsorbate", lambda s: s != "Fe")):
            diffs = []
            for r in results:
                mask = np.array([pick(s) for s in r["symbols"]])
                pred = np.zeros_like(r["f_ref"]) if key is None else r[key]
                diffs.append((pred - r["f_ref"])[mask])
            out[name] = float(np.sqrt((np.concatenate(diffs) ** 2).mean()))
        return out

    report = {
        "model": Path(args.model).name, "multiplicity": args.multiplicity, "training_frames": args.train,
        "frames_at_this_multiplicity": len(rows), "panel_structures_scored": len(results), "groups": len(groups),
        "descriptor_dim": int(rows[0]["x"].shape[1]), "length_scale_reference": length0, "hyperparameters_chosen": chosen,
        "energy": {"uncorrected": energy_scores("e_base_shifted"), "corrected": energy_scores("e_corrected")},
        "force_rmse_ev_per_A": {"zero_baseline": force_scores(None), "uncorrected": force_scores("f_base"),
                                "corrected": force_scores("f_corr")},
        "per_structure": [{"structure": r["structure"], "ubpw91_rel_ev": r["e_ref"] - min(x["e_ref"] for x in results),
                           "uncorrected_error_ev": (r["e_base_shifted"] - r["e_ref"]) - float(np.mean([x["e_base_shifted"] - x["e_ref"] for x in results])),
                           "corrected_error_ev": r["e_corrected"] - r["e_ref"]} for r in results],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_structure", "hyperparameters_chosen")}, indent=1))
    for r in sorted(report["per_structure"], key=lambda r: r["ubpw91_rel_ev"]):
        print(f"  {r['structure']:22s} ref {r['ubpw91_rel_ev']:6.3f}  uncorrected err {r['uncorrected_error_ev']:+7.3f}  corrected err {r['corrected_error_ev']:+7.3f}")


if __name__ == "__main__":
    main()
