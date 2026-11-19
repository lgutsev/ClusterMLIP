"""Held-out spin test on the FenOm warehouse: Fe2 with O, H and N adsorbates (UBPW91, M = 1 to 11).

    python fenom_spin_test.py prepare DATASET_ALL.extxyz WORK [--held-formula Fe2H2O3] [--stride 3]
    python spin_heldout_test.py run   WORK --fold 0 --variant spin|blind --batch-size 32 --epochs 80
    python fenom_spin_test.py score   WORK --out report.json [--device cuda]

Same question as spin_heldout_test.py, on a different system: does a from-scratch MACE with the
total-spin input predict the energy as a function of M for structures it never saw, and does it beat
the same model without the spin input?

Data. `dataset_spin_v0/all.extxyz` of FenOm_Warehouse2: optimisation trajectories of each seed structure
("parent") at several multiplicities. Fe16 and Fe16N2 frames are left out. Each trajectory is thinned
(first frame, every `stride`-th, last). Frames with a force component above `--fmax` eV/A are dropped.
Validation is 5% of the trajectories of the training groups. A 500-frame sample of the training set is
scored too ("train_sample"), to tell a model that did not fit from one that does not transfer.

Split. Parents that share a geometry (same formula and same rounded pair-distance list in any kept
frame) are merged into one group, so a duplicated seed cannot sit on both sides. Two test sets:
  "trajectories"  10% of the trajectories of the training groups (easy: the same minima are in training);
  "parents"  about 15% of the groups of every other formula;
  "formula"  one whole formula (default Fe2H2O3, between Fe2H2O2 and Fe2H2O4 in training).

Scores per test set and variant: energy error per atom, force RMSE against a zero-force baseline, and
for every (parent, charge) with two or more M the adiabatic gaps between neighbouring M (lowest-energy
frame of each M) and whether the model picks the same lowest M as UBPW91.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

FRAME = re.compile(r"__frame(\d+)$")


def fingerprint(a: Atoms) -> tuple:
    s = a.get_chemical_symbols()
    d = a.get_all_distances()
    return (a.get_chemical_formula(),) + tuple(sorted(
        ("".join(sorted((s[i], s[j]))), int(round(d[i, j] / 0.05))) for i in range(len(a)) for j in range(i + 1, len(a))))


def prepare(args) -> None:
    rng = np.random.default_rng(args.seed)
    traj: dict[str, list[tuple[int, Atoms]]] = defaultdict(list)
    dropped = Counter()
    for a in read(args.data, index=":"):
        formula = str(a.info["formula"])
        if formula.startswith("Fe16"):
            dropped["Fe16 frames"] += 1
            continue
        if np.abs(a.arrays["REF_forces"]).max() > args.fmax:
            dropped[f"force above {args.fmax:g} eV/A"] += 1
            continue
        rid = str(a.info["record_id"])
        m = FRAME.search(rid)
        traj[rid[:m.start()] if m else rid].append((int(m.group(1)) if m else 0, a))
    frames: list[Atoms] = []
    for t, items in sorted(traj.items()):
        items.sort(key=lambda x: x[0])
        keep = sorted(set(range(0, len(items), args.stride)) | {len(items) - 1})
        for i in keep:
            a = items[i][1]
            mult = int(a.info["multiplicity"])
            b = Atoms(a.get_chemical_symbols(), positions=a.positions)
            b.info = {"REF_energy": float(a.info["REF_energy"]), "spin": mult, "multiplicity": mult, "charge": int(a.info["charge"]),
                      "formula": str(a.info["formula"]), "parent": str(a.info["parent_record_id"]), "trajectory": t,
                      "step": items[i][0], "last": int(i == len(items) - 1), "record_id": str(a.info["record_id"])}
            b.arrays["REF_forces"] = np.array(a.arrays["REF_forces"])
            frames.append(b)
    # merge parents that share a geometry
    root: dict[str, str] = {}

    def find(x: str) -> str:
        root.setdefault(x, x)
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    seen: dict[tuple, str] = {}
    for b in frames:
        fp, p = fingerprint(b), find(b.info["parent"])
        if fp in seen:
            root[find(seen[fp])] = p
        else:
            seen[fp] = b.info["parent"]
    for b in frames:
        b.info["group"] = find(b.info["parent"])
    by_formula: dict[str, list[str]] = defaultdict(list)
    for g, f in sorted({(b.info["group"], b.info["formula"]) for b in frames}):
        by_formula[f].append(g)
    kind: dict[str, str] = {}
    for f, groups in sorted(by_formula.items()):
        groups = list(rng.permutation(groups))
        if f == args.held_formula:
            kind.update({g: "test_formula" for g in groups})
            continue
        n_test = max(1, round(0.15 * len(groups)))
        for i, g in enumerate(groups):
            kind[g] = "test_parents" if i < n_test else "train"
    # inside the training groups whole trajectories go to validation (5%) and to an easier test set (10%)
    tkind: dict[str, str] = {}
    for t in rng.permutation(sorted({b.info["trajectory"] for b in frames if kind[b.info["group"]] == "train"})):
        u = rng.random()
        tkind[str(t)] = "valid" if u < 0.05 else "test_trajectories" if u < 0.15 else "train"
    split = {"train": [], "valid": [], "test": []}
    for b in frames:
        k = kind[b.info["group"]]
        k = tkind[b.info["trajectory"]] if k == "train" else k
        b.info["test_kind"] = k
        split["test" if k.startswith("test") else k].append(b)
    d = args.work / "fold_0"
    d.mkdir(parents=True, exist_ok=True)
    for name, fr in split.items():
        write(d / f"{name}.extxyz", fr, format="extxyz")
    plan = {"source": str(args.data), "dropped": dict(dropped), "trajectories": len(traj), "kept_frames": len(frames),
            "parents": len({b.info["parent"] for b in frames}), "groups": len(kind), "held_formula": args.held_formula,
            "stride": args.stride, "frames": {n: len(f) for n, f in split.items()},
            "test_frames_by_kind": dict(Counter(b.info["test_kind"] for b in split["test"])),
            "groups_by_kind": dict(Counter(kind.values())),
            "train_by_formula": dict(Counter(b.info["formula"] for b in split["train"])),
            "train_by_multiplicity": {str(k): v for k, v in sorted(Counter(b.info["multiplicity"] for b in split["train"]).items())},
            "folds": [{"fold": 0}]}
    (args.work / "folds.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=1))


def rmse(x) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def score(args) -> None:
    from mace.calculators import MACECalculator

    d = args.work / "fold_0"
    test = read(d / "test.extxyz", index=":")
    train = read(d / "train.extxyz", index=":")
    for a in [train[i] for i in np.random.default_rng(0).choice(len(train), size=min(500, len(train)), replace=False)]:
        a.info["test_kind"] = "train_sample"  # same metrics on frames the model was fitted to
        test.append(a)
    report: dict = {"plan": json.loads((args.work / "folds.json").read_text(encoding="utf-8")), "variants": {}}
    for variant in ("spin", "blind"):
        models = [m for m in sorted((d / variant).glob(f"{variant}_fold0*.model")) if "compiled" not in m.name and "stagetwo" not in m.name]
        if not models:
            continue
        calc = MACECalculator(model_paths=str(models[0]), device=args.device, default_dtype="float64")
        pred_e, pred_f = [], []
        for a in test:
            b = Atoms(a.get_chemical_symbols(), positions=a.positions)
            b.info = {"spin": int(a.info["multiplicity"]), "charge": int(a.info["charge"])}
            b.calc = calc
            pred_e.append(float(b.get_potential_energy())); pred_f.append(np.array(b.get_forces()))
        del calc
        entry: dict = {}
        for kind in ("train_sample", "test_trajectories", "test_parents", "test_formula"):
            idx = [i for i, a in enumerate(test) if a.info["test_kind"] == kind]
            if not idx:
                continue
            de = np.array([(pred_e[i] - test[i].info["REF_energy"]) / len(test[i]) for i in idx])
            df = np.concatenate([(pred_f[i] - test[i].arrays["REF_forces"]).ravel() for i in idx])
            f0 = np.concatenate([test[i].arrays["REF_forces"].ravel() for i in idx])
            # lowest-energy reference frame of every (parent, charge, M); gaps between neighbouring M
            best: dict[tuple, dict[int, int]] = defaultdict(dict)
            for i in idx:
                a = test[i]
                key, m = (a.info["parent"], a.info["formula"], int(a.info["charge"])), int(a.info["multiplicity"])
                if m not in best[key] or a.info["REF_energy"] < test[best[key][m]].info["REF_energy"]:
                    best[key][m] = i
            rows, ground = [], []
            for key, by_m in sorted(best.items()):
                ms = sorted(by_m)
                if len(ms) < 2:
                    continue
                ref = {m: test[by_m[m]].info["REF_energy"] for m in ms}
                got = {m: pred_e[by_m[m]] for m in ms}
                ground.append({"parent": key[0], "formula": key[1], "charge": key[2], "n_M": len(ms),
                               "ref_lowest_M": min(ms, key=ref.get), "pred_lowest_M": min(ms, key=got.get)})
                for lo, hi in zip(ms[:-1], ms[1:]):
                    rows.append({"parent": key[0], "formula": key[1], "charge": key[2], "gap": f"{hi}-{lo}",
                                 "ref_ev": ref[hi] - ref[lo], "pred_ev": got[hi] - got[lo]})
            r, p = np.array([x["ref_ev"] for x in rows]), np.array([x["pred_ev"] for x in rows])
            entry[kind] = {
                "frames": len(idx), "energy_mae_mev_per_atom": float(1000 * np.abs(de).mean()),
                "energy_rmse_mev_per_atom": float(1000 * rmse(de)), "force_rmse_ev_per_A": rmse(df), "force_zero_baseline_ev_per_A": rmse(f0),
                "spin_gaps": {"n": len(rows), "ref_abs_mean_ev": float(np.abs(r).mean()) if len(rows) else None,
                              "mae_ev": float(np.abs(p - r).mean()) if len(rows) else None,
                              "median_abs_error_ev": float(np.median(np.abs(p - r))) if len(rows) else None,
                              "sign_agreement": int((np.sign(p) == np.sign(r)).sum()),
                              "by_formula": {f: {"n": int(sum(x["formula"] == f for x in rows)),
                                                 "mae_ev": float(np.mean([abs(x["pred_ev"] - x["ref_ev"]) for x in rows if x["formula"] == f])),
                                                 "ref_abs_mean_ev": float(np.mean([abs(x["ref_ev"]) for x in rows if x["formula"] == f]))}
                                             for f in sorted({x["formula"] for x in rows})}},
                "lowest_M": {"units": len(ground), "same_as_reference": int(sum(g["ref_lowest_M"] == g["pred_lowest_M"] for g in ground))},
                "gap_rows": rows, "lowest_M_rows": ground}
        report["variants"][variant] = entry
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for v, e in report["variants"].items():
        for kind, x in e.items():
            g, lm = x["spin_gaps"], x["lowest_M"]
            print(f"== {v} / {kind}: {x['frames']} frames; E MAE {x['energy_mae_mev_per_atom']:.1f} meV/atom; "
                  f"F RMSE {x['force_rmse_ev_per_A']:.3f} (zero {x['force_zero_baseline_ev_per_A']:.3f}) eV/A")
            if g["n"]:
                print(f"   spin gaps n={g['n']}: MAE {g['mae_ev']:.3f} eV, median {g['median_abs_error_ev']:.3f} "
                      f"(mean |ref| {g['ref_abs_mean_ev']:.3f}); sign {g['sign_agreement']}/{g['n']}; "
                      f"lowest M right in {lm['same_as_reference']}/{lm['units']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("data", type=Path); p.add_argument("work", type=Path)
    p.add_argument("--held-formula", default="Fe2H2O3"); p.add_argument("--stride", type=int, default=3)
    p.add_argument("--seed", type=int, default=11); p.add_argument("--fmax", type=float, default=10.0)
    p = sub.add_parser("score")
    p.add_argument("work", type=Path); p.add_argument("--out", type=Path, required=True); p.add_argument("--device", default="cuda")
    args = ap.parse_args()
    {"prepare": prepare, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
