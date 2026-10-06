"""Evaluate every finished A/B/C model frame by frame on the 60 training frames and the
90 held-out labelled frames.

Also a sensitivity control for C: the same frames with each frame's local moments
cyclically shifted over atoms (same values, same sum, wrong sites). If C uses s_i,
its errors must grow. The calculator is reset before every call (ASE's cache ignores
custom per-atom arrays).

Writes results/predictions.csv and results/pair_predictions.csv.
"""
from __future__ import annotations

import csv
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

from common import OUT, RESULTS, aligned_match, meta


def main() -> None:
    sets = {"train": read(OUT / "data" / "train.extxyz", ":"),
            "heldout": read(OUT / "data" / "heldout.extxyz", ":")}
    rows = []
    preds = {}
    for run in sorted((OUT / "runs").iterdir()):
        model = run / f"{run.name}.model"
        if not model.is_file() or run.name.startswith("aborted"):
            continue
        regime, letter, seed = run.name.split("_")
        local = letter == "C"
        calc = MACECalculator(model_paths=str(model), device="cuda", default_dtype="float64",
                              arrays_keys={"local_moment": "local_moment"} if local else None)
        variants = ["true", "shifted"] if local else ["true"]
        for split, frames in sets.items():
            for variant in variants:
                for k, a in enumerate(frames):
                    b = a.copy()
                    if variant == "shifted":
                        b.arrays["local_moment"] = np.roll(a.arrays["local_moment"], 5)
                    calc.reset()
                    b.calc = calc
                    e, f = b.get_potential_energy(), b.get_forces()
                    ref = a.arrays["REF_forces"]
                    s = a.arrays["local_moment"]
                    if variant == "true":
                        preds[(run.name, split, k)] = (e, f)
                    rows.append({
                        "run": run.name, "regime": regime, "model": letter, "seed": seed,
                        "split": split, "variant": variant, "k": k,
                        "record_id": a.info["record_id"], "M": int(a.info["multiplicity"]),
                        "tag": a.info.get("subset_tag", ""), "group": meta(a).get("split_group"),
                        "n_negative": int((s < 0).sum()), "n_weak": int((np.abs(s) < 1).sum()),
                        "position": meta(a).get("stage_frame_position"),
                        "dE_meV_atom": 1000 * (e - a.info["REF_energy"]) / len(a),
                        "F_mae_meV_A": 1000 * float(np.abs(f - ref).mean()),
                        "F_rmse_meV_A": 1000 * float(np.sqrt(((f - ref) ** 2).mean())),
                        "F_ref_rms_meV_A": 1000 * float(np.sqrt((ref ** 2).mean())),
                    })
        print("evaluated", run.name, flush=True)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # Step 6: the closest different-pattern same-(q, M) pairs inside the training set.
    tr = sets["train"]
    pair_rows = []
    for i in range(len(tr)):
        for j in range(i + 1, len(tr)):
            if tr[i].info["multiplicity"] != tr[j].info["multiplicity"]:
                continue
            if not (tr[i].info.get("subset_tag") == tr[j].info.get("subset_tag") == "diff_spin_pair"):
                continue
            rmsd, perm, R, *_ = aligned_match(tr[i].positions, tr[j].positions)
            si, sj = tr[i].arrays["local_moment"], tr[j].arrays["local_moment"][perm]
            d_s = float(np.sqrt(((si - sj) ** 2).mean()))
            if d_s < 1.0:
                continue
            Fi, Fj = tr[i].arrays["REF_forces"], tr[j].arrays["REF_forces"][perm] @ R
            dft = Fi - Fj
            for run in sorted({r[0] for r in preds}):
                if (run, "train", i) not in preds:
                    continue
                pi = preds[(run, "train", i)][1]
                pj = preds[(run, "train", j)][1][perm] @ R
                mdl = pi - pj
                pair_rows.append({
                    "run": run, "i": i, "j": j, "M": int(tr[i].info["multiplicity"]),
                    "rmsd_A": rmsd, "D_s": d_s,
                    "D_F_dft_meV_A": 1000 * float(np.sqrt((dft ** 2).mean())),
                    "D_F_model_meV_A": 1000 * float(np.sqrt((mdl ** 2).mean())),
                    "diff_error_meV_A": 1000 * float(np.sqrt(((mdl - dft) ** 2).mean())),
                    "dE_dft_meV": 1000 * (tr[i].info["REF_energy"] - tr[j].info["REF_energy"]),
                    "dE_model_meV": 1000 * (preds[(run, "train", i)][0] - preds[(run, "train", j)][0]),
                })
    if pair_rows:
        with open(RESULTS / "pair_predictions.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pair_rows[0]))
            w.writeheader(); w.writerows(pair_rows)


if __name__ == "__main__":
    main()
