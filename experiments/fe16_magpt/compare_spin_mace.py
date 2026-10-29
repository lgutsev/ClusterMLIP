"""Same fixed-geometry Fe16 test as run_magpt_diagnostic.py, but for ClusterMLIP's own spin-aware MACE models.

Models are the 800-epoch runs of the 2026-10-05 local-spin experiment (branch test/fe16-local-spin-message-passing):
  B = R + Z + total charge + multiplicity (graph-level embedding)
  C = B + per-atom signed Mulliken spin ("local_moment", per-atom continuous embedding)
They were trained on Fe16 UBPW91 frames, so unlike MagPT this is an in-distribution reference, not zero-shot: each
state is tagged train / heldout by matching it against that run's training and held-out sets.

    python experiments/fe16_magpt/compare_spin_mace.py \
        --dataset D:/MLIP_Work_Folder/Cluster_MLIP/fe16_v1/02_fe16_dataset/all.extxyz \
        --runs D:/MLIP_Work_Folder/Cluster_MLIP/fe16_v1/12_local_spin_mp \
        --out experiments/fe16_magpt/report/spin_mace
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster_mlip.magpt_diagnostic import force_metrics, load_states_from_extxyz, spearman  # noqa: E402
from run_magpt_diagnostic import select_groups  # noqa: E402

RUNS = ("long_B_s1", "long_B_s2", "long_C_s1", "long_C_s2")
SCALES = (0.8, 0.9, 1.0, 1.1, 1.2)


def make_atoms(state, local_spins):
    from ase import Atoms

    a = Atoms(state.symbols, positions=state.positions)
    a.info["spin"] = state.multiplicity  # the models' total_spin key holds the multiplicity
    a.info["charge"] = state.charge
    a.arrays["local_moment"] = np.asarray(local_spins, dtype=float)
    return a


def split_of(state, train, heldout) -> str:
    for name, pool in (("train", train), ("heldout", heldout)):
        for b in pool:
            if len(b) == len(state.symbols) and int(b.info["multiplicity"]) == state.multiplicity \
                    and float(np.abs(b.positions - state.positions).max()) < 1e-3:
                return name
    return "unseen"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--runs", required=True, help="12_local_spin_mp folder (data/ and runs/)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    warnings.filterwarnings("ignore")
    import torch
    from ase.io import read
    from mace.calculators import MACECalculator

    torch.set_num_threads(1)
    root = Path(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = select_groups(load_states_from_extxyz(args.dataset))
    train = read(root / "data" / "train.extxyz", ":")
    heldout = read(root / "data" / "heldout.extxyz", ":")
    split = {s.state_id: split_of(s, train, heldout) for g in groups for s in g}

    conditions = [("true", 1.0, "")] + [(f"scale{s:g}", s, "") for s in SCALES if s != 1.0]
    conditions += [("reversed", 1.0, "reverse"), ("shifted", 1.0, "shift")]
    rows, pairs = [], []
    for run in RUNS:
        letter = run.split("_")[1]
        local = letter == "C"
        calc = MACECalculator(model_paths=str(root / "runs" / run / f"{run}.model"), device="cpu",
                              default_dtype="float64", arrays_keys={"local_moment": "local_moment"} if local else None)
        pred = {}
        for g in groups:
            for s in g:
                for cond, scale, ctrl in conditions if local else conditions[:1]:
                    spins = scale * s.local_spins
                    if ctrl == "reverse":
                        spins = -s.local_spins
                    elif ctrl == "shift":
                        spins = np.roll(s.local_spins, 5)  # same values and sum, wrong sites
                    a = make_atoms(s, spins)
                    calc.reset()  # ASE's cache ignores custom per-atom arrays
                    a.calc = calc
                    e, f = float(a.get_potential_energy()), np.array(a.get_forces())
                    pred[(s.state_id, cond)] = (e, f)
                    if cond == "true":
                        fm = force_metrics(s.forces_ev_ang, f)
                        rows.append({"run": run, "model": letter, "geometry_id": s.geometry_id, "state_id": s.state_id,
                                     "multiplicity": s.multiplicity, "in_training": split[s.state_id],
                                     "gaussian_energy_ev": s.energy_ev, "mace_energy_ev": e,
                                     "energy_error_mev_atom": 1000 * (e - s.energy_ev) / len(s.symbols),
                                     "force_mae_ev_ang": fm["mae"], "force_cosine": fm["cosine"],
                                     "gaussian_force_rms_ev_ang": fm["ref_rms"], "mace_force_rms_ev_ang": fm["pred_rms"]})
        for g in groups:
            a, b = g[0], g[1]
            dg = b.energy_ev - a.energy_ev
            dm = {c: pred[(b.state_id, c)][0] - pred[(a.state_id, c)][0] for c, *_ in (conditions if local else conditions[:1])}
            dF_g = b.forces_ev_ang - a.forces_ev_ang
            dF_m = pred[(b.state_id, "true")][1] - pred[(a.state_id, "true")][1]
            pairs.append({"run": run, "model": letter, "geometry_id": a.geometry_id,
                          "step": f"M{a.multiplicity}->M{b.multiplicity}",
                          "in_training": f"{split[a.state_id]}/{split[b.state_id]}",
                          "gaussian_dE_ev": dg, "mace_dE_ev": dm["true"],
                          "sign_agrees": bool(np.sign(dg) == np.sign(dm["true"])),
                          "dE_error_ev": dm["true"] - dg,
                          "dE_by_scale_ev": json.dumps({c: round(dm[c], 5) for c in ("scale0.8", "scale0.9", "true", "scale1.1", "scale1.2")}) if local else "",
                          "dE_reversed_ev": dm.get("reversed", ""), "dE_shifted_ev": dm.get("shifted", ""),
                          "time_reversal_max_abs_diff_ev": max(abs(pred[(s.state_id, "reversed")][0] - pred[(s.state_id, "true")][0]) for s in (a, b)) if local else "",
                          "gaussian_dF_rms_ev_ang": float(np.sqrt(np.mean(dF_g**2))),
                          "mace_dF_rms_ev_ang": float(np.sqrt(np.mean(dF_m**2))),
                          "dF_cosine": force_metrics(dF_g, dF_m)["cosine"]})
        print("evaluated", run, flush=True)

    for name, data in (("states.csv", rows), ("pairs.csv", pairs)):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    summary = {}
    for run in RUNS:
        pr = [p for p in pairs if p["run"] == run]
        st = [r for r in rows if r["run"] == run]
        g = [p["gaussian_dE_ev"] for p in pr]
        m = [p["mace_dE_ev"] for p in pr]
        n_agree = sum(p["sign_agrees"] for p in pr)
        n = len(pr)
        summary[run] = {
            "pairs_sign_agree": n_agree, "n_pairs": n,
            "sign_agreement_binomial_p": sum(math.comb(n, k) for k in range(n_agree, n + 1)) / 2**n,
            "spearman_dE": spearman(g, m), "pearson_dE": float(np.corrcoef(g, m)[0, 1]),
            "dE_mae_mev": 1000 * float(np.mean(np.abs(np.array(m) - np.array(g)))),
            "dE_mae_mev_pairs_fully_in_train": 1000 * float(np.mean([abs(p["dE_error_ev"]) for p in pr if p["in_training"] == "train/train"] or [math.nan])),
            "dE_mae_mev_pairs_with_heldout_state": 1000 * float(np.mean([abs(p["dE_error_ev"]) for p in pr if p["in_training"] != "train/train"] or [math.nan])),
            "state_force_mae_mev_ang": 1000 * float(np.mean([r["force_mae_ev_ang"] for r in st])),
            "dF_cosine_median": float(np.median([p["dF_cosine"] for p in pr])),
        }
        if run.split("_")[1] == "C":
            summary[run]["max_time_reversal_diff_ev"] = max(p["time_reversal_max_abs_diff_ev"] for p in pr)
            summary[run]["pairs_sign_stable_scale_0.8_1.2"] = sum(
                len({np.sign(v) for v in json.loads(p["dE_by_scale_ev"]).values()}) == 1 for p in pr)
            summary[run]["max_dE_shift_vs_true_ev"] = max(abs(p["dE_shifted_ev"] - p["mace_dE_ev"]) for p in pr)
    summary["state_training_split"] = split
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "state_training_split"}, indent=2, default=float))


if __name__ == "__main__":
    main()
