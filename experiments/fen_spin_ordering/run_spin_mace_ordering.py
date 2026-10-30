"""Same Fe_n ordering test as run_magpt_ordering.py, for ClusterMLIP's spin-aware Fe16 MACE models B and C.

    C:/Users/lguts/micromamba/envs/defects/python.exe experiments/fen_spin_ordering/run_spin_mace_ordering.py \
        --states D:/MLIP_Work_Folder/Cluster_MLIP/fen_oct9_states/states.extxyz \
        --runs D:/MLIP_Work_Folder/Cluster_MLIP/fe16_v1/12_local_spin_mp \
        --out experiments/fen_spin_ordering/report

The models (800-epoch runs of the 2026-10-05 local-spin experiment) were trained on 60 neutral Fe16 frames at
M = 47-53, BPW91/6-311++G(d). Here they see every clean Oct9 state (Fe2-Fe25, q = -1/0/+1, many M), so outside
neutral Fe16 at M 47-53 this is a transfer test, not an in-distribution one. Each state is tagged with whether its
(n, q, M) lies in that training range and whether its geometry matches a training frame (sorted-distance RMS).
  B = R + Z + total charge + multiplicity            -> condition "true" only (no per-atom input)
  C = B + per-atom signed Mulliken spin               -> "true", "uniform" ((M-1)/n per atom), "scale0.9", "scale1.1"
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_ordering import fingerprint  # noqa: E402

RUNS = ("long_B_s1", "long_B_s2", "long_C_s1", "long_C_s2")
C_CONDITIONS = ("true", "uniform", "scale0.9", "scale1.1")
TRAIN_MATCH_RMS = 0.02


def spins_for(cond: str, spins: np.ndarray, multiplicity: int) -> np.ndarray:
    if cond == "true":
        return spins
    if cond == "uniform":
        return np.full(len(spins), (multiplicity - 1) / len(spins))
    return spins * float(cond.removeprefix("scale"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", required=True)
    ap.add_argument("--runs", required=True, help="12_local_spin_mp folder (data/ and runs/)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    import torch
    from ase.io import read
    from mace.calculators import MACECalculator

    torch.set_num_threads(4)
    root = Path(args.runs)
    states = [a for a in read(args.states, ":") if a.info["clean"]]
    train = read(root / "data" / "train.extxyz", ":")
    train_fp = [(len(t), int(t.info["multiplicity"]), fingerprint(t.positions)) for t in train]
    train_range = {(len(t), int(t.info["charge"]), int(t.info["multiplicity"])) for t in train}

    rows = {a.info["state_id"]: {"state_id": a.info["state_id"], "n": len(a), "charge": int(a.info["charge"]),
                                 "multiplicity": int(a.info["multiplicity"]),
                                 "in_training_range": (len(a), int(a.info["charge"]), int(a.info["multiplicity"])) in train_range,
                                 "geometry_in_training": any(n == len(a) and float(np.sqrt(np.mean((fp - fingerprint(a.positions)) ** 2))) < TRAIN_MATCH_RMS
                                                             for n, _, fp in train_fp)}
            for a in states}
    for run in RUNS:
        local = run.split("_")[1] == "C"
        calc = MACECalculator(model_paths=str(root / "runs" / run / f"{run}.model"), device="cpu",
                              default_dtype="float64", arrays_keys={"local_moment": "local_moment"} if local else None)
        for a in states:
            spins = np.asarray(a.arrays["mulliken_spin"], float)
            for cond in C_CONDITIONS if local else ("true",):
                b = a.copy()
                b.info = {"spin": int(a.info["multiplicity"]), "charge": int(a.info["charge"])}
                b.arrays["local_moment"] = spins_for(cond, spins, int(a.info["multiplicity"]))
                calc.reset()  # ASE's cache ignores custom per-atom arrays
                b.calc = calc
                rows[a.info["state_id"]][f"{run}_{cond}_ev"] = float(b.get_potential_energy())
        print("evaluated", run, flush=True)

    out = Path(args.out)
    data = list(rows.values())
    with open(out / "spin_mace_states.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    print("wrote", len(data), "states")


if __name__ == "__main__":
    main()
