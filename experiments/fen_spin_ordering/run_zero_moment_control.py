"""Zero-moment control for run_magpt_ordering.py: MagPT energy of every clean state with all moments set to 0.

States in one (n, q) group sit at different geometries, so MagPT can order them by structure alone. This pass
measures that spin-blind ordering; the analysis compares it with the ordering from the real moments.
Resumable like the main run; writes magpt_zero_moment.csv next to magpt_states.csv.

    D:/MLIP_Work_Folder/envs/magpt_dp/python.exe experiments/fen_spin_ordering/run_zero_moment_control.py \
        --states D:/MLIP_Work_Folder/Cluster_MLIP/fen_oct9_states/states.extxyz \
        --checkpoint D:/MLIP_Work_Folder/external/magpt/MagPT/model/MagPT_399168.pth \
        --out experiments/fen_spin_ordering/report
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cluster_mlip.magpt_diagnostic import DeepSpinEvaluator  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    import torch
    from ase.io import read

    torch.set_num_threads(args.threads)
    path = Path(args.out) / "magpt_zero_moment.csv"
    done = set()
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            done = {r["state_id"] for r in csv.DictReader(fh)}
    model = DeepSpinEvaluator(args.checkpoint)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if not done:
            w.writerow(["state_id", "magpt_zero_ev"])
        for a in read(args.states, ":"):
            if not a.info["clean"] or a.info["state_id"] in done:
                continue
            e, *_ = model(a.positions, None, np.zeros((len(a), 3)), a.get_chemical_symbols())
            w.writerow([a.info["state_id"], e])
            fh.flush()
    print("done")


if __name__ == "__main__":
    main()
