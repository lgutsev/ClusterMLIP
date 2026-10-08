"""Steps 4-5: controlled A/B/C memorization runs in MACE's own training loop.

All runs share data, seed, architecture, optimizer, loss weights, epochs and cutoff.
Only the inputs differ:

  A  R + Z                                   (no embedding block)
  B  R + Z + total charge + multiplicity     (ClusterMLIP's current graph-level embedding)
  C  B + per-atom signed Mulliken spin       (per-atom continuous embedding)

MACE adds the joint embedding to node_feats *before* the first interaction block
(mace/modules/models.py, ScaleShiftMACE.forward), so in C the local spin is part of
h_i^(0) and enters every message.

Regimes: `fixed` = the repaired from-scratch defaults (lr 1e-3, forces_weight 10);
`ref` = the settings of the earlier failed 60-frame overfit (lr 5e-3, forces_weight 100,
60 epochs). train_file == valid_file: this is a memorization test.

    python s5_run_abc.py write     # write run scripts
    bash $OUT/runs/run_all.sh       # run sequentially
    bash $OUT/runs/run_long_{0,1}.sh  # 800-epoch B/C follow-up, two queues
"""
from __future__ import annotations

import json
import sys

from common import OUT

MACE = "C:/Users/lguts/micromamba/envs/defects/Scripts/mace_run_train.exe"

GRAPH = {
    "total_spin": {"type": "categorical", "per": "graph", "key": "spin", "in_dim": 1,
                   "emb_dim": 128, "num_classes": 101, "offset": 0},
    "total_charge": {"type": "categorical", "per": "graph", "key": "charge", "in_dim": 1,
                     "emb_dim": 128, "num_classes": 201, "offset": 100},
}
LOCAL = {"local_moment": {"type": "continuous", "per": "atom", "key": "local_moment",
                          "in_dim": 1, "emb_dim": 32}}
MODELS = {"A": None, "B": GRAPH, "C": {**GRAPH, **LOCAL}}
REGIMES = {
    "fixed": {"lr": 0.001, "forces_weight": 10.0, "epochs": 400, "seeds": (1, 2)},
    "ref": {"lr": 0.005, "forces_weight": 100.0, "epochs": 60, "seeds": (1,)},
    # review follow-up: B and C were still improving at epoch 400
    "long": {"lr": 0.001, "forces_weight": 10.0, "epochs": 800, "seeds": (1, 2), "models": ("B", "C")},
}


def argv(model: str, regime: str, seed: int, name: str) -> list[str]:
    r = REGIMES[regime]
    data = (OUT / "data" / "train.extxyz").as_posix()
    a = [
        f"--name={name}", "--model=ScaleShiftMACE",
        f"--train_file={data}", f"--valid_file={data}",
        "--energy_key=REF_energy", "--forces_key=REF_forces",
        "--total_charge_key=charge", "--total_spin_key=spin",
        "--E0s=average",
        "--interaction_first=RealAgnosticResidualNonLinearInteractionBlock",
        "--interaction=RealAgnosticResidualNonLinearInteractionBlock",
        "--num_interactions=2", "--correlation=3", "--max_ell=3", "--r_max=6.0",
        "--num_radial_basis=8", "--hidden_irreps=128x0e + 128x1o + 128x2e", "--MLP_irreps=16x0e",
        "--scaling=rms_forces_scaling", "--loss=weighted", "--energy_weight=1.0",
        f"--forces_weight={r['forces_weight']}", "--stress_weight=0", f"--lr={r['lr']}",
        "--batch_size=8", "--valid_batch_size=8", "--ema", "--ema_decay=0.999",
        "--default_dtype=float64", "--device=cuda", f"--seed={seed}",
        f"--max_num_epochs={r['epochs']}", "--patience=100000", "--eval_interval=5",
        "--error_table=PerAtomMAE",
    ]
    specs = MODELS[model]
    if specs is not None:
        a += [f"--embedding_specs={json.dumps(specs, separators=(',', ':'))}",
              "--use_embedding_readout=True"]
    return a


def write() -> None:
    runs = OUT / "runs"
    lines = ["#!/usr/bin/env bash", "set -u", f'cd "{runs.as_posix()}"']
    for regime, r in REGIMES.items():
        for seed in r["seeds"]:
            for model in r.get("models", MODELS):
                name = f"{regime}_{model}_s{seed}"
                d = runs / name
                d.mkdir(parents=True, exist_ok=True)
                args = argv(model, regime, seed, name)
                (d / "argv.json").write_text(json.dumps(args, indent=1))
                quoted = " ".join("'" + x.replace("'", "'\\''") + "'" for x in args)
                (d / "run.sh").write_text(
                    f'#!/usr/bin/env bash\ncd "$(dirname -- "${{BASH_SOURCE[0]}}")"\n'
                    f'"{MACE}" {quoted} > stdout.txt 2>&1\necho $? > exit_code\n')
                lines.append(f'[ -f {name}/exit_code ] || bash {name}/run.sh')
    # cheap reference-failure runs first, then the long ones
    head, body = lines[:3], lines[3:]
    body.sort(key=lambda s: 0 if "ref_" in s else 1)
    (runs / "run_all.sh").write_text("\n".join(head + [b for b in body if "long_" not in b]) + "\n")
    # 800-epoch follow-up: two parallel queues (GPU otherwise idle)
    longs = [b for b in body if "long_" in b]
    for q in (0, 1):
        (runs / f"run_long_{q}.sh").write_text("\n".join(head + longs[q::2]) + "\n")
    print((runs / "run_long_0.sh").read_text(), (runs / "run_long_1.sh").read_text())


if __name__ == "__main__":
    if sys.argv[1:] == ["write"]:
        write()
