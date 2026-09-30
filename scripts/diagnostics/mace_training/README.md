# MACE training diagnostics (Fe16 baseline, 2026-09-30)

Why the first from-scratch Fe16 MACE (the `cluster-mlip train` defaults) learned nothing,
tested one variable at a time. These scripts drive the real MACE model outside MACE's
training loop, so each factor (loss terms, learning rate, architecture, frame count) can be
isolated. They need `mace-torch` (0.3.16 here) and `torch`, not just the base install.

| Script | What it does |
|---|---|
| `common.py` | load a `.model`, build batches exactly as MACE does (`config_from_atoms` + `AtomicData.from_config`) |
| `memorize.py` | full-batch Adam(amsgrad) with MACE's own parameter groups and weight decays; energy-only / force-only / combined loss; logs weighted loss terms, gradient norms by group, parameter change, per-frame force error. `CHUNK=n` accumulates gradients over chunks (exactly the full-batch gradient: equal atoms per frame), for GPUs that cannot hold the whole batch |
| `fd_check.py` | model forces vs central finite differences of the model's own energy |
| `label_conflicts.py` | force change per unit displacement for cross-stage same-M near-identical pairs (aligned by rotation + atom permutation) vs consecutive optimisation steps |
| `summarize.py` | table of `logs/n*.jsonl` |

## Getting an untrained model with the run's exact statistics

`mace_run_train` with the same arguments plus `--max_num_epochs=1 --lr=0.0` writes a `.model`
whose weights are the initial ones (the epoch-0 loss equals the initial loss exactly).
Zero epochs does not work: MACE then tries to load a checkpoint that was never written.

    python memorize.py init.model sub.extxyz 14 F 400 0.001 none          # 1 frame, force-only
    CHUNK=10 python -u memorize.py init.model sub.extxyz all EF 400 0.001 none

Use `python -u`: when stdout is a file, buffering otherwise hides all progress until exit.

## Findings (Fe16 v1, dataset `02d_fe16_tol05`, 60-frame training subset)

Verified correct: batch energies/forces/positions equal the extxyz; `total_spin` = Gaussian
multiplicity; all 1,358,102 parameters trainable and in exactly one optimizer group; forces
equal −dE/dx (finite differences agree to 4e-7 eV/Å); labels physically consistent
(net torque ≈ 0; dE vs −F·dx along optimisations: corr 0.998, slope 0.99).

Loss in mace-torch 0.3.16 (`WeightedEnergyForcesLoss`): mean over graphs of ((E_ref−E)/N)²
× `energy_weight` + mean over force components of ΔF² × `forces_weight`.

Force-only memorization, 400 full-batch steps (meV/Å):

| frames | lr 0.005 | lr 0.001 | lr 0.0001 | lr 0.001, standard interaction blocks |
|---:|---:|---:|---:|---:|
| 1 | 153 → 42.5 | 153 → **0.37** | 153 → 2.2 | |
| 5 | 122 → 95.5 | 122 → **10.4** | | |
| 15 | | 71 → 19.2 | | |
| 30 | | 54 → 43.2 | | |
| 60 | | 45.8 → 41.2 (oscillating) | 45.8 → 30.5 (monotonic) | 45.3 → **25.4** (monotonic) |

Energy-only memorization succeeds at both learning rates (60 frames: 3.2 / 4.8 meV/atom).

- **In this harness** (full batch, force-only loss): `--lr=0.005` overshoots even on one
  frame, and the `RealAgnosticResidualNonLinearInteractionBlock` fits 60 frames more slowly
  than MACE's standard `RealAgnosticInteractionBlock` + `RealAgnosticResidualInteractionBlock`.
- **In MACE's own loop** (batch 8, combined loss, 60 frames, 500 epochs) the ranking differs:
  every configuration sits on a ~0.2 loss plateau for ~50–75 epochs (400–600 steps) and then
  descends; by epoch ~120–150 the kit blocks at lr 0.001 (0.081) and the standard blocks at
  lr 0.005 (0.088) lead, the standard blocks at lr 0.001 lag (0.166). The harness ranking
  therefore does not transfer, and the block choice is **not** established as a cause. The
  unchanged kit configuration over the same budget is the control (see below).
- **Learning rate, confirmed in MACE's loop** (kit blocks, only `--lr` differs; raw-model
  training loss, 60 frames, batch 8):

  | epoch | 75 | 150 | 225 | 300 |
  |---|---:|---:|---:|---:|
  | `--lr=0.005` (kit default) | 0.199 | 0.195 | 0.167 | 0.145 |
  | `--lr=0.001` | 0.165 | 0.076 | 0.031 | 0.015 |

  At 0.005 the model stays on the ~0.2 plateau (≈ predicting zero force) for ~2,400 steps and
  then only creeps; at 0.001 it leaves the plateau after ~600 steps. The full-data run that
  looked stuck was stopped after 15 epochs (≈1,575 steps), still on the plateau.
  After 500 epochs at 0.001 the 60 training frames are fit to 7.5 meV/Å (16 % relative) but
  only 31 meV/atom in energy — forces are learnable; energy is the next problem.
- **Energy: the loss weighting.** Same run (lr 0.001, 500 epochs), one change each, errors
  on the 60 training frames:

  | change | E (meV/atom) | F (meV/Å) |
  |---|---:|---:|
  | none (`forces_weight` 100, embedding readout on) | 31.2 | 7.5 |
  | `--forces_weight=10` | **6.6** | 9.2 |
  | `--use_embedding_readout=False` | 24.2 | 8.8 |

  `cluster-mlip train` now defaults to lr 0.001 and `forces_weight` 10 from scratch.
- **Not the labels:** 8,869 cross-stage same-M near-identical training pairs follow the same
  |ΔF|/|Δx| distribution as consecutive steps (median 3.4 vs 4.6 eV/Å²); the only outliers
  are exact duplicate frames under two record IDs.
- Secondary: with `--use_embedding_readout=True`, the untrained model adds a
  geometry-independent, multiplicity-dependent energy of ~1.0–1.2 eV/atom that is not
  multiplied by the ScaleShift scale.
- MACE's printed validation errors come from the EMA weights (decay 0.999 ≈ 1000-step
  memory); on a small set with few steps per epoch they lag by hundreds of epochs. Judge
  memorization from the raw-model training loss in `results/*_train.txt` (`"mode": "opt"`).
