# Held-out spin tests

Question: does a from-scratch MACE with the total-spin input predict the energy as a function of the
multiplicity M for structures it never saw, and does it beat the same model without the spin input?

- `spin_heldout_test.py`: Fe16 and Fe16 + CO + 2H (20 fixed geometries, each at M = 49, 47, 45).
- `fenom_spin_test.py`: FenOm warehouse, Fe2 with O, H and N adsorbates (M = 1 to 11).

Both train the same two models per split: "spin" (categorical total-spin and total-charge embedding with
the embedding readout) and "blind" (no embedding). Reference: UBPW91.

## Fe16 + CO + 2H, 2026-11-18

Three folds, whole structure groups held out at every M; the bare Fe16 ladder (M = 49 to 39) always in
training. 120 epochs, one seed, laptop GPU. Scores on the 20 held-out panel geometries.

Spin gaps at fixed geometry (eV):

| gap | n | UBPW91 mean ± sd | spin model mean | spin model MAE | constant-gap MAE | zero-gap MAE (= blind) | sign right | rank correlation |
|---|---|---|---|---|---|---|---|---|
| E(47) − E(49) | 19 | +0.155 ± 0.130 | +0.300 | 0.213 | 0.102 | 0.170 | 17/19 | 0.13 |
| E(45) − E(47) | 20 | +0.389 ± 0.118 | +0.686 | 0.298 | 0.092 | 0.389 | 20/20 | 0.32 |

"Constant gap" is the mean gap of the fold's training structures, used for every held-out structure.

Relative energies between held-out structures at one M, and forces (eV/Å; zero-force baseline in brackets):

| M | spin: MAE (eV), pairs ordered | blind: MAE (eV), pairs ordered | spin: Fe / adsorbate forces | blind: Fe / adsorbate forces |
|---|---|---|---|---|
| 49 | 0.66, 53% | 0.60, 59% | 0.19 (0.27) / 0.79 (1.19) | 0.21 / 0.81 |
| 47 | 0.56, 69% | 0.58, 60% | 0.19 (0.21) / 0.81 (1.11) | 0.18 / 0.80 |
| 45 | 0.60, 53% | 0.59, 59% | 0.18 (0.21) / 0.82 (1.07) | 0.20 / 0.79 |

Reading:

- The spin input is read (forces change by 0.46 eV/Å between M = 49 and 47 at one geometry; the blind model
  changes by zero) and gives the right direction of both gaps.
- The gaps are about twice too large and do not follow the structure: the error is larger than that of one
  constant number per gap. The model has learned an offset per M, not how the gap depends on the geometry.
- Apart from the gaps the two models are the same within noise, and both are poor: relative energies
  0.6 eV with near-random ordering, Fe forces barely better than predicting zero. The fine-tuned foundation
  models reach 0.27 eV on the same panel (docs/acat_cluster_feasibility.md, section 19).
- So on about 200 frames the limit is the from-scratch model as a whole, not the spin input in particular.

Limits: 120 epochs (the best validation epoch was 10 to 80, so longer runs would not have helped), one
seed, validation sets of 6 to 36 frames, fold 0 trains on 113 frames only.

Files: `D:\MLIP_Work_Folder\Cluster_MLIP\spin_heldout_v1\` (`report.json`, `score.log`, `chain.log`).

## FenOm (Fe2 with O, H, N), 2026-11-18/19: run not usable

Source: `FenOm_Warehouse2/dataset_spin_v0/all.extxyz` (17,809 Fe2 frames; Fe16 and Fe16N2 left out; 25
frames above 10 eV/Å dropped). Thinned to 7,321 frames from 1,409 trajectories and 473 parents. Merging
parents that share a geometry leaves 86 groups: 54 train (5,077 frames), 12 validation (949),
12 held-out groups (764) and the 8 groups of the held-out formula Fe2H2O3 (531). 80 epochs, batch 32.

**Neither model learned anything that transfers.** The validation loss was lowest after the first epoch
and rose afterwards (spin: force RMSE 0.29 → 0.37 eV/Å against a zero-force value of 0.32), so the saved
models are the first-epoch states. Their test scores are those of an untrained model and say nothing
about the spin input:

| test set | variant | energy MAE (meV/atom) | force RMSE (zero-force) eV/Å | gap MAE (eV), mean abs. gap 0.30 | lowest M right |
|---|---|---|---|---|---|
| held-out groups | spin | 161 | 0.311 (0.320) | 0.46 | 15/39 |
| held-out groups | blind | 202 | 0.318 (0.320) | 0.30 | 12/39 |
| held-out formula | spin | 60 | 0.208 (0.219) | 0.19 | 17/19 |
| held-out formula | blind | 52 | 0.218 (0.219) | 0.31 | 15/19 |

What is known about the cause (data checks only, no further training):

- Only 54 structure families are in training, and validation and test are other families. The training
  error was not logged, so "fits the training families and does not transfer" and "did not fit at all"
  cannot be told apart from this run.
- The set is far more heterogeneous than the Fe16 one: after the composition fit the energies scatter by
  0.4 eV/atom (Fe2N is 2 eV/atom off), 17% of the frames have a force component above 1 eV/Å, and the
  compositions are unevenly spread (Fe2H2 is in validation and test but not in training).
- `dataset_spin_v0` predates the frame roles and the per-frame SCF filter of the later collects.

For a rerun: log the training error, keep later checkpoints, use the v2 collect with frame roles, add a
split by trajectory inside families beside the split by family, and compare with a fine-tuned foundation model.

Files: `D:\MLIP_Work_Folder\Cluster_MLIP\fenom_spin_heldout_v1\` (`report.json`, `score.log`, `night.log`).

## Cu

No Cu test was possible from data on disk: the warehouses hold Cu structures (Cu13, Cu12, CuOn and others)
only as seed geometries with legacy energies at mixed routes and no forces.

### FenOm rerun, queued 2026-11-19

The per-batch training loss of the first run (`results/*_train.txt`) fell from 1.46 to 0.047 in 80 epochs and
was still falling, while the validation loss started at 0.040: the training set is dominated by a few
extreme frames and the model was still fitting them. So "did not fit" is the better description.

Changes: frames with a force component above 3 eV/Å dropped (270); Huber loss; 150 epochs without early
stop; validation taken from trajectories of the training groups; a third, easy test set (held-out
trajectories of training groups) and a 500-frame training sample are scored. Still `dataset_spin_v0`: the
v2 collect is on QB3, not on the laptop.
