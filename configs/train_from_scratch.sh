#!/usr/bin/env bash
set -euo pipefail

dataset_dir=${1:-dataset}
run_name=${RUN_NAME:-cluster_charge_spin_scratch}
device=${DEVICE:-cuda}

# "spin" stores Gaussian spin multiplicity (1=singlet, 2=doublet, ...).
# Charge is shifted by 100 so negative integer charges become categorical IDs.
mace_run_train \
  --name="$run_name" \
  --model="ScaleShiftMACE" \
  --train_file="$dataset_dir/train.extxyz" \
  --valid_file="$dataset_dir/valid.extxyz" \
  --test_file="$dataset_dir/test.extxyz" \
  --energy_key="REF_energy" \
  --forces_key="REF_forces" \
  --total_charge_key="charge" \
  --total_spin_key="spin" \
  --embedding_specs='{"total_spin":{"type":"categorical","per":"graph","in_dim":1,"emb_dim":128,"num_classes":101,"offset":0},"total_charge":{"type":"categorical","per":"graph","in_dim":1,"emb_dim":128,"num_classes":201,"offset":100}}' \
  --use_embedding_readout \
  --E0s="average" \
  --interaction_first="RealAgnosticResidualNonLinearInteractionBlock" \
  --interaction="RealAgnosticResidualNonLinearInteractionBlock" \
  --num_interactions=2 \
  --correlation=3 \
  --max_ell=3 \
  --r_max=6.0 \
  --num_radial_basis=8 \
  --hidden_irreps="128x0e + 128x1o + 128x2e" \
  --MLP_irreps="16x0e" \
  --loss="weighted" \
  --energy_weight=1 \
  --forces_weight=100 \
  --stress_weight=0 \
  --scaling="rms_forces_scaling" \
  --lr=0.005 \
  --batch_size=8 \
  --valid_batch_size=8 \
  --max_num_epochs=500 \
  --patience=60 \
  --ema \
  --ema_decay=0.999 \
  --default_dtype="float64" \
  --device="$device" \
  --seed=20260811 \
  --keep_checkpoints
