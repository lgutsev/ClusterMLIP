#!/usr/bin/env bash
set -euo pipefail

# UNVERIFIED STARTING POINT -- do not point a real allocation at this
# without a smoke test first.
#
# The README defers foundation-model fine-tuning "until the scratch model
# provides a clean control" -- this script is that deferred next step,
# written from MACE's documented naive fine-tuning flags
# (https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html), not
# from a run that has actually been executed against this project's data.
#
# The specific open question this project adds on top of standard MACE
# fine-tuning: --embedding_specs/--use_embedding_readout add graph-level
# charge/spin categorical embedding *modules* that a public foundation
# checkpoint (mace-mp-0, etc.) was never trained with. Whether
# --foundation_model tolerates new modules being attached alongside its
# pretrained weights (vs. erroring on a state-dict mismatch) depends on the
# installed mace-torch version and is exactly the kind of thing that needs
# checking on a handful of structures before a full campaign -- the same
# spirit as the spin workflow's "code-tested but not human-tested" flag.
# If naive fine-tuning does not tolerate the new embedding modules, the
# multihead-replay path (--multiheads_finetuning True, --pt_train_file) is
# the documented fallback, at the cost of also needing pretraining-replay
# data: https://mace-docs.readthedocs.io/en/latest/guide/multihead_finetuning.html
#
# "spin" stores Gaussian spin multiplicity (1=singlet, 2=doublet, ...).
# Charge is shifted by 100 so negative integer charges become categorical IDs.
# Both match configs/train_from_scratch.sh so a scratch and a fine-tuned run
# stay directly comparable.

dataset_dir=${1:-dataset}
run_name=${RUN_NAME:-cluster_charge_spin_finetune}
device=${DEVICE:-cuda}
foundation_model=${FOUNDATION_MODEL:-medium}

mace_run_train \
  --name="$run_name" \
  --foundation_model="$foundation_model" \
  --multiheads_finetuning=False \
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
  --loss="weighted" \
  --energy_weight=1 \
  --forces_weight=100 \
  --stress_weight=0 \
  --scaling="rms_forces_scaling" \
  --lr=0.0001 \
  --batch_size=8 \
  --valid_batch_size=8 \
  --max_num_epochs=100 \
  --patience=20 \
  --ema \
  --ema_decay=0.99 \
  --amsgrad \
  --default_dtype="float64" \
  --device="$device" \
  --seed=20260811 \
  --keep_checkpoints
