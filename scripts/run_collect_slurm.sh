#!/usr/bin/env bash
#SBATCH --job-name=cluster_mlip_collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=single
#SBATCH --account=loni_perovsk27
#SBATCH --output=collect-%j.stdout
#SBATCH --error=collect-%j.stderr
set -euo pipefail

# Explicitly activate the runtime environment. `sbatch` copies this script
# into a spool directory and runs it from there in a non-interactive,
# non-login shell -- so neither a `conda activate` you ran before
# submitting, nor sourcing a sibling file by relative path (the script's
# own location is no longer this repo's scripts/ folder once Slurm has
# copied it), reaches the job. Override CLUSTER_MLIP_ENV if your env lives
# somewhere other than /project/lgutsev/env/cluster_mlip_runtime (e.g.
#   CLUSTER_MLIP_ENV=/project/lgutsev/env/lgutsev_dev sbatch scripts/run_collect_slurm.sh ...
# to match the environment you use interactively).
CLUSTER_MLIP_ENV=${CLUSTER_MLIP_ENV:-/project/lgutsev/env/cluster_mlip_runtime}
# conda's own activate/deactivate hooks (e.g. deactivate-gxx_linux-64.sh)
# reference variables like CONDA_BACKUP_CXX without guarding them -- a known
# conda/`set -u` incompatibility, nothing wrong with this environment.
# Nounset is relaxed for just this block and restored immediately after.
set +u
for conda_sh in "$HOME"/miniforge3/etc/profile.d/conda.sh \
                "$HOME"/miniconda3/etc/profile.d/conda.sh \
                "$HOME"/anaconda3/etc/profile.d/conda.sh; do
  if [ -f "$conda_sh" ]; then
    # shellcheck disable=SC1090
    source "$conda_sh"
    break
  fi
done
if command -v conda >/dev/null 2>&1 && [ -d "$CLUSTER_MLIP_ENV" ]; then
  conda activate "$CLUSTER_MLIP_ENV"
fi
set -u

# Single-core batch job for `cluster-mlip collect`: it parses a finished (or
# partially finished) Gaussian campaign's .log/.out files into the MACE
# extxyz training dataset. It is a read-only downstream consumer -- it never
# touches checkpoints, .gjf inputs, inputs.txt, or scheduler state, so it is
# NOT part of a restart and is normally run once, at the end of a campaign.
# It needs no network access and the parse is single-threaded, so it belongs
# on the serial queue rather than an interactive session or a whole node.
#
# collect holds every labeled frame in memory before it writes the splits;
# a large spin warehouse can exceed the 16G requested above. If the job is
# killed for memory, resubmit with e.g. `sbatch --mem=48G ...`.
#
# The first argument is the campaign/outputs directory; every argument after
# it is forwarded verbatim, e.g.:
#   sbatch scripts/run_collect_slurm.sh "$W2" \
#     -o /ddnB/work/lgutsev/ClusterMLIP/campaigns/FenOm_Warehouse2/dataset_spin_v1 \
#     --frames converged --allow-partial
#
# Any #SBATCH directive above can be overridden on the command line without
# editing this file, e.g.:
#   sbatch --time=04:00:00 --mem=32G scripts/run_collect_slurm.sh "$W2" -o dataset

outputs=${1:?usage: sbatch run_collect_slurm.sh <campaign_dir> -o <dataset_dir> [collect args...]}
shift

if command -v cluster-mlip >/dev/null 2>&1; then
  cluster_mlip=(cluster-mlip)
else
  cluster_mlip=(python -m cluster_mlip.cli)
fi

echo "Collect source: $outputs"
echo "Forwarded args: $*"
echo "Host: $(hostname)"
date
"${cluster_mlip[@]}" collect "$outputs" "$@"
date
