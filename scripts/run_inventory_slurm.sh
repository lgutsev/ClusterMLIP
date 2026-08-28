#!/usr/bin/env bash
#SBATCH --job-name=cluster_mlip_inventory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --partition=checkpt
#SBATCH --account=loni_perovsk27
#SBATCH --output=inventory-%j.stdout
#SBATCH --error=inventory-%j.stderr
set -euo pipefail

# Explicitly activate the runtime environment. `sbatch` copies this script
# into a spool directory and runs it from there in a non-interactive,
# non-login shell -- so neither a `conda activate` you ran before
# submitting, nor sourcing a sibling file by relative path (the script's
# own location is no longer this repo's scripts/ folder once Slurm has
# copied it), reaches the job. Override CLUSTER_MLIP_ENV if your env lives
# somewhere other than /project/lgutsev/env/cluster_mlip_runtime.
CLUSTER_MLIP_ENV=${CLUSTER_MLIP_ENV:-/project/lgutsev/env/cluster_mlip_runtime}
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

# Single-node batch job for `cluster-mlip inventory` against a folder of many
# warehouse ZIPs -- this is the one command in the pipeline slow enough (a
# large/nested warehouse can genuinely take hours to parse) to need
# submitting rather than running interactively. `literature-gap` is
# deliberately NOT bundled in here: it needs live internet access, which
# LONI compute nodes typically do not have, so run it separately from a
# login node once this job's output exists (see README.md).
#
# Any #SBATCH directive above can be overridden on the command line without
# editing this file, e.g.:
#   sbatch --time=04:00:00 --account=myaccount scripts/run_inventory_slurm.sh

# Defaults to the directory this job was submitted from, so the common case
# is: cd into the folder of warehouse ZIPs, then `sbatch
# scripts/run_inventory_slurm.sh` with no arguments.
folder=${1:-.}
output=${2:-inventory}

if command -v cluster-mlip >/dev/null 2>&1; then
  cluster_mlip=(cluster-mlip)
else
  cluster_mlip=(python -m cluster_mlip.cli)
fi

jobs=${SLURM_CPUS_PER_TASK:-1}
"${cluster_mlip[@]}" inventory "$folder" -o "$output" --recursive --jobs "$jobs"
