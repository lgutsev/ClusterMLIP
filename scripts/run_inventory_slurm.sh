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
#
# Activate whatever environment has `cluster-mlip` installed before
# submitting (e.g. `source .venv/bin/activate`) -- sbatch inherits the
# environment of the shell that submits it.

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
