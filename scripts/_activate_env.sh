#!/usr/bin/env bash
# Sourced (not executed directly) by the other scripts/run_*_slurm.sh jobs.
#
# `sbatch` runs your job in a non-interactive, non-login shell that does NOT
# read ~/.bashrc -- so a `conda activate` you ran before `sbatch` is *not*
# inherited by the job itself, even though plain environment variables like
# PATH sometimes are. Without this, the job silently falls back to whatever
# bare `python`/`cluster-mlip` happens to be first on PATH (e.g. the base
# miniforge install), which won't have this package or its dependencies
# installed -- surfacing as "No module named 'cluster_mlip'" deep inside a
# batch job instead of at submission time.
#
# Override CLUSTER_MLIP_ENV to activate a different conda env/prefix without
# editing this file, e.g.:
#   CLUSTER_MLIP_ENV=/path/to/other/env sbatch scripts/run_inventory_slurm.sh
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
