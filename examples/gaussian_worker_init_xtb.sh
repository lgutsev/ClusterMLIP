#!/usr/bin/env bash
# Optional worker initialization for Gaussian External jobs that call xTB.
# Pass this file to `cluster-mlip prepare-slurm --worker-init ...`.

set -euo pipefail

conda_sh=${CLUSTER_MLIP_CONDA_SH:-/home/lgutsev/miniconda3/etc/profile.d/conda.sh}
xtb_env=${CLUSTER_MLIP_XTB_ENV:-xtb}
wrapper_dir=${CLUSTER_MLIP_XTB_WRAPPER_DIR:-$HOME/xtb-gaussian}

if [[ ! -f $conda_sh ]]; then
  echo "xTB worker initialization cannot find conda.sh: $conda_sh" >&2
  return 2
fi

# shellcheck source=/dev/null
source "$conda_sh"
conda activate "$xtb_env"
export PATH="$wrapper_dir:$PATH"
