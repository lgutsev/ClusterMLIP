#!/usr/bin/env bash
#SBATCH --job-name=fe16_spin_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=single
#SBATCH --account=loni_perovsk27
#SBATCH --output=spin_audit-%j.out
#SBATCH --error=spin_audit-%j.err
set -euo pipefail
# Submit from the updated checkout, NOT from an old fe16_loni_kit snapshot.
# sbatch spools the script elsewhere; use the submission directory explicitly.
AUDIT_REPO=${CLUSTER_MLIP_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
AUDIT_PYTHON=${CLUSTER_MLIP_PYTHON:-/project/lgutsev/env/lgutsev_dev/bin/python}
[[ -f "$AUDIT_REPO/src/cluster_mlip/spin_audit.py" ]] || {
  echo "Set CLUSTER_MLIP_REPO to the updated ClusterMLIP checkout" >&2; exit 1;
}
[[ -x "$AUDIT_PYTHON" ]] || { echo "Missing Python: $AUDIT_PYTHON" >&2; exit 1; }
export PYTHONPATH="$AUDIT_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$AUDIT_PYTHON" -c 'import numpy, cluster_mlip; print("numpy", numpy.__version__, "code", cluster_mlip.__file__)'
date -Is
"$AUDIT_PYTHON" -m cluster_mlip.cli audit-spin-labels "$@"
date -Is
