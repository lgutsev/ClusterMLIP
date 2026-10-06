#!/usr/bin/env bash
#SBATCH --job-name=chk_inventory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=single
#SBATCH --account=loni_perovsk27
#SBATCH --output=chk_inventory-%j.out
#SBATCH --error=chk_inventory-%j.err
# Read-only check that each converged stage's .chk still holds that stage's solution (rattle parents).
#
# Run this metadata/inventory job from QB4. It calls formchk but does NOT launch a
# Gaussian reference calculation; QB3 remains the runner for the actual G09 jobs.
# Load whichever Gaussian module on QB4 provides formchk before submitting.
#
# usage (from the updated ClusterMLIP checkout on QB4):
#   FORMCHK=formchk sbatch scripts/run_chk_inventory_slurm.sh COLLECT_DATASET CAMPAIGN_DIR OUT_DIR
set -euo pipefail
REPO=${CLUSTER_MLIP_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
PY=${CLUSTER_MLIP_PYTHON:-/project/lgutsev/env/cluster_mlip_runtime/bin/python}
FORMCHK=${FORMCHK:-formchk}
[[ -f "$REPO/scripts/diagnostics/rattle/chk_inventory.py" ]] || {
  echo "run from the updated ClusterMLIP checkout or set CLUSTER_MLIP_REPO" >&2; exit 1;
}
[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }
command -v "${FORMCHK%% *}" >/dev/null || {
  echo "formchk not found: load the Gaussian module on QB4 or set FORMCHK" >&2; exit 1;
}
[[ -e "$3" ]] && { echo "$3 exists; choose a new OUT_DIR" >&2; exit 1; }
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
git log --oneline -1
"$PY" -V
date -Is
"$PY" scripts/diagnostics/rattle/chk_inventory.py "$1" "$2" "$3" --formchk "$FORMCHK"
date -Is
