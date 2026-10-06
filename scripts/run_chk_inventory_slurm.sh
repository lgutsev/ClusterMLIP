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
# usage (from the ClusterMLIP checkout, with Gaussian's formchk available, e.g. after `module load g09`):
#   FORMCHK=formchk sbatch scripts/run_chk_inventory_slurm.sh COLLECT_DATASET CAMPAIGN_DIR OUT_DIR
set -euo pipefail
REPO=${CLUSTER_MLIP_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
PY=${CLUSTER_MLIP_PYTHON:-/project/lgutsev/env/lgutsev_dev/bin/python}
FORMCHK=${FORMCHK:-formchk}
command -v "${FORMCHK%% *}" >/dev/null || { echo "formchk not found: load the Gaussian module or set FORMCHK" >&2; exit 1; }
[[ -e "$3" ]] && { echo "$3 exists; choose a new OUT_DIR" >&2; exit 1; }
cd "$REPO"; git log --oneline -1; date -Is
"$PY" scripts/diagnostics/rattle/chk_inventory.py "$1" "$2" "$3" --formchk "$FORMCHK"
date -Is
