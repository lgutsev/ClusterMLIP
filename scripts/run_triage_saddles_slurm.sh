#!/usr/bin/env bash
#SBATCH --job-name=triage_saddles
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --partition=single
#SBATCH --account=loni_perovsk27
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Read-only: classify every started saddle (TS) job in a campaign and
# recommend a rescue. Converged saddles are listed too.
#
#   sbatch scripts/run_triage_saddles_slurm.sh CAMPAIGN REPORT_DIR
#
# Logs land in the submission directory. Point CLUSTER_MLIP_REPO and
# CLUSTER_MLIP_PYTHON at the git checkout and its environment.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 CAMPAIGN REPORT_DIR" >&2; exit 1; }
CAMPAIGN=$1 REPORT=$2
REPO=${CLUSTER_MLIP_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
PY=${CLUSTER_MLIP_PYTHON:-python3}
[[ -f "$REPO/src/cluster_mlip/saddle_triage.py" ]] || {
  echo "Set CLUSTER_MLIP_REPO to the ClusterMLIP checkout (got $REPO)" >&2; exit 1;
}
[[ -e "$REPORT" ]] && { echo "refusing: $REPORT already exists" >&2; exit 1; }
export PYTHONPATH="$REPO/src"
echo "host $(hostname)  start $(date -Is)"
echo "code $(git -C "$REPO" rev-parse HEAD)  dirty=$(git -C "$REPO" status --porcelain | wc -l)"
"$PY" -c 'import sys, cluster_mlip; print("python", sys.version.split()[0], cluster_mlip.__file__)'
"$PY" -m cluster_mlip.cli triage-saddles "$CAMPAIGN" -o "$REPORT" --include-converged
echo "triage done $(date -Is)"
