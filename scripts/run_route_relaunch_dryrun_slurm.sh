#!/usr/bin/env bash
#SBATCH --job-name=route_relaunch_dry
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=single
#SBATCH --account=loni_perovsk27
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Read-only route check of a Gaussian campaign: audit-routes into REPORT_DIR,
# then relaunch-routes --dry-run. Nothing in the campaign is changed.
#
#   sbatch scripts/run_route_relaunch_dryrun_slurm.sh CAMPAIGN REPORT_DIR [SEEDS.extxyz]
#
# SEEDS.extxyz (optional) lets higher_order_saddle records take their
# imaginary-mode order from the seeds the campaign was prepared from.
# Logs land in the submission directory. Point CLUSTER_MLIP_REPO and
# CLUSTER_MLIP_PYTHON at the git checkout and its environment (export them once
# in ~/.bashrc; sbatch passes the environment through).
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: $0 CAMPAIGN REPORT_DIR [SEEDS.extxyz]" >&2; exit 1; }
CAMPAIGN=$1 REPORT=$2 SEEDS=${3:-}
REPO=${CLUSTER_MLIP_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
PY=${CLUSTER_MLIP_PYTHON:-python3}
[[ -f "$REPO/src/cluster_mlip/relaunch.py" ]] || {
  echo "Set CLUSTER_MLIP_REPO to the ClusterMLIP checkout (got $REPO)" >&2; exit 1;
}
[[ -e "$REPORT" ]] && { echo "refusing: $REPORT already exists" >&2; exit 1; }
# Only the checkout on the path: an inherited PYTHONPATH (or its empty entry,
# which means the current directory) must not shadow cluster_mlip.
export PYTHONPATH="$REPO/src"
echo "host $(hostname)  start $(date -Is)"
echo "code $(git -C "$REPO" rev-parse HEAD)  dirty=$(git -C "$REPO" status --porcelain | wc -l)"
"$PY" -c 'import sys, cluster_mlip; print("python", sys.version.split()[0], cluster_mlip.__file__)'

# audit-routes exits 2 when anything must be relaunched: a finding, not a failure.
rc=0; "$PY" -m cluster_mlip.cli audit-routes "$CAMPAIGN" -o "$REPORT" || rc=$?
[[ $rc -le 2 ]] || { echo "audit-routes failed (exit $rc)" >&2; exit "$rc"; }
echo "audit done $(date -Is) (exit $rc; 2 = relaunches needed)"

extra=()
[[ -n "$SEEDS" ]] && extra=(--saddle-order-from "$SEEDS")
"$PY" -m cluster_mlip.cli relaunch-routes "$CAMPAIGN" --dry-run ${extra[@]+"${extra[@]}"}
echo "dry run done $(date -Is)"
