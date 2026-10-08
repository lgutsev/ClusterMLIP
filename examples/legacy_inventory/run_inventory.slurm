#!/bin/bash
#SBATCH --job-name=cluster-legacy-inventory
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/inventory-%j.out
#SBATCH --error=logs/inventory-%j.err
# 16 cores for memory, not speed (4 cores, ~16 GB, was OOM-killed in job 1072686) (no --mem on QB4; the run holds every parsed record until it writes):
# the source is ~52 GB in 33,131 Gaussian .out files (2026-10-06), too much for 1 core in 2 h
# conda's activation scripts and the module function read unset variables: -u only after them
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the package directory}"
mkdir -p outputs
rm -f outputs/DONE
save_progress() {
    if [ -n "${RUN_ROOT:-}" ]; then
        for file in PROGRESS.json progress.jsonl; do
            [ ! -f "$RUN_ROOT/$file" ] || cp "$RUN_ROOT/$file" "outputs/$file" || true
        done
    fi
}
trap 'code=$?; save_progress; printf "%s\n" "$code" > outputs/exit_code' EXIT
# Slurm ends a job at its time limit with SIGTERM: record 143, not the 0 the EXIT trap saw (job 1071655)
trap 'save_progress; printf "143\n" > outputs/exit_code; trap - EXIT; exit 143' TERM
printf 'started\n' > outputs/STARTED
source /home/lgutsev/miniforge3/etc/profile.d/conda.sh
conda activate "${CLUSTER_MLIP_ENV:-/project/lgutsev/env/cluster_mlip_runtime}"
module load gaussian/g16-c01
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/runtime${PYTHONPATH:+:$PYTHONPATH}"
SOURCE_ROOT=${LEGACY_SOURCE_ROOT:-/ddnB/project/ramu/gutsev/glg}
CAMPAIGNS_ROOT=${CLUSTER_CAMPAIGNS_ROOT:-/ddnB/work/lgutsev/ClusterMLIP/campaigns}
RUN_ROOT="$CAMPAIGNS_ROOT/Gutsev_Legacy_2026-10/run_${SLURM_JOB_ID:?}"
printf '%s\n' "$RUN_ROOT" > outputs/result_path.txt
python -u inventory.py --self-test
python -u inventory.py --source "$SOURCE_ROOT" --campaigns "$CAMPAIGNS_ROOT" --output "$RUN_ROOT"
for file in summary.json report.md files.csv record_provenance.csv incoming_file_duplicates.csv overlap.csv inclusion_proposals.csv RUN_STATUS.json; do
    cp "$RUN_ROOT/$file" "outputs/$file"
done
printf '0\n' > outputs/exit_code
touch outputs/DONE
