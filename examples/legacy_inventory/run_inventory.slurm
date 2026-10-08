#!/bin/bash
#SBATCH --job-name=cluster-legacy-inventory
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/inventory-%j.out
#SBATCH --error=logs/inventory-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the package directory}"
mkdir -p outputs
rm -f outputs/DONE
trap 'code=$?; printf "%s\n" "$code" > outputs/exit_code' EXIT
printf 'started\n' > outputs/STARTED
source /home/lgutsev/miniforge3/etc/profile.d/conda.sh
conda activate "${CLUSTER_MLIP_ENV:-/project/lgutsev/env/cluster_mlip_runtime}"
module load gaussian/g16-c01
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/runtime${PYTHONPATH:+:$PYTHONPATH}"
SOURCE_ROOT=${LEGACY_SOURCE_ROOT:-/ddnB/project/ramu/gutsev/glg}
CAMPAIGNS_ROOT=${CLUSTER_CAMPAIGNS_ROOT:-/ddnB/work/lgutsev/ClusterMLIP/campaigns}
RUN_ROOT="$CAMPAIGNS_ROOT/Gutsev_Legacy_2026-10/run_${SLURM_JOB_ID:?}"
printf '%s\n' "$RUN_ROOT" > outputs/result_path.txt
python inventory.py --self-test
python inventory.py --source "$SOURCE_ROOT" --campaigns "$CAMPAIGNS_ROOT" --output "$RUN_ROOT"
for file in summary.json report.md files.csv record_provenance.csv incoming_file_duplicates.csv overlap.csv inclusion_proposals.csv RUN_STATUS.json; do
    cp "$RUN_ROOT/$file" "outputs/$file"
done
printf '0\n' > outputs/exit_code
touch outputs/DONE
