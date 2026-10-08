#!/bin/bash
#SBATCH --job-name=cluster-root-check
#SBATCH --account=loni_perovsk27
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/root-check-%j.out
#SBATCH --error=logs/root-check-%j.err
# 8 cores for memory, not speed: ~1.4 M record headers (energies) are held in memory while the
# ~2 GB overlap.csv streams. Package 26 needed 16 cores only because it kept full geometries.
# conda's activation scripts read unset variables: -u only after them
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the package directory}"
mkdir -p outputs
rm -f outputs/DONE
trap 'code=$?; printf "%s\n" "$code" > outputs/exit_code' EXIT
trap 'printf "143\n" > outputs/exit_code; trap - EXIT; exit 143' TERM
printf 'started\n' > outputs/STARTED
source /home/lgutsev/miniforge3/etc/profile.d/conda.sh
conda activate "${CLUSTER_MLIP_ENV:-/project/lgutsev/env/cluster_mlip_runtime}"
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CAMPAIGNS_ROOT=${CLUSTER_CAMPAIGNS_ROOT:-/ddnB/work/lgutsev/ClusterMLIP/campaigns}
INVENTORY_RUN=${INVENTORY_RUN:-$CAMPAIGNS_ROOT/Gutsev_Legacy_2026-10/run_1075007}
SOURCE_ROOT=${LEGACY_SOURCE_ROOT:-/ddnB/project/ramu/gutsev/glg}
RUN_ROOT="$CAMPAIGNS_ROOT/Gutsev_Legacy_2026-10/root_check_${SLURM_JOB_ID:?}"
printf '%s\n' "$RUN_ROOT" > outputs/result_path.txt
python -u root_check.py --self-test
python -u root_check.py --run "$INVENTORY_RUN" --campaigns "$CAMPAIGNS_ROOT" \
    --source "$SOURCE_ROOT" --output "$RUN_ROOT"
for file in summary.json report.md pairs_flagged.csv bins.csv RUN_STATUS.json; do
    cp "$RUN_ROOT/$file" "outputs/$file"
done
printf '0\n' > outputs/exit_code
touch outputs/DONE
