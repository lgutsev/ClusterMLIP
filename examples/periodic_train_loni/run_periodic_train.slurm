#!/bin/bash
#SBATCH --job-name=periodic-train
#SBATCH --account=loni_perovsk27
#SBATCH --partition=gpu2
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Written by ClusterMLIP examples/periodic_train_loni/make_loni_package.py. One array task per variant:
# 0 scratch seed 1, 1 scratch seed 2, 2 fine-tune of MACE-MP-0 small. Each task re-ingests the
# finished package-27 A3 VASP jobs on LONI (no DFT data travels through git), trains, and evaluates.
# Resubmitting resumes from the last MACE checkpoint (--restart_latest).
# conda's activation scripts read unset variables: -u only after them
set -eo pipefail
module purge
source /home/lgutsev/miniforge3/etc/profile.d/conda.sh
conda activate "${PERIODIC_MACE_ENV:-/project/lgutsev/env/mace_env}"
set -u
export XDG_CACHE_HOME=/project/lgutsev/cache TRITON_CACHE_DIR=/project/lgutsev/cache/triton \
    CUDA_CACHE_PATH=/project/lgutsev/cache/nv MPLCONFIGDIR=/project/lgutsev/cache/matplotlib
cd "${SLURM_SUBMIT_DIR:?submit from the package directory}"
PKG=$PWD
export PYTHONPATH="$PKG/runtime${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

VARIANTS=(scratch_s1 scratch_s2 ft_mp0small)
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
OUT=$PKG/outputs/$V
WORK=$PKG/work/$V
mkdir -p "$OUT" "$WORK"
rm -f "$OUT/DONE"
trap 'code=$?; printf "%s\n" "$code" > "$OUT/exit_code"' EXIT

# Package 27 may have been sorted into a batch folder; PKG27_OUTPUTS overrides.
PKG27=${PKG27_OUTPUTS:-}
if [ -z "$PKG27" ]; then
    for candidate in "$PKG"/../27_doe2_fe3_stageA/outputs "$PKG"/../batch*/27_doe2_fe3_stageA/outputs; do
        if [ -d "$candidate" ]; then PKG27=$candidate; break; fi
    done
fi
if [ -z "$PKG27" ] || [ ! -d "$PKG27" ]; then
    echo "No package 27 outputs found; export PKG27_OUTPUTS=/abs/path/to/27_doe2_fe3_stageA/outputs" >&2
    exit 2
fi
JOBS=A3_slab_fix,A3_fe3_M11,A3_fe3_M9,A3_fe3_free,A3_fe3n2_side,A3_fe3n2_end

case $V in
    scratch_s1) ARGS=(--variant scratch --seed 1 --max-epochs 300 --lr 0.001) ;;
    scratch_s2) ARGS=(--variant scratch --seed 2 --max-epochs 300 --lr 0.001) ;;
    ft_mp0small)
        source "$PKG/../model_paths.sh"
        FOUNDATION=$(resolve_smoke_mace "$PKG")
        ARGS=(--variant finetune --foundation "$FOUNDATION" --seed 1 --max-epochs 100 --lr 0.0001) ;;
esac

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import mace, torch; print('mace', mace.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
printf 'package 27 outputs: %s\n' "$(readlink -f "$PKG27")" | tee "$OUT/source.txt"

RUN_SH=$(python periodic_train.py --outputs "$PKG27" --jobs "$JOBS" --valid-jobs A3_fe3_free \
    --test-jobs A3_fe3n2_end --work "$WORK" "${ARGS[@]}" | tee "$OUT/prepare.log" | tail -1)
cp "$WORK/dataset/ingest_summary.json" "$WORK/dataset/jobs.csv" "$WORK/campaign/train_manifest.json" "$OUT/"

bash "$RUN_SH"
SEED_DIR=$(dirname "$RUN_SH")
MODEL=$(ls "$SEED_DIR"/periodic_*.model | grep -v _compiled | head -1)
printf 'model: %s\n' "$MODEL" | tee -a "$OUT/source.txt"

# Evaluate on every frame and on the held-out job, against the DFT labels.
for split in all test; do
    REACTIONS=()
    if [ "$split" = all ]; then REACTIONS=(--reactions reactions_A3.json); fi
    python zero_shot.py predict "$WORK/dataset/$split.extxyz" --backend mace --model "$MODEL" \
        --name "$V" -o "$OUT/preds_$split"
    python zero_shot.py report "$WORK/dataset/$split.extxyz" "$OUT/preds_$split/$V.npz" \
        "${REACTIONS[@]}" -o "$OUT/report_$split"
done

cp "$MODEL" "$OUT/"
cp -r "$SEED_DIR/logs" "$SEED_DIR/results" "$OUT/" 2>/dev/null || true
printf '0\n' > "$OUT/exit_code"
touch "$OUT/DONE"
