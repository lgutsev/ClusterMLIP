#!/usr/bin/env bash
#SBATCH --job-name=cluster_mlip_pdf_index
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=single
#SBATCH --account=loni_perovsk27
#SBATCH --output=pdf_index-%j.stdout
#SBATCH --error=pdf_index-%j.stderr
set -euo pipefail

# Explicitly activate the runtime environment. `sbatch` copies this script
# into a spool directory and runs it from there in a non-interactive,
# non-login shell -- so neither a `conda activate` you ran before
# submitting, nor sourcing a sibling file by relative path (the script's
# own location is no longer this repo's scripts/ folder once Slurm has
# copied it), reaches the job. Override CLUSTER_MLIP_ENV if your env lives
# somewhere other than /project/lgutsev/env/cluster_mlip_runtime.
CLUSTER_MLIP_ENV=${CLUSTER_MLIP_ENV:-/project/lgutsev/env/cluster_mlip_runtime}
for conda_sh in "$HOME"/miniforge3/etc/profile.d/conda.sh \
                "$HOME"/miniconda3/etc/profile.d/conda.sh \
                "$HOME"/anaconda3/etc/profile.d/conda.sh; do
  if [ -f "$conda_sh" ]; then
    # shellcheck disable=SC1090
    source "$conda_sh"
    break
  fi
done
if command -v conda >/dev/null 2>&1 && [ -d "$CLUSTER_MLIP_ENV" ]; then
  conda activate "$CLUSTER_MLIP_ENV"
fi

# Single-core batch job for `cluster-mlip pdf-index` against a local corpus
# of paper PDFs (a ZIP of PDFs, a folder of them, or a single .pdf). Text
# extraction over a large/scanned-image-heavy corpus can genuinely take
# hours, and needs no network access, so this belongs on LONI's `single`
# (serial) queue rather than run interactively -- unlike `literature-gap`,
# which needs live internet and stays a login-node command (see README.md).
#
# One core is enough: pypdf's text extraction is single-threaded per PDF,
# and this job processes PDFs one at a time rather than in parallel.
#
# Any #SBATCH directive above can be overridden on the command line without
# editing this file, e.g.:
#   sbatch --time=04:00:00 --account=myaccount scripts/run_pdf_index_slurm.sh

# Defaults to the directory this job was submitted from, so the common case
# is: cd into the folder holding the PDF ZIP, then `sbatch
# scripts/run_pdf_index_slurm.sh <papers.zip>`.
source=${1:?usage: sbatch run_pdf_index_slurm.sh <pdfs.zip|folder|file.pdf> [output_dir]}
output=${2:-pdf_index}

if command -v cluster-mlip >/dev/null 2>&1; then
  cluster_mlip=(cluster-mlip)
else
  cluster_mlip=(python -m cluster_mlip.cli)
fi

"${cluster_mlip[@]}" pdf-index "$source" -o "$output"
