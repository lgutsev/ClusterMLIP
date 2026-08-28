#!/usr/bin/env bash
#SBATCH --job-name=cluster_mlip_literature_gap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=single
#SBATCH --account=loni_perovsk27
#SBATCH --output=literature_gap-%j.stdout
#SBATCH --error=literature_gap-%j.stderr
set -euo pipefail

# Explicitly activate the runtime environment. `sbatch` copies this script
# into a spool directory and runs it from there in a non-interactive,
# non-login shell -- so neither a `conda activate` you ran before
# submitting, nor sourcing a sibling file by relative path (the script's
# own location is no longer this repo's scripts/ folder once Slurm has
# copied it), reaches the job. Override CLUSTER_MLIP_ENV if your env lives
# somewhere other than /project/lgutsev/env/cluster_mlip_runtime.
CLUSTER_MLIP_ENV=${CLUSTER_MLIP_ENV:-/project/lgutsev/env/cluster_mlip_runtime}
# conda's own activate/deactivate hooks (e.g. deactivate-gxx_linux-64.sh)
# reference variables like CONDA_BACKUP_CXX without guarding them -- a known
# conda/`set -u` incompatibility, nothing wrong with this environment.
# Nounset is relaxed for just this block and restored immediately after.
set +u
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
set -u

# `literature-gap` is the one command in the pipeline that needs live
# internet access (it queries the OpenAlex API). Whether LONI's `single`
# partition allows outbound internet from a compute node depends on site
# policy -- this job fails fast with a clear error if it doesn't
# ("could not reach https://api.openalex.org/works ... run it from a login
# node"), it will not hang, so it's safe to try. If that happens, just run
# the same `cluster-mlip literature-gap ...` command directly on a login
# node instead -- it normally finishes in well under a minute (an author's
# full bibliography is one or two paginated API requests), so this batch
# job exists purely to get it off an interactive session, not because the
# work itself is heavy.
#
# Every argument after the script name is forwarded verbatim, e.g.:
#   sbatch scripts/run_literature_gap_slurm.sh inventory_output \
#     -o literature_gap_output --orcid 0000-0001-7752-5567 \
#     --contact-email you@example.edu --pdf-index pdf_index/pdf_index.json
#
# Any #SBATCH directive above can be overridden on the command line without
# editing this file, e.g.:
#   sbatch --time=00:30:00 --account=myaccount scripts/run_literature_gap_slurm.sh ...

if command -v cluster-mlip >/dev/null 2>&1; then
  cluster_mlip=(cluster-mlip)
else
  cluster_mlip=(python -m cluster_mlip.cli)
fi

"${cluster_mlip[@]}" literature-gap "$@"
