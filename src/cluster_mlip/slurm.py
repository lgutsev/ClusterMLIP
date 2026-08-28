from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict


_NPROC_RE = re.compile(r"^\s*%nprocshared\s*=\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class SlurmConfig:
    jobs_per_batch: int = 30
    concurrent_jobs: int = 4
    cpus_per_job: int = 16
    time_limit: str = "72:00:00"
    partition: str = "checkpt"
    account: str = "loni_perovsk27"
    gaussian_module: str = "gaussian/g16-c01"
    gaussian_command: str = "g16"
    job_name: str = "cluster_mlip_g16"
    memory_per_node: str | None = None
    scratch_root: str = "/work/$USER/g16-scr"


class SlurmPlan(TypedDict):
    campaign: str
    manifest: str
    input_count: int
    batch_count: int
    config: dict[str, object]
    worker_init: str | None
    input_fingerprint: str
    warnings: list[str]


@dataclass(frozen=True)
class ExtractSlurmConfig:
    time_limit: str = "12:00:00"
    partition: str = "checkpt"
    account: str = "loni_perovsk27"
    gaussian_module: str = "gaussian/g16-c01"
    job_name: str = "cluster_mlip_extract"
    cluster_mlip_command: str = "cluster-mlip"
    runtime_env: str = "/project/lgutsev/env/cluster_mlip_runtime"


class ExtractSlurmPlan(TypedDict):
    source: str
    source_sha256: str
    output: str
    config: dict[str, object]
    extract_arguments: list[str]
    sbatch_script: str
    submit_script: str


def _safe_directive(value: str, name: str) -> str:
    if not value or any(character in value for character in "\r\n"):
        raise ValueError(f"{name} must be a non-empty single line")
    return value


def _path_sha256(path: Path) -> str:
    """Hash one source file or a directory tree without depending on mtimes."""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _extract_sbatch_script(
    source: Path,
    output: Path,
    config: ExtractSlurmConfig,
    extract_arguments: list[str],
) -> str:
    directives = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={_safe_directive(config.job_name, 'job name')}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH --time={_safe_directive(config.time_limit, 'time limit')}",
        f"#SBATCH --partition={_safe_directive(config.partition, 'partition')}",
        f"#SBATCH --account={_safe_directive(config.account, 'account')}",
        "#SBATCH --output=extract-%j.stdout",
        "#SBATCH --error=extract-%j.stderr",
    ]
    command = [
        _safe_directive(config.cluster_mlip_command, "cluster-mlip command"),
        "extract",
        str(source),
        "-o",
        str(output),
        *extract_arguments,
    ]
    module_line = (
        f"set +u; module load {shlex.quote(config.gaussian_module)}; set -u"
        if config.gaussian_module
        else ""
    )
    runtime_env = shlex.quote(config.runtime_env)
    source_message = shlex.quote(f"Source: {source}")
    output_message = shlex.quote(f"Output: {output}")
    body = f"""
set -euo pipefail

runtime_env={runtime_env}
if [[ -n $runtime_env ]]; then
  set +u
  for conda_sh in "$HOME"/miniforge3/etc/profile.d/conda.sh \
                  "$HOME"/miniconda3/etc/profile.d/conda.sh \
                  "$HOME"/anaconda3/etc/profile.d/conda.sh; do
    if [[ -f $conda_sh ]]; then
      source "$conda_sh"
      break
    fi
  done
  if command -v conda >/dev/null 2>&1 && [[ -d $runtime_env ]]; then
    conda activate "$runtime_env"
  elif [[ -d $runtime_env/bin ]]; then
    export PATH="$runtime_env/bin:$PATH"
  fi
  set -u
fi
{module_line}
printf '%s\n' {source_message}
printf '%s\n' {output_message}
echo "Host: $(hostname)"
date
{' '.join(shlex.quote(part) for part in command)}
date
"""
    return "\n".join(directives) + body


def prepare_extract_slurm(
    source: Path,
    output: Path,
    config: ExtractSlurmConfig,
    *,
    extract_arguments: list[str] | None = None,
) -> ExtractSlurmPlan:
    """Persist an auditable one-node Slurm job for the sequential extractor."""
    source = source.resolve()
    output = output.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir() and (output == source or source in output.parents):
        raise ValueError("extract output must not be inside a source directory")
    source_sha256 = _path_sha256(source)
    output.mkdir(parents=True, exist_ok=True)
    arguments = list(extract_arguments or [])
    sbatch_path = output / "run_extract.sbatch"
    submit_path = output / "submit_extract.sh"
    sbatch_path.write_text(
        _extract_sbatch_script(source, output, config, arguments), encoding="utf-8"
    )
    submit_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

output=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
sbatch "$@" --chdir="$output" "$output/run_extract.sbatch"
""",
        encoding="utf-8",
    )
    sbatch_path.chmod(0o755)
    submit_path.chmod(0o755)
    plan: ExtractSlurmPlan = {
        "source": str(source),
        "source_sha256": source_sha256,
        "output": str(output),
        "config": asdict(config),
        "extract_arguments": arguments,
        "sbatch_script": sbatch_path.name,
        "submit_script": submit_path.name,
    }
    (output / "extract_slurm_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return plan


def submit_extract_slurm(output: Path) -> str:
    """Submit a previously persisted extract job and retain sbatch's job-id response."""
    output = output.resolve()
    submit_path = output / "submit_extract.sh"
    if not submit_path.is_file():
        raise FileNotFoundError(submit_path)
    result = subprocess.run(
        ["bash", str(submit_path)], check=True, capture_output=True, text=True
    )
    response = result.stdout.strip()
    (output / "extract_submission.txt").write_text(response + "\n", encoding="utf-8")
    return response


def _manifest_inputs(campaign: Path) -> tuple[Path, list[str]]:
    manifest = campaign / "jobs.csv"
    if not manifest.exists():
        manifest = campaign / "spin_jobs.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"no jobs.csv or spin_jobs.csv found under {campaign}; run prepare or prepare-spins first"
        )

    inputs: list[str] = []
    seen: set[str] = set()
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "input" not in reader.fieldnames:
            raise ValueError(f"{manifest} has no input column")
        for row_number, row in enumerate(reader, start=2):
            raw = (row.get("input") or "").strip()
            if not raw:
                raise ValueError(f"{manifest}:{row_number} has an empty input path")
            relative = Path(raw)
            if relative.is_absolute():
                raise ValueError(f"{manifest}:{row_number} input must be relative to the campaign: {raw}")
            resolved = (campaign / relative).resolve()
            if campaign != resolved and campaign not in resolved.parents:
                raise ValueError(f"{manifest}:{row_number} input escapes the campaign: {raw}")
            if not resolved.is_file():
                raise FileNotFoundError(f"input listed by {manifest}:{row_number} does not exist: {raw}")
            normalized = str(relative)
            if normalized not in seen:
                seen.add(normalized)
                inputs.append(normalized)

    if not inputs:
        raise ValueError(f"{manifest} contains no inputs")
    stems: dict[str, str] = {}
    for item in inputs:
        stem = Path(item).stem
        if stem in stems:
            raise ValueError(
                f"input stems must be unique for output collection: {stems[stem]!r} and {item!r}"
            )
        stems[stem] = item
    return manifest, inputs


def _worker_script(default_command: str) -> str:
    command = shlex.quote(default_command)
    return f"""#!/usr/bin/env bash
set -euo pipefail

if (( $# != 5 )); then
  echo "Usage: $0 INPUT OUTPUT STATUS RC_FILE SCRATCH_DIR" >&2
  exit 2
fi

input=$1
output=$2
status=$3
rc_file=$4
scratch_dir=$5
gaussian_command=${{GAUSSIAN_COMMAND:-{command}}}

mkdir -p -- "$scratch_dir" "$(dirname -- "$output")"
cleanup_scratch() {{
  if [[ ${{KEEP_GAUSSIAN_SCRATCH:-0}} != 1 ]]; then
    rm -rf -- "$scratch_dir"
  fi
}}
trap cleanup_scratch EXIT

if [[ -n ${{GAUSSIAN_WORKER_INIT:-}} ]]; then
  if [[ ! -f $GAUSSIAN_WORKER_INIT ]]; then
    echo "Worker init file not found: $GAUSSIAN_WORKER_INIT" >&2
    printf 'ERROR worker_init_missing\n' > "$status"
    printf '2\n' > "$rc_file"
    exit 2
  fi
  # shellcheck source=/dev/null
  source "$GAUSSIAN_WORKER_INIT"
fi

export GAUSS_SCRDIR="$scratch_dir"
export GAUSS_USE_SLASHSCR=1
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export GAUSS_PDEF="${{SLURM_CPUS_PER_TASK:-1}}"
ulimit -s unlimited

started=$(date --iso-8601=seconds 2>/dev/null || date)
printf '%s\n' "$started" > "${{status%.status}}.started"
echo ">>> Starting $(basename -- "$input") on $(hostname) at $started"

set +e
time "$gaussian_command" < "$input" > "$output" 2>&1
command_rc=$?
set -e
printf '%s\n' "$command_rc" > "$rc_file"

finished=$(date --iso-8601=seconds 2>/dev/null || date)
printf '%s\n' "$finished" > "${{status%.status}}.finished"
if (( command_rc == 0 )) && grep -q 'Normal termination of Gaussian' "$output"; then
  printf 'OK\n' > "$status"
  echo "<<< Finished $(basename -- "$input") OK at $finished"
  exit 0
fi

if (( command_rc == 0 )); then
  printf 'ERROR missing_normal_termination\n' > "$status"
  echo "<<< Gaussian returned zero but no normal termination was found" >&2
  exit 1
fi
printf 'ERROR %s\n' "$command_rc" > "$status"
echo "<<< Gaussian failed with exit code $command_rc" >&2
exit "$command_rc"
"""


def _batch_script(config: SlurmConfig, batch_index: int) -> str:
    directives = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={_safe_directive(config.job_name, 'job name')}_{batch_index:04d}",
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks-per-node={config.concurrent_jobs}",
        f"#SBATCH --cpus-per-task={config.cpus_per_job}",
        f"#SBATCH --time={_safe_directive(config.time_limit, 'time limit')}",
        f"#SBATCH --partition={_safe_directive(config.partition, 'partition')}",
        f"#SBATCH --account={_safe_directive(config.account, 'account')}",
        "#SBATCH --output=scheduler-%j.stdout",
        "#SBATCH --error=scheduler-%j.stderr",
    ]
    if config.memory_per_node:
        directives.append(f"#SBATCH --mem={_safe_directive(config.memory_per_node, 'memory per node')}")
    module_line = ""
    if config.gaussian_module:
        # Lmod's `module` bash function references variables it doesn't
        # always guard (e.g. $LMOD_SETTARG_CMD) -- the same class of bug as
        # conda's activate/deactivate hooks referencing $CONDA_BACKUP_CXX --
        # so nounset must be relaxed around the load, not just here but for
        # the rest of the script too, since `module load` also unsets some
        # variables it never (re-)declared.
        module_line = f"set +u; module load {shlex.quote(config.gaussian_module)}; set -u"
    scratch_template = shlex.quote(config.scratch_root)
    body = f"""
set -euo pipefail

campaign_root=${{CLUSTER_MLIP_CAMPAIGN_ROOT:?Use the generated submit.sh so the campaign root is exported}}
campaign_root=$(cd -- "$campaign_root" && pwd -P)
batch_dir=${{CLUSTER_MLIP_BATCH_DIR:?Use the generated submit.sh so the batch directory is exported}}
batch_dir=$(cd -- "$batch_dir" && pwd -P)
batch_file="$batch_dir/inputs.txt"
worker="$campaign_root/run_gaussian_worker.sh"
run_policy=${{RUN_POLICY:-resume}}
cpus_per_job=${{SLURM_CPUS_PER_TASK:-{config.cpus_per_job}}}
runtime_user=${{USER:-$(id -un)}}
scratch_template={scratch_template}
scratch_template=${{scratch_template//\\$USER/$runtime_user}}
scratch_root=${{GAUSSIAN_SCRATCH_ROOT:-$scratch_template}}
scratch_root=${{scratch_root//\\$USER/$runtime_user}}

case "$run_policy" in
  resume|all) ;;
  *) echo "RUN_POLICY must be 'resume' or 'all', got: $run_policy" >&2; exit 2 ;;
esac
[[ -f $batch_file ]] || {{ echo "Missing batch input list: $batch_file" >&2; exit 2; }}
[[ -x $worker ]] || {{ echo "Missing worker: $worker" >&2; exit 2; }}
mkdir -p -- "$scratch_root/${{SLURM_JOB_ID}}"

{module_line}
export GAUSSIAN_COMMAND=${{GAUSSIAN_COMMAND:-{shlex.quote(config.gaussian_command)}}}
export GAUSS_USE_SLASHSCR=1
ulimit -s unlimited

mapfile -t inputs < "$batch_file"
echo "Batch: $(basename -- "$batch_dir"); ${{#inputs[@]}} inputs; policy=$run_policy"
echo "Campaign: $campaign_root"
echo "Host: $(hostname)"
date

fifo="${{TMPDIR:-/tmp}}/cluster_mlip_gaussian.${{SLURM_JOB_ID}}.fifo"
rm -f -- "$fifo"
mkfifo "$fifo"
exec 9<>"$fifo"
rm -f -- "$fifo"
for ((slot=0; slot<{config.concurrent_jobs}; slot++)); do printf '\n' >&9; done
cleanup() {{ exec 9>&- 9<&- || true; }}
trap cleanup EXIT

for input_name in "${{inputs[@]}}"; do
  [[ -n $input_name ]] || continue
  input="$batch_dir/$input_name"
  base=$(basename -- "${{input_name%.*}}")
  output="$batch_dir/${{base}}.log"
  status="$batch_dir/${{base}}.status"
  rc_file="$batch_dir/${{base}}.rc"

  if [[ $run_policy == resume && -s $output ]] && grep -q 'Normal termination of Gaussian' "$output"; then
    printf 'OK\n' > "$status"
    echo "SKIP complete: $input_name"
    continue
  fi

  read -r -u 9
  {{
    trap 'printf "\\n" >&9' EXIT
    job_scratch="$scratch_root/${{SLURM_JOB_ID}}/${{base}}"
    set +e
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="$cpus_per_job" \
      --kill-on-bad-exit=0 "$worker" "$input" "$output" "$status" "$rc_file" "$job_scratch"
    worker_rc=$?
    set -e
    if (( worker_rc != 0 )); then
      echo "Worker failed for $input_name (rc=$worker_rc)" >&2
    fi
  }} &
done
wait

complete=0
failed=0
for input_name in "${{inputs[@]}}"; do
  [[ -n $input_name ]] || continue
  base=$(basename -- "${{input_name%.*}}")
  output="$batch_dir/${{base}}.log"
  status="$batch_dir/${{base}}.status"
  if [[ -s $output ]] && grep -q 'Normal termination of Gaussian' "$output"; then
    ((complete += 1))
  else
    ((failed += 1))
    echo "INCOMPLETE: $base ($(<"$status" 2>/dev/null || echo no-status))" >&2
  fi
done

echo "$(basename -- "$batch_dir") summary: complete=$complete incomplete=$failed"
date
(( failed == 0 ))
"""
    return "\n".join(directives) + body


def _batch_submit_script(worker_init_name: str | None) -> str:
    worker_default = f'"$campaign_root/{worker_init_name}"' if worker_init_name else ""
    return f"""#!/usr/bin/env bash
set -euo pipefail

batch_dir=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd -P)
campaign_root=$(cd -- "$batch_dir/../.." && pwd -P)
run_policy=${{RUN_POLICY:-resume}}
worker_init=${{GAUSSIAN_WORKER_INIT:-{worker_default}}}
export_spec="ALL,CLUSTER_MLIP_CAMPAIGN_ROOT=$campaign_root,CLUSTER_MLIP_BATCH_DIR=$batch_dir,RUN_POLICY=$run_policy"
if [[ -n $worker_init ]]; then
  export_spec+=",GAUSSIAN_WORKER_INIT=$worker_init"
fi
sbatch "$@" --chdir="$batch_dir" --export="$export_spec" "$batch_dir/run_batch.sbatch"
"""


def _submit_all_script(batch_count: int) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

campaign_root=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd -P)
run_policy=${{RUN_POLICY:-resume}}
submitted=0
skipped=0
for batch_number in $(seq 1 {batch_count}); do
  batch_dir=$(printf '%s/slurm_batches/batch_%04d' "$campaign_root" "$batch_number")
  if [[ $run_policy == resume ]]; then
    incomplete=0
    while IFS= read -r input_name; do
      [[ -n $input_name ]] || continue
      base=${{input_name%.*}}
      output="$batch_dir/${{base}}.log"
      if [[ ! -s $output ]] || ! grep -q 'Normal termination of Gaussian' "$output"; then
        incomplete=1
        break
      fi
    done < "$batch_dir/inputs.txt"
    if (( incomplete == 0 )); then
      echo "SKIP complete batch: $(basename -- "$batch_dir")"
      ((skipped += 1))
      continue
    fi
  fi
  result=$("$batch_dir/submit.sh" "$@")
  echo "$(basename -- "$batch_dir"): $result"
  ((submitted += 1))
done
echo "Submitted $submitted batch job(s); skipped $skipped complete batch(es)."
"""


def _status_script(batch_count: int) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

campaign_root=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd -P)
total_planned=0
total_normal=0
total_errors=0
printf '%-12s %8s %8s %10s %8s\n' batch planned normal incomplete errors
for batch_number in $(seq 1 {batch_count}); do
  batch_dir=$(printf '%s/slurm_batches/batch_%04d' "$campaign_root" "$batch_number")
  planned=$(awk 'NF {{count++}} END {{print count+0}}' "$batch_dir/inputs.txt")
  normal=0
  errors=0
  while IFS= read -r input_name; do
    [[ -n $input_name ]] || continue
    base=${{input_name%.*}}
    output="$batch_dir/${{base}}.log"
    status="$batch_dir/${{base}}.status"
    if [[ -s $output ]] && grep -q 'Normal termination of Gaussian' "$output"; then
      ((normal += 1))
    elif [[ -f $status ]] && grep -q '^ERROR' "$status"; then
      ((errors += 1))
    fi
  done < "$batch_dir/inputs.txt"
  incomplete=$((planned - normal))
  printf '%-12s %8d %8d %10d %8d\n' "$(basename -- "$batch_dir")" "$planned" "$normal" "$incomplete" "$errors"
  total_planned=$((total_planned + planned))
  total_normal=$((total_normal + normal))
  total_errors=$((total_errors + errors))
done
printf '%-12s %8d %8d %10d %8d\n' TOTAL "$total_planned" "$total_normal" "$((total_planned-total_normal))" "$total_errors"
"""


def prepare_slurm_batches(
    campaign: Path,
    config: SlurmConfig,
    *,
    worker_init: Path | None = None,
    allow_nproc_mismatch: bool = False,
) -> SlurmPlan:
    campaign = campaign.resolve()
    if not campaign.is_dir():
        raise NotADirectoryError(campaign)
    for name, value in (
        ("jobs_per_batch", config.jobs_per_batch),
        ("concurrent_jobs", config.concurrent_jobs),
        ("cpus_per_job", config.cpus_per_job),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    manifest, inputs = _manifest_inputs(campaign)
    warnings: list[str] = []
    mismatches: list[str] = []
    missing_nproc: list[str] = []
    for item in inputs:
        text = (campaign / item).read_text(encoding="utf-8", errors="ignore")
        match = _NPROC_RE.search(text)
        if match is None:
            missing_nproc.append(item)
        elif int(match.group(1)) != config.cpus_per_job:
            mismatches.append(f"{item} declares {match.group(1)}")
    if missing_nproc:
        warnings.append(
            f"{len(missing_nproc)} input(s) have no %nprocshared directive; Slurm still allocates "
            f"{config.cpus_per_job} CPUs/job"
        )
    if mismatches and not allow_nproc_mismatch:
        preview = ", ".join(mismatches[:5])
        if len(mismatches) > 5:
            preview += f", ... ({len(mismatches)} total)"
        raise ValueError(
            f"--cpus-per-job={config.cpus_per_job} disagrees with Gaussian inputs: {preview}; "
            "regenerate inputs with matching --nproc, change --cpus-per-job, or explicitly pass "
            "--allow-nproc-mismatch"
        )
    if mismatches:
        warnings.append(f"{len(mismatches)} input(s) disagree with --cpus-per-job")

    input_fingerprint = hashlib.sha256("\0".join(inputs).encode()).hexdigest()
    old_plan_path = campaign / "slurm_plan.json"
    old_output_root = campaign / "slurm_outputs"
    batch_root = campaign / "slurm_batches"
    outputs_exist = (
        old_output_root.exists() and any(path.is_file() for path in old_output_root.rglob("*"))
    ) or (
        batch_root.exists()
        and any(
            path.is_file() and path.suffix in {".log", ".status", ".rc"}
            for path in batch_root.rglob("*")
        )
    )
    if outputs_exist:
        if not old_plan_path.exists():
            raise RuntimeError(
                "Slurm batch outputs already exist but no prior slurm_plan.json exists; "
                "move or archive those outputs before generating a new batch map"
            )
        old_plan = json.loads(old_plan_path.read_text(encoding="utf-8"))
        old_config = old_plan.get("config", {})
        same_mapping = (
            old_plan.get("input_fingerprint") == input_fingerprint
            and old_config.get("jobs_per_batch") == config.jobs_per_batch
        )
        if not same_mapping:
            raise RuntimeError(
                "refusing to change the input list or jobs-per-batch after Slurm outputs exist; "
                "a reshuffle could duplicate labels across batch directories"
            )
    batches = [
        inputs[start : start + config.jobs_per_batch]
        for start in range(0, len(inputs), config.jobs_per_batch)
    ]
    worker_init_name = None
    if worker_init is not None:
        worker_init = worker_init.resolve()
        if not worker_init.is_file():
            raise FileNotFoundError(worker_init)
        worker_init_name = "gaussian_worker_init.sh"
        destination = campaign / worker_init_name
        if worker_init != destination:
            shutil.copy2(worker_init, destination)
        destination.chmod(0o755)

    batch_root.mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(batches, start=1):
        batch_dir = batch_root / f"batch_{index:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        for stale_link in batch_dir.iterdir():
            # Symlink on Linux/LONI; a real copy on Windows when unprivileged
            # symlink creation isn't available (see the copy2 fallback
            # below) -- either way, this directory only ever holds inputs
            # this function itself generated, so both are safe to clear
            # before repopulating.
            if stale_link.suffix.lower() in {".gjf", ".com"} and (
                stale_link.is_symlink() or stale_link.is_file()
            ):
                stale_link.unlink()
        input_names: list[str] = []
        for item in batch:
            source = campaign / item
            input_name = source.name
            destination = batch_dir / input_name
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(
                    f"refusing to replace non-generated batch input path: {destination}"
                )
            try:
                destination.symlink_to(os.path.relpath(source, batch_dir))
            except OSError:
                # Unprivileged symlink creation is disabled by default on
                # Windows (no admin rights / Developer Mode) -- harmless to
                # fall back to a real copy there, since this is purely a
                # local-development convenience: the target deployment
                # platform (Linux/LONI) always supports symlinks for a
                # regular user, so this path is never exercised in
                # production and never silently changes production behavior.
                shutil.copy2(source, destination)
            input_names.append(input_name)
        (batch_dir / "inputs.txt").write_text(
            "\n".join(input_names) + "\n", encoding="utf-8"
        )
        for name, content in {
            "run_batch.sbatch": _batch_script(config, index),
            "submit.sh": _batch_submit_script(worker_init_name),
        }.items():
            path = batch_dir / name
            path.write_text(content, encoding="utf-8")
            path.chmod(0o755)

    generated = {
        "run_gaussian_worker.sh": _worker_script(config.gaussian_command),
        "submit_gaussian_batches.sh": _submit_all_script(len(batches)),
        "gaussian_batch_status.sh": _status_script(len(batches)),
    }
    for name, content in generated.items():
        path = campaign / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)
    for obsolete in (
        "run_gaussian_array.sbatch",
        "submit_gaussian_array.sh",
        "gaussian_array_status.sh",
    ):
        obsolete_path = campaign / obsolete
        if obsolete_path.exists():
            obsolete_path.unlink()

    plan: SlurmPlan = {
        "campaign": str(campaign),
        "manifest": manifest.name,
        "input_count": len(inputs),
        "batch_count": len(batches),
        "config": asdict(config),
        "worker_init": worker_init_name,
        "input_fingerprint": input_fingerprint,
        "warnings": warnings,
    }
    (campaign / "slurm_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return plan
