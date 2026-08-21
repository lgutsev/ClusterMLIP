from __future__ import annotations

import csv
import hashlib
import json
import re
import shlex
import shutil
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
    account: str = "loni_dspm_25"
    gaussian_module: str = "gaussian/g16-c01"
    gaussian_command: str = "g16"
    job_name: str = "cluster_mlip_g16"
    memory_per_node: str | None = None
    array_concurrency: int | None = None
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


def _safe_directive(value: str, name: str) -> str:
    if not value or any(character in value for character in "\r\n"):
        raise ValueError(f"{name} must be a non-empty single line")
    return value


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


def _array_script(config: SlurmConfig, batch_count: int) -> str:
    array = f"1-{batch_count}"
    if config.array_concurrency is not None:
        array += f"%{config.array_concurrency}"
    directives = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={_safe_directive(config.job_name, 'job name')}",
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks-per-node={config.concurrent_jobs}",
        f"#SBATCH --cpus-per-task={config.cpus_per_job}",
        f"#SBATCH --time={_safe_directive(config.time_limit, 'time limit')}",
        f"#SBATCH --partition={_safe_directive(config.partition, 'partition')}",
        f"#SBATCH --account={_safe_directive(config.account, 'account')}",
        f"#SBATCH --array={array}",
        "#SBATCH --output=slurm_logs/gaussian-%A_%a.out",
        "#SBATCH --error=slurm_logs/gaussian-%A_%a.err",
    ]
    if config.memory_per_node:
        directives.append(f"#SBATCH --mem={_safe_directive(config.memory_per_node, 'memory per node')}")
    module_line = ""
    if config.gaussian_module:
        module_line = f"module load {shlex.quote(config.gaussian_module)}"
    scratch_template = shlex.quote(config.scratch_root)
    body = f"""
set -euo pipefail

campaign_root=${{CLUSTER_MLIP_CAMPAIGN_ROOT:?Use ./submit_gaussian_array.sh so the campaign root is exported}}
campaign_root=$(cd -- "$campaign_root" && pwd -P)
batch_id=$(printf '%04d' "${{SLURM_ARRAY_TASK_ID:?}}")
batch_file="$campaign_root/slurm_batches/batch_${{batch_id}}.txt"
output_dir="$campaign_root/slurm_outputs/batch_${{batch_id}}"
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
[[ -f $batch_file ]] || {{ echo "Missing batch manifest: $batch_file" >&2; exit 2; }}
[[ -x $worker ]] || {{ echo "Missing worker: $worker" >&2; exit 2; }}
mkdir -p -- "$output_dir" "$scratch_root/${{SLURM_JOB_ID}}"

{module_line}
export GAUSSIAN_COMMAND=${{GAUSSIAN_COMMAND:-{shlex.quote(config.gaussian_command)}}}
export GAUSS_USE_SLASHSCR=1
ulimit -s unlimited

mapfile -t inputs < "$batch_file"
echo "Array task $SLURM_ARRAY_TASK_ID: ${{#inputs[@]}} inputs; policy=$run_policy"
echo "Campaign: $campaign_root"
echo "Host: $(hostname)"
date

fifo="${{TMPDIR:-/tmp}}/cluster_mlip_gaussian.${{SLURM_JOB_ID}}.${{SLURM_ARRAY_TASK_ID}}.fifo"
rm -f -- "$fifo"
mkfifo "$fifo"
exec 9<>"$fifo"
rm -f -- "$fifo"
for ((slot=0; slot<{config.concurrent_jobs}; slot++)); do printf '\n' >&9; done
cleanup() {{ exec 9>&- 9<&- || true; }}
trap cleanup EXIT

for input_rel in "${{inputs[@]}}"; do
  [[ -n $input_rel ]] || continue
  input="$campaign_root/$input_rel"
  base=$(basename -- "${{input_rel%.*}}")
  output="$output_dir/${{base}}.log"
  status="$output_dir/${{base}}.status"
  rc_file="$output_dir/${{base}}.rc"

  if [[ $run_policy == resume && -s $output ]] && grep -q 'Normal termination of Gaussian' "$output"; then
    printf 'OK\n' > "$status"
    echo "SKIP complete: $input_rel"
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
      echo "Worker failed for $input_rel (rc=$worker_rc)" >&2
    fi
  }} &
done
wait

complete=0
failed=0
for input_rel in "${{inputs[@]}}"; do
  [[ -n $input_rel ]] || continue
  base=$(basename -- "${{input_rel%.*}}")
  output="$output_dir/${{base}}.log"
  status="$output_dir/${{base}}.status"
  if [[ -s $output ]] && grep -q 'Normal termination of Gaussian' "$output"; then
    ((complete += 1))
  else
    ((failed += 1))
    echo "INCOMPLETE: $base ($(<"$status" 2>/dev/null || echo no-status))" >&2
  fi
done

echo "Batch $batch_id summary: complete=$complete incomplete=$failed"
date
(( failed == 0 ))
"""
    return "\n".join(directives) + body


def _submit_script(worker_init_name: str | None) -> str:
    worker_default = f'"$campaign_root/{worker_init_name}"' if worker_init_name else ""
    return f"""#!/usr/bin/env bash
set -euo pipefail

campaign_root=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd -P)
mkdir -p -- "$campaign_root/slurm_logs"
run_policy=${{RUN_POLICY:-resume}}
worker_init=${{GAUSSIAN_WORKER_INIT:-{worker_default}}}
export_spec="ALL,CLUSTER_MLIP_CAMPAIGN_ROOT=$campaign_root,RUN_POLICY=$run_policy"
if [[ -n $worker_init ]]; then
  export_spec+=",GAUSSIAN_WORKER_INIT=$worker_init"
fi
sbatch "$@" --chdir="$campaign_root" --export="$export_spec" \
  "$campaign_root/run_gaussian_array.sbatch"
"""


def _status_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

campaign_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
planned=$(awk 'NF {count++} END {print count+0}' "$campaign_root"/slurm_batches/batch_*.txt)
normal=0
errors=0
if [[ -d $campaign_root/slurm_outputs ]]; then
  while IFS= read -r -d '' output; do
    if grep -q 'Normal termination of Gaussian' "$output"; then ((normal += 1)); fi
  done < <(find "$campaign_root/slurm_outputs" -type f -name '*.log' -print0)
  while IFS= read -r -d '' status; do
    if grep -q '^ERROR' "$status"; then ((errors += 1)); fi
  done < <(find "$campaign_root/slurm_outputs" -type f -name '*.status' -print0)
fi
pending=$((planned - normal))
(( pending < 0 )) && pending=0
printf 'planned=%d normal=%d incomplete=%d error_status=%d\n' "$planned" "$normal" "$pending" "$errors"
if (( errors > 0 )); then
  echo "Error status files:"
  grep -l '^ERROR' "$campaign_root"/slurm_outputs/batch_*/*.status 2>/dev/null || true
fi
"""


def prepare_slurm_array(
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
    if config.array_concurrency is not None and config.array_concurrency < 1:
        raise ValueError("array_concurrency must be at least 1")

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
    outputs_exist = (campaign / "slurm_outputs").exists() and any(
        path.is_file() for path in (campaign / "slurm_outputs").rglob("*")
    )
    if outputs_exist:
        if not old_plan_path.exists():
            raise RuntimeError(
                "slurm_outputs already contains files but no prior slurm_plan.json exists; "
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
    batch_dir = campaign / "slurm_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for stale in batch_dir.glob("batch_*.txt"):
        stale.unlink()
    for index, batch in enumerate(batches, start=1):
        (batch_dir / f"batch_{index:04d}.txt").write_text("\n".join(batch) + "\n", encoding="utf-8")

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

    generated = {
        "run_gaussian_worker.sh": _worker_script(config.gaussian_command),
        "run_gaussian_array.sbatch": _array_script(config, len(batches)),
        "submit_gaussian_array.sh": _submit_script(worker_init_name),
        "gaussian_array_status.sh": _status_script(),
    }
    for name, content in generated.items():
        path = campaign / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)

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
