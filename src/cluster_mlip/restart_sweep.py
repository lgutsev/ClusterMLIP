"""One unattended pass that restarts optimizer step-limit failures.

A minimum search that stopped because Gaussian ran out of geometry steps
(link 9999 after "Number of steps exceeded") has a good checkpoint and needs
nothing but more steps. ``restart_sweep`` finds those inputs and hands them to
``prepare_spin_restarts``: same route, same limits, a checkpoint restart in
the input's own batch folder. Everything else is left alone and reported.

The pass is deliberately narrow:

* only batches with no Slurm job, in two queue snapshots, the second taken
  just before anything is written; if the queue cannot be read, nothing is
  written;
* only minimum searches (saddles need a person to read the triage plan),
  and never a point on a reaction path, which should not be optimized at all;
* a stage that is itself the first stage of a restart is never restarted
  again: one attempt per stage;
* a failed step with more than half of its SCFs unconverged is parked, not
  restarted;
* a cumulative cap on restarted stages, kept in a ledger in the campaign;
* a lock file, so two passes never write the manifest at once.

It prepares restarts and lists the batches to resubmit; it never submits.
Resubmitting a batch reruns every listed input that is not complete, so the
report also names the failed inputs it left alone in those batches
(``rerun_on_resubmit``): hold them first, or they start again from scratch.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import socket
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .gaussian import _CM_RE, _SCF_NOT_CONVERGED_RE, _SCF_RE
from .io import write_text_lf
from .restart import _active, _batch_locations, _read_csv, first_unfinished_stage, prepare_spin_restarts
from .routes import (
    config_type_from_stem,
    geometry_role_for_config_type,
    input_stage_routes,
    intended_stationary_point,
    route_search_kind,
)
from .spin import parse_spin_diagnostics

EXIT_CONTINUE = 0
EXIT_STOPPED = 10
EXIT_NOTHING_DONE = 11

LOCK_NAME = "restart_sweep.lock"
STATE_DIR = "restart_sweep"
LEDGER_NAME = "ledger.json"
PENDING_NAME = "pending_submit.txt"
PARKED_NAME = "parked.csv"

REPEAT_MINIMUM_DECIDED = 8
REPEAT_MAXIMUM_FRACTION = 0.25
PARK_UNCONVERGED_FRACTION = 0.5

_STEPS_EXCEEDED_RE = re.compile(r"Number of steps exceeded", re.I)
_LINK_9999_RE = re.compile(
    r"Error termination request processed by link 9999"
    r"|Error termination via Lnk1e in \S*?l9999\.exe",
    re.I,
)
_ERROR_RE = re.compile(r"Error termination", re.I)
_JOB_NAME_RE = re.compile(r"^#SBATCH\s+--job-name=(\S+)", re.M)

QueueReader = Callable[[], list[dict[str, str]]]


class QueueUnreadable(RuntimeError):
    """The scheduler could not be asked which jobs exist."""


def squeue_reader(commands: list[list[str]] | None = None) -> QueueReader:
    """Read every job of this user from one or more squeue commands.

    Each command must print ``jobid|name|state|workdir`` lines (the default
    does). Give one command per scheduler that can run this campaign.
    """
    commands = commands or [["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%j|%T|%Z"]]

    def read() -> list[dict[str, str]]:
        jobs: list[dict[str, str]] = []
        for command in commands:
            try:
                done = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
            except (OSError, subprocess.SubprocessError) as error:
                raise QueueUnreadable(f"{' '.join(command)}: {error}") from error
            if done.returncode != 0:
                raise QueueUnreadable(
                    f"{' '.join(command)}: exit {done.returncode}: {done.stderr.strip()[:200]}")
            for line in done.stdout.splitlines():
                if not line.strip():
                    continue
                parts = line.strip().split("|", 3)
                if len(parts) != 4:
                    raise QueueUnreadable(f"{' '.join(command)}: unexpected line: {line[:200]}")
                jobs.append(dict(zip(("job_id", "name", "state", "workdir"), parts)))
        return jobs

    return read


def _busy_batches(campaign: Path, jobs: list[dict[str, str]]) -> dict[str, str]:
    """``batch name -> job id`` for every batch folder a queued or running job uses.

    A job belongs to a batch when it runs in that folder or carries the job
    name the batch script declares; either is enough.
    """
    busy: dict[str, str] = {}
    by_dir: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for job in jobs:
        try:
            by_dir[str(Path(job["workdir"]).resolve())] = job["job_id"]
        except OSError:
            pass
        by_name[job["name"]] = job["job_id"]
    for batch in sorted((campaign / "slurm_batches").glob("batch_*")):
        job_id = by_dir.get(str(batch.resolve()), "")
        script = batch / "run_batch.sbatch"
        if not job_id and script.is_file():
            declared = _JOB_NAME_RE.search(script.read_text(encoding="utf-8", errors="replace"))
            if declared:
                job_id = by_name.get(declared.group(1), "")
        if job_id:
            busy[batch.name] = job_id
    return busy


def _failing_section(text: str) -> str:
    """The log text of the last charge/multiplicity section Gaussian started."""
    sections = list(_CM_RE.finditer(text))
    return text[sections[-1].start():] if sections else text


def classify_log(text: str) -> dict[str, Any]:
    """How the last section of a log ended, and how its SCFs behaved."""
    section = _failing_section(text)
    scf_unconverged = len(_SCF_NOT_CONVERGED_RE.findall(section))
    # An SCF that gave up without printing "SCF Done" still ran: count it, so
    # the unconverged ones are never more than the total.
    scf_total = max(len(_SCF_RE.findall(section)), scf_unconverged)
    if not _ERROR_RE.search(section):
        ending = "no_error_termination"
    elif _LINK_9999_RE.search(section) and _STEPS_EXCEEDED_RE.search(section):
        ending = "step_limit"
    else:
        ending = "other_error"
    return {"ending": ending, "scf_total": scf_total, "scf_unconverged": scf_unconverged}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_ledger(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema_version": 1, "stopped": "", "restarted_inputs": 0, "restarted_stages": 0,
            "restarts": [], "parked": {}, "sweeps": []}


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write_text_lf(temporary, json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _restart_outcomes(campaign: Path, ledger: dict[str, Any]) -> Counter[str]:
    """What became of the restarts earlier sweeps wrote: ok, repeat, other, pending."""
    outcomes: Counter[str] = Counter()
    for entry in ledger["restarts"]:
        log = campaign / entry["new_output"]
        if not log.is_file() or log.stat().st_size == 0:
            outcomes["pending"] += 1
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        diagnostics = parse_spin_diagnostics(text)
        first = diagnostics[0] if diagnostics else None
        if first is not None and first.normal_termination and first.optimized:
            outcomes["ok"] += 1
            continue
        # Only the first stage counts: it is the one that ran out of steps.
        sections = list(_CM_RE.finditer(text))
        first_text = text[:sections[1].start()] if len(sections) > 1 else text
        ending = classify_log(first_text)["ending"]
        outcomes[{"step_limit": "repeat", "other_error": "other"}.get(ending, "pending")] += 1
    return outcomes


def restart_sweep(
    campaign: Path,
    *,
    max_stages: int,
    read_queue: QueueReader | None = None,
    execute: bool = False,
    start: int = 1,
    end: int | None = None,
    parked_inputs: set[str] | None = None,
    clear_stop: str = "",
) -> dict[str, Any]:
    """Run one sweep. Returns the report; ``report["exit_code"]`` is the verdict.

    Without ``execute`` nothing in the campaign changes: the report says what
    a real pass would restart. ``max_stages`` is the cumulative cap on
    restarted stages over every sweep of this campaign.
    """
    campaign = campaign.resolve()
    read_queue = read_queue or squeue_reader()
    state = campaign / STATE_DIR
    ledger_path = state / LEDGER_NAME
    lock = campaign / LOCK_NAME
    report: dict[str, Any] = {
        "campaign": str(campaign), "started_utc": _now(), "execute": execute,
        "max_stages": max_stages, "status": "", "exit_code": EXIT_CONTINUE,
        "plan": [], "left_alone": [], "counts": {}, "pending_submit": [],
        "rerun_on_resubmit": [],
    }

    def finish(status: str, code: int) -> dict[str, Any]:
        report["status"], report["exit_code"] = status, code
        report["finished_utc"] = _now()
        return report

    if execute:
        state.mkdir(exist_ok=True)
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            report["lock"] = lock.read_text(encoding="utf-8", errors="replace").strip()
            return finish("lock_held", EXIT_NOTHING_DONE)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(f"restart-sweep pid {os.getpid()} on {socket.gethostname()} at {_now()}\n")
    elif lock.exists():
        report["lock"] = lock.read_text(encoding="utf-8", errors="replace").strip()
    try:
        return _sweep(campaign, report, finish, ledger_path, max_stages=max_stages,
                      read_queue=read_queue, execute=execute, start=start, end=end,
                      parked_inputs=parked_inputs or set(), clear_stop=clear_stop)
    finally:
        if execute:
            try:
                _write_json(state / f"report_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json", report)
            finally:
                lock.unlink()


def _sweep(
    campaign: Path, report: dict[str, Any], finish: Callable[[str, int], dict[str, Any]],
    ledger_path: Path, *, max_stages: int, read_queue: QueueReader, execute: bool,
    start: int, end: int | None, parked_inputs: set[str], clear_stop: str,
) -> dict[str, Any]:
    state = ledger_path.parent
    ledger = _load_ledger(ledger_path)

    def stop(reason: str) -> dict[str, Any]:
        if execute:
            ledger["stopped"] = reason
            ledger["stopped_utc"] = _now()
            _write_json(ledger_path, ledger)
        report["stopped"] = reason
        return finish("stopped", EXIT_STOPPED)

    if ledger["stopped"]:
        if not clear_stop:
            report["stopped"] = ledger["stopped"]
            return finish("stopped", EXIT_STOPPED)
        ledger.setdefault("cleared_stops", []).append(
            {"stopped": ledger["stopped"], "cleared_utc": _now(), "note": clear_stop})
        ledger["stopped"] = ""

    outcomes = _restart_outcomes(campaign, ledger)
    report["earlier_restarts"] = dict(outcomes)
    decided = outcomes["ok"] + outcomes["repeat"] + outcomes["other"]
    if decided >= REPEAT_MINIMUM_DECIDED and outcomes["repeat"] > REPEAT_MAXIMUM_FRACTION * decided:
        return stop(f"repeat_step_limit_rate: {outcomes['repeat']} of {decided} restarted stages "
                    "ran out of steps again")

    try:
        busy = _busy_batches(campaign, read_queue())
    except QueueUnreadable as error:
        report["queue_error"] = str(error)
        return finish("queue_unreadable", EXIT_NOTHING_DONE)

    manifest = campaign / "spin_jobs.csv"
    manifest_read_sha = _sha256(manifest)
    _, all_rows = _read_csv(manifest)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        if _active(row):
            grouped[row["input"]].append(row)
    locations = _batch_locations(campaign)
    parked: dict[str, str] = dict(ledger["parked"])
    candidates: list[dict[str, Any]] = []
    left: list[dict[str, str]] = report["left_alone"]
    counts: Counter[str] = Counter()

    for input_path, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["stage_index"]))
        name = Path(input_path).name
        batch = locations.get(name)
        if batch is None:
            continue
        number = int(batch.name[6:])
        if number < start or (end is not None and number > end):
            continue
        log = batch / f"{Path(name).stem}.log"
        if not log.is_file() or log.stat().st_size == 0:
            counts["not_started"] += 1
            continue
        if batch.name in busy:
            counts["batch_has_job"] += 1
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        source_input = campaign / input_path
        input_text = source_input.read_text(encoding="utf-8", errors="replace") if source_input.is_file() else ""
        unfinished = first_unfinished_stage(rows, input_text, parse_spin_diagnostics(text))
        if unfinished is None:
            counts["complete"] += 1
            continue
        verdict = classify_log(text)

        def leave(reason: str) -> None:
            counts[reason.split(":")[0]] += 1
            left.append({"input": name, "batch": batch.name, "reason": reason})

        if verdict["ending"] != "step_limit":
            leave(verdict["ending"])
            continue
        row = rows[unfinished]
        routes = input_stage_routes(input_text)
        route = routes[unfinished] if unfinished < len(routes) else ""
        config_type = row.get("config_type") or config_type_from_stem(name)
        if geometry_role_for_config_type(config_type) == "irc_point":
            # A point on a reaction path is a fixed geometry. Optimizing it was
            # the mistake; giving that optimization more steps would repeat it.
            leave("path_point")
            continue
        if (route_search_kind(route) != "minimum"
                or intended_stationary_point(config_type) == "saddle"):
            leave("not_a_minimum_search")
            continue
        if name in parked_inputs or name in parked:
            leave("parked")
            continue
        total, bad = verdict["scf_total"], verdict["scf_unconverged"]
        if total and bad > PARK_UNCONVERGED_FRACTION * total:
            parked[name] = f"{bad} of {total} SCFs unconverged in the failed step"
            leave(f"parked: {parked[name]}")
            continue
        if unfinished == 0 and row.get("initialization") == "interrupted_checkpoint_restart":
            leave("repeat_step_limit: this stage was already restarted once")
            continue
        candidates.append({
            "input": name, "input_path": input_path, "batch": batch.name,
            "multiplicity": row["intended_multiplicity"], "stages": len(rows) - unfinished,
            "scf_total": total, "scf_unconverged": bad,
        })

    report["counts"] = dict(counts)
    report["busy_batches"] = len(busy)
    budget = max_stages - int(ledger["restarted_stages"])
    chosen: list[dict[str, Any]] = []
    over_cap = False
    for candidate in candidates:
        if candidate["stages"] > budget:
            over_cap = True
            left.append({"input": candidate["input"], "batch": candidate["batch"], "reason": "cap"})
            continue
        budget -= candidate["stages"]
        chosen.append(candidate)

    dry: dict[str, Any] = {"plan": [], "skipped": []}
    if chosen:
        try:
            dry = dict(prepare_spin_restarts(
                campaign, start=start, end=end, dry_run=True,
                only_inputs={item["input"] for item in chosen}))
        except FileExistsError as error:
            # A restart name already on disk: the folder is not in the state
            # the manifest describes. Nothing has been written.
            report["refusal"] = str(error)
            return stop(f"refusal: {error}")
    planned = {Path(row["archived_input"]).name: row for row in dry["plan"]}
    wrong_mode = [row for row in dry["plan"] if row.get("mode") != "checkpoint_restart"]
    if wrong_mode:
        return stop(f"unexpected_restart_mode: {wrong_mode[0].get('mode')} for "
                    f"{Path(wrong_mode[0]['archived_input']).name}")
    for row in dry["skipped"]:
        left.append({"input": Path(row["input"]).name, "batch": row["batch"],
                     "reason": f"restart_skipped: {row['reason']}"})
    chosen = [item for item in chosen if item["input"] in planned]
    for item in chosen:
        item.update({key: planned[item["input"]][key] for key in (
            "source_checkpoint", "seed_checkpoint", "new_input", "new_output")})
    report["plan"] = chosen

    if not execute:
        if over_cap:
            report["cap_would_be_reached"] = True
        return finish("dry_run", EXIT_CONTINUE)

    if parked != ledger["parked"]:
        ledger["parked"] = parked
        write_text_lf(state / PARKED_NAME, "input,reason\n" + "".join(
            f"{name},{reason}\n" for name, reason in sorted(parked.items())))
    if chosen:
        try:
            busy_again = _busy_batches(campaign, read_queue())
        except QueueUnreadable as error:
            report["queue_error"] = str(error)
            _write_json(ledger_path, ledger)
            return finish("queue_unreadable", EXIT_NOTHING_DONE)
        started = [item for item in chosen if item["batch"] in busy_again]
        for item in started:
            left.append({"input": item["input"], "batch": item["batch"],
                         "reason": "batch_got_a_job_between_snapshots"})
        chosen = [item for item in chosen if item["batch"] not in busy_again]
        report["plan"] = chosen
    if chosen:
        before = _sha256(manifest)
        if before != manifest_read_sha:
            return stop("manifest_changed_during_sweep")
        _write_json(state / f"plan_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json", chosen)
        try:
            result = prepare_spin_restarts(
                campaign, start=start, end=end,
                only_inputs={item["input"] for item in chosen})
        except FileExistsError as error:
            report["refusal"] = str(error)
            return stop(f"refusal: {error}")
        report["manifest_sha256_before"] = before
        report["manifest_sha256_after"] = _sha256(manifest)
        written = {Path(row["archived_input"]).name for row in result["plan"]}
        if written != {item["input"] for item in chosen}:
            report["written"] = sorted(written)
            return stop("restart_wrote_a_different_set_than_planned")
        ledger["restarted_inputs"] += len(chosen)
        ledger["restarted_stages"] += sum(item["stages"] for item in chosen)
        ledger["restarts"].extend({**item, "sweep_utc": report["started_utc"]} for item in chosen)
        pending_path = state / PENDING_NAME
        pending = set(pending_path.read_text(encoding="utf-8").split()) if pending_path.is_file() else set()
        pending |= {item["batch"] for item in chosen}
        write_text_lf(pending_path, "".join(f"{name}\n" for name in sorted(pending)))
        report["pending_submit"] = sorted(pending)
        # A resubmitted batch reruns every listed input that is not complete,
        # from scratch. Name the ones this pass left alone in those batches,
        # so they can be held (hold-inputs --with-output) before the submit.
        touched = {item["batch"] for item in chosen}
        report["rerun_on_resubmit"] = [row for row in left if row["batch"] in touched]
    ledger["sweeps"].append({
        "utc": report["started_utc"], "restarted_inputs": len(chosen),
        "restarted_stages": sum(item["stages"] for item in chosen), "counts": dict(counts),
    })
    _write_json(ledger_path, ledger)
    report["ledger"] = {key: ledger[key] for key in ("restarted_inputs", "restarted_stages")}
    if over_cap:
        return stop(f"cap_reached: {ledger['restarted_stages']} of {max_stages} stages used and "
                    "the next restart does not fit")
    return finish("swept", EXIT_CONTINUE)
