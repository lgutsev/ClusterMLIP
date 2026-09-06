"""Read-only inspection of generated batch folders; no scheduler assumptions."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gaussian import gaussian_job_complete, parse_final_force_frame
from .spin import parse_spin_diagnostics

STATES = ('complete', 'failed', 'incomplete', 'activity_unconfirmed', 'not_started', 'missing_input')


def _optional(path: Path) -> str:
    return path.read_text(errors='replace').strip() if path.is_file() else ''


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_batch_progress(campaign: Path, destination: Path | None = None, *,
                         start: int = 1, end: int | None = None, audit: bool = False) -> dict[str, Any]:
    campaign = campaign.resolve()
    manifest = campaign / 'spin_jobs.csv'
    if not manifest.is_file():
        manifest = campaign / 'jobs.csv'
    if not manifest.is_file():
        raise FileNotFoundError(f'No jobs.csv or spin_jobs.csv in {campaign}')
    with manifest.open(newline='', encoding='utf-8') as handle:
        manifest_rows = list(csv.DictReader(handle))
    by_input: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in manifest_rows:
        by_input[Path(row['input']).name].append(row)
    plan_path = campaign / 'slurm_plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}
    directories = {int(p.name[6:]): p for p in (campaign / 'slurm_batches').glob('batch_*')
                   if p.is_dir() and p.name[6:].isdigit()}
    final = int(plan.get('batch_count', max(directories, default=0)))
    end = final if end is None else end
    if not 1 <= start <= end <= final:
        raise ValueError(f'Batch range must satisfy 1 <= start <= end <= {final}')
    issues: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    seen: set[str] = set()

    def issue(batch: str, name: str, severity: str, detail: str) -> None:
        issues.append(dict(batch=batch, input=name, severity=severity, detail=detail))

    for number in range(start, end + 1):
        batch_name = f'batch_{number:04d}'
        directory = campaign / 'slurm_batches' / batch_name
        names = _optional(directory / 'inputs.txt').splitlines()
        names = [name.strip() for name in names if name.strip()]
        if not names:
            issue(batch_name, '', 'error', 'Missing or empty inputs.txt; batch progress is unavailable')
        members: list[dict[str, Any]] = []
        for name in names:
            if Path(name).name != name or name in ('.', '..'):
                issue(batch_name, name, 'error', 'Invalid batch input name')
                continue
            if name in seen:
                issue(batch_name, name, 'error', 'Input occurs in more than one selected batch/list entry')
            seen.add(name)
            rows = by_input.get(name, [])
            if not rows:
                issue(batch_name, name, 'error', 'Input is absent from the campaign manifest')
            inp = directory / name
            if inp.is_file() and not inp.resolve().is_relative_to(campaign):
                issue(batch_name, name, 'error', 'Input link escapes campaign')
                continue
            input_text = _optional(inp)
            expected = 1 + len(re.findall(r'^\s*--link1--\s*$', input_text, re.I | re.M))
            output = directory / (Path(name).stem + '.log')
            text = _optional(output)
            rc = _optional(output.with_suffix('.rc'))
            status = _optional(output.with_suffix('.status'))
            started = output.with_suffix('.started')
            finished = output.with_suffix('.finished')
            # A resumed attempt must not inherit the previous attempt's status.
            if started.is_file():
                attempt_time = started.stat().st_mtime
                if output.is_file() and output.stat().st_mtime < attempt_time:
                    text = ''
                for suffix in ('.rc', '.status'):
                    marker = output.with_suffix(suffix)
                    if marker.is_file() and marker.stat().st_mtime < attempt_time:
                        if suffix == '.rc':
                            rc = ''
                        else:
                            status = ''
            diagnostics = parse_spin_diagnostics(text) if text else []
            if not inp.is_file():
                state = 'missing_input'
                issue(batch_name, name, 'error', 'Batch input is missing or its link is broken')
            elif gaussian_job_complete(text, expected) and (not rc or rc == '0'):
                state = 'complete'
            elif 'Error termination' in text or status.startswith('ERROR') or (rc and rc != '0'):
                state = 'failed'
            elif started.is_file() and (not finished.is_file() or started.stat().st_mtime > finished.stat().st_mtime):
                state = 'activity_unconfirmed'
            elif text:
                state = 'incomplete'
            else:
                state = 'not_started'
            spin_rows = [r for r in rows if r.get('intended_multiplicity')]
            completed_stages = sum(any(
                d.charge == int(r['intended_charge'])
                and d.multiplicity == int(r['intended_multiplicity'])
                and d.normal_termination for d in diagnostics) for r in spin_rows)
            if not spin_rows:
                completed_stages = expected if state == 'complete' else 0
            last = diagnostics[-1] if diagnostics else None
            entry = dict(batch=batch_name, input=name, state=state,
                         completed_stages=completed_stages, planned_stages=len(spin_rows) or expected,
                         observed_multiplicity=last.multiplicity if last else '',
                         last_energy_hartree=last.energy_hartree if last else '',
                         optimization_step=(re.findall(r'Step number\s+(\d+)', text)[-1: ] or [''])[0],
                         output_bytes=output.stat().st_size if output.is_file() else 0,
                         output_updated_utc=datetime.fromtimestamp(output.stat().st_mtime, timezone.utc).isoformat()
                             if output.is_file() else '',
                         return_code=rc, output=str(output.relative_to(campaign)))
            members.append(entry)
            if audit:
                if spin_rows and len(spin_rows) != expected:
                    issue(batch_name, name, 'error', 'Number of Link1 stages differs from spin manifest')
                if state == 'failed':
                    errors = re.findall(r'^.*Error termination.*$', text, re.M)
                    issue(batch_name, name, 'error', errors[-1].strip() if errors else f'Worker failure: {status}; rc={rc}')
                if re.search(r'Guess\s*=\s*\((?=[^)]*\bRead\b)(?=[^)]*\bAlways\b)[^)]*\)', input_text, re.I):
                    issue(batch_name, name, 'error', 'Contradictory Guess=(Read,Always); regenerate inputs')
                nprocs = re.findall(r'^\s*%nprocshared\s*=\s*(\d+)', input_text, re.I | re.M)
                cpus = plan.get('config', {}).get('cpus_per_job')
                if cpus and any(int(n) != int(cpus) for n in nprocs):
                    issue(batch_name, name, 'error', 'Gaussian nprocshared disagrees with saved Slurm CPUs/job')
                if inp.is_file():
                    digest = hashlib.sha256(inp.read_bytes()).hexdigest()
                    if any(r.get('input_sha256') and r['input_sha256'] != digest for r in rows):
                        issue(batch_name, name, 'error', 'Input SHA256 differs from manifest')
                if 'convergence failure' in text.lower():
                    issue(batch_name, name, 'warning', 'SCF convergence warning; inspect labels (IOP(5/13=1) may permit continuation)')
                if state == 'complete':
                    try:
                        frame = parse_final_force_frame(text, output)
                        if frame is None:
                            issue(batch_name, name, 'error', 'Completed output has no parseable final force frame')
                        elif last and (frame.record.charge, frame.record.multiplicity) != (last.charge, last.multiplicity):
                            issue(batch_name, name, 'error', 'Final force frame belongs to an earlier charge/multiplicity section')
                    except Exception as exc:
                        issue(batch_name, name, 'error', f'Force parser: {exc}')
        jobs.extend(members)
        counts = collections.Counter(row['state'] for row in members)
        batches.append(dict(batch=batch_name, planned=len(members),
                            **{state: counts[state] for state in STATES},
                            completed_stages=sum(row['completed_stages'] for row in members),
                            planned_stages=sum(row['planned_stages'] for row in members),
                            errors=sum(i['batch'] == batch_name and i['severity'] == 'error' for i in issues),
                            warnings=sum(i['batch'] == batch_name and i['severity'] == 'warning' for i in issues)))
    destination = (destination or campaign / 'monitoring').resolve()
    destination.mkdir(parents=True, exist_ok=True)
    summary = dict(campaign=str(campaign), generated_utc=datetime.now(timezone.utc).isoformat(),
                   start=start, end=end, audit=audit, scheduler_queried=False,
                   planned=len(jobs), by_state=dict(collections.Counter(j['state'] for j in jobs)),
                   errors=sum(i['severity'] == 'error' for i in issues),
                   warnings=sum(i['severity'] == 'warning' for i in issues), batches=batches)
    _write_csv(destination / 'batch_progress.csv', batches, list(batches[0]))
    _write_csv(destination / 'job_progress.csv', jobs, list(jobs[0]) if jobs else ['batch', 'input', 'state'])
    _write_csv(destination / 'audit_issues.csv', issues, ['batch', 'input', 'severity', 'detail'])
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return dict(summary=summary, batches=batches, jobs=jobs, issues=issues, destination=destination)
