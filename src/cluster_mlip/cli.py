from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

from .active_learning import predict_committee_forces, write_next_batch
from .analysis import write_analysis
from .audit import run_private_audit
from .batch_inventory import build_inventory
from .batch_progress import write_batch_progress
from .dataset import grouped_split, read_jobs_manifest, read_labeled_extxyz, write_labeled_extxyz
from .doctor import MISSING_REQUIRED, format_report, run_checks
from .evaluate import predict_with_mace, write_evaluation_report
from .gaussian import extract_document_records, gaussian_job_complete, parse_force_frames
from .io import iter_documents, read_document, read_extxyz, source_tree, write_extxyz, write_manifest
from .jobs import (
    DEFAULT_LINK1_ROUTE,
    DEFAULT_RATTLE_ROUTE,
    DEFAULT_ROUTE,
    DEFAULT_SADDLE_ROUTE,
    expanded_records,
    write_gaussian_jobs,
)
from .label_report import write_label_report
from .literature import DEFAULT_KEYWORDS, run_literature_gap
from .paper_pdfs import load_pdf_compositions, write_pdf_index
from .mace_glue import MaceUnavailable
from .manifest import write_experiment_manifest
from .models import Record, composition_allowed, geometry_signature
from .physical_checks import write_physical_checks_report
from .progress import write_campaign_progress
from .relaunch import prepare_route_relaunch
from .restart import prepare_spin_restarts
from .route_audit import audit_campaign_routes, resolve_config_type
from .routes import FINDINGS, inspect_job, intended_stationary_point
from .spin import (
    DEFAULT_SPIN_ROUTE,
    parse_spin_diagnostics,
    route_with_frequency,
    validate_fragment_specification_shape,
    validate_spin_campaign,
    write_automatic_fe_spin_jobs,
    write_spin_inventory,
    write_spin_jobs,
)
from .slurm import (
    ExtractSlurmConfig,
    SlurmConfig,
    prepare_extract_slurm,
    prepare_slurm_batches,
    submit_extract_slurm,
)
from .stratify import STRATA_FIELDS
from .training import DEFAULT_SEED, TrainingConfig, write_training_campaign


def _elements(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _record_allowed(record: Record, args: argparse.Namespace) -> bool:
    allowed = _elements(getattr(args, "elements", None))
    required = _elements(getattr(args, "require_elements", None))
    symbols = {a.symbol for a in record.atoms}
    if not composition_allowed(record.atoms, allowed):
        return False
    if required is not None and not required.issubset(symbols):
        return False
    if getattr(args, "min_atoms", None) is not None and len(record.atoms) < args.min_atoms:
        return False
    if getattr(args, "max_atoms", None) is not None and len(record.atoms) > args.max_atoms:
        return False
    return True


def command_extract(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[Record] = []
    errors: list[tuple[str, str]] = []
    with source_tree(source) as root:
        documents = [source] if source.suffix.lower() != ".zip" and source.is_file() else list(iter_documents(root))
        for document in documents:
            try:
                relative = str(document.relative_to(root)) if document.is_relative_to(root) else document.name
                text = read_document(document)
                for record in extract_document_records(text, relative):
                    if not _record_allowed(record, args):
                        continue
                    if args.types and record.config_type not in args.types:
                        continue
                    records.append(record)
            except Exception as exc:  # preserve a complete audit instead of aborting a 2k-file archive
                errors.append((str(document), str(exc)))
    # De-duplicate document/log copies by state-aware geometric identity.
    unique: dict[tuple, Record] = {}
    for record in records:
        key = (
            geometry_signature(record.atoms),
            record.charge,
            record.multiplicity,
            record.config_type,
            record.irc_path,
            record.irc_point,
        )
        unique.setdefault(key, record)
    records = sorted(unique.values(), key=lambda r: (r.config_type, r.formula, r.source, r.record_id))
    write_extxyz(records, output / "seeds.extxyz")
    write_manifest(records, output / "manifest.csv")
    with (output / "errors.tsv").open("w", encoding="utf-8") as handle:
        for name, message in errors:
            handle.write(f"{name}\t{message}\n")
    counts = collections.Counter(r.config_type for r in records)
    print(f"Extracted {len(records)} unique seeds from {source}")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")
    print(f"Unreadable documents: {len(errors)}")
    return 0


def command_extract_slurm(args: argparse.Namespace) -> int:
    extract_arguments: list[str] = []
    for flag, value in (
        ("--elements", args.elements),
        ("--require-elements", args.require_elements),
        ("--min-atoms", args.min_atoms),
        ("--max-atoms", args.max_atoms),
    ):
        if value is not None:
            extract_arguments.extend([flag, str(value)])
    if args.types:
        extract_arguments.append("--types")
        extract_arguments.extend(args.types)
    config = ExtractSlurmConfig(
        time_limit=args.time,
        partition=args.partition,
        account=args.account,
        gaussian_module=args.gaussian_module,
        job_name=args.job_name,
        cluster_mlip_command=args.cluster_mlip_command,
        runtime_env=args.runtime_env,
    )
    plan = prepare_extract_slurm(
        Path(args.source),
        Path(args.output),
        config,
        extract_arguments=extract_arguments,
    )
    print(f"Prepared extract Slurm job: {Path(plan['output']) / plan['sbatch_script']}")
    print(f"Source SHA-256: {plan['source_sha256']}")
    if args.submit:
        print(submit_extract_slurm(Path(plan["output"])))
    else:
        print(f"Submit: {Path(plan['output']) / plan['submit_script']}")
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    summary = write_analysis(
        source,
        output,
        record_filter=lambda record: (
            _record_allowed(record, args)
            and (not args.types or record.config_type in args.types)
        ),
        jobs=args.jobs,
    )
    structures = summary["structures"]
    files = summary["files"]
    print(
        f"Analyzed {files['total']} files from {source} "
        f"({files['compatible']} compatible inputs)"
    )
    print(
        f"Structures: {structures['records']} records, "
        f"{structures['unique_geometry_state']} unique geometry/state entries"
    )
    print(f"Report: {output / 'report.md'}")
    return 0


def command_audit(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    output = (
        Path(args.output).resolve()
        if args.output
        else (Path.cwd() / "private_audits" / source.stem).resolve()
    )
    result = run_private_audit(
        source,
        output,
        elements=_elements(args.elements),
        required_elements=_elements(args.require_elements),
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        types=set(args.types) if args.types else None,
        jobs=args.jobs,
    )
    full = result["full"]["structures"]
    print(
        f"Full audit: {full['records']} records, "
        f"{full['unique_geometry_state']} unique geometry/state entries"
    )
    if result["selection"] is not None:
        selected = result["selection"]["structures"]
        print(
            f"Selection: {selected['records']} records, "
            f"{selected['unique_geometry_state']} unique geometry/state entries"
        )
    print(f"Private audit: {output}")
    return 0


def command_audit_routes(args: argparse.Namespace) -> int:
    result = audit_campaign_routes(
        Path(args.campaign),
        Path(args.output) if args.output else None,
        include_inactive=args.include_inactive,
    )
    summary = result["summary"]
    print(f"Inputs audited: {summary['inputs_audited']}")
    print("Route intent: " + ", ".join(
        f"{name}={count}" for name, count in sorted(summary["by_intent"].items())
    ))
    if summary["findings"]:
        print("Findings:")
        for code, count in sorted(summary["findings"].items(), key=lambda item: -item[1]):
            severity, relaunch, explanation = FINDINGS[code]
            print(f"  {count:5} {severity:7} {code}: {explanation}")
            if relaunch:
                print("        -> results unusable; these jobs must be relaunched")
    else:
        print("Every job's route matches its structural label.")
    print(f"Must relaunch: {summary['must_relaunch']} "
          f"({summary['completed_but_invalid']} of them already finished)")
    sources = summary["must_relaunch_by_geometry_source"]
    if sources:
        print("  How they must be rebuilt:")
        for name, count in sorted(sources.items()):
            explanation = {
                "input_coordinates": "corrected in place; spin-flip chain preserved",
                "checkpoint": "spin-ladder restart with no coordinates; "
                              "root ladder rebuilt, whole restart lineage retired",
                "unknown": "geometry source unclear; skipped for manual review",
            }.get(name, "")
            print(f"    {count:5} {name}: {explanation}")
    for item in summary["unreadable"]:
        print(f"NOT AUDITED: {item['input']}: {item['reason']}", file=sys.stderr)
    print(f"Reports: {result['destination']}")
    return 2 if summary["must_relaunch"] else 0


def command_relaunch_routes(args: argparse.Namespace) -> int:
    result = prepare_route_relaunch(
        Path(args.campaign),
        start=args.start,
        end=args.end,
        assume_stopped=args.assume_stopped,
        dry_run=args.dry_run,
        saddle_order=args.saddle_order,
    )
    verb = "Would relaunch" if args.dry_run else "Relaunched"
    print(f"{verb} {result['input_count']} input(s), {result['job_row_count']} manifest row(s)")
    for row in result["plan"]:
        print(f"  {row['batch'] or '-'} {row['original_input']} [{row['config_type']}, "
              f"was {row['previous_state']}] -> {row['new_input']}")
        print(f"      route: {row['route_before']}")
        print(f"          -> {row['route_after']}")
    for row in result["skipped"]:
        print(f"SKIPPED {row['input']}: {row['reason']}", file=sys.stderr)
    if args.dry_run:
        print("Dry run: nothing was written. Re-run without --dry-run to apply.")
    else:
        print(f"Manifest and batch listings backed up under: {result['backup']}")
        print("Resubmit the same batch range with the existing head launcher; do not "
              "re-run prepare-slurm.")
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    records = read_extxyz(Path(args.seeds))
    records = [r for r in records if _record_allowed(r, args)]
    if args.types:
        records = [r for r in records if r.config_type in args.types]
    if args.max_seeds is not None:
        records = records[:args.max_seeds]
    jobs = expanded_records(records, args.rattles_per_seed, args.rattle_sigma, args.seed)
    write_gaussian_jobs(
        jobs,
        Path(args.output),
        args.route,
        args.memory,
        args.nproc,
        args.rattle_route,
        args.link1_route,
        args.saddle_route,
    )
    saddles = sum(
        1 for record in jobs
        if "rattle_index" not in record.metadata
        and intended_stationary_point(record.config_type) == "saddle"
    )
    print(f"Prepared {len(jobs)} Gaussian force jobs from {len(records)} seeds")
    print(f"Seed route: {args.route}")
    if saddles:
        print(f"Saddle route ({saddles} labeled saddle seeds): {args.saddle_route}")
    print(f"Rattled route: {args.rattle_route}")
    print(f"Link1 force route: {args.link1_route}")
    return 0


def command_prepare_slurm(args: argparse.Namespace) -> int:
    config = SlurmConfig(
        jobs_per_batch=args.jobs_per_batch,
        concurrent_jobs=args.concurrent_jobs,
        cpus_per_job=args.cpus_per_job,
        time_limit=args.time,
        partition=args.partition,
        account=args.account,
        gaussian_module=args.gaussian_module,
        gaussian_command=args.gaussian_command,
        job_name=args.job_name,
        memory_per_node=args.memory_per_node,
        scratch_root=args.scratch_root,
    )
    plan = prepare_slurm_batches(
        Path(args.campaign),
        config,
        worker_init=Path(args.worker_init) if args.worker_init else None,
        allow_nproc_mismatch=args.allow_nproc_mismatch,
    )
    print(
        f"Prepared {plan['input_count']} Gaussian inputs in {plan['batch_count']} Slurm batches "
        f"({args.jobs_per_batch} inputs/batch, {args.concurrent_jobs} concurrent jobs/node)"
    )
    print(f"Submit: {Path(args.campaign).resolve() / 'submit_gaussian_batches.sh'}")
    for warning in plan["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


def command_campaign_status(args: argparse.Namespace) -> int:
    if args.by_batch or args.audit:
        batch_result = write_batch_progress(Path(args.campaign), Path(args.output) if args.output else None,
                                      start=args.start, end=args.end, audit=args.audit)
        print("Batch          Inputs   Done Failed Partial Active? Waiting Stages     Issues")
        for row in batch_result["batches"]:
            print(f"{row['batch']:14} {row['planned']:6} {row['complete']:6} {row['failed']:6} "
                  f"{row['incomplete']:7} {row['activity_unconfirmed']:7} {row['not_started']:7} "
                  f"{row['completed_stages']:4}/{row['planned_stages']:<4} "
                  f"{row['errors']} errors, {row['warnings']} warnings")
        print("Active? = start marker without a matching finish; scheduler state is unconfirmed.")
        print("Waiting = no calculation started; queued and unsubmitted jobs cannot be distinguished from files.")
        print(f"Reports: {batch_result['destination']}")
        return 2 if args.audit and batch_result['summary']['errors'] else 0
    if args.start != 1 or args.end is not None:
        raise ValueError("--start/--end require --by-batch or --audit")
    result = write_campaign_progress(
        Path(args.campaign), Path(args.output) if args.output else None
    )
    summary = result["summary"]
    counts = summary["by_state"]
    assert isinstance(counts, dict)
    print(f"Campaign jobs: {summary['total']}")
    print("Status: " + ", ".join(f"{name}={count}" for name, count in counts.items()))
    print(f"Progress CSV: {result['csv_path']}")
    print(f"Summary JSON: {result['summary_path']}")
    return 0


def command_collect(args: argparse.Namespace) -> int:
    output_roots = [Path(item).resolve() for item in args.outputs]
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    for outputs in output_roots:
        if not outputs.is_dir():
            raise NotADirectoryError(outputs)
    if (args.valid_fraction < 0 or args.test_fraction < 0
            or args.valid_fraction + args.test_fraction >= 1):
        raise ValueError("split fractions must be nonnegative and sum to less than one")
    if args.allow_partial and args.frames != "converged":
        raise ValueError(
            "--allow-partial requires --frames converged so interrupted optimization steps "
            "cannot enter the dataset"
        )
    frames = []
    failures: list[tuple[str, str]] = []
    for outputs in output_roots:
        spin_campaign = (outputs / "spin_jobs.csv").is_file()
        manifest_path = outputs / ("spin_jobs.csv" if spin_campaign else "jobs.csv")
        manifest = read_jobs_manifest(manifest_path)
        manifest_by_output: dict[str, list[dict[str, str]]] = {}
        for row in manifest.values():
            key = Path(row.get("output") or row["job_id"]).stem
            manifest_by_output.setdefault(key, []).append(row)
        paths = sorted(list(outputs.rglob("*.log")) + list(outputs.rglob("*.out")))
        counts = collections.Counter(path.stem for path in paths if not path.is_symlink())
        for path in paths:
            if path.is_symlink():
                continue
            rows = manifest_by_output.get(path.stem, [])
            if manifest and not rows:
                continue  # Scheduler logs and unrelated files are not Gaussian labels.
            try:
                if counts[path.stem] > 1:
                    raise ValueError("ambiguous duplicate output name within campaign")
                rc_path = path.with_suffix(".rc")
                bad_rc = rc_path.is_file() and rc_path.read_text().strip() != "0"
                if bad_rc and not args.allow_partial:
                    raise ValueError("Gaussian worker recorded a nonzero or invalid exit code")
                text = path.read_text(errors="ignore")
                expected = len(rows) if spin_campaign else 1
                input_text = ""
                if rows and rows[0].get("input"):
                    input_path = outputs / rows[0]["input"]
                    if input_path.is_file():
                        input_text = input_path.read_text(errors="ignore")
                        expected = 1 + len(re.findall(
                            r"^\s*--link1--\s*$", input_text, re.I | re.M
                        ))
                # A job that searched for the wrong stationary point terminates
                # normally and yields a parseable force frame, so completeness
                # checks cannot catch it: a transition state run with a plain
                # Opt contributes a minimum labeled as a saddle, which is worse
                # for training than having no frame at all.
                invalidated = (rows[0].get("route_invalidated", "") if rows else "")
                verdict = None
                if input_text and rows:
                    config_type, _ = resolve_config_type(rows[0], Path(rows[0]["input"]).name)
                    if config_type:
                        verdict = inspect_job(config_type, input_text, text)
                if not args.allow_route_mismatch and (
                        invalidated or (verdict and verdict["must_relaunch"])):
                    codes = invalidated or ";".join(verdict["findings"])
                    raise ValueError(
                        f"route does not search for the labeled stationary point ({codes}); "
                        "relaunch with cluster-mlip relaunch-routes, or pass "
                        "--allow-route-mismatch to accept the label anyway"
                    )
                complete = gaussian_job_complete(text, expected) and not bad_rc
                if not complete and not args.allow_partial:
                    raise ValueError("incomplete Gaussian job (including Link1 stages)")
                parsed = parse_force_frames(text, path)
                if not parsed:
                    raise ValueError("no complete energy/geometry/force frame")
                diagnostics = parse_spin_diagnostics(text)
                grouped: dict[str, list] = {}
                for frame in parsed:
                    if spin_campaign:
                        matches = [row for row in rows
                                   if int(row["intended_charge"]) == frame.record.charge
                                   and int(row["intended_multiplicity"]) == frame.record.multiplicity]
                        if len(matches) != 1:
                            raise ValueError("force frame has missing or ambiguous spin-stage provenance")
                        row = matches[0]
                    else:
                        row = rows[0] if rows else {}
                    job_id = row.get("job_id", path.stem)
                    record = frame.record
                    record.record_id = job_id
                    record.source = row.get("source", str(path))
                    record.config_type = row.get("config_type") or "labeled"
                    record.route = row.get("legacy_route", "")
                    if row.get("legacy_energy_hartree"):
                        record.legacy_energy_hartree = float(row["legacy_energy_hartree"])
                    record.metadata.update({key: value for key, value in row.items() if value})
                    record.metadata["parent_record_id"] = row.get("parent_record_id") or job_id
                    record.metadata["gaussian_output"] = str(path.relative_to(outputs))
                    record.metadata["collection_campaign"] = str(outputs)
                    record.metadata["source_job_complete"] = complete
                    stage_diagnostics = [item for item in diagnostics
                                         if item.charge in (None, record.charge)
                                         and item.multiplicity in (None, record.multiplicity)]
                    record.metadata["spin_stage_normal_termination"] = any(
                        item.normal_termination for item in stage_diagnostics
                    )
                    record.metadata["spin_stage_optimized"] = any(
                        item.optimized for item in stage_diagnostics
                    )
                    grouped.setdefault(job_id, []).append(frame)
                if spin_campaign and complete and set(grouped) != {row["job_id"] for row in rows}:
                    raise ValueError("one or more planned spin stages have no force label")
                selected = []
                for job_id, members in grouped.items():
                    if args.frames == "all":
                        for frame in members:
                            frame.record.record_id = f"{job_id}__frame{frame.record.metadata['force_frame_index']:06d}"
                        selected.extend(members)
                    elif args.frames == "converged":
                        final = members[-1]
                        if (final.record.metadata["spin_stage_normal_termination"]
                                and final.record.metadata["spin_stage_optimized"]):
                            selected.append(final)
                    else:
                        selected.append(members[-1])
                frames.extend(selected)
            except Exception as exc:
                failures.append((str(path), str(exc)))
    write_labeled_extxyz(frames, destination / "all.extxyz")
    splits = grouped_split(
        frames, args.valid_fraction, args.test_fraction, args.seed, stratify_by=args.stratify_by
    )
    for name, subset in splits.items():
        write_labeled_extxyz(subset, destination / f"{name}.extxyz")
    with (destination / "failed_outputs.tsv").open("w", encoding="utf-8") as handle:
        for name, message in failures:
            handle.write(f"{name}\t{message}\n")
    print(f"Collected {len(frames)} labeled frames; rejected {len(failures)} outputs")
    print("Split: " + ", ".join(f"{name}={len(values)}" for name, values in splits.items()))
    label_summary = write_label_report(
        frames, destination, args.force_outlier_threshold,
        splits=splits, stratify_by=args.stratify_by or (),
    )
    print(
        f"Label report: {len(label_summary['outliers'])} force-RMS outliers "
        f"(> {args.force_outlier_threshold} eV/Angstrom) -- see {destination / 'label_report.md'}"
    )
    return 0 if frames else 2


def _stratify_by(value: str) -> tuple[str, ...]:
    fields = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [field for field in fields if field not in STRATA_FIELDS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown --stratify-by field(s) {unknown}; choose from {STRATA_FIELDS}"
        )
    return fields


def _multiplicities(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("multiplicities must be comma-separated integers") from exc


def command_prepare_spin_restarts(args: argparse.Namespace) -> int:
    result = prepare_spin_restarts(
        Path(args.campaign), start=args.start, end=args.end,
        assume_stopped=args.assume_stopped, dry_run=args.dry_run,
    )
    print(
        f"Prepared {result['input_count']} shortened restart input(s) "
        f"covering {result['stage_count']} unfinished stage(s)"
    )
    print(f"Campaign: {Path(args.campaign).resolve()}")
    if args.dry_run:
        for row in result.get("plan", []):
            print(
                f"  {row['batch']}: {Path(row['archived_input']).name} -> "
                f"m{row['restart_multiplicity']} ({row['remaining_stages']} stage(s))"
            )
    skipped = result.get("skipped", [])
    if skipped:
        reasons = collections.Counter(row["reason"] for row in skipped)
        print("Skipped: " + ", ".join(f"{name}={count}" for name, count in sorted(reasons.items())))
    if args.dry_run:
        print("Dry run only; no files changed.")
    else:
        print("Existing batch inputs were updated in place; use the same launchers.")
    return 0


def command_spin_extract(args: argparse.Namespace) -> int:
    count = write_spin_inventory(Path(args.source).resolve(), Path(args.output).resolve())
    print(f"Extracted {count} spin-state records")
    print(f"Inventory: {Path(args.output).resolve() / 'spin_inventory.csv'}")
    return 0


def command_prepare_spins(args: argparse.Namespace) -> int:
    print(
        "WARNING: prepare-spins is experimental and has not been human-tested on a production "
        "Gaussian campaign; inspect generated inputs before submission.",
        file=sys.stderr,
    )
    records = read_extxyz(Path(args.seeds))
    if args.auto_from_data and args.record_ids:
        raise ValueError(
            "--auto-from-data needs complete formula/charge groups; do not combine it with --record-id"
        )
    if args.record_ids:
        requested = set(args.record_ids)
        available = {record.record_id for record in records}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"--record-id not found in {args.seeds}: {unknown}")
        records = [record for record in records if record.record_id in requested]
    records = [record for record in records if _record_allowed(record, args)]
    route = route_with_frequency(args.route) if args.freq else args.route
    specifications = None
    if args.fragment_spec:
        payload = json.loads(Path(args.fragment_spec).read_text(encoding="utf-8"))
        specifications = payload.get("guesses", payload) if isinstance(payload, dict) else payload
        shape_errors = validate_fragment_specification_shape(specifications)
        if shape_errors:
            raise ValueError(
                f"{args.fragment_spec} does not match the expected shape "
                "(see examples/spin_fragments.schema.json):\n  - " + "\n  - ".join(shape_errors)
            )
    if args.auto_from_data:
        if args.high_spin is not None or args.targets is not None:
            raise ValueError("--auto-from-data infers --high-spin and --targets; do not pass them")
        if specifications is not None or args.strategy not in {"auto", "ladder"}:
            raise ValueError("--auto-from-data prepares ladders only and cannot infer fragment guesses")
        stages = write_automatic_fe_spin_jobs(
            records=records,
            output=Path(args.output),
            route=route,
            memory=args.memory,
            nproc=args.nproc,
        )
        print(f"Plan: {Path(args.output).resolve() / 'spin_plan.csv'}")
        print(f"Plan summary: {Path(args.output).resolve() / 'spin_plan_summary.json'}")
    else:
        if args.high_spin is None or args.targets is None:
            raise ValueError("manual spin preparation requires both --high-spin and --targets")
        stages = write_spin_jobs(
            records=records,
            output=Path(args.output),
            high_spin=args.high_spin,
            targets=args.targets,
            route=route,
            memory=args.memory,
            nproc=args.nproc,
            fragment_specifications=specifications,
            strategy=args.strategy,
        )
    print(f"Prepared {stages} traceable spin stages")
    print(f"Manifest: {Path(args.output).resolve() / 'spin_jobs.csv'}")
    return 0


def command_validate_spins(args: argparse.Namespace) -> int:
    summary = validate_spin_campaign(
        Path(args.original).resolve(),
        Path(args.new_outputs).resolve(),
        Path(args.output).resolve(),
        args.geometry_tolerance,
        args.spin_tolerance,
        args.s2_tolerance,
    )
    print(
        f"Validated {summary['legacy_records']} legacy states against {summary['new_records']} new states: "
        f"missing={summary['missing']}, alternative_roots={summary['alternative_root']}"
    )
    print(f"Report: {Path(args.output).resolve() / 'report.md'}")
    strict_failures = (
        summary["missing"]
        + summary["alternative_root"]
        + summary["new_calculation_incomplete"]
        + summary["planned_stages_missing"]
        + summary["lineage_errors"]
        + summary["untracked_new_states"]
        + summary["planned_stages_incomplete"]
        + summary["planned_states_uncharacterized"]
        + summary["fragment_alignment_mismatches"]
        + summary["fragment_alignment_unresolved"]
    )
    if args.require_stability:
        strict_failures += summary["planned_states_without_stability"]
    return 2 if (args.strict or args.require_stability) and strict_failures else 0


def command_doctor(args: argparse.Namespace) -> int:
    checks = run_checks()
    report, worst = format_report(checks)
    print(report)
    return 1 if worst == MISSING_REQUIRED else 0


def command_manifest(args: argparse.Namespace) -> int:
    manifest = write_experiment_manifest(
        Path(args.dataset),
        Path(args.output),
        config=Path(args.config) if args.config else None,
        notes=args.notes or "",
    )
    print(f"Dataset files hashed: {len(manifest['dataset_files'])}")
    print(f"Git commit: {manifest['git_commit'] or '(not a git checkout)'}")
    print(f"Manifest: {Path(args.output).resolve()}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    frames = read_labeled_extxyz(Path(args.labeled))
    if not frames:
        print("No labeled frames found; nothing to evaluate.", file=sys.stderr)
        return 1
    try:
        predictions = predict_with_mace(Path(args.model), frames, device=args.device)
    except MaceUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary = write_evaluation_report(frames, predictions, Path(args.output))
    overall = summary["overall"]
    print(
        f"Evaluated {summary['n_frames']} frames: "
        f"energy MAE {overall['energy_mae_ev_per_atom']:.4f} eV/atom, "
        f"force MAE {overall['force_mae_ev_ang']:.4f} eV/Angstrom"
    )
    print(f"Report: {Path(args.output).resolve() / 'report.md'}")
    if not args.skip_physical_checks:
        checks = write_physical_checks_report(frames, predictions, Path(args.output))
        for check in checks:
            status = "n/a" if check["passed"] is None else ("pass" if check["passed"] else "FAIL")
            print(f"  [{status}] {check['name']} ({check['n_frames_considered']} considered)")
        print(f"Physical checks: {Path(args.output).resolve() / 'physical_checks.md'}")
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = TrainingConfig(
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output),
        run_name=args.run_name,
        mode="finetune" if args.finetune else "scratch",
        seeds=tuple(args.seed) if args.seed else (DEFAULT_SEED,),
        foundation_model=args.foundation_model,
        e0s=args.e0s,
        device=args.device,
        multiheads_finetuning=args.multiheads_finetuning,
        forces_weight=args.forces_weight,
        energy_weight=args.energy_weight,
        max_num_epochs=args.max_num_epochs,
        spin_num_classes=args.spin_num_classes,
        spin_offset=args.spin_offset,
        charge_num_classes=args.charge_num_classes,
        charge_offset=args.charge_offset,
        allow_mixed_method=args.allow_mixed_method,
        force=args.force,
        extra_args=tuple(args.extra_arg or ()),
    )
    try:
        plan = write_training_campaign(config)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    output = Path(args.output).resolve()
    print(
        f"Wrote {len(plan['seed_runs'])} training script(s) "
        f"({plan['mode']}, {plan['model']}"
        + (f", foundation={plan['foundation_model']}" if plan["foundation_model"] else "")
        + ") to " + str(output)
    )
    print(
        f"  charge range {plan['charge_range']}, multiplicity range "
        f"{plan['multiplicity_range']}, label routes: {len(plan['label_routes'])}"
    )
    for run in plan["seed_runs"]:
        print(f"  {run['script']}")
    for warning in plan["warnings"]:
        print(f"  warning: {warning}", file=sys.stderr)
    if plan["mode"] == "finetune":
        print(f"  read {output / 'PREFLIGHT.md'} before submitting")
    print(f"  manifest: {output / 'train_manifest.json'}")
    return 0


def command_select_next_batch(args: argparse.Namespace) -> int:
    if len(args.models) < 2:
        print("error: --models needs at least two checkpoints to form a committee", file=sys.stderr)
        return 1
    candidates = read_extxyz(Path(args.candidates))
    if not candidates:
        print("No candidate structures found; nothing to rank.", file=sys.stderr)
        return 1
    try:
        committee_forces = predict_committee_forces(
            [Path(model) for model in args.models], candidates, device=args.device
        )
    except MaceUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    selected = write_next_batch(candidates, committee_forces, Path(args.output), args.top_k)
    print(f"Ranked {len(candidates)} candidates by {len(args.models)}-model committee disagreement")
    if selected:
        print(
            f"Selected top {len(selected)}: worst disagreement "
            f"{selected[0][1]:.4f} eV/Angstrom, weakest of the selection {selected[-1][1]:.4f} eV/Angstrom"
        )
    print(f"Next batch: {Path(args.output).resolve() / 'next_batch.extxyz'}")
    return 0


def command_inventory(args: argparse.Namespace) -> int:
    result = build_inventory(
        Path(args.folder).resolve(), Path(args.output).resolve(),
        recursive=args.recursive, jobs=args.jobs,
    )
    print(f"Inventoried {len(result['zips'])} ZIP files")
    print(f"Unique formula/charge/multiplicity/state combinations: {len(result['master'])}")
    print(f"Inventory: {Path(args.output).resolve() / 'inventory.md'}")
    return 0


def command_pdf_index(args: argparse.Namespace) -> int:
    try:
        result = write_pdf_index(Path(args.source).resolve(), Path(args.output).resolve())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    counts = result["counts"]
    assert isinstance(counts, dict)
    print(
        f"Indexed {result['n_pdfs']} PDFs -- "
        f"ok: {counts.get('ok', 0)}  |  "
        f"no DOI found: {counts.get('no_doi_found', 0)}  |  "
        f"unreadable: {counts.get('unreadable', 0)}"
    )
    top_errors = result.get("top_errors")
    if isinstance(top_errors, list) and top_errors:
        print("Most common errors among unreadable PDFs:")
        for message, count in top_errors:
            print(f"  {count}x  {message}")
    print(f"PDF index: {result['output']}")
    return 0


def command_literature_gap(args: argparse.Namespace) -> int:
    pdf_compositions = None
    if args.pdf_index:
        pdf_compositions = load_pdf_compositions(Path(args.pdf_index).resolve())
    try:
        summary = run_literature_gap(
            Path(args.source).resolve(), Path(args.output).resolve(),
            author_ids=args.author_id or (),
            orcids=args.orcid or (),
            keywords=args.keywords or DEFAULT_KEYWORDS,
            contact_email=args.contact_email, jobs=args.jobs,
            author_name=args.author_name,
            pdf_compositions=pdf_compositions,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    counts = summary["counts"]
    assert isinstance(counts, dict)
    n_fetched = summary.get("n_fetched")
    if isinstance(n_fetched, int) and n_fetched != summary["n_papers"]:
        print(f"OpenAlex returned {n_fetched} papers by this author; {summary['n_papers']} look relevant")
    else:
        print(f"Checked {summary['n_papers']} papers")
    print(
        f"On file: {counts.get('on_file', 0)}  |  "
        f"May be missing: {counts.get('possible_gap', 0)}  |  "
        f"Not sure: {counts.get('unclear', 0)}"
    )
    print(f"Literature gap report: {Path(args.output).resolve() / 'literature_gap.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cluster-mlip", description="Legacy Gaussian cluster-to-MACE pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="inventory and summarize an archive/database")
    analyze.add_argument("source", help="archive ZIP, document, or directory")
    analyze.add_argument("-o", "--output", default="analysis")
    analyze.add_argument("--elements", help="keep only systems composed of these comma-separated elements")
    analyze.add_argument("--require-elements", help="require all of these comma-separated elements")
    analyze.add_argument("--min-atoms", type=int)
    analyze.add_argument("--max-atoms", type=int)
    analyze.add_argument("--types", nargs="+", help="keep selected config_type values")
    analyze.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="parse files in this many worker processes (default: 1, sequential)",
    )
    analyze.set_defaults(func=command_analyze)

    audit = sub.add_parser(
        "audit",
        help="generate a private full audit plus an optional filtered selection",
    )
    audit.add_argument("source", help="archive ZIP, document, or directory")
    audit.add_argument(
        "-o", "--output",
        help="output directory (default: private_audits/<source-name>)",
    )
    audit.add_argument("--elements", help="selection element allow-list")
    audit.add_argument("--require-elements", help="elements required in the selection")
    audit.add_argument("--min-atoms", type=int)
    audit.add_argument("--max-atoms", type=int)
    audit.add_argument("--types", nargs="+", help="configuration types for the selection")
    audit.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="parse files in this many worker processes (default: 1, sequential)",
    )
    audit.set_defaults(func=command_audit)

    extract = sub.add_parser("extract", help="extract stationary points and explicit IRC points")
    extract.add_argument("source", help="archive ZIP, document, or directory")
    extract.add_argument("-o", "--output", default="extracted")
    extract.add_argument("--elements", help="keep only systems composed of these comma-separated elements")
    extract.add_argument("--require-elements", help="require all of these comma-separated elements")
    extract.add_argument("--min-atoms", type=int)
    extract.add_argument("--max-atoms", type=int)
    extract.add_argument("--types", nargs="+", help="keep selected config_type values")
    extract.set_defaults(func=command_extract)

    extract_slurm = sub.add_parser(
        "extract-slurm",
        help="generate and optionally submit a reproducible one-node Slurm extract job",
    )
    extract_slurm.add_argument("source", help="archive ZIP, document, or directory")
    extract_slurm.add_argument("-o", "--output", default="extracted")
    extract_slurm.add_argument("--elements", help="keep only systems composed of these comma-separated elements")
    extract_slurm.add_argument("--require-elements", help="require all of these comma-separated elements")
    extract_slurm.add_argument("--min-atoms", type=int)
    extract_slurm.add_argument("--max-atoms", type=int)
    extract_slurm.add_argument("--types", nargs="+", help="keep selected config_type values")
    extract_slurm.add_argument("--time", default="12:00:00")
    extract_slurm.add_argument("--partition", default="single")
    extract_slurm.add_argument("--account", default="loni_perovsk27")
    extract_slurm.add_argument("--gaussian-module", default="gaussian/g16-c01")
    extract_slurm.add_argument("--job-name", default="cluster_mlip_extract")
    extract_slurm.add_argument(
        "--cluster-mlip-command", default="cluster-mlip",
        help="executable available inside the Slurm job (default: cluster-mlip from exported PATH)",
    )
    extract_slurm.add_argument(
        "--runtime-env", default="/project/lgutsev/env/cluster_mlip_runtime",
        help="conda environment activated inside the batch job; pass an empty string to disable",
    )
    extract_slurm.add_argument(
        "--submit", action="store_true",
        help="submit immediately after persisting the script and provenance manifest",
    )
    extract_slurm.set_defaults(func=command_extract_slurm)

    spin_extract = sub.add_parser(
        "spin-extract",
        help="extract geometries plus multiplicity, <S^2>, local-spin, and convergence evidence",
    )
    spin_extract.add_argument("source", help="legacy archive, Gaussian output, or directory")
    spin_extract.add_argument("-o", "--output", default="spin_inventory")
    spin_extract.set_defaults(func=command_spin_extract)

    prepare = sub.add_parser("prepare", help="generate Gaussian energy+force jobs")
    prepare.add_argument("seeds", help="seeds.extxyz from extract")
    prepare.add_argument("-o", "--output", default="gaussian_jobs")
    prepare.add_argument("--elements", help="comma-separated element allow-list, e.g. Fe,N,O")
    prepare.add_argument("--require-elements", help="require all of these comma-separated elements")
    prepare.add_argument("--min-atoms", type=int)
    prepare.add_argument("--max-atoms", type=int)
    prepare.add_argument("--types", nargs="+", help="configuration types to include")
    prepare.add_argument("--max-seeds", type=int)
    prepare.add_argument("--rattles-per-seed", type=int, default=4)
    prepare.add_argument("--rattle-sigma", type=float, default=0.05, help="Cartesian Gaussian sigma in Angstrom")
    prepare.add_argument("--seed", type=int, default=20260811)
    prepare.add_argument(
        "--route",
        default=DEFAULT_ROUTE,
        help="first-stage route for unperturbed seeds (default: legacy BPW91 optimization/frequency)",
    )
    prepare.add_argument(
        "--saddle-route",
        default=DEFAULT_SADDLE_ROUTE,
        help="first-stage route for seeds labeled transition_state/first_order_saddle/"
             "higher_order_saddle; must request a TS/QST search, not a plain Opt",
    )
    prepare.add_argument(
        "--rattle-route",
        default=DEFAULT_RATTLE_ROUTE,
        help="first-stage route for rattled structures; must not optimize away the displacement",
    )
    prepare.add_argument(
        "--link1-route",
        default=DEFAULT_LINK1_ROUTE,
        help="checkpoint-linked diffuse-basis force route used for final labels",
    )
    prepare.add_argument("--memory", default="16GB")
    prepare.add_argument("--nproc", type=int, default=16)
    prepare.set_defaults(func=command_prepare)

    prepare_slurm = sub.add_parser(
        "prepare-slurm",
        help="generate resumable per-folder Slurm jobs for a prepared Gaussian campaign",
    )
    prepare_slurm.add_argument(
        "campaign", help="directory containing jobs.csv or spin_jobs.csv plus generated inputs"
    )
    prepare_slurm.add_argument("--jobs-per-batch", type=int, default=30)
    prepare_slurm.add_argument(
        "--concurrent-jobs", type=int, default=4,
        help="Gaussian jobs multiplexed inside each one-node batch job",
    )
    prepare_slurm.add_argument("--cpus-per-job", type=int, default=16)
    prepare_slurm.add_argument(
        "--allow-nproc-mismatch", action="store_true",
        help="generate despite %%nprocshared/Slurm CPU disagreement (normally rejected)",
    )
    prepare_slurm.add_argument("--time", default="72:00:00")
    prepare_slurm.add_argument("--partition", default="checkpt")
    prepare_slurm.add_argument("--account", default="loni_perovsk27")
    prepare_slurm.add_argument(
        "--gaussian-module",
        default="gaussian/g16-c01",
        help="Gaussian module, or a quoted whitespace-separated load sequence such as "
        "'mvapich2 gaussian/g09-d01'",
    )
    prepare_slurm.add_argument("--gaussian-command", default="g16")
    prepare_slurm.add_argument("--job-name", default="cluster_mlip_g16")
    prepare_slurm.add_argument(
        "--memory-per-node",
        help="optional Slurm --mem value; Gaussian per-job memory remains controlled by the .gjf files",
    )
    prepare_slurm.add_argument(
        "--scratch-root", default="/work/$USER/g16-scr",
        help="node-visible Gaussian scratch parent (shell variables are expanded at run time)",
    )
    prepare_slurm.add_argument(
        "--worker-init",
        help="optional shell file copied into the campaign and sourced inside every srun worker",
    )
    prepare_slurm.set_defaults(func=command_prepare_slurm)

    campaign_status = sub.add_parser(
        "campaign-status",
        help="write a job-by-job progress/provenance table for a Gaussian campaign",
    )
    campaign_status.add_argument(
        "campaign", help="prepared Gaussian campaign containing jobs.csv or spin_jobs.csv"
    )
    campaign_status.add_argument(
        "-o", "--output", help="progress CSV path, or report directory with --by-batch/--audit (default: CAMPAIGN/monitoring)"
    )
    campaign_status.add_argument("--by-batch", action="store_true", help="summarize each batch and write CSV reports")
    campaign_status.add_argument("--audit", action="store_true", help="also check inputs, hashes, CPU settings and final force labels; exit 2 for errors")
    campaign_status.add_argument("--start", type=int, default=1, help="first batch to inspect (inclusive)")
    campaign_status.add_argument("--end", type=int, help="last batch to inspect (inclusive)")
    campaign_status.set_defaults(func=command_campaign_status)

    audit_routes = sub.add_parser(
        "audit-routes",
        help="check every job's route against its structural label (finds transition "
             "states launched as ordinary Opt); exit 2 when any job must be relaunched",
    )
    audit_routes.add_argument(
        "campaign", help="prepared Gaussian campaign containing jobs.csv or spin_jobs.csv"
    )
    audit_routes.add_argument(
        "-o", "--output", help="report directory (default: CAMPAIGN/monitoring)"
    )
    audit_routes.add_argument(
        "--include-inactive", action="store_true",
        help="also audit superseded manifest rows from earlier restarts/relaunches",
    )
    audit_routes.set_defaults(func=command_audit_routes)

    relaunch_routes = sub.add_parser(
        "relaunch-routes",
        help="rebuild and activate corrected inputs for jobs whose route searched for "
             "the wrong stationary point",
    )
    relaunch_routes.add_argument("campaign", help="prepared Gaussian campaign")
    relaunch_routes.add_argument("--start", type=int, default=1, help="first batch (inclusive)")
    relaunch_routes.add_argument("--end", type=int, help="last batch (inclusive)")
    relaunch_routes.add_argument(
        "--assume-stopped", action="store_true",
        help="proceed for inputs left with an unmatched .started marker; pass only after "
             "squeue confirms the allocations are gone",
    )
    relaunch_routes.add_argument(
        "--saddle-order", type=int,
        help="imaginary-mode order for higher_order_saddle records (Opt=(Saddle=N)); "
             "without it those records are skipped rather than retargeted to order 1",
    )
    relaunch_routes.add_argument("--dry-run", action="store_true")
    relaunch_routes.set_defaults(func=command_relaunch_routes)

    prepare_spins = sub.add_parser(
        "prepare-spins",
        help="prepare high-spin-to-low-spin Link1 ladders and optional fragment AFM guesses",
    )
    prepare_spins.add_argument("seeds", help="seeds.extxyz from spin-extract or extract")
    prepare_spins.add_argument("-o", "--output", default="gaussian_spin_jobs")
    prepare_spins.add_argument(
        "--auto-from-data", action="store_true",
        help=(
            "plan every Fe oxide from archived Fe spin densities or, when absent, the highest "
            "of multiple observed multiplicities; unsupported groups are skipped"
        ),
    )
    prepare_spins.add_argument(
        "--record-id", dest="record_ids", action="append",
        help="exact parent record_id from spin_inventory.csv; repeat to select multiple parents",
    )
    prepare_spins.add_argument("--high-spin", type=int, help="trusted high-spin multiplicity")
    prepare_spins.add_argument(
        "--targets", type=_multiplicities,
        help="comma-separated low-spin multiplicities; skipped intermediate values are inserted",
    )
    prepare_spins.add_argument(
        "--fragment-spec",
        help="JSON file with explicit atom-to-fragment maps and fragment charge/spin orientations",
    )
    prepare_spins.add_argument(
        "--strategy", choices=("auto", "ladder", "fragment", "both"), default="auto",
        help=(
            "state preparation: ladder, fragment, both, or auto (both when a fragment spec is "
            "supplied; otherwise ladder)"
        ),
    )
    prepare_spins.add_argument("--elements", help="comma-separated element allow-list")
    prepare_spins.add_argument("--require-elements", help="require all listed elements")
    prepare_spins.add_argument("--min-atoms", type=int)
    prepare_spins.add_argument("--max-atoms", type=int)
    prepare_spins.add_argument("--route", default=DEFAULT_SPIN_ROUTE)
    prepare_spins.add_argument(
        "--freq", action="store_true",
        help=(
            "add a Freq calculation to every stage's route (off by default); cheap on small "
            "clusters and useful for flagging poor optimizations via imaginary modes, but not "
            "recommended for larger systems in the same campaign"
        ),
    )
    prepare_spins.add_argument("--memory", default="16GB")
    prepare_spins.add_argument("--nproc", type=int, default=16)
    prepare_spins.set_defaults(func=command_prepare_spins)

    collect = sub.add_parser("collect", help="collect completed Gaussian force outputs into MACE extxyz")
    collect.add_argument("outputs", nargs="+", help="one or more campaign directories containing Gaussian outputs and manifests")
    collect.add_argument("-o", "--output", default="dataset")
    collect.add_argument("--valid-fraction", type=float, default=0.10)
    collect.add_argument("--test-fraction", type=float, default=0.10)
    collect.add_argument("--seed", type=int, default=20260811)
    collect.add_argument(
        "--frames", choices=("final", "all", "converged"), default="final",
        help=(
            "final force frame per completed job/spin stage (default), every force-bearing step, "
            "or only the final frame of normally terminated optimized stages"
        ),
    )
    collect.add_argument(
        "--allow-partial", action="store_true",
        help=(
            "recover completed stages from interrupted jobs; requires --frames converged so "
            "unfinished optimization steps are excluded"
        ),
    )
    collect.add_argument(
        "--allow-route-mismatch", action="store_true",
        help=(
            "ingest labels from jobs whose route searched for a different stationary point "
            "than their label claims (for example a transition state run with a plain Opt). "
            "Off by default: such a frame is a minimum labeled as a saddle"
        ),
    )
    collect.add_argument(
        "--force-outlier-threshold", type=float, default=5.0,
        help="flag frames whose force RMS (eV/Angstrom) exceeds this in label_report.md/json",
    )
    collect.add_argument(
        "--stratify-by", type=_stratify_by, default="pes_region,charge_spin_class",
        help=(
            "comma-separated axes from stratify.STRATA_FIELDS to split on so a small "
            "class (e.g. a handful of transition states) isn't left to chance -- "
            f"choose from {', '.join(STRATA_FIELDS)}; empty string falls back to one "
            "pseudo-stratum containing everything"
        ),
    )
    collect.set_defaults(func=command_collect)

    restart_spins = sub.add_parser(
        "prepare-spin-restarts",
        help="archive interrupted attempts and activate shortened inputs in the same batches",
    )
    restart_spins.add_argument("campaign", help="original prepared spin campaign")
    restart_spins.add_argument("--start", type=int, default=1, help="first original batch to inspect")
    restart_spins.add_argument("--end", type=int, help="last original batch to inspect (inclusive)")
    restart_spins.add_argument(
        "--assume-stopped", action="store_true",
        help="allow copying checkpoints with unmatched .started markers after independently confirming jobs stopped",
    )
    restart_spins.add_argument(
        "--dry-run", action="store_true",
        help="report restart candidates without renaming or writing files",
    )
    restart_spins.set_defaults(func=command_prepare_spin_restarts)

    validate_spins = sub.add_parser(
        "validate-spins",
        help="check legacy-state coverage and preserve alternative newly converged SCF roots",
    )
    validate_spins.add_argument("original", help="legacy archive/directory or extracted extxyz")
    validate_spins.add_argument("new_outputs", help="directory containing new Gaussian outputs")
    validate_spins.add_argument("-o", "--output", default="spin_validation")
    validate_spins.add_argument("--geometry-tolerance", type=float, default=0.05)
    validate_spins.add_argument("--spin-tolerance", type=float, default=0.25)
    validate_spins.add_argument("--s2-tolerance", type=float, default=0.25)
    validate_spins.add_argument(
        "--strict", action="store_true",
        help="exit nonzero for missing/wrong roots, incomplete jobs, or unverified spin provenance",
    )
    validate_spins.add_argument(
        "--require-stability", action="store_true",
        help="also require explicit wavefunction stability results (off for routine spin campaigns)",
    )
    validate_spins.set_defaults(func=command_validate_spins)

    doctor = sub.add_parser(
        "doctor", help="check for strings/formchk/Gaussian/mace-torch before a large run"
    )
    doctor.set_defaults(func=command_doctor)

    manifest = sub.add_parser(
        "manifest",
        help="bundle a dataset's checksums, config, and git commit into one experiment manifest",
    )
    manifest.add_argument("dataset", help="dataset directory from `collect` (train/valid/test/all.extxyz)")
    manifest.add_argument("-o", "--output", default="manifest.json", help="output manifest JSON path")
    manifest.add_argument("--config", help="training config file to checksum alongside the dataset")
    manifest.add_argument("--notes", help="free-text notes to record in the manifest")
    manifest.set_defaults(func=command_manifest)

    evaluate = sub.add_parser(
        "evaluate",
        help="score a trained MACE model's energy/force error by geometry class, charge, and multiplicity",
    )
    evaluate.add_argument("labeled", help="labeled extxyz from `collect`, e.g. dataset/test.extxyz")
    evaluate.add_argument("--model", required=True, help="path to a trained MACE model checkpoint")
    evaluate.add_argument("-o", "--output", default="evaluation")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument(
        "--skip-physical-checks", action="store_true",
        help="skip the five stratify-class physical sanity checks (they're cheap, but this is a fast path)",
    )
    evaluate.set_defaults(func=command_evaluate)

    train = sub.add_parser(
        "train",
        help="generate a charge/spin MACE training campaign from a `collect` dataset",
    )
    train.add_argument("dataset", help="dataset directory from `collect` (train/valid/test.extxyz)")
    train.add_argument("-o", "--output", default="models/run", help="campaign output directory")
    train.add_argument("--run-name", default="cluster_charge_spin", dest="run_name")
    train.add_argument(
        "--seed", type=int, action="append",
        help="training seed; repeat to generate a committee (default: one seed)",
    )
    train.add_argument(
        "--finetune", action="store_true",
        help="fine-tune a foundation model instead of training from scratch",
    )
    train.add_argument(
        "--foundation-model", default="polar-1-m", dest="foundation_model",
        help="foundation checkpoint for --finetune: polar-1-{s,m,l} (native charge/spin, "
        "wB97M-V), mace-omol, small/medium/large, or a path",
    )
    train.add_argument(
        "--multiheads-finetuning", action="store_true", dest="multiheads_finetuning",
        help="use multihead-replay fine-tuning (needs --pt_train_file via --extra-arg)",
    )
    train.add_argument("--e0s", default="average", help="MACE --E0s (default: average)")
    train.add_argument("--device", default="cuda")
    train.add_argument("--energy-weight", type=float, default=1.0, dest="energy_weight")
    train.add_argument("--forces-weight", type=float, default=100.0, dest="forces_weight")
    train.add_argument("--max-num-epochs", type=int, default=None, dest="max_num_epochs")
    train.add_argument("--spin-num-classes", type=int, default=101, dest="spin_num_classes")
    train.add_argument("--spin-offset", type=int, default=0, dest="spin_offset")
    train.add_argument("--charge-num-classes", type=int, default=201, dest="charge_num_classes")
    train.add_argument("--charge-offset", type=int, default=100, dest="charge_offset")
    train.add_argument(
        "--allow-mixed-method", action="store_true", dest="allow_mixed_method",
        help="proceed even if the dataset mixes force-label routes (unsound unless equivalent)",
    )
    train.add_argument("--force", action="store_true", help="overwrite a non-empty output directory")
    train.add_argument(
        "--extra-arg", action="append", dest="extra_arg",
        help="append a raw flag to every mace_run_train command (repeatable)",
    )
    train.set_defaults(func=command_train)

    select_next_batch = sub.add_parser(
        "select-next-batch",
        help="rank unlabeled candidates by committee force disagreement for active-learning DFT labeling",
    )
    select_next_batch.add_argument("candidates", help="extxyz of unlabeled candidate structures to rank")
    select_next_batch.add_argument(
        "--models", nargs="+", required=True,
        help="two or more MACE checkpoints trained on different seeds/subsets (a committee)",
    )
    select_next_batch.add_argument("-o", "--output", default="next_batch")
    select_next_batch.add_argument("--top-k", type=int, default=50)
    select_next_batch.add_argument("--device", default="cpu")
    select_next_batch.set_defaults(func=command_select_next_batch)

    inventory = sub.add_parser(
        "inventory",
        help="inventory every ZIP in a folder of warehouse deliveries, plus one merged master list",
    )
    inventory.add_argument("folder", help="folder containing warehouse ZIP files")
    inventory.add_argument("-o", "--output", default="inventory")
    inventory.add_argument(
        "--recursive", action="store_true", help="also search subfolders for ZIP files"
    )
    inventory.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="parse each ZIP's files in this many worker processes (default: 1, sequential)",
    )
    inventory.set_defaults(func=command_inventory)

    literature_gap = sub.add_parser(
        "literature-gap",
        help="compare a warehouse inventory against an author's published cluster papers (needs internet)",
    )
    literature_gap.add_argument(
        "source",
        help="an `inventory` output directory, or a raw folder of warehouse ZIPs to inventory inline",
    )
    literature_gap.add_argument("-o", "--output", default="literature_gap")
    literature_gap.add_argument(
        "--author-id", dest="author_id", action="append",
        help=(
            "verified OpenAlex author id, e.g. A5029253658 (repeat for duplicate profiles)"
        ),
    )
    literature_gap.add_argument(
        "--orcid", action="append",
        help="ORCID iD or URL, e.g. 0000-0002-1825-0097 (repeatable; alternative to --author-id)",
    )
    literature_gap.add_argument(
        "--author-name",
        help="optional human-readable author name for the report heading (never used to resolve identity)",
    )
    literature_gap.add_argument(
        "--keywords", nargs="+",
        help=(
            "fallback relevance words checked locally against title/abstract text when a paper's "
            "formulas don't overlap the local warehouse's elements (default: cluster, iron, oxide) "
            "-- this does not restrict the OpenAlex query itself, since this field's titles are "
            "usually chemical formulas rather than English phrases"
        ),
    )
    literature_gap.add_argument(
        "--contact-email",
        help="added to the OpenAlex request as its documented 'polite pool' contact (optional, better rate limits)",
    )
    literature_gap.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="if `source` is a raw ZIP folder, parse it with this many worker processes",
    )
    literature_gap.add_argument(
        "--pdf-index",
        help=(
            "a pdf_index.json from `cluster-mlip pdf-index` -- when given, a paper whose DOI "
            "matches an indexed PDF is also checked against that PDF's full text, not just "
            "OpenAlex's title/abstract"
        ),
    )
    literature_gap.set_defaults(func=command_literature_gap)

    pdf_index = sub.add_parser(
        "pdf-index",
        help=(
            "extract full text from a local corpus of paper PDFs and mine it for DOIs and "
            "chemical formulas, for `literature-gap --pdf-index` (offline; needs the `pdf` extra)"
        ),
    )
    pdf_index.add_argument("source", help="a ZIP of PDFs, a folder of PDFs, or a single .pdf file")
    pdf_index.add_argument("-o", "--output", default="pdf_index")
    pdf_index.set_defaults(func=command_pdf_index)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "literature-gap" and not args.author_id and not args.orcid:
        parser.error("literature-gap requires at least one --author-id or --orcid")
    if getattr(args, "valid_fraction", 0) + getattr(args, "test_fraction", 0) >= 1:
        parser.error("valid_fraction + test_fraction must be < 1")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
