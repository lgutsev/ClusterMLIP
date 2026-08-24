from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from .active_learning import predict_committee_forces, write_next_batch
from .analysis import write_analysis
from .audit import run_private_audit
from .batch_inventory import build_inventory
from .dataset import grouped_split, read_jobs_manifest, read_labeled_extxyz, write_labeled_extxyz
from .doctor import MISSING_REQUIRED, format_report, run_checks
from .evaluate import predict_with_mace, write_evaluation_report
from .gaussian import extract_document_records, parse_final_force_frame
from .io import iter_documents, read_document, read_extxyz, source_tree, write_extxyz, write_manifest
from .jobs import (
    DEFAULT_LINK1_ROUTE,
    DEFAULT_RATTLE_ROUTE,
    DEFAULT_ROUTE,
    expanded_records,
    write_gaussian_jobs,
)
from .label_report import write_label_report
from .literature import DEFAULT_KEYWORDS, run_literature_gap
from .mace_glue import MaceUnavailable
from .manifest import write_experiment_manifest
from .models import Record, composition_allowed, geometry_signature
from .physical_checks import write_physical_checks_report
from .progress import write_campaign_progress
from .spin import (
    DEFAULT_SPIN_ROUTE,
    validate_fragment_specification_shape,
    validate_spin_campaign,
    write_automatic_fe_spin_jobs,
    write_spin_inventory,
    write_spin_jobs,
)
from .slurm import SlurmConfig, prepare_slurm_batches
from .stratify import STRATA_FIELDS


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
    )
    print(f"Prepared {len(jobs)} Gaussian force jobs from {len(records)} seeds")
    print(f"Seed route: {args.route}")
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
    outputs = Path(args.outputs)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = read_jobs_manifest(outputs / "jobs.csv")
    manifest_by_output = {
        Path(row.get("output", "")).stem: row
        for row in manifest.values()
        if row.get("output")
    }
    frames = []
    failures: list[tuple[str, str]] = []
    for path in sorted(list(outputs.rglob("*.log")) + list(outputs.rglob("*.out"))):
        row = manifest.get(path.stem, {}) or manifest_by_output.get(path.stem, {})
        job_id = row.get("job_id", path.stem)
        seed = None
        if row:
            provenance_keys = (
                "human_id",
                "source_record_id",
                "parent_record_id",
                "variant",
                "rattle_index",
                "rattle_sigma_angstrom",
                "campaign_seed",
                "resolved_rattle_seed",
                "parent_geometry_sha256",
                "input_geometry_sha256",
                "input_sha256",
                "legacy_energy_hartree",
                "legacy_route",
                "first_route",
                "link1_route",
            )
            metadata = {key: row[key] for key in provenance_keys if row.get(key)}
            metadata["parent_record_id"] = row.get("parent_record_id", job_id)
            seed = Record(
                record_id=job_id,
                source=row.get("source", str(path)),
                atoms=[],
                charge=int(row.get("charge", 0)),
                multiplicity=int(row.get("multiplicity", 1)),
                config_type=row.get("config_type", "labeled"),
                route=row.get("legacy_route", ""),
                legacy_energy_hartree=(
                    float(row["legacy_energy_hartree"])
                    if row.get("legacy_energy_hartree")
                    else None
                ),
                metadata=metadata,
            )
        try:
            frame = parse_final_force_frame(path.read_text(errors="ignore"), path, seed)
            if frame is None:
                failures.append((str(path), "no complete energy/geometry/force frame"))
            else:
                frames.append(frame)
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
    if frames:
        label_summary = write_label_report(
            frames, destination, args.force_outlier_threshold,
            splits=splits, stratify_by=args.stratify_by,
        )
        print(
            f"Label report: {len(label_summary['outliers'])} force-RMS outliers "
            f"(> {args.force_outlier_threshold} eV/Angstrom) -- see {destination / 'label_report.md'}"
        )
    return 0


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
            route=args.route,
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
            route=args.route,
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
        + summary["planned_states_without_stability"]
    )
    return 2 if args.strict and strict_failures else 0


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


def command_literature_gap(args: argparse.Namespace) -> int:
    try:
        summary = run_literature_gap(
            Path(args.source).resolve(), Path(args.output).resolve(),
            author_ids=args.author_id or (),
            orcids=args.orcid or (),
            keywords=args.keywords or DEFAULT_KEYWORDS,
            contact_email=args.contact_email, jobs=args.jobs,
            author_name=args.author_name,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    counts = summary["counts"]
    assert isinstance(counts, dict)
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
    prepare_slurm.add_argument("--account", default="loni_dspm_25")
    prepare_slurm.add_argument("--gaussian-module", default="gaussian/g16-c01")
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
        "-o", "--output", help="progress CSV path (default: CAMPAIGN/progress.csv)"
    )
    campaign_status.set_defaults(func=command_campaign_status)

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
    prepare_spins.add_argument("--memory", default="16GB")
    prepare_spins.add_argument("--nproc", type=int, default=16)
    prepare_spins.set_defaults(func=command_prepare_spins)

    collect = sub.add_parser("collect", help="collect completed Gaussian force outputs into MACE extxyz")
    collect.add_argument("outputs", help="directory containing .log/.out files and optional jobs.csv")
    collect.add_argument("-o", "--output", default="dataset")
    collect.add_argument("--valid-fraction", type=float, default=0.10)
    collect.add_argument("--test-fraction", type=float, default=0.10)
    collect.add_argument("--seed", type=int, default=20260811)
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
        help="title/abstract keywords to search for (default: iron/iron-oxide/transition-metal cluster terms)",
    )
    literature_gap.add_argument(
        "--contact-email",
        help="added to the OpenAlex request as its documented 'polite pool' contact (optional, better rate limits)",
    )
    literature_gap.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="if `source` is a raw ZIP folder, parse it with this many worker processes",
    )
    literature_gap.set_defaults(func=command_literature_gap)
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
