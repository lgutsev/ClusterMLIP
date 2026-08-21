from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from .analysis import write_analysis
from .audit import run_private_audit
from .dataset import grouped_split, read_jobs_manifest, write_labeled_extxyz
from .gaussian import extract_document_records, parse_final_force_frame
from .io import iter_documents, read_document, read_extxyz, source_tree, write_extxyz, write_manifest
from .jobs import DEFAULT_ROUTE, expanded_records, write_gaussian_jobs
from .models import Record, composition_allowed, geometry_signature
from .spin import (
    DEFAULT_SPIN_ROUTE,
    validate_spin_campaign,
    write_spin_inventory,
    write_spin_jobs,
)


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
    write_gaussian_jobs(jobs, Path(args.output), args.route, args.memory, args.nproc)
    print(f"Prepared {len(jobs)} Gaussian force jobs from {len(records)} seeds")
    print(f"Route: {args.route}")
    return 0


def command_collect(args: argparse.Namespace) -> int:
    outputs = Path(args.outputs)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = read_jobs_manifest(outputs / "jobs.csv")
    frames = []
    failures: list[tuple[str, str]] = []
    for path in sorted(list(outputs.rglob("*.log")) + list(outputs.rglob("*.out"))):
        job_id = path.stem
        row = manifest.get(job_id, {})
        seed = None
        if row:
            seed = Record(
                record_id=job_id,
                source=row.get("source", str(path)),
                atoms=[],
                charge=int(row.get("charge", 0)),
                multiplicity=int(row.get("multiplicity", 1)),
                config_type=row.get("config_type", "labeled"),
                metadata={"parent_record_id": row.get("parent_record_id", job_id)},
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
    splits = grouped_split(frames, args.valid_fraction, args.test_fraction, args.seed)
    for name, subset in splits.items():
        write_labeled_extxyz(subset, destination / f"{name}.extxyz")
    with (destination / "failed_outputs.tsv").open("w", encoding="utf-8") as handle:
        for name, message in failures:
            handle.write(f"{name}\t{message}\n")
    print(f"Collected {len(frames)} labeled frames; rejected {len(failures)} outputs")
    print("Split: " + ", ".join(f"{name}={len(values)}" for name, values in splits.items()))
    return 0


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
    records = [record for record in records if _record_allowed(record, args)]
    specifications = None
    if args.fragment_spec:
        payload = json.loads(Path(args.fragment_spec).read_text(encoding="utf-8"))
        specifications = payload.get("guesses", payload) if isinstance(payload, dict) else payload
        if not isinstance(specifications, list):
            raise ValueError("fragment specification must be a list or an object containing 'guesses'")
    stages = write_spin_jobs(
        records,
        Path(args.output),
        args.high_spin,
        args.targets,
        args.route,
        args.memory,
        args.nproc,
        specifications,
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
    )
    return 2 if args.strict and strict_failures else 0


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
    prepare.add_argument("--route", default=DEFAULT_ROUTE)
    prepare.add_argument("--memory", default="16GB")
    prepare.add_argument("--nproc", type=int, default=16)
    prepare.set_defaults(func=command_prepare)

    prepare_spins = sub.add_parser(
        "prepare-spins",
        help="prepare high-spin-to-low-spin Link1 ladders and optional fragment AFM guesses",
    )
    prepare_spins.add_argument("seeds", help="seeds.extxyz from spin-extract or extract")
    prepare_spins.add_argument("-o", "--output", default="gaussian_spin_jobs")
    prepare_spins.add_argument("--high-spin", type=int, required=True, help="trusted high-spin multiplicity")
    prepare_spins.add_argument(
        "--targets", type=_multiplicities, required=True,
        help="comma-separated low-spin multiplicities; skipped intermediate values are inserted",
    )
    prepare_spins.add_argument(
        "--fragment-spec",
        help="JSON file with explicit atom-to-fragment maps and fragment charge/spin orientations",
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
        help="exit nonzero when a legacy state is missing or converges to an alternative root",
    )
    validate_spins.set_defaults(func=command_validate_spins)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "valid_fraction", 0) + getattr(args, "test_fraction", 0) >= 1:
        parser.error("valid_fraction + test_fraction must be < 1")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
