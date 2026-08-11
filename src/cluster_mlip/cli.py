from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from .analysis import write_analysis
from .dataset import grouped_split, read_jobs_manifest, write_labeled_extxyz
from .gaussian import extract_document_records, parse_final_force_frame
from .io import iter_documents, read_document, read_extxyz, source_tree, write_extxyz, write_manifest
from .jobs import DEFAULT_ROUTE, expanded_records, write_gaussian_jobs
from .models import Record, composition_allowed


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
            tuple((a.symbol, round(a.x, 6), round(a.y, 6), round(a.z, 6)) for a in record.atoms),
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
    summary = write_analysis(source, output)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cluster-mlip", description="Legacy Gaussian cluster-to-MACE pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="inventory and summarize an archive/database")
    analyze.add_argument("source", help="archive ZIP, document, or directory")
    analyze.add_argument("-o", "--output", default="analysis")
    analyze.set_defaults(func=command_analyze)

    extract = sub.add_parser("extract", help="extract stationary points and explicit IRC points")
    extract.add_argument("source", help="archive ZIP, document, or directory")
    extract.add_argument("-o", "--output", default="extracted")
    extract.add_argument("--elements", help="keep only systems composed of these comma-separated elements")
    extract.add_argument("--require-elements", help="require all of these comma-separated elements")
    extract.add_argument("--min-atoms", type=int)
    extract.add_argument("--max-atoms", type=int)
    extract.add_argument("--types", nargs="+", help="keep selected config_type values")
    extract.set_defaults(func=command_extract)

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

    collect = sub.add_parser("collect", help="collect completed Gaussian force outputs into MACE extxyz")
    collect.add_argument("outputs", help="directory containing .log/.out files and optional jobs.csv")
    collect.add_argument("-o", "--output", default="dataset")
    collect.add_argument("--valid-fraction", type=float, default=0.10)
    collect.add_argument("--test-fraction", type=float, default=0.10)
    collect.add_argument("--seed", type=int, default=20260811)
    collect.set_defaults(func=command_collect)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "valid_fraction", 0) + getattr(args, "test_fraction", 0) >= 1:
        parser.error("valid_fraction + test_fraction must be < 1")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
