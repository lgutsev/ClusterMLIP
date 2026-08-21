from __future__ import annotations

import collections
import csv
import hashlib
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from .gaussian import extract_document_records
from .io import SUPPORTED_SUFFIXES, read_document, source_tree
from .models import Record, geometry_signature


@dataclass
class FileResult:
    source: str
    suffix: str
    size_bytes: int
    status: str
    record_count: int
    error: str = ""


def geometry_key(record: Record) -> str:
    payload = f"{record.charge}|{record.multiplicity}|{geometry_signature(record.atoms)}"
    return hashlib.sha256(payload.encode()).hexdigest()


def scan_source(
    source: Path,
    record_filter: Callable[[Record], bool] | None = None,
) -> tuple[list[FileResult], list[Record]]:
    files: list[FileResult] = []
    records: list[Record] = []
    with source_tree(source) as root:
        documents = (
            [source]
            if source.is_file() and source.suffix.lower() != ".zip"
            else [
                path for path in sorted(root.rglob("*"))
                if path.is_file() and not path.name.startswith("._")
            ]
        )
        for document in documents:
            relative = (
                str(document.relative_to(root))
                if document.is_relative_to(root)
                else document.name
            )
            if document.suffix.lower() not in SUPPORTED_SUFFIXES:
                files.append(
                    FileResult(
                        relative,
                        document.suffix.lower() or "[none]",
                        document.stat().st_size,
                        "unsupported",
                        0,
                    )
                )
                continue
            try:
                parsed = extract_document_records(read_document(document), relative)
                records.extend(
                    record for record in parsed
                    if record_filter is None or record_filter(record)
                )
                files.append(
                    FileResult(
                        relative,
                        document.suffix.lower() or "[none]",
                        document.stat().st_size,
                        "parsed" if parsed else "no_structure",
                        len(parsed),
                    )
                )
            except Exception as exc:
                files.append(
                    FileResult(
                        relative,
                        document.suffix.lower() or "[none]",
                        document.stat().st_size,
                        "error",
                        0,
                        str(exc),
                    )
                )
    return files, records


def _counts(values: Iterable[object]) -> dict[str, int]:
    return dict(sorted(collections.Counter(str(value) for value in values).items()))


def summarize(files: list[FileResult], records: list[Record]) -> dict[str, object]:
    geometry_counts = collections.Counter(geometry_key(record) for record in records)
    unique = list({geometry_key(record): record for record in records}.values())
    atom_counts = [len(record.atoms) for record in unique]
    elements = collections.Counter(atom.symbol for record in unique for atom in record.atoms)
    duplicate_groups = [count for count in geometry_counts.values() if count > 1]
    routes = collections.Counter(record.route.strip() for record in records if record.route.strip())
    return {
        "files": {
            "total": len(files),
            "compatible": sum(file.suffix in SUPPORTED_SUFFIXES for file in files),
            "total_bytes": sum(file.size_bytes for file in files),
            "by_suffix": _counts(file.suffix for file in files),
            "by_status": _counts(file.status for file in files),
        },
        "structures": {
            "records": len(records),
            "unique_geometry_state": len(unique),
            "duplicate_records": len(records) - len(unique),
            "duplicate_groups": len(duplicate_groups),
            "largest_duplicate_group": max(duplicate_groups, default=1 if records else 0),
            "atom_count": {
                "min": min(atom_counts, default=0),
                "median": statistics.median(atom_counts) if atom_counts else 0,
                "max": max(atom_counts, default=0),
            },
        },
        "chemistry": {
            "element_atom_counts": dict(sorted(elements.items())),
            "formulas": _counts(record.formula for record in unique),
            "charge_multiplicity": _counts(
                f"q={record.charge}, mult={record.multiplicity}" for record in unique
            ),
            "configuration_types": _counts(record.config_type for record in unique),
            "irc_paths": _counts(
                record.config_type for record in unique if record.config_type.startswith("irc")
            ),
        },
        "legacy_routes": dict(routes.most_common(50)),
    }


def _markdown_table(counts: dict[str, int], limit: int = 30) -> str:
    rows = ["| Value | Count |", "|---|---:|"]
    for key, value in list(counts.items())[:limit]:
        escaped_key = key.replace("|", "\\|")
        rows.append(f"| {escaped_key} | {value} |")
    return "\n".join(rows)


def write_analysis(
    source: Path,
    output: Path,
    record_filter: Callable[[Record], bool] | None = None,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    files, records = scan_source(source, record_filter=record_filter)
    summary = summarize(files, records)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "files.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FileResult.__annotations__)
        writer.writeheader()
        writer.writerows(asdict(file) for file in files)
    with (output / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = [
            "record_id", "source", "formula", "n_atoms", "charge", "multiplicity",
            "config_type", "irc_path", "irc_point", "legacy_energy_hartree", "route",
            "geometry_state_sha256", "state_inference",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "record_id": record.record_id,
                "source": record.source,
                "formula": record.formula,
                "n_atoms": len(record.atoms),
                "charge": record.charge,
                "multiplicity": record.multiplicity,
                "config_type": record.config_type,
                "irc_path": record.irc_path,
                "irc_point": record.irc_point,
                "legacy_energy_hartree": record.legacy_energy_hartree,
                "route": record.route,
                "geometry_state_sha256": geometry_key(record),
                # Blank means the state came directly from an explicit Gaussian
                # "Charge = x Multiplicity = y" line, not a filename convention
                # or fallback guess -- see gaussian.extract_warehouse_record.
                "state_inference": record.metadata.get("state_inference", ""),
            })
    file_summary = summary["files"]
    structure_summary = summary["structures"]
    chemistry = summary["chemistry"]
    report = f"""# Cluster database analysis

Source: `{source}`

## Coverage

- Files inventoried: {file_summary['total']}
- Compatible input files: {file_summary['compatible']}
- Parsed structure records: {structure_summary['records']}
- Unique geometry/charge/multiplicity states: {structure_summary['unique_geometry_state']}
- Duplicate records: {structure_summary['duplicate_records']} across {structure_summary['duplicate_groups']} groups
- Atom-count range: {structure_summary['atom_count']['min']}–{structure_summary['atom_count']['max']} (median {structure_summary['atom_count']['median']})

### File status

{_markdown_table(file_summary['by_status'])}

### File formats

{_markdown_table(file_summary['by_suffix'])}

## Chemistry

### Configuration types

{_markdown_table(chemistry['configuration_types'])}

### Charge and multiplicity

{_markdown_table(chemistry['charge_multiplicity'])}

### Element atom counts

{_markdown_table(chemistry['element_atom_counts'])}

### Most common formulas

{_markdown_table(chemistry['formulas'])}

## Interpretation notes

- `first_order_saddle` is inferred from exactly one imaginary frequency; `transition_state` requires an explicit TS/QST marker.
- IRC records are split only when point/path markers are present. A lone checkpoint contributes only its stored geometry.
- Duplicate identity includes Cartesian coordinates rounded to 1e-6 Å, charge, and multiplicity.
- Files with `error` status need attention; binary `.chk` files require Gaussian `formchk` in `PATH`.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    return summary
