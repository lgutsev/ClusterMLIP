from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from .analysis import DatabaseSummary, summarize_and_write
from .analysis import scan_source as scan_one_source
from .models import Record


class ZipEntry(TypedDict):
    source: str
    summary: DatabaseSummary


class MasterEntry(TypedDict):
    formula: str
    charge: int
    multiplicity: int
    config_type: str
    n_atoms: int
    sources: list[str]


class InventoryResult(TypedDict):
    zips: list[ZipEntry]
    master: list[MasterEntry]


def find_zips(folder: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*.zip" if recursive else "*.zip"
    return sorted(folder.glob(pattern))


def _master_key(record: Record) -> tuple[str, int, int, str]:
    return (record.formula, record.charge, record.multiplicity, record.config_type)


def build_inventory(
    folder: Path, output: Path, recursive: bool = False, jobs: int = 1
) -> InventoryResult:
    """Inventory every ZIP directly under `folder` (or, with `recursive`,
    under any subfolder), writing one per-ZIP report (same files
    `cluster-mlip analyze` writes for a single source) into
    `output/by_source/<zip-stem>/`, plus one merged master list across all of
    them at the top level.

    Each ZIP is scanned exactly once (via analysis.scan_source), then that
    same scan result is used both to write the per-ZIP report
    (analysis.summarize_and_write) and to fold into the merged master list --
    a large LONI-scale warehouse ZIP is expensive enough to parse that
    scanning it twice would be a real cost, not just an inefficiency.
    """
    zips = find_zips(folder, recursive=recursive)
    if not zips:
        raise ValueError(f"no *.zip files found under {folder}" + (" (recursively)" if recursive else ""))

    output.mkdir(parents=True, exist_ok=True)
    by_source_dir = output / "by_source"

    zip_entries: list[ZipEntry] = []
    master: dict[tuple[str, int, int, str], MasterEntry] = {}
    for zip_path in zips:
        files, records = scan_one_source(zip_path, jobs=jobs)
        summary = summarize_and_write(zip_path, files, records, by_source_dir / zip_path.stem)
        zip_entries.append({"source": zip_path.name, "summary": summary})
        for record in records:
            key = _master_key(record)
            master_entry = master.setdefault(
                key,
                {
                    "formula": record.formula,
                    "charge": record.charge,
                    "multiplicity": record.multiplicity,
                    "config_type": record.config_type,
                    "n_atoms": len(record.atoms),
                    "sources": [],
                },
            )
            if zip_path.name not in master_entry["sources"]:
                master_entry["sources"].append(zip_path.name)

    master_rows = sorted(
        master.values(), key=lambda row: (row["formula"], row["charge"], row["multiplicity"], row["config_type"])
    )
    for row in master_rows:
        row["sources"].sort()
    result: InventoryResult = {"zips": zip_entries, "master": master_rows}

    (output / "inventory.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Warehouse inventory",
        "",
        f"- ZIP files inventoried: {len(zip_entries)}",
        f"- Unique formula/charge/multiplicity/state combinations: {len(master_rows)}",
        "",
        "## Per-ZIP summary",
        "",
        "| ZIP | Files | Structure records | Unique states |",
        "|---|---:|---:|---:|",
    ]
    for entry in zip_entries:
        structures = entry["summary"]["structures"]
        files_summary = entry["summary"]["files"]
        lines.append(
            f"| {entry['source']} | {files_summary['total']} | {structures['records']} | "
            f"{structures['unique_geometry_state']} |"
        )
    lines += [
        "",
        "See `by_source/<zip-name>/report.md` for each ZIP's own full analysis.",
        "",
        "## Master list: every formula/charge/multiplicity/state we have, and where",
        "",
        "| Formula | Charge | Multiplicity | Type | Atoms | Found in |",
        "|---|---:|---:|---|---:|---|",
    ]
    for row in master_rows:
        lines.append(
            f"| {row['formula']} | {row['charge']} | {row['multiplicity']} | {row['config_type']} | "
            f"{row['n_atoms']} | {', '.join(row['sources'])} |"
        )
    (output / "inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def known_formulas(inventory: InventoryResult) -> set[str]:
    """The coarse "do we have any calculation of this composition at all"
    set literature.py compares against -- a paper's title/abstract almost
    never states charge/multiplicity/config_type, so matching at that finer
    grain (available in `inventory["master"]` for a human to inspect) would
    under-match essentially everything.
    """
    return {row["formula"] for row in inventory["master"]}
