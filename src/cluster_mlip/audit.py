from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from .analysis import write_analysis
from .models import Record, composition_allowed


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_private_audit(
    source: Path,
    output: Path,
    *,
    elements: set[str] | None = None,
    required_elements: set[str] | None = None,
    min_atoms: int | None = None,
    max_atoms: int | None = None,
    types: set[str] | None = None,
) -> dict[str, object]:
    """Generate a complete private audit and an optional filtered selection."""
    source = source.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    full = write_analysis(source, output / "full")
    constraints = any(
        value is not None
        for value in (elements, required_elements, min_atoms, max_atoms, types)
    )
    selected = None
    if constraints:
        def keep(record: Record) -> bool:
            symbols = {atom.symbol for atom in record.atoms}
            return (
                composition_allowed(record.atoms, elements)
                and (required_elements is None or required_elements.issubset(symbols))
                and (min_atoms is None or len(record.atoms) >= min_atoms)
                and (max_atoms is None or len(record.atoms) <= max_atoms)
                and (types is None or record.config_type in types)
            )

        selected = write_analysis(source, output / "selection", record_filter=keep)

    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "source": str(source),
        "source_size_bytes": source.stat().st_size if source.is_file() else None,
        "source_sha256": _sha256(source),
        "filters": {
            "elements": sorted(elements) if elements else None,
            "required_elements": sorted(required_elements) if required_elements else None,
            "min_atoms": min_atoms,
            "max_atoms": max_atoms,
            "types": sorted(types) if types else None,
        },
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"full": full, "selection": selected, "provenance": provenance}
