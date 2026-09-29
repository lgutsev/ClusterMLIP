"""Minimal periodic extended-XYZ reader/writer for supported-cluster structures.

`io.read_extxyz` handles the gas-phase `Record` format only (no cell, no per-atom
columns beyond species/positions). Supported clusters need the lattice, the pbc
flags and a per-atom marker of which atoms belong to the cluster, and they are
usually written by ASE. This module reads and writes that subset of the format
without importing ASE, so the VASP campaign tooling keeps the package free of
third-party dependencies.

Per-atom conventions used across the supported-cluster tooling:

* ``cluster`` (integer 0/1) -- 1 for cluster atoms, 0 for support atoms. This is
  the authoritative marker; ``tags`` (ASE: 1 = cluster) is accepted as a fallback.
* ``initial_magmoms`` / ``atomic_spins`` (real) -- optional starting moments.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io import parse_extxyz_info_line, quote_extxyz

_TYPE_CODES = {"S": str, "R": float, "I": int, "L": None}


def _parse_bool(value: str) -> bool:
    if value in {"T", "True", "true", "1"}:
        return True
    if value in {"F", "False", "false", "0"}:
        return False
    raise ValueError(f"not a logical value: {value!r}")


@dataclass
class Structure:
    """One periodic (or free) structure with arbitrary per-atom columns."""

    symbols: list[str]
    positions: list[tuple[float, float, float]]
    cell: list[tuple[float, float, float]] | None = None
    pbc: tuple[bool, bool, bool] = (False, False, False)
    info: dict[str, Any] = field(default_factory=dict)
    # name -> one value (or tuple of values) per atom
    arrays: dict[str, list[Any]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.symbols)

    @property
    def structure_id(self) -> str:
        for key in ("structure_id", "record_id", "name"):
            if self.info.get(key) not in (None, ""):
                return str(self.info[key])
        return ""

    def cluster_mask(self, cluster_elements: set[str] | None = None) -> list[bool]:
        """Return the cluster/support split, refusing to guess when it is ambiguous."""
        if "cluster" in self.arrays:
            return [bool(int(v)) for v in self.arrays["cluster"]]
        if "tags" in self.arrays and any(int(v) for v in self.arrays["tags"]):
            return [int(v) == 1 for v in self.arrays["tags"]]
        if cluster_elements:
            mask = [s in cluster_elements for s in self.symbols]
            if not any(mask) or all(mask):
                raise ValueError(
                    f"{self.structure_id or 'structure'}: cluster elements {sorted(cluster_elements)} "
                    "select none or all of the atoms"
                )
            return mask
        raise ValueError(
            f"{self.structure_id or 'structure'}: no 'cluster' column or tags; "
            "pass the cluster elements explicitly"
        )


def _parse_properties(spec: str) -> list[tuple[str, str, int]]:
    parts = spec.split(":")
    if len(parts) % 3:
        raise ValueError(f"malformed Properties={spec!r}")
    return [(parts[i], parts[i + 1], int(parts[i + 2])) for i in range(0, len(parts), 3)]


def _info_value(raw: str) -> Any:
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    if raw in {"T", "F"}:
        return raw == "T"
    return raw


def read_structures(path: Path) -> list[Structure]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[Structure] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n_atoms = int(lines[i].split()[0])
        raw_info = parse_extxyz_info_line(lines[i + 1])
        props = _parse_properties(raw_info.pop("Properties", "species:S:1:pos:R:3"))
        cell = None
        if "Lattice" in raw_info:
            values = [float(v) for v in str(raw_info.pop("Lattice")).split()]
            cell = [tuple(values[k:k + 3]) for k in (0, 3, 6)]
        pbc: tuple[bool, bool, bool] = (False, False, False)
        if "pbc" in raw_info:
            flags = str(raw_info.pop("pbc")).split()
            pbc = tuple(_parse_bool(f) for f in flags)  # type: ignore[assignment]
        elif cell is not None:
            pbc = (True, True, True)
        info: dict[str, Any] = {}
        for key, value in raw_info.items():
            if key == "metadata":
                try:
                    info[key] = json.loads(value)
                    continue
                except (TypeError, ValueError):
                    pass
            info[key] = _info_value(value) if isinstance(value, str) else value
        columns: dict[str, list[Any]] = {name: [] for name, _, _ in props}
        for line in lines[i + 2:i + 2 + n_atoms]:
            tokens = line.split()
            k = 0
            for name, code, width in props:
                chunk = tokens[k:k + width]
                k += width
                if code == "L":
                    vals: list[Any] = [_parse_bool(t) for t in chunk]
                else:
                    cast = _TYPE_CODES[code]
                    vals = [cast(t) for t in chunk]  # type: ignore[misc]
                columns[name].append(vals[0] if width == 1 else tuple(vals))
        symbols = [str(s) for s in columns.pop("species")]
        positions = [tuple(p) for p in columns.pop("pos")]
        out.append(Structure(symbols, positions, cell, pbc, info, columns))  # type: ignore[arg-type]
        i += n_atoms + 2
    return out


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.10f}"
    return str(value)


def _column_code(values: list[Any]) -> tuple[str, int]:
    first = values[0]
    width = len(first) if isinstance(first, tuple) else 1
    sample = first[0] if isinstance(first, tuple) else first
    if isinstance(sample, bool):
        return "L", width
    if isinstance(sample, int):
        return "I", width
    if isinstance(sample, float):
        return "R", width
    return "S", width


def write_structures(structures: list[Structure], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for s in structures:
            names = list(s.arrays)
            spec = ["species:S:1", "pos:R:3"]
            codes = {}
            for name in names:
                code, width = _column_code(s.arrays[name])
                codes[name] = code
                spec.append(f"{name}:{code}:{width}")
            fields = []
            if s.cell is not None:
                lattice = " ".join(f"{v:.10f}" for row in s.cell for v in row)
                fields.append(f'Lattice="{lattice}"')
            fields.append(f"Properties={':'.join(spec)}")
            for key, value in s.info.items():
                if isinstance(value, (dict, list)):
                    fields.append(f"{key}={quote_extxyz(json.dumps(value, sort_keys=True))}")
                elif isinstance(value, str):
                    fields.append(f"{key}={quote_extxyz(value)}")
                elif isinstance(value, float):
                    fields.append(f"{key}={value:.12g}")
                else:
                    fields.append(f"{key}={_format_value(value)}")
            fields.append('pbc="' + " ".join("T" if p else "F" for p in s.pbc) + '"')
            handle.write(f"{len(s)}\n{' '.join(fields)}\n")
            for index, (symbol, pos) in enumerate(zip(s.symbols, s.positions)):
                row = [f"{symbol:3s}", *(f"{x: .10f}" for x in pos)]
                for name in names:
                    value = s.arrays[name][index]
                    items = value if isinstance(value, tuple) else (value,)
                    row.extend(_format_value(v) for v in items)
                handle.write(" ".join(row) + "\n")
