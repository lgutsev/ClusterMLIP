"""Export saddle-point geometries with their imaginary-mode displacement vectors.

Every Gaussian frequency analysis prints, per normal mode, the frequency, the
reduced mass and a displacement vector for each atom. For a saddle point the
imaginary modes are the directions that lead off it: the reaction coordinate of
a first-order saddle, and the surplus directions of a higher-order one. This
module reads those vectors out of finished logs so they can be looked at,
animated, or used to displace a geometry along a mode.

One record is written per (frequency analysis, imaginary mode):

- ``saddle_modes.extxyz``: the geometry of the analysis, with a per-atom
  ``mode`` column (the displacement vector as Gaussian prints it: normalized,
  not mass-weighted, Cartesian, two decimals) and the frequency, reduced mass,
  charge, multiplicity and source in the comment line;
- ``saddle_modes.csv``: one row per record, without the vectors.

The geometry is the last orientation table printed before the analysis.
Gaussian reports normal modes in the standard orientation when symmetry is on,
so that table is preferred when the job printed one; with ``NoSymm`` only the
input orientation exists and both frames coincide.

Only the default mode table is read (``Atom  AN  X  Y  Z``). ``Freq=HPModes``
prints a different, higher-precision table, which is reported as unsupported
rather than misread.

Read-only: nothing in the campaign is modified.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gaussian import _CM_RE, _SCF_RE, _orientation_tables
from .models import Atom
from .route_audit import audit_campaign_routes
from .routes import _HARMONIC_RE

HARTREE_TO_EV = 27.211386245988

SUMMARY_COLUMNS = [
    "input", "batch", "config_type", "intent", "state", "formula", "charge",
    "multiplicity", "analysis_index", "analyses_in_log", "imaginary_modes",
    "mode_index", "frequency_cm1", "reduced_mass_amu", "largest_atom_displacement",
    "largest_atom", "orientation", "energy_ev", "output",
]

_MODE_NUMBERS_RE = re.compile(r"^\s+\d+(?:\s+\d+)*\s*$")
_VALUES_RE = re.compile(r"^\s*(Frequencies|Red\. masses)\s+--\s+(.*)$")
_ATOM_HEADER_RE = re.compile(r"^\s*Atom\s+AN\s+X\s+Y\s+Z")
_HPMODES_RE = re.compile(r"^\s*Coord\s+Atom\s+Element", re.M)
_END_RE = re.compile(r"Thermochemistry|Normal termination|Error termination", re.I)


@dataclass
class ImaginaryMode:
    index: int                      # 1-based mode number in the analysis
    frequency_cm1: float            # negative
    reduced_mass_amu: float | None
    vectors: list[tuple[float, float, float]]


@dataclass
class FrequencyAnalysis:
    position: int
    charge: int | None
    multiplicity: int | None
    energy_ev: float | None
    atoms: list[Atom]
    orientation: str
    frequencies: list[float]
    imaginary: list[ImaginaryMode]
    unsupported: str = ""


def _floats(text: str) -> list[float]:
    values = []
    for token in text.split():
        try:
            values.append(float(token.replace("D", "E").replace("d", "e")))
        except ValueError:
            pass
    return values


def _parse_modes(block: str, n_atoms: int) -> tuple[list[float], dict[int, tuple[float | None, list[tuple[float, float, float]]]]]:
    """All frequencies of one analysis, and (reduced mass, vectors) per mode number."""
    frequencies: list[float] = []
    modes: dict[int, tuple[float | None, list[tuple[float, float, float]]]] = {}
    lines = block.splitlines()
    i = 0
    numbers: list[int] = []
    masses: list[float] = []
    group_freqs: list[float] = []
    while i < len(lines):
        line = lines[i]
        if _MODE_NUMBERS_RE.match(line) and i + 2 < len(lines) and any(
                "Frequencies --" in candidate for candidate in lines[i + 1:i + 3]):
            numbers = [int(token) for token in line.split()]
            masses, group_freqs = [], []
        value = _VALUES_RE.match(line)
        if value:
            if value.group(1) == "Frequencies":
                group_freqs = _floats(value.group(2))
                frequencies.extend(group_freqs)
            else:
                masses = _floats(value.group(2))
        if _ATOM_HEADER_RE.match(line) and numbers and len(group_freqs) == len(numbers):
            rows = lines[i + 1:i + 1 + n_atoms]
            vectors: list[list[tuple[float, float, float]]] = [[] for _ in numbers]
            complete = len(rows) == n_atoms
            for row in rows:
                parts = row.split()
                if len(parts) != 2 + 3 * len(numbers):
                    complete = False
                    break
                try:
                    values = [float(token) for token in parts[2:]]
                except ValueError:
                    complete = False
                    break
                for column in range(len(numbers)):
                    x, y, z = values[3 * column:3 * column + 3]
                    vectors[column].append((x, y, z))
            if complete:
                for column, number in enumerate(numbers):
                    mass = masses[column] if column < len(masses) else None
                    modes[number] = (mass, vectors[column])
            i += n_atoms
            numbers = []
        i += 1
    return frequencies, modes


def frequency_analyses(text: str) -> list[FrequencyAnalysis]:
    """Every harmonic analysis in a Gaussian log, with its imaginary modes."""
    headers = [match.start() for match in _HARMONIC_RE.finditer(text)]
    if not headers:
        return []
    tables = _orientation_tables(text)
    charges = list(_CM_RE.finditer(text))
    energies = list(_SCF_RE.finditer(text))
    analyses: list[FrequencyAnalysis] = []
    for number, start in enumerate(headers):
        limit = headers[number + 1] if number + 1 < len(headers) else len(text)
        end_match = _END_RE.search(text, start, limit)
        block = text[start:end_match.start() if end_match else limit]
        cm = next((m for m in reversed(charges) if m.end() < start), None)
        section_start = cm.start() if cm else 0
        before = [table for table in tables if section_start <= table[0] and table[1] < start]
        if not before:
            continue
        last = before[-1]
        # Modes are printed in the standard orientation when the job has one:
        # take the standard table of the same geometry (printed just before or
        # after the input table), else the last table of any kind.
        standard = [table for table in before[-2:] if table[3].lower().startswith("standard")]
        _, _, atoms, orientation = standard[-1] if standard else last
        energy = next((m for m in reversed(energies) if m.end() < start), None)
        energy_ev = None
        if energy is not None and energy.start() >= section_start:
            values = _floats(energy.group(1))
            energy_ev = values[0] * HARTREE_TO_EV if values else None
        analysis = FrequencyAnalysis(
            position=start,
            charge=int(cm.group(1)) if cm else None,
            multiplicity=int(cm.group(2)) if cm else None,
            energy_ev=energy_ev, atoms=atoms, orientation=orientation,
            frequencies=[], imaginary=[],
        )
        if _HPMODES_RE.search(block):
            analysis.unsupported = "Freq=HPModes table is not read"
            analyses.append(analysis)
            continue
        frequencies, modes = _parse_modes(block, len(atoms))
        analysis.frequencies = frequencies
        for index, frequency in enumerate(frequencies, 1):
            if frequency >= 0:
                continue
            if index not in modes:
                analysis.unsupported = f"no displacement table for imaginary mode {index}"
                continue
            mass, vectors = modes[index]
            analysis.imaginary.append(ImaginaryMode(index, frequency, mass, vectors))
        analyses.append(analysis)
    return analyses


def _formula(atoms: list[Atom]) -> str:
    counts: dict[str, int] = {}
    for atom in atoms:
        counts[atom.symbol] = counts.get(atom.symbol, 0) + 1
    return "".join(f"{symbol}{count if count > 1 else ''}" for symbol, count in sorted(counts.items()))


def _extxyz_frame(analysis: FrequencyAnalysis, mode: ImaginaryMode, info: dict[str, Any]) -> str:
    fields = {
        "frequency_cm1": f"{mode.frequency_cm1:.4f}",
        "mode_index": mode.index,
        "imaginary_modes": len(analysis.imaginary),
        "reduced_mass_amu": "" if mode.reduced_mass_amu is None else f"{mode.reduced_mass_amu:.4f}",
        "charge": "" if analysis.charge is None else analysis.charge,
        "multiplicity": "" if analysis.multiplicity is None else analysis.multiplicity,
        "energy": "" if analysis.energy_ev is None else f"{analysis.energy_ev:.8f}",
        "orientation": json.dumps(analysis.orientation),
        **{key: json.dumps(str(value)) for key, value in info.items()},
    }
    comment = "Properties=species:S:1:pos:R:3:mode:R:3 " + " ".join(
        f"{key}={value}" for key, value in fields.items() if value != ""
    )
    lines = [str(len(analysis.atoms)), comment]
    for atom, (dx, dy, dz) in zip(analysis.atoms, mode.vectors):
        lines.append(
            f"{atom.symbol:<2} {atom.x:14.8f} {atom.y:14.8f} {atom.z:14.8f} "
            f"{dx:8.2f} {dy:8.2f} {dz:8.2f}"
        )
    return "\n".join(lines) + "\n"


def export_campaign_saddle_modes(
    campaign: Path,
    destination: Path,
    *,
    all_intents: bool = False,
) -> dict[str, Any]:
    """Write ``saddle_modes.extxyz``/``.csv`` for a campaign's finished logs.

    By default only jobs labeled as saddles are read; ``all_intents`` also
    reports imaginary modes found in jobs labeled as minima or unconstrained
    (a "minimum" with an imaginary mode is itself a finding).
    """
    campaign = campaign.resolve()
    audit = audit_campaign_routes(campaign, write_reports=False)
    rows: list[dict[str, Any]] = []
    frames: list[str] = []
    notes: list[dict[str, str]] = []
    logs_read = 0
    for audit_row in audit["rows"]:
        if audit_row["state"] == "not_started" or not audit_row["output"]:
            continue
        if not all_intents and audit_row["intent"] != "saddle":
            continue
        output = campaign / audit_row["output"]
        if not output.is_file():
            continue
        logs_read += 1
        analyses = frequency_analyses(output.read_text(encoding="utf-8", errors="replace"))
        for index, analysis in enumerate(analyses):
            if analysis.unsupported:
                notes.append({"output": audit_row["output"], "note": analysis.unsupported})
            for mode in analysis.imaginary:
                size, atom_number = max(
                    ((x * x + y * y + z * z) ** 0.5, number)
                    for number, (x, y, z) in enumerate(mode.vectors, 1)
                )
                info = {
                    "config_type": audit_row["config_type"], "batch": audit_row["batch"],
                    "analysis_index": index, "source": audit_row["output"],
                }
                frames.append(_extxyz_frame(analysis, mode, info))
                rows.append({
                    "input": audit_row["input"], "batch": audit_row["batch"],
                    "config_type": audit_row["config_type"], "intent": audit_row["intent"],
                    "state": audit_row["state"], "formula": _formula(analysis.atoms),
                    "charge": analysis.charge, "multiplicity": analysis.multiplicity,
                    "analysis_index": index, "analyses_in_log": len(analyses),
                    "imaginary_modes": len(analysis.imaginary), "mode_index": mode.index,
                    "frequency_cm1": f"{mode.frequency_cm1:.4f}",
                    "reduced_mass_amu": (
                        "" if mode.reduced_mass_amu is None else f"{mode.reduced_mass_amu:.4f}"
                    ),
                    "largest_atom_displacement": f"{size:.2f}",
                    "largest_atom": f"{analysis.atoms[atom_number - 1].symbol}{atom_number}",
                    "orientation": analysis.orientation,
                    "energy_ev": "" if analysis.energy_ev is None else f"{analysis.energy_ev:.8f}",
                    "output": audit_row["output"],
                })
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "saddle_modes.extxyz").write_text("".join(frames), encoding="utf-8", newline="\n")
    with (destination / "saddle_modes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    analyses_with_modes = len({(row["output"], row["analysis_index"]) for row in rows})
    summary = {
        "campaign": str(campaign), "logs_read": logs_read, "records": len(rows),
        "analyses_with_imaginary_modes": analyses_with_modes,
        "first_order": sum(1 for row in rows if row["imaginary_modes"] == 1),
        "notes": notes, "destination": str(destination),
    }
    (destination / "saddle_modes.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
