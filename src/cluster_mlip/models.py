from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Atom:
    symbol: str
    x: float
    y: float
    z: float


@dataclass
class Record:
    record_id: str
    source: str
    atoms: list[Atom]
    charge: int
    multiplicity: int
    config_type: str
    route: str = ""
    legacy_energy_hartree: float | None = None
    imaginary_frequencies: int | None = None
    irc_path: int | None = None
    irc_point: int | None = None
    electronic_state: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def total_spin(self) -> int:
        """Return 2S = multiplicity - 1."""
        return self.multiplicity - 1

    @property
    def formula(self) -> str:
        counts: dict[str, int] = {}
        for atom in self.atoms:
            counts[atom.symbol] = counts.get(atom.symbol, 0) + 1
        if "C" in counts:
            order = ["C"] + (["H"] if "H" in counts else []) + sorted(s for s in counts if s not in {"C", "H"})
        else:
            order = sorted(counts)
        return "".join(s + (str(counts[s]) if counts[s] != 1 else "") for s in order)


@dataclass
class LabeledFrame:
    record: Record
    energy_ev: float
    forces_ev_ang: list[tuple[float, float, float]]
    output_file: Path


def composition_allowed(atoms: Iterable[Atom], allowed: set[str] | None) -> bool:
    return allowed is None or {a.symbol for a in atoms}.issubset(allowed)


def geometry_signature(atoms: Iterable[Atom]) -> str:
    """Canonical geometry identity string, coordinates rounded to 1e-6 Angstrom.

    This is the single source of truth for "same geometry" used across the
    pipeline (duplicate detection in analysis/audit, record-id derivation in
    gaussian parsing, seed de-duplication in the extract command). Keeping
    one implementation avoids the three copies silently drifting apart.
    """
    return ";".join(f"{a.symbol}:{a.x:.6f},{a.y:.6f},{a.z:.6f}" for a in atoms)
