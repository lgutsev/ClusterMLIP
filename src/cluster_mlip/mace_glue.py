"""Optional MACE/ASE/torch integration, isolated from the rest of the package.

Nothing else in cluster_mlip imports torch/mace/ase at module load time --
those are only pulled in from inside the functions here, which are only
called from `evaluate.predict_with_mace` and
`active_learning.predict_committee_forces`. That keeps `analyze`, `extract`,
`spin-extract`, etc. fully usable with the zero-dependency base install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import Record

TRAIN_EXTRA_HINT = "pip install -e '.[train]'  (mace-torch, torch, ase)"


class MaceUnavailable(RuntimeError):
    """Raised when a command needs the optional training stack and it isn't installed."""


def require_mace() -> None:
    try:
        import ase  # noqa: F401
        import mace.calculators  # noqa: F401
    except ImportError as exc:
        raise MaceUnavailable(
            f"this command needs the optional training stack: {TRAIN_EXTRA_HINT}"
        ) from exc


def record_to_atoms(record: Record) -> Any:
    """Build an ASE Atoms object matching the charge/spin keys MACE was
    trained with (see configs/train_from_scratch.sh: total_charge_key=charge,
    total_spin_key=spin).
    """
    from ase import Atoms

    atoms = Atoms(
        symbols=[atom.symbol for atom in record.atoms],
        positions=[(atom.x, atom.y, atom.z) for atom in record.atoms],
    )
    atoms.info["charge"] = record.charge
    atoms.info["spin"] = record.multiplicity
    return atoms


def load_calculator(model_paths: list[Path], device: str = "cpu") -> Any:
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=[str(path) for path in model_paths], device=device)


def predict_forces_and_energy(calculator: Any, record: Record) -> tuple[float, list[tuple[float, float, float]]]:
    atoms = record_to_atoms(record)
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    forces = [(float(fx), float(fy), float(fz)) for fx, fy, fz in atoms.get_forces()]
    return energy, forces
