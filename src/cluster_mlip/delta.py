"""ASE calculators for the supported-cluster Delta-model.

    E(AB) = E_gas(A; q, M) + E_support(B) + dE_int(AB; q, M)

* ``DeltaCalculator`` evaluates the three terms on the cluster atoms (A, as a
  free molecule), the support atoms (B, periodic) and the whole system, and adds
  the forces atom by atom.
* ``SubtractiveInteraction`` is the stand-in interaction term before VASP
  labels exist: dE_int = E(AB) - E(A) - E(B) from one total-energy model. With
  MACE-MP-0 in all three slots the Delta-model reduces exactly to MACE-MP-0
  on the whole system, which is what the example's baseline run uses.
* ``load_model`` turns a short spec (``mace-mp:medium+d3``, ``mace-polar:polar-1-m``,
  or a path to a trained ``.model``) into a calculator, recording whether the
  model reads the total charge/multiplicity.

The cluster/support split comes from the per-atom ``cluster`` array (1 = cluster),
the convention shared with `periodic.py` and the VASP tooling. Charge and
multiplicity are read from ``atoms.info["charge"]`` and ``atoms.info["spin"]``
(the multiplicity, as in `mace_glue.record_to_atoms`); the support is passed as
a closed-shell neutral system.

This module needs ASE (and mace-torch for `load_model`); the rest of the
package does not import it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


def cluster_mask(atoms: Atoms, cluster_elements: Iterable[str] | None = None) -> np.ndarray:
    """Boolean mask of the cluster atoms: `cluster` array, else tags == 1, else explicit elements."""
    if "cluster" in atoms.arrays:
        return np.asarray(atoms.arrays["cluster"]).astype(bool)
    tags = atoms.get_tags()
    if tags.any():
        return tags == 1
    if cluster_elements:
        elements = set(cluster_elements)
        mask = np.array([s in elements for s in atoms.get_chemical_symbols()])
        if mask.any() and not mask.all():
            return mask
    raise ValueError("cannot tell cluster from support: set atoms.arrays['cluster'] (1 = cluster)")


def mark_cluster(atoms: Atoms, mask: np.ndarray) -> Atoms:
    atoms.arrays["cluster"] = np.asarray(mask, dtype=int)
    return atoms


def _strip(atoms: Atoms) -> Atoms:
    a = atoms.copy()
    a.set_constraint()
    a.calc = None
    return a


def cluster_part(atoms: Atoms, mask: np.ndarray) -> Atoms:
    """The cluster atoms as a free molecule, keeping the system's charge and multiplicity."""
    a = _strip(atoms)[mask]
    a.pbc = False
    a.cell = None
    a.info = {"charge": int(atoms.info.get("charge", 0)), "spin": int(atoms.info.get("spin", 1))}
    return a


def support_part(atoms: Atoms, mask: np.ndarray) -> Atoms:
    """The support atoms in the original cell, closed-shell and neutral."""
    a = _strip(atoms)[~mask]
    a.info = {"charge": 0, "spin": 1}
    return a


def reset_calculator(calc: Any) -> None:
    """Drop a calculator's cached results, including every part of a SumCalculator.

    ASE's cache check compares positions/cell/numbers but not ``atoms.info``, so the same
    geometry at another multiplicity would silently reuse the old result. Mixing
    calculators (MACE-MP + D3 is a SumCalculator) have no ``reset()`` of their own.
    """
    for part in getattr(getattr(calc, "mixer", None), "calcs", ()):
        reset_calculator(part)
    if hasattr(calc, "reset"):
        calc.reset()
    else:
        calc.results = {}
        calc.atoms = None


def _evaluate(calc: Calculator, atoms: Atoms) -> tuple[float, np.ndarray]:
    reset_calculator(calc)  # the cache ignores atoms.info["spin"]
    a = atoms.copy()
    a.calc = calc
    e = float(a.get_potential_energy())
    f = np.array(a.get_forces(apply_constraint=False))
    a.calc = None  # a shared calculator's cache belongs to whichever system it saw last
    return e, f


class SubtractiveInteraction(Calculator):
    """dE_int = E(AB) - E(A) - E(B) from a single total-energy calculator."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, total: Calculator, cluster_elements: Iterable[str] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.total = total
        self.cluster_elements = tuple(cluster_elements or ())

    def calculate(self, atoms: Atoms | None = None, properties: list[str] | None = None,
                  system_changes: list[str] = all_changes) -> None:
        super().calculate(atoms, properties or ["energy"], system_changes)
        assert self.atoms is not None
        mask = cluster_mask(self.atoms, self.cluster_elements)
        full = _strip(self.atoms)
        e_ab, f_ab = _evaluate(self.total, full)
        e_a, f_a = _evaluate(self.total, cluster_part(self.atoms, mask))
        e_b, f_b = _evaluate(self.total, support_part(self.atoms, mask))
        forces = f_ab.copy()
        forces[mask] -= f_a
        forces[~mask] -= f_b
        e = e_ab - e_a - e_b
        self.results = {"energy": e, "free_energy": e, "forces": forces}


class DeltaCalculator(Calculator):
    """E = E_gas(cluster) + E_support(support) + dE_int(whole), forces summed per atom.

    After each calculation ``self.terms`` holds the three energies, so a script can
    decompose an adsorption energy without re-evaluating anything.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, cluster: Calculator, support: Calculator, interaction: Calculator,
                 cluster_elements: Iterable[str] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.cluster = cluster
        self.support = support
        self.interaction = interaction
        self.cluster_elements = tuple(cluster_elements or ())
        self.terms: dict[str, float] = {}

    def calculate(self, atoms: Atoms | None = None, properties: list[str] | None = None,
                  system_changes: list[str] = all_changes) -> None:
        super().calculate(atoms, properties or ["energy"], system_changes)
        assert self.atoms is not None
        mask = cluster_mask(self.atoms, self.cluster_elements)
        e_a, f_a = _evaluate(self.cluster, cluster_part(self.atoms, mask))
        e_b, f_b = _evaluate(self.support, support_part(self.atoms, mask))
        full = _strip(self.atoms)
        e_i, f_i = _evaluate(self.interaction, full)
        forces = f_i.copy()
        forces[mask] += f_a
        forces[~mask] += f_b
        e = e_a + e_b + e_i
        self.terms = {"cluster": e_a, "support": e_b, "interaction": e_i}
        self.results = {"energy": e, "free_energy": e, "forces": forces}


# --------------------------------------------------------------------------- models
@dataclass
class LoadedModel:
    spec: str
    calculator: Any  # an ASE calculator (MACE, or a SumCalculator with D3)
    spin_aware: bool

    @property
    def tag(self) -> str:
        """A short file-name-safe label for outputs produced with this model."""
        text = self.spec
        if "/" in text or "\\" in text:  # a model file: its name, not its whole path
            text = Path(text.split("+d3")[0].removesuffix(":blind")).stem + ("-blind" if ":blind" in text else "") + ("+d3" if text.endswith("+d3") else "")
        elif text.endswith(".model"):
            text = Path(text).stem
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def load_model(spec: str, *, device: str = "cpu", dtype: str = "float32") -> LoadedModel:
    """Build a calculator from a spec.

    ``mace-mp:<size>[+d3]``      MACE-MP-0 foundation model (spin-blind), optional D3(BJ)
    ``mace-polar:<name>``        MACE-POLAR-1 (reads charge and multiplicity)
    ``<path>.model[+d3]``        a model trained by `cluster-mlip train` (reads charge/spin)
    ``<path>.model:blind``       a trained model without charge/spin inputs
    """
    d3 = spec.endswith("+d3")
    base = spec[:-3] if d3 else spec
    if base.startswith("mace-mp:"):
        from mace.calculators import mace_mp

        mp = mace_mp(model=base.split(":", 1)[1], dispersion=d3, default_dtype=dtype, device=device)
        return LoadedModel(spec, mp, spin_aware=False)
    if base.startswith("mace-polar:"):
        from mace.calculators import mace_polar

        if d3:
            raise ValueError("MACE-POLAR-1 already includes long-range terms; drop +d3")
        return LoadedModel(spec, mace_polar(base.split(":", 1)[1], device=device, default_dtype=dtype),
                           spin_aware=True)
    blind = base.endswith(":blind")
    path = Path(base[:-6] if blind else base)
    if not path.is_file():
        raise FileNotFoundError(f"model spec {spec!r}: {path} is not a file")
    from mace.calculators import MACECalculator

    calc: Any = MACECalculator(model_paths=[str(path)], device=device, default_dtype=dtype)
    if d3:
        calc = _with_d3(calc, device, dtype)
    return LoadedModel(spec, calc, spin_aware=not blind)


def _with_d3(calc: Any, device: str, dtype: str) -> Any:
    from ase.calculators.mixing import SumCalculator
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

    import torch

    d3 = TorchDFTD3Calculator(device=device, damping="bj", xc="pbe",
                              dtype=torch.float64 if dtype == "float64" else torch.float32)
    return SumCalculator([calc, d3])


# --------------------------------------------------------------------------- spin scans
@dataclass
class SpinScanResult:
    atoms: Atoms
    multiplicity: int | None
    energy: float
    by_multiplicity: dict[int, float]
    converged: bool


def relax_over_multiplicities(
    atoms: Atoms,
    calc: Calculator,
    multiplicities: Iterable[int] | None,
    relax: Callable[[Atoms], bool],
    energy_of: Callable[[Atoms], float] | None = None,
) -> SpinScanResult:
    """Relax `atoms` at each multiplicity and keep the lowest (adiabatic) state.

    `relax(atoms)` optimises in place (atoms.calc already set) and returns whether it
    converged. Pass ``multiplicities=None`` for a spin-blind calculator: one relaxation,
    the multiplicity in ``atoms.info`` is left untouched.
    """
    energy_of = energy_of or (lambda a: float(a.get_potential_energy()))
    mults = list(multiplicities) if multiplicities else [None]
    best: SpinScanResult | None = None
    energies: dict[int, float] = {}
    for m in mults:
        a = atoms.copy()
        if m is not None:
            a.info["spin"] = int(m)
            a.info.setdefault("charge", 0)
        reset_calculator(calc)  # the cache ignores atoms.info (see reset_calculator)
        a.calc = calc
        ok = relax(a)
        e = energy_of(a)
        a.calc = None
        if m is not None:
            energies[int(m)] = e
        if best is None or e < best.energy:
            best = SpinScanResult(a, m, e, energies, ok)
    assert best is not None
    best.by_multiplicity = energies
    return best
