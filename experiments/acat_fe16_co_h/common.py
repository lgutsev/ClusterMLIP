"""Shared pieces of the Fe16 + CO + 2H pilot: catalyst, site list, calculators."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.io import read  # noqa: E402

from cluster_mlip.adsorbate_search import ROLE_CATALYST, SiteReport, find_cluster_sites  # noqa: E402

CATALYST_FILE = HERE / "catalyst" / "fe16_200_49_gf_m49.extxyz"


def load_catalyst(path: Path = CATALYST_FILE) -> Atoms:
    atoms = read(path)
    atoms.calc = None
    atoms.set_array("role", np.full(len(atoms), ROLE_CATALYST))
    return atoms


def load_sites(catalyst: Atoms) -> SiteReport:
    return find_cluster_sites(catalyst, bond_cutoff=3.0)


def make_calculator(spec: str, device: str = "cuda", dtype: str = "float32") -> Any:
    """``mace:<path-or-name>`` (foundation checkpoint, no charge/spin input) or ``toy``."""
    if spec == "toy":
        from toy_calculator import ToyPairCalculator
        return ToyPairCalculator()
    if spec.startswith("mace:"):
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=spec[5:], device=device, default_dtype=dtype)
    raise ValueError(f"unknown calculator spec {spec!r}")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
