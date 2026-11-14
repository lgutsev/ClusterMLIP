"""Single-point parity for PERIODIC cells: native MACECalculator against the toolkit.

    python parity_periodic.py MODEL --out report.json [--dtype float64]

A small first test, not a benchmark. Eight generic periodic structures built with ASE
(bulk, strained triclinic bulk, slabs with and without an adsorbate, mixed pbc), all rattled:
native vs batch of 1, batch of N, shuffled order and a different batch composition.
The yardstick is native against native. Other projects should repeat this on their own cells.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc111, molecule

from alchemi_backend import load_wrapper, single_points


def structures() -> list[tuple[str, Atoms]]:
    rng = np.random.default_rng(7)
    out: list[tuple[str, Atoms]] = []

    def add(name: str, a: Atoms, rattle: float = 0.05) -> None:
        a = a.copy()
        a.positions += rng.normal(0, rattle, a.positions.shape)
        out.append((name, a))

    add("Fe_bcc_2x2x2_cubic", bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2))
    add("Fe_bcc_3x3x3_primitive", bulk("Fe", "bcc", a=2.87) * (3, 3, 3))
    strained = bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2)
    cell = np.array(strained.cell) @ (np.eye(3) + np.array([[0.02, 0.03, 0.0], [0.0, -0.01, 0.04], [0.01, 0.0, 0.015]]))
    strained.set_cell(cell, scale_atoms=True)
    add("Cu_fcc_2x2x2_triclinic_strained", strained)
    add("MgO_rocksalt_2x2x2", bulk("MgO", "rocksalt", a=4.21, cubic=True) * (2, 2, 2))
    add("Si_diamond_2x2x2", bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2))
    slab = fcc111("Cu", size=(3, 3, 3), vacuum=8.0)
    slab.pbc = True
    add("Cu111_slab_3x3x3_pbc_xyz", slab)
    slab_co = fcc111("Cu", size=(3, 3, 3), vacuum=8.0)
    add_adsorbate(slab_co, molecule("CO"), 1.9, "ontop")
    slab_co.pbc = True
    add("Cu111_slab_CO_pbc_xyz", slab_co)
    slab2 = fcc111("Cu", size=(2, 2, 4), vacuum=8.0)
    slab2.pbc = (True, True, False)
    add("Cu111_slab_2x2x4_pbc_xy_only", slab2)
    return out


def diff(a: dict, b: dict) -> tuple[float, float]:
    return abs(a["energy"] - b["energy"]), float(np.abs(a["forces"] - b["forces"]).max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    args = ap.parse_args()
    from mace.calculators import MACECalculator

    named = structures()
    atoms = [a for _, a in named]
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    calc = MACECalculator(model_paths=args.model, device="cuda", default_dtype=args.dtype)

    def native() -> list[dict]:
        res = []
        for a in atoms:
            b = a.copy()
            b.calc = calc
            res.append({"energy": float(b.get_potential_energy()), "forces": np.array(b.get_forces())})
        return res

    n1, n2 = native(), native()
    model = load_wrapper(args.model, device="cuda", dtype=dtype)
    b1 = [single_points(model, [a], dtype=dtype)[0] for a in atoms]
    bn = single_points(model, atoms, dtype=dtype)
    order = [5, 2, 7, 0, 3, 6, 1, 4]
    shuffled = single_points(model, [atoms[i] for i in order], dtype=dtype)
    shuffled = [shuffled[order.index(i)] for i in range(len(atoms))]
    half = single_points(model, atoms[::2], dtype=dtype)

    def worst(xs: list[dict], ys: list[dict]) -> dict:
        d = [diff(x, y) for x, y in zip(xs, ys)]
        return {"max_abs_dE_eV": max(e for e, _ in d), "max_force_component_diff_eV_per_A": max(f for _, f in d)}

    report = {
        "model": Path(args.model).name, "dtype": args.dtype,
        "structures": [{"name": n, "atoms": len(a), "pbc": [bool(x) for x in a.pbc],
                        "native_energy_eV": n1[i]["energy"], "native_max_force_eV_per_A": float(np.abs(n1[i]["forces"]).max()),
                        "native_vs_batchN_dE_eV": diff(n1[i], bn[i])[0], "native_vs_batchN_dF": diff(n1[i], bn[i])[1],
                        "edges_toolkit": bn[i].get("n_edges")} for i, (n, a) in enumerate(named)],
        "native_vs_native": worst(n1, n2), "native_vs_batch1": worst(n1, b1), "native_vs_batchN": worst(n1, bn),
        "batchN_vs_shuffled": worst(bn, shuffled), "batchN_vs_half_batch": worst(bn[::2], half),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for s in report["structures"]:
        print(f"{s['name']:36s} {s['atoms']:3d} atoms pbc={s['pbc']}  dE={s['native_vs_batchN_dE_eV']:.2e} eV  dF={s['native_vs_batchN_dF']:.2e} eV/A"
              f"  (max |F| {s['native_max_force_eV_per_A']:.2f})")
    for k in ("native_vs_native", "native_vs_batch1", "native_vs_batchN", "batchN_vs_shuffled", "batchN_vs_half_batch"):
        print(k, report[k])


if __name__ == "__main__":
    main()
