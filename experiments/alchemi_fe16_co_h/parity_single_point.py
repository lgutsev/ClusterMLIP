"""Single-point agreement: native MACE calculator vs the toolkit wrapper.

    python parity_single_point.py FIXTURE.extxyz MODEL --out DIR [--dtype float64]

Arms, all at identical coordinates with the same checkpoint file and dtype:
  native   ASE MACECalculator, one structure at a time (evaluated twice: repeatability)
  batch1   toolkit wrapper, one structure per batch (evaluated twice: repeatability)
  batchN   toolkit wrapper, all structures in one batch
  permuted same batch in a different order, forces mapped back
  mixed    each structure batched with different companions (halves of the panel)
Also: rigid translation + rotation of every structure (energy invariance, force
covariance) through the wrapper, and neighbor counts against ASE's neighbor list.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from ase.neighborlist import neighbor_list
from scipy.spatial.transform import Rotation

from alchemi_backend import load_wrapper, single_points


def diff(a, b):
    de = abs(a["energy"] - b["energy"])
    df = a["forces"] - b["forces"]
    return de, float(np.abs(df).max()), float(np.sqrt((df ** 2).mean()))


def summarize(pairs):
    de, fmax, frms = zip(*pairs)
    return {"max_abs_dE_eV": float(max(de)), "max_force_component_diff_eV_per_A": float(max(fmax)),
            "rms_force_component_diff_eV_per_A": float(np.sqrt(np.mean(np.square(frms))))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture", type=Path)
    ap.add_argument("model")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    frames = read(args.fixture, index=":")
    ids = [a.info["fixture_id"] for a in frames]
    n = len(frames)

    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=args.model, device=args.device, default_dtype=args.dtype)

    def native_all():
        out = []
        for a in frames:
            b = a.copy()
            b.calc = calc
            out.append({"energy": float(b.get_potential_energy()), "forces": np.array(b.get_forces(), dtype=float)})
            calc.reset()
        return out

    native, native2 = native_all(), native_all()
    model = load_wrapper(args.model, device=args.device, dtype=dtype)
    cutoff = float(model.cutoff)
    one = [single_points(model, [a], device=args.device, dtype=dtype)[0] for a in frames]
    one2 = [single_points(model, [a], device=args.device, dtype=dtype)[0] for a in frames]
    full = single_points(model, frames, device=args.device, dtype=dtype)
    perm = np.random.default_rng(7).permutation(n)
    permuted_raw = single_points(model, [frames[i] for i in perm], device=args.device, dtype=dtype)
    permuted = [None] * n
    for pos, i in enumerate(perm):
        permuted[i] = permuted_raw[pos]
    mixed = [None] * n
    for group in (list(range(0, n, 2)), list(range(1, n, 2))):
        res = single_points(model, [frames[i] for i in group], device=args.device, dtype=dtype)
        for i, r in zip(group, res):
            mixed[i] = r

    # atom-order permutation inside each structure: forces must follow the atoms
    atom_perm = []
    rng = np.random.default_rng(11)
    for a, ref in zip(frames, one):
        p = rng.permutation(len(a))
        r = single_points(model, [a[p]], device=args.device, dtype=dtype)[0]
        back = np.empty_like(r["forces"])
        back[p] = r["forces"]
        atom_perm.append(diff(ref, {"energy": r["energy"], "forces": back}))

    # rigid motion (isolated cluster): energy unchanged, forces rotate with the structure
    rigid = []
    for k, (a, ref) in enumerate(zip(frames, one)):
        rot = Rotation.from_rotvec(np.array([0.3, -0.7, 0.5]) * (1 + 0.1 * k)).as_matrix()
        b = a.copy()
        centre = b.positions.mean(axis=0)
        b.positions = (b.positions - centre) @ rot.T + centre + np.array([1.3, -0.4, 2.2])
        r = single_points(model, [b], device=args.device, dtype=dtype)[0]
        rigid.append(diff({"energy": ref["energy"], "forces": ref["forces"] @ rot.T}, r))

    # neighbor construction: edge counts against ASE at the model cutoff; no edge may join two systems
    edge_rows = []
    for a, r1, rN in zip(frames, one, full):
        i_ase = neighbor_list("i", a, cutoff)
        edge_rows.append({"ase": int(len(i_ase)), "batch1": r1["edges"], "batchN": rN["edges"]})

    e_native = np.array([r["energy"] for r in native])
    rel = lambda res: np.array([r["energy"] for r in res]) - res[1]["energy"]  # noqa: E731  relative to F01
    report = {
        "fixture": str(args.fixture), "ids": ids, "model": args.model, "dtype": args.dtype, "device": args.device,
        "cutoff_A": cutoff, "n_structures": n, "atoms_per_structure": [len(a) for a in frames],
        "repeatability": {"native_vs_native": summarize([diff(a, b) for a, b in zip(native, native2)]),
                          "batch1_vs_batch1": summarize([diff(a, b) for a, b in zip(one, one2)])},
        "native_vs_batch1": summarize([diff(a, b) for a, b in zip(native, one)]),
        "native_vs_batchN": summarize([diff(a, b) for a, b in zip(native, full)]),
        "batch1_vs_batchN": summarize([diff(a, b) for a, b in zip(one, full)]),
        "batchN_vs_permuted_order": summarize([diff(a, b) for a, b in zip(full, permuted)]),
        "batchN_vs_other_companions": summarize([diff(a, b) for a, b in zip(full, mixed)]),
        "atom_order_permutation": summarize(atom_perm),
        "rigid_translation_rotation": summarize(rigid),
        "relative_energy_max_diff_native_vs_batchN_eV": float(np.abs(rel(native) - rel(full)).max()),
        "per_atom_energy_max_diff_native_vs_batchN_eV": float(max(abs(a["energy"] - b["energy"]) / len(f)
                                                                  for a, b, f in zip(native, full, frames))),
        "edges": edge_rows, "edge_counts_match_ase": all(r["ase"] == r["batch1"] == r["batchN"] for r in edge_rows),
        "cross_system_edges": int(full[0]["cross_system_edges"]),
        "native_energies_eV": e_native.tolist(),
        "ranking_native": [ids[i] for i in np.argsort(e_native[1:]) + 1],
        "ranking_batchN": [ids[i] for i in np.argsort([r["energy"] for r in full][1:]) + 1],
    }
    report["ranking_identical"] = report["ranking_native"] == report["ranking_batchN"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"parity_{args.dtype}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("edges", "ids", "native_energies_eV",
                                                                  "ranking_native", "ranking_batchN",
                                                                  "atoms_per_structure")}, indent=1))


if __name__ == "__main__":
    main()
