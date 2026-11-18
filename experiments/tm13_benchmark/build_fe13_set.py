"""Build the held-out Fe13 set: one source geometry plus four deterministic displacements.

    python experiments/tm13_benchmark/build_fe13_set.py --source-xyz SRC.xyz --source-label LABEL \
        --provenance "where SRC came from" --out experiments/tm13_benchmark/sets/LABEL
    python experiments/tm13_benchmark/build_fe13_set.py --legacy-states STATES.extxyz --state-id 2fdecba6 \
        --source-label fe13_legacy_ubpw91_m45 --out experiments/tm13_benchmark/sets/fe13_legacy_ubpw91_m45

Configurations (atom order of the source is kept; coordinates in Angstrom):
  c0_source     the source geometry, unchanged
  c1_expand     every atom moved radially out from the centroid, scale 1 + s
  c2_compress   scale 1 - s; s is set so the mean shell-atom displacement is --radial (default 0.04 A)
  c3_surface    one surface atom (lowest index that is not the central atom) moved --surface A (default 0.05)
                along the tangential direction towards its nearest surface neighbour
  c4_random     seeded Gaussian displacement of all atoms, net translation removed, then rescaled so the
                per-atom RMS displacement is exactly --random (default 0.04 A)
All five are test data only: never add them to a training set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tm13_common import BOND_CUTOFF, central_atom, describe, sha256  # noqa: E402


def write_xyz(path: Path, symbols: list[str], pos: np.ndarray, comment: str) -> None:
    lines = [str(len(symbols)), comment] + [f"{s:2s} {x:16.10f} {y:16.10f} {z:16.10f}" for s, (x, y, z) in zip(symbols, pos)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-xyz")
    ap.add_argument("--legacy-states", help="fen_oct9_states/states.extxyz")
    ap.add_argument("--state-id", help="state_id prefix in --legacy-states")
    ap.add_argument("--source-label", required=True)
    ap.add_argument("--provenance", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--radial", type=float, default=0.04)
    ap.add_argument("--surface", type=float, default=0.05)
    ap.add_argument("--random", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    from ase.io import read

    source: dict = {"label": args.source_label, "provenance": args.provenance}
    if args.legacy_states:
        hits = [a for a in read(args.legacy_states, ":") if str(a.info["state_id"]).startswith(args.state_id)]
        if len(hits) != 1:
            raise SystemExit(f"{len(hits)} states match {args.state_id}")
        atoms = hits[0]
        source.update(file=str(args.legacy_states), file_sha256=sha256(args.legacy_states),
                      state_id=str(atoms.info["state_id"]), legacy_doc=str(atoms.info["source"]),
                      level=str(atoms.info["level"]), charge=int(atoms.info["charge"]),
                      multiplicity=int(atoms.info["multiplicity"]), energy_eV=float(atoms.info["REF_energy"]),
                      mulliken_spins=[float(x) for x in atoms.arrays["mulliken_spin"]])
    else:
        atoms = read(args.source_xyz)
        source.update(file=str(args.source_xyz), file_sha256=sha256(args.source_xyz))
    symbols = atoms.get_chemical_symbols()
    if len(symbols) != 13 or len(set(symbols)) != 1:
        raise SystemExit(f"expected 13 atoms of one element, got {atoms.get_chemical_formula()}")
    pos0 = np.array(atoms.positions, float)
    cutoff = BOND_CUTOFF[symbols[0]]
    c = central_atom(pos0)
    centroid = pos0.mean(0)
    shell = [i for i in range(13) if i != c]
    radii = np.linalg.norm(pos0[shell] - centroid, axis=1)
    s = args.radial / radii.mean()

    configs: dict[str, tuple[np.ndarray, dict]] = {"c0_source": (pos0, {"kind": "source"})}
    for name, sign in (("c1_expand", 1), ("c2_compress", -1)):
        configs[name] = (centroid + (pos0 - centroid) * (1 + sign * s),
                         {"kind": "radial", "scale": 1 + sign * s, "mean_shell_displacement_A": args.radial})
    atom = shell[0]
    d = np.linalg.norm(pos0[shell] - pos0[atom], axis=1)
    d[0] = np.inf
    neighbour = shell[int(np.argmin(d))]
    radial = (pos0[atom] - centroid) / np.linalg.norm(pos0[atom] - centroid)
    t = pos0[neighbour] - pos0[atom]
    t -= radial * (t @ radial)
    t /= np.linalg.norm(t)
    p = pos0.copy()
    p[atom] += args.surface * t
    configs["c3_surface"] = (p, {"kind": "one_surface_atom", "atom_index_1based": atom + 1,
                                 "towards_atom_1based": neighbour + 1, "direction": t.tolist(),
                                 "displacement_A": args.surface, "translation_removed": False})
    rng = np.random.default_rng(args.seed)
    dx = rng.normal(size=(13, 3))
    dx -= dx.mean(0)
    dx *= args.random / np.sqrt(np.mean(np.sum(dx**2, axis=1)))
    configs["c4_random"] = (pos0 + dx, {"kind": "random", "seed": args.seed, "generator": "numpy default_rng (PCG64) normal",
                                         "per_atom_rms_displacement_A": args.random, "translation_removed": True,
                                         "rotation_removed": False})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"set": args.source_label, "element": symbols[0], "n_atoms": 13, "units": "Angstrom",
                "role": "held-out test only; never train on these", "source": source,
                "central_atom_1based": c + 1, "configurations": []}
    for name, (p, how) in configs.items():
        f = out / f"{name}.xyz"
        write_xyz(f, symbols, p, f"{args.source_label} {name}")
        disp = np.linalg.norm(p - pos0, axis=1)
        manifest["configurations"].append({"name": name, "file": f.name, "sha256": sha256(f), **how,
                                           "rms_displacement_A": float(np.sqrt(np.mean(disp**2))),
                                           "max_displacement_A": float(disp.max()), **describe(p, cutoff)})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    for row in manifest["configurations"]:
        print(f"{row['name']:12s} rms {row['rms_displacement_A']:.4f} max {row['max_displacement_A']:.4f}  "
              f"{row['motif']}  mean nn {row['mean_nn_A']:.4f}")


if __name__ == "__main__":
    main()
