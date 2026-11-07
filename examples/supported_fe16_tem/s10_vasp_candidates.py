"""Step 10: pick supported structures for VASP interaction labels.

Takes, from one s2/s3 output directory,
  * the best landing of every basin on every support (supported_<support>.extxyz), and
  * every --md-every-th MD frame (md_<support>.extxyz), which samples the thermal
    distortions the Delta-model has to handle in MD,
and writes vasp_candidates.extxyz with the per-atom 'cluster' column, a unique
structure_id and the multiplicity, ready for

    cluster-mlip vasp-prepare <out>/vasp_candidates.extxyz -o vasp_fe16_supported

``--spin-neighbours`` adds M-2 and M+2 for every structure (fixed-moment single
points), so the interaction term also learns how the support shifts the spin ladder.

usage: s10_vasp_candidates.py [--out DIR] [--md-every 20] [--spin-neighbours]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write

HERE = Path(__file__).parent
parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--out", default=str(HERE / "out"), help="an s2/s3 output directory")
parser.add_argument("--supports", default="graphene,MgO100")
parser.add_argument("--md-every", type=int, default=20, help="0 = no MD frames")
parser.add_argument("--spin-neighbours", action="store_true")
args = parser.parse_args()
out = Path(args.out)

candidates = []
res_path = out / "results.json"
gas_info = json.loads(res_path.read_text())["gas"] if res_path.is_file() else {}


def multiplicity(a) -> int:
    """The frame's M; files written before s2 recorded M (the committed baseline, or a
    spin-blind run's M=1 placeholder) fall back to the basin's M / DFT M in results.json."""
    recorded = int(a.info.get("multiplicity", a.info.get("spin", 0)))
    if recorded > 1:
        return recorded
    iso = gas_info.get(a.info.get("isomer", ""), {})
    m = iso.get("M") or iso.get("dft_mult") or recorded
    if not m:
        raise SystemExit(f"no multiplicity for {a.info.get('isomer')}; pass --multiplicities to vasp-prepare")
    return int(m)


def add(atoms, sid: str, source: str) -> None:
    a = atoms.copy()
    a.calc = None
    if "cluster" not in a.arrays:  # baseline files predate the column; there the cluster is all Fe
        a.arrays["cluster"] = np.array([s == "Fe" for s in a.get_chemical_symbols()], dtype=int)
    m = multiplicity(a)
    keep = {k: a.info[k] for k in ("support", "isomer") if k in a.info}
    mults = [m - 2, m, m + 2] if args.spin_neighbours else [m]
    for mult in (x for x in mults if x >= 1):
        b = a.copy()
        b.info = dict(keep, structure_id=sid if mult == m else f"{sid}_m{mult}",
                      charge=0, multiplicity=mult, spin=mult, source=source)
        candidates.append(b)


for sname in args.supports.split(","):
    landed = out / f"supported_{sname}.extxyz"
    if landed.is_file():
        for a in read(landed, ":"):
            add(a, f"{sname}_{a.info['isomer']}_landed", landed.name)
    md = out / f"md_{sname}.extxyz"
    if args.md_every and md.is_file():
        frames = read(md, ":")
        for k in range(args.md_every - 1, len(frames), args.md_every):
            add(frames[k], f"{sname}_{frames[k].info.get('isomer', 'X')}_md{k:04d}", md.name)

if not candidates:
    raise SystemExit(f"no supported_*.extxyz / md_*.extxyz in {out}")
path = out / "vasp_candidates.extxyz"
write(path, candidates)
n_atoms = sorted({len(a) for a in candidates})
print(f"{len(candidates)} structures ({n_atoms} atoms) -> {path}")
print(f"next: cluster-mlip vasp-prepare {path} -o vasp_fe16_supported")
