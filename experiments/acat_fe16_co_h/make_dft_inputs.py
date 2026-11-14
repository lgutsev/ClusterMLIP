"""Draft Gaussian inputs for the DFT shortlist. PREPARATION ONLY: nothing is staged or submitted.

    python make_dft_inputs.py SHORTLIST.extxyz --out DRAFT_DIR [--multiplicities 49 47 45]

Each structure gets one multi-link input, the flip-down ladder that worked for bare
Fe16 (packages 91 rev 2 and 99): at fixed geometry, for each multiplicity from high
to low, a ``Stable=Opt`` link followed by a ``Force`` link, all on one checkpoint with
``Guess=Read`` after the first link. Rules carried over from those packages:

* ``IOP(5/13=1,5/36=1,8/11=1)`` stays in every link; unconverged SCFs are screened
  from the logs afterwards ("Convergence criterion not met").
* one job-type keyword per link (never ``SP`` together with ``Stable=Opt``).
* atom order = catalyst atoms 0..N-1 exactly as in the catalyst file, then adsorbates.

The draft directory is not a dispatch package. A submission goes through the desk
with its own package generator, numbering and hand-off note.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from ase.io import read

ROUTE_HEAD = "#p UBPW91/6-311++G* SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm"
ROUTE_TAIL = "Pop=Hirshfeld IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine"
JOB_TYPES = ("SP", "Stable=Opt", "Force", "Opt", "Freq")


def route(job: str, read_checkpoint: bool) -> str:
    text = f"{ROUTE_HEAD} {job} {ROUTE_TAIL}" + (" Geom=Checkpoint Guess=Read" if read_checkpoint else "")
    kinds = [k for k in JOB_TYPES if f" {k} " in f" {text} "]
    if len(kinds) != 1:
        raise ValueError(f"a link must have exactly one job type, got {kinds}: {text}")
    if "IOP(5/13=1" not in text:
        raise ValueError("IOP(5/13=1) must stay in every link")
    return text


def gaussian_input(name: str, atoms, charge: int, multiplicities: list[int], mem: str, cores: int) -> str:
    links = []
    first = True
    for mult in multiplicities:
        for job in ("Stable=Opt", "Force"):
            head = [f"%chk={name}.chk", f"%mem={mem}", f"%nprocshared={cores}", route(job, not first), "",
                    f"{name} M={mult} {job}", "", f"{charge} {mult}"]
            if first:
                head += [f"{s:<2s} {x:16.10f} {y:16.10f} {z:16.10f}"
                         for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions)]
            links.append("\n".join(head) + "\n\n")
            first = False
    return "--Link1--\n".join(links)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shortlist", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--multiplicities", nargs="+", type=int, default=[49, 47, 45])
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--mem", default="24GB")
    ap.add_argument("--cores", type=int, default=12)
    args = ap.parse_args()
    if "loni_package" in {p.lower() for p in args.out.resolve().parts}:
        raise SystemExit("refusing to write into loni_package: drafts are not dispatch packages")
    frames = read(args.shortlist, index=":")
    (args.out / "inputs").mkdir(parents=True, exist_ok=True)
    rows = []
    for k, atoms in enumerate(frames):
        electrons = int(atoms.numbers.sum()) - args.charge
        for mult in args.multiplicities:
            if (electrons - (mult - 1)) % 2:
                raise SystemExit(f"multiplicity {mult} is impossible for {electrons} electrons")
        role = atoms.arrays["role"]
        n_cat = int((role != 2).sum())
        if not (role[:n_cat] != 2).all() or not (role[n_cat:] == 2).all():
            raise SystemExit("catalyst atoms must come first")
        name = f"fe16coh2_{k:03d}"
        text = gaussian_input(name, atoms, args.charge, args.multiplicities, args.mem, args.cores)
        path = args.out / "inputs" / f"{name}.gjf"
        with open(path, "w", encoding="ascii", newline="\n") as fh:
            fh.write(text)
        rows.append({"job": name, "shortlist_id": atoms.info["shortlist_id"], "reason": atoms.info.get("reason", ""),
                     "motif": atoms.info.get("motif", ""), "formula": atoms.get_chemical_formula(),
                     "charge": args.charge, "multiplicities": " ".join(map(str, args.multiplicities)),
                     "links": 2 * len(args.multiplicities), "catalyst_atoms": f"1-{n_cat}",
                     "adsorbate_atoms": " ".join(f"{i + 1}:{atoms.symbols[i]}:mol{atoms.arrays['mol_id'][i]}"
                                                 for i in range(n_cat, len(atoms))),
                     "input_sha256": hashlib.sha256(text.encode()).hexdigest()})
    with open(args.out / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out / "README_DRAFT_NOT_STAGED.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# Draft Gaussian inputs: Fe16 + CO + 2H reference panel\n\n"
                 f"**Draft only. Not a dispatch package, not staged, not submitted.**\n\n"
                 f"{len(rows)} inputs, {2 * len(args.multiplicities)} links each "
                 f"(M = {', '.join(map(str, args.multiplicities))}: Stable=Opt then Force, fixed geometry).\n"
                 f"Source: `{args.shortlist.name}`. Atom numbering in `frames.csv` is Gaussian's (1-based).\n")
    print(f"wrote {len(rows)} draft inputs to {args.out}")


if __name__ == "__main__":
    main()
