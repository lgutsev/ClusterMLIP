"""Review a returned reference package (fixed-geometry ladder: Stable=Opt then Force per M).

    python review_reference_batch.py PACKAGE_DIR OUTPUTS_DIR --models mace:PATH [mace:PATH ...] --out report.json

Per frame and multiplicity: link count, Normal/Error terminations, SCF convergence flags
("Convergence criterion not met"), whether the Force link reproduces the Stable=Opt energy,
stability verdict, <S**2>, UBPW91 energy and forces. Then, on the same geometries, each
model's energies relative to frame 0 and its force errors against UBPW91, next to the
error of predicting zero force. Models here take no spin input, so they are compared with
every multiplicity separately.

An energy or force from a link whose own SCF did not converge is flagged and not used.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import make_calculator  # noqa: E402

HARTREE_EV = 27.211386245988
BOHR_A = 0.529177210903
ROUTE_KINDS = ("Stable=Opt", "Force")


def read_input(path: Path) -> tuple[Atoms, list[int]]:
    links = path.read_text(encoding="utf-8").split("--Link1--\n")
    mults = []
    for link in links:
        m = re.search(r"^(-?\d+) (\d+)$", link, flags=re.M)
        if m is None:
            raise ValueError(f"{path}: no charge/multiplicity line")
        mults.append(int(m.group(2)))
    symbols, positions = [], []
    for line in links[0].splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].isalpha():
            symbols.append(parts[0])
            positions.append([float(x) for x in parts[1:]])
    return Atoms(symbols, positions=positions), mults


def split_links(text: str) -> list[str]:
    """One chunk per link, each ending at its termination line."""
    chunks, start = [], 0
    for m in re.finditer(r"^ (Normal|Error) termination.*$", text, flags=re.M):
        chunks.append(text[start:m.end()])
        start = m.end()
    return chunks


def parse_link(chunk: str, natoms: int) -> dict:
    # the log wraps the route at 70 columns, also inside a keyword: join the lines without a separator
    block = re.search(r"^ (#.*\n(?: .*\n)*?) -{5,}", chunk, flags=re.M)
    route = "".join(line[1:] for line in block.group(1).splitlines()) if block else ""
    kind = next((k for k in ROUTE_KINDS if f" {k} " in f" {route} "), "?")
    scf = re.findall(r"SCF Done:\s+E\(\S+\) =\s+(-?\d+\.\d+)\s+A\.U\. after\s+(\d+) cycles", chunk)
    s2 = re.findall(r"S\*\*2 before annihilation\s+(-?\d+\.\d+),\s+after\s+(-?\d+\.\d+)", chunk)
    out = {"kind": kind, "normal": " Normal termination" in chunk, "n_scf": len(scf),
           "scf_not_converged": chunk.count("Convergence criterion not met"),
           "energy_ha": float(scf[-1][0]) if scf else None, "scf_cycles": int(scf[-1][1]) if scf else None,
           "s2_after": float(s2[-1][1]) if s2 else None,
           "stable": ("The wavefunction is stable" in chunk) if kind == "Stable=Opt" else None,
           "instability_found": "wavefunction has an" in chunk}
    m = re.search(r"Forces \(Hartrees/Bohr\)\n.*\n -+\n((?:\s+\d+\s+\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\n)+)", chunk)
    if m:
        rows = [line.split() for line in m.group(1).strip().splitlines()]
        forces = np.array([[float(x) for x in r[2:]] for r in rows]) * HARTREE_EV / BOHR_A
        if len(forces) != natoms:
            raise ValueError("force block has the wrong number of atoms")
        out["forces"] = forces
    return out


def rmse(a: np.ndarray) -> float:
    return float(np.sqrt((a ** 2).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("package", type=Path)
    ap.add_argument("outputs", type=Path)
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with open(args.package / "inputs" / "frames.csv", newline="", encoding="utf-8") as fh:
        frames = list(csv.DictReader(fh))

    report: dict = {"package": args.package.name, "frames": [], "models": {}}
    data = []
    for row in frames:
        atoms, mults = read_input(args.package / "inputs" / f"{row['frame']}.gjf")
        text = (args.outputs / f"{row['frame']}.log").read_text(encoding="utf-8", errors="replace")
        links = [parse_link(c, len(atoms)) for c in split_links(text)]
        watchdog = (args.outputs / f"{row['frame']}.watchdog").read_text(encoding="utf-8").strip()
        entry = {"frame": row["frame"], "draft_job": row["draft_job"], "why": row["why"], "motif": row["motif"],
                 "links": len(links), "normal_terminations": sum(l["normal"] for l in links),
                 "watchdog": watchdog or None, "seconds": int((args.outputs / f"{row['frame']}.seconds").read_text()),
                 "states": {}}
        usable = {}
        for i in range(0, len(links) - 1, 2):
            stab, force = links[i], links[i + 1]
            mult = mults[i]
            if (stab["kind"], force["kind"]) != ROUTE_KINDS or mults[i + 1] != mult:
                raise ValueError(f"{row['frame']}: unexpected link order at link {i + 1}")
            ok = force["normal"] and force["scf_not_converged"] == 0 and "forces" in force
            state = {"stable_opt": {k: stab[k] for k in ("normal", "scf_not_converged", "scf_cycles", "energy_ha", "stable",
                                                         "instability_found")},
                     "force": {k: force[k] for k in ("normal", "scf_not_converged", "scf_cycles", "energy_ha", "s2_after")},
                     "force_link_reproduces_energy_ha": abs(stab["energy_ha"] - force["energy_ha"]),
                     "s2_expected": (mult - 1) / 2 * ((mult - 1) / 2 + 1),
                     "usable": bool(ok and stab["stable"] and not stab["instability_found"]),
                     "note": "Stable=Opt SCF hit the cycle cap; energy and forces are taken from the converged Force link"
                     if stab["scf_not_converged"] and ok else None}
            if state["usable"]:
                state["energy_ev"] = force["energy_ha"] * HARTREE_EV
                f = force["forces"]
                sym = np.array(atoms.get_chemical_symbols())
                state["force_rms_ev_a"] = {"Fe": rmse(f[sym == "Fe"]), "adsorbate": rmse(f[sym != "Fe"]),
                                           "max_norm": float(np.linalg.norm(f, axis=1).max())}
                usable[mult] = force
            entry["states"][str(mult)] = state
        if usable:
            lowest = min(usable, key=lambda m: usable[m]["energy_ha"])
            entry["lowest_multiplicity"] = lowest
            entry["spin_gaps_ev"] = {str(m): (usable[m]["energy_ha"] - usable[lowest]["energy_ha"]) * HARTREE_EV for m in usable}
        report["frames"].append(entry)
        data.append((atoms, usable))

    mults_all = sorted({m for _, u in data for m in u}, reverse=True)
    ref = {m: [u[m]["energy_ha"] * HARTREE_EV if m in u else None for _, u in data] for m in mults_all}
    report["ubpw91_relative_to_frame0_ev"] = {
        str(m): [None if e is None or ref[m][0] is None else e - ref[m][0] for e in ref[m]] for m in mults_all}
    lowest_e = [min(u[m]["energy_ha"] for m in u) * HARTREE_EV if u else None for _, u in data]
    report["ubpw91_lowest_state_relative_to_frame0_ev"] = [None if e is None else e - lowest_e[0] for e in lowest_e]

    for spec in args.models:
        calc = make_calculator(spec)
        energies, forces = [], []
        for atoms, _ in data:
            a = atoms.copy()
            a.calc = calc
            energies.append(float(a.get_potential_energy()))
            forces.append(np.array(a.get_forces()))
        entry = {"relative_to_frame0_ev": [e - energies[0] for e in energies], "forces_vs_ubpw91": {}}
        for m in mults_all:
            parts: dict[str, list] = {"Fe": [], "adsorbate": []}
            zero: dict[str, list] = {"Fe": [], "adsorbate": []}
            for (atoms, u), f in zip(data, forces):
                if m not in u:
                    continue
                sym = np.array(atoms.get_chemical_symbols())
                for key, mask in (("Fe", sym == "Fe"), ("adsorbate", sym != "Fe")):
                    parts[key].append((f - u[m]["forces"])[mask])
                    zero[key].append(u[m]["forces"][mask])
            entry["forces_vs_ubpw91"][str(m)] = {
                key: {"model_rmse_ev_a": rmse(np.concatenate(parts[key])), "zero_baseline_rmse_ev_a": rmse(np.concatenate(zero[key])),
                      "frames": len(parts[key])} for key in parts if parts[key]}
        report["models"][Path(spec.split(":", 1)[-1]).name] = entry
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=1))
    for fr in report["frames"]:
        print(fr["frame"], fr["normal_terminations"], "/", fr["links"], "lowest M", fr.get("lowest_multiplicity"),
              {m: (s["usable"], s["stable_opt"]["scf_not_converged"], round(s["force_link_reproduces_energy_ha"], 9))
               for m, s in fr["states"].items()}, fr.get("spin_gaps_ev"))


if __name__ == "__main__":
    main()
