"""Review the frozen-Fe optimization package (ladder links, then one Opt=ModRedundant link).

    python review_opt_package.py PACKAGE_DIR OUTPUTS_DIR --out report.json [--xyz final.extxyz]

Per frame: Normal terminations, watchdog reason, and for the optimization link every step's
energy, SCF convergence flag and the four convergence tests; whether a stationary point was
found; how far each atom moved between the input and the last geometry (Fe atoms must not
move); <S**2>. Steps whose SCF did not converge are counted and never used as the result.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

HARTREE_EV = 27.211386245988
SYMBOL = {26: "Fe", 6: "C", 8: "O", 1: "H"}


def input_geometry(path: Path) -> tuple[list[str], np.ndarray]:
    sym, xyz = [], []
    for line in path.read_text(encoding="utf-8").split("--Link1--\n")[0].splitlines():
        p = line.split()
        if len(p) == 4 and p[0].isalpha():
            sym.append(p[0])
            xyz.append([float(x) for x in p[1:]])
    return sym, np.array(xyz)


def orientations(chunk: str) -> list[np.ndarray]:
    out = []
    for m in re.finditer(r"Input orientation:\s*\n -+\n.*\n.*\n -+\n((?:\s+\d+\s+\d+\s+\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\n)+)", chunk):
        out.append(np.array([[float(x) for x in line.split()[3:]] for line in m.group(1).strip().splitlines()]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("package", type=Path)
    ap.add_argument("outputs", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--xyz", type=Path)
    args = ap.parse_args()
    with open(args.package / "inputs" / "frames.csv", newline="", encoding="utf-8") as fh:
        frames = list(csv.DictReader(fh))
    report, finals = [], []
    for row in frames:
        sym, xyz0 = input_geometry(args.package / "inputs" / f"{row['frame']}.gjf")
        text = (args.outputs / f"{row['frame']}.log").read_text(encoding="utf-8", errors="replace")
        ends = [m.end() for m in re.finditer(r"^ (Normal|Error) termination.*$", text, flags=re.M)]
        last = text[ends[-1]:] if ends and len(ends) < int(row["links"]) else (text[ends[-2]:] if len(ends) > 1 else text)
        entry = {"frame": row["frame"], "draft_job": row["draft_job"], "target_multiplicity": int(row["target_multiplicity"]),
                 "links_expected": int(row["links"]), "normal_terminations": text.count(" Normal termination"),
                 "error_terminations": text.count(" Error termination"),
                 "watchdog": (args.outputs / f"{row['frame']}.watchdog").read_text(encoding="utf-8").strip() or None,
                 "hours": round(int((args.outputs / f"{row['frame']}.seconds").read_text()) / 3600, 2)}
        if "Opt=ModRedundant" in row["job_types"]:
            scf = [(float(e), int(n)) for e, n in re.findall(r"SCF Done:\s+E\(\S+\) =\s+(-?\d+\.\d+)\s+A\.U\. after\s+(\d+) cycles", last)]
            # an SCF flag belongs to the SCF Done line that follows it
            flags = [("Convergence criterion not met" in seg) for seg in re.split(r"SCF Done:", last)[:-1]]
            tests = re.findall(r"Maximum Force\s+(\S+)\s+\S+\s+(YES|NO)\s*\n\s*RMS\s+Force\s+(\S+)\s+\S+\s+(YES|NO)\s*\n"
                               r"\s*Maximum Displacement\s+(\S+)\s+\S+\s+(YES|NO)\s*\n\s*RMS\s+Displacement\s+(\S+)\s+\S+\s+(YES|NO)", last)
            geoms = orientations(last)
            s2 = re.findall(r"S\*\*2 before annihilation\s+(-?\d+\.\d+),\s+after\s+(-?\d+\.\d+)", last)
            move = np.linalg.norm(geoms[-1] - xyz0, axis=1) if geoms else np.zeros(len(sym))
            fe = np.array([s == "Fe" for s in sym])
            good = [i for i, bad in enumerate(flags) if not bad]
            entry.update({
                "opt_steps": len(tests), "scf_evaluations": len(scf), "scf_not_converged_steps": int(sum(flags)),
                "stationary_point_found": "Stationary point found" in last,
                "last_step_tests": None if not tests else {
                    "max_force": float(tests[-1][0]), "max_force_ok": tests[-1][1] == "YES", "rms_force_ok": tests[-1][3] == "YES",
                    "max_displacement": float(tests[-1][4]), "max_displacement_ok": tests[-1][5] == "YES",
                    "rms_displacement": float(tests[-1][6]), "rms_displacement_ok": tests[-1][7] == "YES"},
                "first_energy_ha": scf[0][0] if scf else None,
                "last_energy_ha": scf[-1][0] if scf else None, "last_scf_converged": bool(flags and not flags[-1]),
                "lowest_converged_energy_ha": min(scf[i][0] for i in good) if good else None,
                "relaxation_ev": (scf[-1][0] - scf[0][0]) * HARTREE_EV if scf else None,
                "energy_change_last_step_ev": (scf[-1][0] - scf[-2][0]) * HARTREE_EV if len(scf) > 1 else None,
                "s2_after_last": float(s2[-1][1]) if s2 else None,
                "max_fe_move_A": float(move[fe].max()), "adsorbate_move_A": [round(float(x), 3) for x in move[~fe]],
            })
            if geoms:
                finals.append((row, sym, geoms[-1], entry))
        else:
            chunks, start = [], 0
            for e in ends:
                chunks.append(text[start:e])
                start = e
            entry["links"] = [{"scf_not_converged": c.count("Convergence criterion not met"),
                               "energy_ha": float(re.findall(r"SCF Done:\s+E\(\S+\) =\s+(-?\d+\.\d+)", c)[-1]),
                               "cycles": int(re.findall(r"after\s+(\d+) cycles", c)[-1]),
                               "stable": "The wavefunction is stable" in c} for c in chunks]
        report.append(entry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.xyz:
        with open(args.xyz, "w", encoding="utf-8", newline="\n") as fh:
            for row, sym, xyz, entry in finals:
                fh.write(f"{len(sym)}\n")
                fh.write(f'Properties=species:S:1:pos:R:3 pbc="F F F" frame={row["frame"]} draft_job={row["draft_job"]} '
                         f'multiplicity={row["target_multiplicity"]} charge=0 energy_ha={entry["last_energy_ha"]} '
                         f'stationary_point={entry["stationary_point_found"]} level=UBPW91/6-311++G* fe_frozen=T\n')
                for s, p in zip(sym, xyz):
                    fh.write(f"{s:<2s} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}\n")
    for e in report:
        print(json.dumps(e))


if __name__ == "__main__":
    main()
