"""Collect the returned TM13 reference package into labels and spin/SCF diagnostics.

    python experiments/tm13_benchmark/collect_dft.py --package RETURNED_PACKAGE_DIR --set experiments/tm13_benchmark/sets/fe13_legacy_ubpw91_m45 --out experiments/tm13_benchmark/report

Per Gaussian link: SCF energy and cycle count, "Convergence criterion not met" count, <S**2> before/after
annihilation, stability outcome, Hirshfeld spin populations, normal termination; Force links also give forces.

A state (configuration, M, route to it = ladder or direct) is usable when both of its links terminated
normally with a converged SCF, the Stable=Opt link ended on a stable wavefunction, and the Force link kept
the Stable=Opt link's solution (|dE| < 1e-6 Ha and |d<S**2>| < 1e-3).

Outputs: dft_links.csv (every link, failed ones included), dft_states.csv (every state, with the energy
relative to the lowest usable state at the same geometry, and the force difference between ladder and direct
solutions of the same M), dft_labels.extxyz (usable states; branch="selected" marks, per configuration and
M, the lowest-energy usable solution; "alternate" the other). Energies in eV, forces in eV/A.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

HARTREE = 27.211386245988
BOHR = 0.529177210903


def parse_links(text: str) -> list[dict]:
    chunks = re.split(r"\n Entering Link 1 = ", text)[1:]
    links = []
    for chunk in chunks:
        d: dict = {"normal": "Normal termination of Gaussian" in chunk,
                   "not_converged": chunk.count("Convergence criterion not met")}
        m = re.search(r"\n -+\n (#p.*?)\n -+\n", chunk, re.S)
        route = re.sub(r"\n ", "", m.group(1)) if m else ""
        d["job"] = "Stable=Opt" if "Stable=Opt" in route else "Force" if "Force" in route else "other"
        m = re.search(r"Multiplicity =\s*(\d+)", chunk)
        d["multiplicity"] = int(m.group(1)) if m else None
        scf = re.findall(r"SCF Done:\s+E\(\S+\)\s*=\s*(-?\d+\.\d+)\s+A\.U\. after\s+(\d+) cycles", chunk)
        d["energy_Ha"] = float(scf[-1][0]) if scf else None
        d["scf_cycles"] = sum(int(c) for _, c in scf)
        d["n_scf"] = len(scf)
        s2 = re.findall(r"S\*\*2 before annihilation\s+(-?\d+\.\d+),\s+after\s+(-?\d+\.\d+)", chunk)
        d["s2_before"], d["s2_after"] = (float(s2[-1][0]), float(s2[-1][1])) if s2 else (None, None)
        if "The wavefunction is stable under the perturbations considered" in chunk:
            d["stability"] = "stable_after_reopt" if "has an internal instability" in chunk or "has an RHF -> UHF" in chunk else "stable"
        elif "instability" in chunk:
            d["stability"] = "unstable"
        else:
            d["stability"] = ""
        blocks = chunk.split("Hirshfeld charges, spin densities, dipoles, and CM5 charges")
        spins = []
        if len(blocks) > 1:
            for line in blocks[-1].splitlines()[2:]:
                p = line.split()
                if len(p) >= 4 and p[0].isdigit():
                    spins.append(float(p[3]))
                elif spins:
                    break
        d["hirshfeld_spins"] = spins
        f = re.search(r"Forces \(Hartrees/Bohr\)\n.*?\n -+\n(.*?)\n -+\n", chunk, re.S)
        d["forces"] = (np.array([[float(x) for x in line.split()[2:5]] for line in f.group(1).splitlines()]) * HARTREE / BOHR
                       if f else None)
        links.append(d)
    return links


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from ase.io import read, write

    pkg, setdir, out = Path(args.package), Path(args.set), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = list(csv.DictReader(open(pkg / "inputs" / "frames.csv", encoding="utf-8")))
    link_rows, states = [], []
    for fr in frames:
        log = pkg / "outputs" / f"{fr['frame']}.log"
        expected = [(int(m), job) for m in fr["multiplicities"].split() for job in ("Stable=Opt", "Force")]
        links = parse_links(log.read_text(errors="replace")) if log.exists() else []
        watchdog = (pkg / "outputs" / f"{fr['frame']}.watchdog")
        reason = watchdog.read_text().strip() if watchdog.exists() else ""
        for i, (m, job) in enumerate(expected):
            d = links[i] if i < len(links) else None
            link_rows.append({"frame": fr["frame"], "config": fr["config"], "task": fr["task"], "link": i, "multiplicity": m,
                              "job": job, "status": "missing" if d is None else "ok" if d["normal"] and not d["not_converged"] else "failed",
                              "normal_termination": "" if d is None else d["normal"],
                              "scf_not_converged": "" if d is None else d["not_converged"],
                              "energy_Ha": "" if d is None or d["energy_Ha"] is None else f"{d['energy_Ha']:.8f}",
                              "scf_cycles": "" if d is None else d["scf_cycles"],
                              "s2_before": "" if d is None or d["s2_before"] is None else d["s2_before"],
                              "s2_after": "" if d is None or d["s2_after"] is None else d["s2_after"],
                              "stability": "" if d is None else d["stability"],
                              "hirshfeld_spins": "" if d is None else json.dumps(d["hirshfeld_spins"]),
                              "watchdog": reason})
        for k, m in enumerate(int(x) for x in fr["multiplicities"].split()):
            st, fo = (links[2 * k] if 2 * k < len(links) else None), (links[2 * k + 1] if 2 * k + 1 < len(links) else None)
            why = []
            if st is None or fo is None:
                why.append("link missing")
            else:
                for name, d in (("stable link", st), ("force link", fo)):
                    if not d["normal"]:
                        why.append(f"{name}: no normal termination")
                    if d["not_converged"]:
                        why.append(f"{name}: SCF not converged")
                if st["multiplicity"] != m or fo["multiplicity"] != m:
                    why.append("multiplicity mismatch")
                if not st["stability"].startswith("stable"):
                    why.append("stability not confirmed")
                if fo["forces"] is None:
                    why.append("no forces")
                if None in (st["energy_Ha"], fo["energy_Ha"], st["s2_before"], fo["s2_before"]):
                    why.append("energy or S2 missing")
                elif abs(st["energy_Ha"] - fo["energy_Ha"]) > 1e-6 or abs(st["s2_before"] - fo["s2_before"]) > 1e-3:
                    why.append("force link left the stable link's solution")
            states.append({"config": fr["config"], "multiplicity": m, "route": fr["task"], "frame": fr["frame"],
                           "usable": not why, "why_not": "; ".join(why), "st": st, "fo": fo})

    by_geom: dict[tuple[str, int], list[dict]] = {}
    for s in states:
        if s["usable"]:
            by_geom.setdefault((s["config"], s["multiplicity"]), []).append(s)
    for group in by_geom.values():
        best = min(group, key=lambda s: s["fo"]["energy_Ha"])
        for s in group:
            s["branch"] = "selected" if s is best else "alternate"
            s["dE_vs_selected_eV"] = (s["fo"]["energy_Ha"] - best["fo"]["energy_Ha"]) * HARTREE
            s["force_rms_diff_vs_selected_eV_A"] = float(np.sqrt(np.mean((s["fo"]["forces"] - best["fo"]["forces"]) ** 2)))
    lowest = {}
    for (config, m), group in by_geom.items():
        e = min(s["fo"]["energy_Ha"] for s in group)
        lowest[config] = min(lowest.get(config, np.inf), e)

    state_rows, labels = [], []
    for s in states:
        fo = s["fo"]
        row = {"config": s["config"], "multiplicity": s["multiplicity"], "route": s["route"], "frame": s["frame"],
               "usable": s["usable"], "why_not": s["why_not"], "branch": s.get("branch", ""),
               "energy_Ha": "" if not fo or fo["energy_Ha"] is None else f"{fo['energy_Ha']:.8f}",
               "E_rel_lowest_M_same_geometry_eV": "" if not s["usable"] else f"{(fo['energy_Ha'] - lowest[s['config']]) * HARTREE:.5f}",
               "dE_vs_selected_same_M_eV": "" if not s["usable"] else f"{s['dE_vs_selected_eV']:.6f}",
               "force_rms_diff_vs_selected_same_M_eV_A": "" if not s["usable"] else f"{s['force_rms_diff_vs_selected_eV_A']:.5f}",
               "s2_before": "" if not fo else fo["s2_before"], "s2_expected": (s["multiplicity"] ** 2 - 1) / 4,
               "stability": "" if not s["st"] else s["st"]["stability"],
               "force_rms_eV_A": "" if not fo or fo["forces"] is None else f"{float(np.sqrt(np.mean(fo['forces'] ** 2))):.5f}",
               "hirshfeld_spin_sum": "" if not fo or not fo["hirshfeld_spins"] else f"{sum(fo['hirshfeld_spins']):.3f}",
               "hirshfeld_spins": "" if not fo else json.dumps(fo["hirshfeld_spins"])}
        state_rows.append(row)
        if s["usable"]:
            a = read(setdir / f"{s['config']}.xyz")
            a.info = {"config": s["config"], "multiplicity": s["multiplicity"], "spin": s["multiplicity"], "charge": 0,
                      "route": s["route"], "branch": s["branch"], "usable": True, "REF_energy": fo["energy_Ha"] * HARTREE,
                      "s2_before": fo["s2_before"], "level": "UBPW91/6-311++G*", "config_type": "tm13_heldout_test"}
            a.arrays["REF_forces"] = fo["forces"]
            if len(fo["hirshfeld_spins"]) == len(a):
                a.arrays["hirshfeld_spin"] = np.array(fo["hirshfeld_spins"])
            labels.append(a)

    for name, data in (("dft_links.csv", link_rows), ("dft_states.csv", state_rows)):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    if labels:
        write(out / "dft_labels.extxyz", labels)
    print(f"links {len(link_rows)} ({sum(r['status'] == 'ok' for r in link_rows)} ok), states {len(states)} "
          f"({len(labels)} usable)")


if __name__ == "__main__":
    main()
