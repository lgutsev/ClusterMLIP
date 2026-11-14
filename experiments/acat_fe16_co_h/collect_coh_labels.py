"""Collect the Fe16 + CO + 2H reference packages (desk 112, 116, 118, 124) into labeled extxyz.

    PYTHONPATH=src python collect_coh_labels.py OUT_DIR PKG112_DIR PKG116_DIR PKG118_DIR PKG124_DIR

These packages have no campaign manifests, so `cluster-mlip collect` cannot read them. As for the
bare-Fe16 ladder labels (packages 91 and 99), this calls the project parser
(`gaussian.parse_force_frames`) on every log and keeps each force-bearing frame that passes an
independent SCF screen.

Frame kinds (``config_type``):
- ``panel_fixed``            Force link at a search-model geometry (112, 116)
- ``opt_fe_frozen``          step of an optimization with the 16 Fe atoms frozen (118)
- ``stationary_fe_frozen``   last step of such an optimization that reported a stationary point
- ``opt_all_free``           step of an unconstrained optimization (124)
- ``stationary_all_free``    its last step, stationary point reported (no frequency check was run)

Forces are Gaussian's full Cartesian forces in every case, also on atoms that were frozen during
an optimization; ``frozen_atoms`` in the metadata records which atoms the optimizer held.

Screening, independent of the parser: a frame is dropped when ">>>>>>>>>> Convergence criterion
not met" belongs to the SCF that produced it. IOP(5/13=1) lets Gaussian continue past such SCFs,
so a Normal termination proves nothing. Duplicates are removed by (multiplicity, geometry rounded
to 1e-6 Angstrom); the first source wins (118 and 124 repeat earlier ladder links).

Spin: ``hirshfeld_spins`` is attached when the frame's own link printed a Hirshfeld table for that
geometry (Force links and finished optimizations). ``spin_state`` is ``all_alpha`` when no Fe has a
Hirshfeld spin below -0.5; for intermediate optimization steps it is taken from the link's start
and marked ``spin_state_inferred``.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from cluster_mlip.dataset import write_labeled_extxyz
from cluster_mlip.gaussian import _FORCE_HEADER_RE, parse_force_frames
from cluster_mlip.models import geometry_signature

HARTREE_EV = 27.211386245988
_ROUTE_RE = re.compile(r"(?m)^ #p UBPW91")
_NOT_MET = ">>>>>>>>>> Convergence criterion not met"
_HIRSH_RE = re.compile(r"Hirshfeld charges, spin densities.*?\n(.*?)\n\s*Tot", re.S)
PACKAGES = ["112", "116", "118", "124"]  # priority order for de-duplication


def hirshfeld(block: str) -> list[float]:
    return [float(row.split()[3]) for row in block.strip().splitlines()[1:]]


def state_of(spins: list[float], symbols: list[str]) -> str:
    beta = [i + 1 for i, (s, el) in enumerate(zip(spins, symbols)) if el == "Fe" and s < -0.5]
    return "all_alpha" if not beta else "beta_" + "_".join(map(str, beta))


def route_of(body: str) -> str:
    """The echoed route, re-joined: the log wraps it at 70 columns, also inside a keyword."""
    block = re.split(r"(?m)^ -{10,}$", body, maxsplit=1)[0]
    return "".join(line[1:] for line in block.splitlines())


def collect(log: Path, package: str, row: dict) -> list:
    text = log.read_text(encoding="utf-8", errors="replace")
    links = [m.start() for m in _ROUTE_RE.finditer(text)] + [len(text)]
    headers = [m.start() for m in _FORCE_HEADER_RE.finditer(text)]
    structure = row.get("draft_job") or row.get("name")
    frozen = row.get("frozen_atoms", "")
    kept = []
    for frame in parse_force_frames(text, log):
        pos = headers[frame.record.metadata["force_frame_index"]]
        link = max(i for i, start in enumerate(links[:-1]) if start < pos)
        body = text[links[link]:links[link + 1]]
        route = route_of(body)
        is_opt = (" Opt " in f" {route} " or " Opt=ModRedundant " in f" {route} ")
        prior = text[links[link]:pos]
        last_scf = prior.rfind("SCF Done:")
        if last_scf < 0 or _NOT_MET in prior[max(0, last_scf - 200):last_scf]:
            continue
        if frame.record.metadata.get("scf_convergence_warning"):
            continue
        symbols = [a.symbol for a in frame.record.atoms]
        blocks = _HIRSH_RE.findall(body)
        own = None
        steps = [m.start() for m in _FORCE_HEADER_RE.finditer(body)]
        is_last = bool(steps) and links[link] + steps[-1] == pos
        stationary = is_opt and is_last and "Stationary point found" in body
        if not is_opt and blocks:
            own = hirshfeld(blocks[-1])
        elif stationary and blocks:
            own = hirshfeld(blocks[-1])
        earlier = _HIRSH_RE.findall(text[:links[link]])
        start_state = state_of(hirshfeld(earlier[-1]), symbols) if earlier else None
        meta = frame.record.metadata
        meta.update({"package": package, "package_frame": row["frame"], "structure": structure, "link": link + 1,
                     "link_kind": "opt" if is_opt else "force", "frozen_atoms": frozen if is_opt else "",
                     "catalyst_basis": "fe16-200-49-gf", "method": "UBPW91/6-311++G* G09 D.01"})
        if own is not None and len(own) == len(symbols):
            meta["hirshfeld_spins"] = own
            meta["spin_state"], meta["spin_state_inferred"] = state_of(own, symbols), False
        else:
            meta["spin_state"], meta["spin_state_inferred"] = start_state, True
        if not is_opt:
            frame.record.config_type = "panel_fixed" if package in ("112", "116") else "ladder_at_relaxed_geometry"
        elif frozen:
            frame.record.config_type = "stationary_fe_frozen" if stationary else "opt_fe_frozen"
        else:
            frame.record.config_type = "stationary_all_free" if stationary else "opt_all_free"
        kept.append(frame)
    return kept


def main() -> None:
    out = Path(sys.argv[1])
    roots = dict(zip(PACKAGES, (Path(a) for a in sys.argv[2:6])))
    out.mkdir(parents=True, exist_ok=True)
    seen, frames, rows = set(), [], []
    for package in PACKAGES:
        with open(roots[package] / "inputs" / "frames.csv", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                log = roots[package] / "outputs" / f"{row['frame']}.log"
                got = collect(log, package, row)
                new = 0
                for frame in got:
                    key = (frame.record.multiplicity, geometry_signature(frame.record.atoms))
                    if key in seen:
                        continue
                    seen.add(key)
                    m = frame.record.metadata
                    frame.record.record_id = (f"fe16coh2-p{package}-{row['frame']}-l{m['link']:02d}"
                                              f"-f{m['force_frame_index']:03d}")
                    frames.append(frame)
                    new += 1
                rows.append([package, row["frame"], row.get("draft_job") or row.get("name"), len(got), new])
    write_labeled_extxyz(frames, out / "fe16_coh2_frames.extxyz")
    e_min = min(f.energy_ev for f in frames)
    with (out / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(["record_id", "package", "package_frame", "structure", "link", "link_kind", "config_type", "multiplicity",
                    "energy_ev", "e_minus_lowest_ev", "max_force_ev_ang", "max_fe_force_ev_ang", "max_adsorbate_force_ev_ang",
                    "frozen_atoms", "s2_before", "spin_state", "spin_state_inferred"])
        for f in frames:
            m = f.record.metadata
            norms = [(fx * fx + fy * fy + fz * fz) ** 0.5 for fx, fy, fz in f.forces_ev_ang]
            fe = [n for n, a in zip(norms, f.record.atoms) if a.symbol == "Fe"]
            ads = [n for n, a in zip(norms, f.record.atoms) if a.symbol != "Fe"]
            w.writerow([f.record.record_id, m["package"], m["package_frame"], m["structure"], m["link"], m["link_kind"],
                        f.record.config_type, f.record.multiplicity, f"{f.energy_ev:.6f}", f"{f.energy_ev - e_min:.4f}",
                        f"{max(norms):.4f}", f"{max(fe):.4f}", f"{max(ads):.4f}", m["frozen_atoms"], m.get("s2_before", ""),
                        m["spin_state"], m["spin_state_inferred"]])
    with (out / "sources.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(["package", "frame", "structure", "frames_passing_screen", "new_after_dedup"])
        w.writerows(rows)
    by: dict = {}
    for f in frames:
        k = (f.record.config_type, f.record.multiplicity)
        by[k] = by.get(k, 0) + 1
    summary = {"frames": len(frames), "structures": len({f.record.metadata["structure"] for f in frames}),
               "lowest_energy_ha": e_min / HARTREE_EV,
               "by_config_and_multiplicity": {f"{k[0]}|M{k[1]}": v for k, v in sorted(by.items())},
               "spin_states": sorted({str(f.record.metadata["spin_state"]) for f in frames})}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for r in rows:
        print("  source", r)


if __name__ == "__main__":
    main()
