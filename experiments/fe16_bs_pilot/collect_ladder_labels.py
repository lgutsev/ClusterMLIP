"""Collect the Fe16 ladder (package 91) and ladder-then-Opt (package 99) logs into labeled extxyz.

    PYTHONPATH=<ClusterMLIP src with parse_force_frames + local spins> \
        python collect_ladder_labels.py OUT_DIR PKG91_DIR PKG99_DIR

These packages have no campaign manifests, so `cluster-mlip collect` cannot read them. This script
calls the same parser (`gaussian.parse_force_frames`) on each log and keeps every force-bearing frame.
Frames come from two kinds of link:
- fixed-geometry Force links: package 91's vertical rungs at the M = 49 geometry;
- Opt steps: package 99's relaxations at M = 47, 45 and 41.

Each frame is screened independently of the parser: a frame is dropped when ">>>>>>>>>> Convergence
criterion not met" appears between its link start and its force table. IOP(5/13=1) lets Gaussian
continue past such SCFs, so a Normal termination proves nothing.

Duplicates are removed by (multiplicity, geometry rounded to 1e-6 Angstrom); the first source wins.
Package 99 repeats package 91's ladder links, and package 91's XQC frame repeats the standard frame
for M = 49 to 45.

Spin state:
- `hirshfeld_spins` (metadata) is attached when the same link printed a Hirshfeld table for that
  geometry. Force links and finished Opts print one; intermediate Opt steps do not.
- `spin_state` comes from the Hirshfeld table of the frame or of its link's start: `all_alpha`, or
  `central_beta` (Fe4 below -0.5).
- `spin_state_inferred=true` marks Opt steps whose state is taken from the link start.
- <S^2> stayed constant through every Opt (REVIEW_99.md), so no state switched.
- Mulliken columns are written as usual (provenance). The central Fe4 has the known Mulliken artifact,
  so use Hirshfeld for local moments.
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

FM_REFERENCE_EV = -20221.7600899 * 27.211386245988
_ROUTE_RE = re.compile(r"(?m)^ #p UBPW91")
_NOT_MET = ">>>>>>>>>> Convergence criterion not met"
_HIRSH_RE = re.compile(r"Hirshfeld charges, spin densities.*?\n(.*?)\n\s*Tot", re.S)

LOGS = [  # (package, frame, description), in priority order for de-duplication
    ("91", "frame_0000", "ladder 49->35, standard route"),
    ("91", "frame_0001", "ladder 49->43, XQC fallback"),
    ("99", "frame_0000", "ladder 49->47 then Opt M=47"),
    ("99", "frame_0001", "ladder 49->45 then Opt M=45"),
    ("99", "frame_0002", "ladder 49->41 then Opt M=41"),
]


def hirshfeld(block: str) -> list[float]:
    return [float(row.split()[3]) for row in block.strip().splitlines()[1:]]


def state_of(spins: list[float]) -> str:
    beta = [i + 1 for i, s in enumerate(spins) if s < -0.5]
    if not beta:
        return "all_alpha"
    if beta == [4]:
        return "central_beta"
    return "beta_" + "_".join(map(str, beta))


def collect(log: Path, package: str, frame_name: str) -> list:
    text = log.read_text(encoding="utf-8", errors="replace")
    links = [m.start() for m in _ROUTE_RE.finditer(text)] + [len(text)]
    headers = [m.start() for m in _FORCE_HEADER_RE.finditer(text)]
    frames = parse_force_frames(text, log)
    kept = []
    for frame in frames:
        pos = headers[frame.record.metadata["force_frame_index"]]
        link = max(i for i, start in enumerate(links[:-1]) if start < pos)
        body = text[links[link]:links[link + 1]]
        # Gaussian wraps the echoed route over several lines, up to the next dashed rule
        route = " ".join(line.strip() for line in re.split(r"(?m)^ -{10,}$", body, maxsplit=1)[0].splitlines())
        kind = "opt" if " Opt " in f" {route} " and "Stable=Opt" not in route else "force"
        # independent SCF screen: G09 prints the not-met banner on the line right before the
        # "SCF Done" it belongs to; check the SCF that produced this frame's energy
        prior = text[links[link]:pos]
        last_scf = prior.rfind("SCF Done:")
        if last_scf < 0 or _NOT_MET in prior[max(0, last_scf - 200):last_scf]:
            continue
        if frame.record.metadata.get("scf_convergence_warning"):
            continue
        blocks = _HIRSH_RE.findall(body)
        # a Force link prints its Hirshfeld table after the force table, for that geometry;
        # an Opt link prints one at its end, for the final geometry only
        own = None
        if kind == "force" and blocks:
            own = hirshfeld(blocks[-1])
        elif kind == "opt" and blocks and "Stationary point found" in body:
            steps = [m.start() for m in _FORCE_HEADER_RE.finditer(body)]
            if steps and links[link] + steps[-1] == pos:
                own = hirshfeld(blocks[-1])
        start_state = None
        earlier = _HIRSH_RE.findall(text[:links[link]])
        if earlier:
            start_state = state_of(hirshfeld(earlier[-1]))
        meta = frame.record.metadata
        meta.update({
            "package": package, "package_frame": frame_name, "link": link + 1, "link_kind": kind,
            "geometry_basis": "fe16-200-49-gf", "method": "UBPW91/6-311++G* G09 D.01",
        })
        if own is not None:
            meta["hirshfeld_spins"] = own
            meta["spin_state"], meta["spin_state_inferred"] = state_of(own), False
        else:
            meta["spin_state"], meta["spin_state_inferred"] = start_state, True
        if kind == "force":
            frame.record.config_type = "ladder_vertical" if frame.record.multiplicity != 49 else "minimum"
        else:
            final = own is not None
            frame.record.config_type = "minimum" if final else "ladder_opt"
        kept.append(frame)
    return kept


def main() -> None:
    out, pkg91, pkg99 = (Path(a) for a in sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    roots = {"91": pkg91, "99": pkg99}
    seen, frames, rows = set(), [], []
    for package, frame_name, desc in LOGS:
        log = roots[package] / "outputs" / f"{frame_name}.log"
        got = collect(log, package, frame_name)
        new = 0
        for frame in got:
            key = (frame.record.multiplicity, geometry_signature(frame.record.atoms))
            if key in seen:
                continue
            seen.add(key)
            frame.record.record_id = f"fe16-ladder-p{package}-{frame_name}-l{frame.record.metadata['link']:02d}-f{frame.record.metadata['force_frame_index']:03d}"
            frames.append(frame)
            new += 1
        rows.append([package, frame_name, desc, len(got), new])
    write_labeled_extxyz(frames, out / "fe16_ladder_frames.extxyz")
    with (out / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(["record_id", "package", "package_frame", "link", "link_kind", "config_type", "multiplicity",
                    "energy_ev", "e_minus_fm_ev", "max_force_ev_ang", "s2_before", "spin_state",
                    "spin_state_inferred", "hirshfeld_fe4"])
        for f in frames:
            m = f.record.metadata
            fmax = max((fx * fx + fy * fy + fz * fz) ** 0.5 for fx, fy, fz in f.forces_ev_ang)
            w.writerow([f.record.record_id, m["package"], m["package_frame"], m["link"], m["link_kind"],
                        f.record.config_type, f.record.multiplicity, f"{f.energy_ev:.6f}",
                        f"{f.energy_ev - FM_REFERENCE_EV:.4f}", f"{fmax:.4f}", m.get("s2_before", ""),
                        m["spin_state"], m["spin_state_inferred"],
                        f"{m['hirshfeld_spins'][3]:.2f}" if "hirshfeld_spins" in m else ""])
    with (out / "sources.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(["package", "frame", "description", "frames_passing_screen", "new_after_dedup"])
        w.writerows(rows)
    by = {}
    for f in frames:
        k = (f.record.multiplicity, f.record.metadata["spin_state"], f.record.config_type)
        by[k] = by.get(k, 0) + 1
    (out / "summary.json").write_text(json.dumps(
        {"frames": len(frames), "by_multiplicity_state_config": {f"M{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(by.items(), reverse=True)},
         "sources": rows}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(frames)} frames to {out}")
    for k, v in sorted(by.items(), reverse=True):
        print(f"  M={k[0]:>2} {k[1]:13} {k[2]:15} {v}")
    for r in rows:
        print("  source", r)


if __name__ == "__main__":
    main()
