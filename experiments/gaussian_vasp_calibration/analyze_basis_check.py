"""Which side of the package-62 force gap is off: analysis of the G09 basis-set check package.

Reads the basis-check results (tasks.csv, gaussian/outputs/<task>.log) and the package-62 results (its G09
PW91 logs for the control, its VASP prod_pw91 outputs for the comparison) and writes, into --output:

  links.csv     one row per (task, link): energy, SCF status, <S**2>, Fe Hirshfeld pattern, NBasis
  report.json   every metric and the verdict
  report.md     the same, readable

Rules (docs/gaussian-basis-check.md):

- SCF screen: IOP(5/13=1) lets an unconverged SCF run on to Normal termination, so a link is
  unconverged if it printed "Convergence criterion not met", its final Conv= is above --scf-conv-max, or
  it used all MaxCyc cycles. Unconverged links never enter a metric. Link 1 (UBPW91) is only a guess source.
- Same state: a larger-basis link counts only if its Fe Hirshfeld sign pattern equals the 6-311++G* link's
  and its <S**2> is within --s2-tol of it.
- Control: link 2 must reproduce package 62's G_PW91 energy (|dE| < 1 meV) and forces (max < 1 meV/A).
- Per state and basis B: force RMSE against VASP PW91, and the fraction of the package-62 gap that the basis
  change explains, alpha = <F_B - F_B0, F_V - F_B0> / |F_V - F_B0|^2 (1 = moves all the way to VASP).
- Verdict on the pure-Fe minima at the largest converged basis of each: median alpha >= 0.5 -> the
  6-311++G* labels carry basis error; <= 0.2 -> the Gaussian side is basis-converged, look at VASP
  (PAW Fe_pv frozen core / augmentation); between -> partial. Not converged if the TZVPP->QZVPP step moves
  forces by more than 25% of the gap.

    python experiments/gaussian_vasp_calibration/analyze_basis_check.py \
        --cluster-mlip-src .claude/worktrees/periodic-vasp-ingest/src \
        --calibration D:/MLIP_Work_Folder/hpc_smoke_tests/62_clustermlip_gaussian_vasp_calibration \
        D:/MLIP_Work_Folder/hpc_smoke_tests/<desk number>_clustermlip_gaussian_basis_check \
        -o D:/MLIP_Work_Folder/Cluster_MLIP/basis_check_<desk number>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_calibration as ac  # noqa: E402  (sibling: Gaussian/VASP parsers, pattern, lstsq, rms)

B0 = "6-311++G*"
AMU_EV_A2_TO_CM1 = 521.4708  # sqrt(eV / (A^2 amu)) in cm^-1
FE_MASS = 55.845
FE2_EXP = {"re_a": 2.02, "re_err_a": 0.02, "where": "Ne matrix, EXAFS",
           "ref": "Purdum, Montano, Shenoy, Morrison, Phys. Rev. B 25, 4412 (1982), doi:10.1103/PhysRevB.25.4412"}
PURE_FE_MINIMA = ("s000_Fe2_m7_minimum", "s024_Fe8_m25_minimum", "s028_Fe10_m29_minimum", "s032_Fe16_m51_minimum")
REL_PAIRS = (("s023_Fe8_m23_minimum", "s024_Fe8_m25_minimum", "Fe8 spin gap M23-M25"),
             ("s025_Fe8_m27_minimum", "s024_Fe8_m25_minimum", "Fe8 spin gap M27-M25"),
             ("s007_Fe2N2O2_m9_offeq", "s005_Fe2N2O2_m9_minimum", "Fe2N2O2 off-eq minus minimum"))
ALPHA_BASIS, ALPHA_NOT = 0.5, 0.2
LADDER_STEP_MAX = 0.25


def read_geometry(gjf: Path) -> tuple[list[str], list[list[float]]]:
    block = re.search(r"\n0 \d+\n(.*?)\n\n", gjf.read_text(encoding="utf-8").split("--Link1--")[0], re.S).group(1)
    rows = [line.split() for line in block.splitlines()]
    return [r[0] for r in rows], [[float(v) for v in r[1:4]] for r in rows]


def nbasis(log: Path) -> list[int]:
    text = log.read_text(encoding="utf-8", errors="replace")
    return [int(m) for m in re.findall(r"\n\s+(\d+) basis functions,", text)]


def diff_rms(fa, fb, symbols, only=None) -> float | None:
    d = [c for el, a, b in zip(symbols, fa, fb) if only is None or el == only for c in (p - q for p, q in zip(a, b))]
    return 1000 * math.sqrt(sum(c * c for c in d) / (len(d) / 3)) if d else None  # per-atom vector RMS, meV/A


def alpha(fb, f0, fv) -> float | None:
    gap = [q - p for a, b in zip(f0, fv) for p, q in zip(a, b)]
    move = [q - p for a, b in zip(f0, fb) for p, q in zip(a, b)]
    g2 = sum(g * g for g in gap)
    return sum(m * g for m, g in zip(move, gap)) / g2 if g2 else None


def radial(forces, positions) -> float:
    """Mean outward radial force (meV/A); negative = the cluster wants to shrink."""
    c = [sum(p[k] for p in positions) / len(positions) for k in range(3)]
    tot = 0.0
    for f, p in zip(forces, positions):
        r = math.dist(p, c)
        tot += sum(fi * (p[k] - c[k]) / r for k, fi in enumerate(f)) if r > 1e-6 else 0.0
    return 1000 * tot / len(forces)


def bond_derivative(forces, positions) -> tuple[float, float]:
    """Fe2: (r, dE/dr in eV/A) from the two atomic forces."""
    a, b = positions
    r = math.dist(a, b)
    u = [(x - y) / r for x, y in zip(a, b)]
    fa = sum(f * k for f, k in zip(forces[0], u))
    fb = sum(f * k for f, k in zip(forces[1], u))
    return r, -(fa - fb) / 2


def fit_re(points: list[tuple[float, float, float]]) -> dict | None:
    """points (r, dE/dr, E): quadratic fit of dE/dr about the mean r; re = its root, k = slope there."""
    if len(points) < 4:
        return None
    r0 = sum(p[0] for p in points) / len(points)
    c = ac.lstsq([[1.0, p[0] - r0, (p[0] - r0) ** 2] for p in points], [p[1] for p in points])
    x = -c[0] / c[1]
    for _ in range(20):  # Newton on the quadratic
        x -= (c[0] + c[1] * x + c[2] * x * x) / (c[1] + 2 * c[2] * x)
    k = c[1] + 2 * c[2] * x
    mu = FE_MASS / 2
    resid = max(abs(p[1] - (c[0] + c[1] * (p[0] - r0) + c[2] * (p[0] - r0) ** 2)) for p in points)
    return {"re_a": r0 + x, "k_ev_a2": k, "we_cm1": AMU_EV_A2_TO_CM1 * math.sqrt(k / mu) if k > 0 else None,
            "n_points": len(points), "fit_resid_max_ev_a": resid}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", help="the basis-check package results directory")
    ap.add_argument("--calibration", required=True, help="package-62 results directory")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--cluster-mlip-src", required=True, help="src/ of a checkout that has cluster_mlip/vasp_ingest.py")
    ap.add_argument("--max-cycles", type=int, default=200)
    ap.add_argument("--scf-conv-max", type=float, default=1e-6)
    ap.add_argument("--s2-tol", type=float, default=0.1, help="largest <S**2> change from the 6-311++G* link")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(args.cluster_mlip_src).resolve()))
    from cluster_mlip import vasp_ingest  # noqa: E402  (branch-only module)

    res, cal, out = Path(args.results), Path(args.calibration), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tasks = list(csv.DictReader((res / "tasks.csv").open(encoding="utf-8")))
    flat, per_task, missing = [], {}, []
    for t in tasks:
        log = res / "gaussian" / "outputs" / f"{t['task']}.log"
        symbols, positions = read_geometry(res / "gaussian" / "inputs" / f"{t['task']}.gjf")
        labels = t["links"].split(";")
        if not log.is_file():
            missing.append(t["task"]); continue
        text = log.read_text(encoding="utf-8", errors="replace")
        n_normal = text.count("Normal termination of Gaussian")
        links = ac.parse_gaussian_log(log, len(symbols), args.max_cycles, args.scf_conv_max)
        nb = nbasis(log)
        lv = {}
        for i, (label, link) in enumerate(zip(labels, links)):
            link["pattern"] = ac.pattern(symbols, link["hirshfeld"]) if link["hirshfeld"] else ""
            link["nbasis"] = nb[i] if i < len(nb) else None
            lv[label] = link
        base = lv.get(f"PW91/{B0}")
        for label, link in lv.items():
            same = True
            if label.startswith("PW91/") and label != f"PW91/{B0}" and base:
                same = link["pattern"] == base["pattern"] and abs((link["s2"] or 0) - (base["s2"] or 0)) <= args.s2_tol
            link["same_state"] = same
            link["usable"] = link["scf_converged"] and same and bool(link["forces"])
            flat.append({"task": t["task"], "state_id": t["state_id"], "link": label, "energy_ev": f"{link['energy_ev']:.6f}",
                         "scf_converged": link["scf_converged"], "scf_flags": link["scf_flags"], "cycles": link["cycles"],
                         "conv": link["conv"], "s2": link["s2"], "fe_pattern": link["pattern"], "nbasis": link["nbasis"],
                         "same_state": same, "usable": link["usable"]})
        per_task[t["task"]] = {"row": t, "symbols": symbols, "positions": positions, "links": lv,
                               "n_normal": n_normal, "n_links_expected": int(t["n_links"]),
                               "seconds": (res / "gaussian" / "outputs" / f"{t['task']}.seconds").read_text().strip()
                               if (res / "gaussian" / "outputs" / f"{t['task']}.seconds").is_file() else None}

    # --- control and per-state force comparison against package 62 --------------------------------
    control, states = {}, {}
    for name, d in per_task.items():
        row = d["row"]
        if not row["pkg62_frame"]:
            continue
        sid, syms = row["state_id"], d["symbols"]
        g62 = ac.parse_gaussian_log(cal / "gaussian" / "outputs" / f"{row['pkg62_frame']}.log", len(syms),
                                    args.max_cycles, args.scf_conv_max)[1]
        b0 = d["links"].get(f"PW91/{B0}")
        if b0:
            control[sid] = {"dE_mev": 1000 * (b0["energy_ev"] - g62["energy_ev"]),
                            "max_dF_mev_a": max(1000 * math.dist(a, b) for a, b in zip(b0["forces"], g62["forces"])),
                            "scf_converged": b0["scf_converged"]}
            control[sid]["ok"] = (abs(control[sid]["dE_mev"]) < 1 and control[sid]["max_dF_mev_a"] < 1
                                  and b0["scf_converged"])
        v = ac.read_vasp(cal, "prod_pw91", sid, vasp_ingest)
        if not v or "forces" not in v:
            continue
        v_pattern = ac.pattern(syms, v["moments"]) if v["moments"] else ""
        st = {"formula": row["formula"], "vasp_scf_converged": v["scf_converged"], "vasp_pattern": v_pattern,
              "vasp_energy_ev": v["energy_ev"], "radial_vasp_mev_a": radial(v["forces"], d["positions"]), "bases": {}}
        for label, link in d["links"].items():
            if not label.startswith("PW91/"):
                continue
            basis = label.split("/", 1)[1]
            e = {"usable": link["usable"], "energy_ev": link["energy_ev"], "nbasis": link["nbasis"],
                 "pattern_matches_vasp": link["pattern"] == v_pattern}
            if link["usable"] and b0 and b0["usable"] and v["scf_converged"]:
                e.update({"force_rmse_vs_vasp_mev_a": {el: diff_rms(link["forces"], v["forces"], syms, el)
                                                       for el in sorted(set(syms))},
                          "radial_mev_a": radial(link["forces"], d["positions"]),
                          "alpha": alpha(link["forces"], b0["forces"], v["forces"]),
                          "shift_from_b0_mev_a": diff_rms(link["forces"], b0["forces"], syms),
                          "dE_from_b0_mev_atom": 1000 * (link["energy_ev"] - b0["energy_ev"]) / len(syms)})
            st["bases"][basis] = e
        tz, qz = d["links"].get("PW91/def2-TZVPP"), d["links"].get("PW91/def2-QZVPP")
        if tz and qz and tz["usable"] and qz["usable"] and b0 and b0["usable"]:
            gap = diff_rms(v["forces"], b0["forces"], syms)
            st["ladder_step_tz_to_qz"] = diff_rms(qz["forces"], tz["forces"], syms) / gap if gap else None
        states[sid] = st

    # --- relative energies --------------------------------------------------------------------------
    rel = {}
    for a, b, what in REL_PAIRS:
        if a not in states or b not in states:
            continue
        entry = {"vasp_mev": 1000 * (states[a]["vasp_energy_ev"] - states[b]["vasp_energy_ev"])}
        for basis in states[b]["bases"]:
            ea, eb = states[a]["bases"].get(basis), states[b]["bases"].get(basis)
            if ea and eb and ea["usable"] and eb["usable"]:
                entry[basis + "_mev"] = 1000 * (ea["energy_ev"] - eb["energy_ev"])
        rel[what] = entry

    # --- Fe2 scan -----------------------------------------------------------------------------------
    fe2 = {}
    for d in per_task.values():
        if d["row"]["formula"] != "Fe2":
            continue
        for label, link in d["links"].items():
            if label.startswith("PW91/") and link["usable"]:
                r, g = bond_derivative(link["forces"], d["positions"])
                fe2.setdefault(label.split("/", 1)[1], []).append((r, g, link["energy_ev"]))
    fe2_fit = {basis: fit_re(sorted(p)) for basis, p in fe2.items()}
    v_fe2 = ac.read_vasp(cal, "prod_pw91", "s000_Fe2_m7_minimum", vasp_ingest)
    fe2_vasp = None
    if v_fe2 and "forces" in v_fe2 and "s000_Fe2_m7_minimum" in states:
        d0 = next(d for d in per_task.values() if d["row"]["state_id"] == "s000_Fe2_m7_minimum" and d["row"]["pkg62_frame"])
        r0, g_v = bond_derivative(v_fe2["forces"], d0["positions"])
        best = next((fe2_fit[b] for b in ("def2-QZVPP", "def2-TZVPP", B0) if fe2_fit.get(b)), None)
        if best:  # harmonic estimate with the Gaussian curvature: VASP has only the one point
            fe2_vasp = {"r0_a": r0, "dEdr_ev_a": g_v, "re_estimate_a": r0 - g_v / best["k_ev_a2"]}

    # --- verdict --------------------------------------------------------------------------------------
    alphas, steps = {}, []
    for sid in PURE_FE_MINIMA:
        st = states.get(sid)
        if not st:
            continue
        for basis in ("def2-QZVPP", "def2-TZVPP"):
            e = st["bases"].get(basis)
            if e and e.get("alpha") is not None:
                alphas[sid] = {"basis": basis, "alpha": e["alpha"]}
                break
        if st.get("ladder_step_tz_to_qz") is not None:
            steps.append(st["ladder_step_tz_to_qz"])
    failed_controls = [k for k, v in control.items() if not v["ok"]]
    med = statistics.median(a["alpha"] for a in alphas.values()) if alphas else None
    if failed_controls:
        verdict = ("control_failed", "link 2 did not reproduce package 62 for " + ", ".join(failed_controls))
    elif med is None:
        verdict = ("no_data", "no pure-Fe minimum has a usable larger-basis link")
    elif steps and max(steps) > LADDER_STEP_MAX:
        verdict = ("ladder_not_converged", f"TZVPP->QZVPP moves forces by {max(steps):.2f} of the gap")
    elif med >= ALPHA_BASIS:
        verdict = ("gaussian_basis", f"median alpha {med:.2f}: the 6-311++G* labels carry basis error")
    elif med <= ALPHA_NOT:
        verdict = ("not_basis", f"median alpha {med:.2f}: Gaussian is basis-converged; look at the VASP side "
                                "(PAW Fe_pv frozen core / augmentation)")
    else:
        verdict = ("partial", f"median alpha {med:.2f}: basis explains part of the gap")

    seconds = {k: d["seconds"] for k, d in per_task.items()}
    terminations = {k: f"{d['n_normal']}/{d['n_links_expected']}" for k, d in per_task.items()
                    if d["n_normal"] != d["n_links_expected"]}
    unconverged = [f"{r['task']} {r['link']} ({r['scf_flags']})" for r in flat if not r["scf_converged"]]
    state_changed = [f"{r['task']} {r['link']}" for r in flat if not r["same_state"]]
    report = {"results": str(res), "calibration": str(cal), "missing_logs": missing, "terminations_short": terminations,
              "scf_unconverged_links": unconverged, "state_changed_links": state_changed, "control": control,
              "states": states, "relative_energies": rel, "fe2_scan_fit": fe2_fit, "fe2_vasp_estimate": fe2_vasp,
              "fe2_experiment": FE2_EXP, "alpha_pure_fe": alphas, "alpha_median": med, "ladder_steps": steps,
              "verdict": {"outcome": verdict[0], "why": verdict[1]}, "seconds": seconds,
              "thresholds": {"scf_conv_max": args.scf_conv_max, "max_cycles": args.max_cycles, "s2_tol": args.s2_tol,
                             "alpha_basis": ALPHA_BASIS, "alpha_not": ALPHA_NOT, "ladder_step_max": LADDER_STEP_MAX}}
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    if flat:
        with (out / "links.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0]), lineterminator="\n")
            w.writeheader(); w.writerows(flat)
    (out / "report.md").write_text(render(report), encoding="utf-8", newline="\n")
    print(json.dumps({"verdict": report["verdict"], "unconverged": len(unconverged), "missing": missing}, indent=1))
    return 0


def _f(v, fmt="{:.0f}"):
    return "-" if v is None else fmt.format(v)


def render(r: dict) -> str:
    L = ["# G09 basis-set check on the package-62 force gap", "",
         f"Results: `{r['results']}`; package 62: `{r['calibration']}`.", "",
         f"**Verdict: {r['verdict']['outcome']}** — {r['verdict']['why']}.", "",
         "## Screens", "",
         f"- missing logs: {', '.join(r['missing_logs']) or 'none'}",
         f"- tasks with fewer Normal terminations than links: {r['terminations_short'] or 'none'}",
         f"- unconverged SCF links (never used): {'; '.join(r['scf_unconverged_links']) or 'none'}",
         f"- larger-basis links that changed state (never used): {'; '.join(r['state_changed_links']) or 'none'}", "",
         "## Control: link 2 vs package 62 G_PW91", "", "| state | dE (meV) | max dF (meV/A) | ok |", "|---|---:|---:|---|"]
    L += [f"| {k} | {v['dE_mev']:.3f} | {v['max_dF_mev_a']:.2f} | {v['ok']} |" for k, v in r["control"].items()]
    L += ["", "## Forces vs VASP PW91 (alpha = share of the package-62 gap the basis change explains)", "",
          "| state | basis | NBasis | Fe RMSE vs VASP (meV/A) | radial G / V (meV/A) | alpha | shift from 6-311++G* (meV/A) | dE (meV/atom) |",
          "|---|---|---:|---:|---|---:|---:|---:|"]
    for sid, st in r["states"].items():
        for basis, e in st["bases"].items():
            fe = (e.get("force_rmse_vs_vasp_mev_a") or {}).get("Fe")
            L.append(f"| {sid} | {basis} | {_f(e['nbasis'])} | {_f(fe)} | {_f(e.get('radial_mev_a'))} / "
                     f"{_f(st['radial_vasp_mev_a'])} | {_f(e.get('alpha'), '{:.2f}')} | {_f(e.get('shift_from_b0_mev_a'))} | "
                     f"{_f(e.get('dE_from_b0_mev_atom'), '{:.1f}')} |" + ("" if e["usable"] else " (not usable)"))
    L += ["", f"TZVPP->QZVPP step / gap: {', '.join(f'{s:.2f}' for s in r['ladder_steps']) or '-'}", "",
          "## Relative energies (meV)", ""]
    for what, e in r["relative_energies"].items():
        L.append(f"- {what}: " + ", ".join(f"{k.removesuffix('_mev')} {v:.1f}" for k, v in e.items()))
    L += ["", "## Fe2 bond length (M = 7)", "", "| basis | re (A) | k (eV/A^2) | we (cm^-1) | points |", "|---|---:|---:|---:|---:|"]
    for basis, fit in r["fe2_scan_fit"].items():
        if fit:
            L.append(f"| {basis} | {fit['re_a']:.4f} | {fit['k_ev_a2']:.2f} | {_f(fit['we_cm1'])} | {fit['n_points']} |")
    if r["fe2_vasp_estimate"]:
        L.append(f"| VASP PW91 (harmonic estimate from one point) | {r['fe2_vasp_estimate']['re_estimate_a']:.4f} | | | 1 |")
    x = r["fe2_experiment"]
    L += ["", f"Experiment: re = {x['re_a']:.2f} ± {x['re_err_a']:.2f} A ({x['where']}; {x['ref']}). A matrix value, "
               "not gas phase: an anchor at the ±0.02 A level only.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
