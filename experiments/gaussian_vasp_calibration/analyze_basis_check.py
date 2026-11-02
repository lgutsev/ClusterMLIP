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
import make_basis_check_package as mbc  # noqa: E402  (sibling: POSCAR reader)

B0 = "6-311++G*"
AMU_EV_A2_TO_CM1 = 521.4708  # sqrt(eV / (A^2 amu)) in cm^-1
FE_MASS = 55.845
FE2_EXP = {"re_a": 2.02, "re_err_a": 0.02, "where": "Ne matrix, EXAFS",
           "ref": "Purdum, Montano, Shenoy, Morrison, Phys. Rev. B 25, 4412 (1982), doi:10.1103/PhysRevB.25.4412"}
FE2_STATE, FE8_STATE = "s000_Fe2_m7_minimum", "s024_Fe8_m25_minimum"
PURE_FE_MINIMA = ("s000_Fe2_m7_minimum", "s024_Fe8_m25_minimum", "s028_Fe10_m29_minimum", "s032_Fe16_m51_minimum")
REL_PAIRS = (("s023_Fe8_m23_minimum", "s024_Fe8_m25_minimum", "Fe8 spin gap M23-M25"),
             ("s025_Fe8_m27_minimum", "s024_Fe8_m25_minimum", "Fe8 spin gap M27-M25"),
             ("s007_Fe2N2O2_m9_offeq", "s005_Fe2N2O2_m9_minimum", "Fe2N2O2 off-eq minus minimum"))
ALPHA_BASIS, ALPHA_NOT = 0.5, 0.2
LADDER_STEP_MAX = 0.25
# --- beyond def2-QZVPP (make_basis_check_package.py --ladder beyond-qz) ----------------------------
LADDER_STEPS = (("def2-TZVPP", "def2-QZVPP"), ("def2-QZVPP", "pcseg-4(-i)"), ("def2-QZVPP", "cc-pVQZ"), ("pcseg-3", "pcseg-4(-i)"),
                ("pcseg-4(-i)", "pcseg-4"), ("unc-def2-QZVPP", "pcseg-4(-i)"))
GEN_CONTROL = ("def2-QZVPP", "def2-QZVPP(gen)")  # the same basis built in and through Gen: must agree
G_LIMIT = ("pcseg-4", "pcseg-4(-i)")  # the Gaussian basis-limit stand-in, largest usable first
# Two independent families that must agree for the limit to count. Package 83's unc-def2-QZVPP failed G09's
# quadrature check, so the revision (r2) compares def2-QZVPP with pcseg-4(-i); cc-pVQZ is reported as a third.
FAMILIES = ("def2-QZVPP", "pcseg-4(-i)")
# Scalar relativity: the cc-pVQZ / cc-pVQZ-DK pair. G09 D.01 has no DKH2 gradients, so the DKH2 link is a single
# point and relativity is read from energies: the Fe2 re from an E(r) fit, the Fe8 radial force from a breathing
# stretch (scales 0.99/1.00/1.01). Both links of the pair are fitted the same way.
DKH = ("cc-pVQZ", "cc-pVQZ-DK+DKH2")
SP_SUFFIX = "+DKH2"  # energy-only links
ATTRIB_STATES = (FE2_STATE, FE8_STATE)
VASP_PV, VASP_SV = "pw91_pv", ("pw91_sv_encut650", "pw91_sv")  # Fe_sv: the ENCUT-650 run if usable
CONV_MAX = 0.10  # family disagreement or last ladder step, as a share of the package-62 gap
SHARE_MIN = 0.20  # a component counts when it explains at least this share of the gap
RESIDUAL_MAX = 0.20  # what neither side explains, as a share of the gap
SV_ENCUT_MAX = 0.05  # Fe_sv ENCUT 520 -> 650 shift, as a share of the gap


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


def _plus(fa, fb, sign=1.0):
    return [[p + sign * q for p, q in zip(a, b)] for a, b in zip(fa, fb)]


def force_shares(fs: dict, f_sv) -> dict | None:
    """Shares of one state's package-62 force gap (6-311++G* -> VASP Fe_pv), each a projection on that gap:
    basis = the Gaussian limit stand-in; relativity = the DKH2 shift at fixed basis; vasp_paw = how far Fe_sv
    moves VASP back toward G09; residual = |G limit (+DKH2) - VASP Fe_sv| / |gap|."""
    b, f0, fv, syms = fs["bases"], fs["bases"].get(B0), fs["vasp"], fs["symbols"]
    lim = next((x for x in G_LIMIT if x in b), None)
    if not (f0 and fv and lim):
        return None
    gap = diff_rms(fv, f0, syms)
    out = {"g_limit": lim, "gap_mev_a": gap, "basis": alpha(b[lim], f0, fv)}
    if all(x in b for x in FAMILIES):
        out["family_diff"] = diff_rms(b[FAMILIES[1]], b[FAMILIES[0]], syms) / gap
    if "pcseg-3" in b and "pcseg-4(-i)" in b:
        out["last_step"] = diff_rms(b["pcseg-4(-i)"], b["pcseg-3"], syms) / gap
    best = b[lim]
    if all(x in b for x in DKH):
        rel = _plus(b[DKH[1]], b[DKH[0]], -1.0)
        out["relativity"] = alpha(_plus(f0, rel), f0, fv)
        best = _plus(best, rel)
    if f_sv:
        out["vasp_paw"] = alpha(f_sv, fv, f0)
        out["residual"] = diff_rms(best, f_sv, syms) / gap
    return out


def fit_re_energy(points: list[tuple[float, float]]) -> dict | None:
    """points (r, E in eV): cubic fit of E(r) about the mean r; re = the minimum, k = the curvature there."""
    if len(points) < 5:
        return None
    r0 = sum(p[0] for p in points) / len(points)
    c = ac.lstsq([[1.0, p[0] - r0, (p[0] - r0) ** 2, (p[0] - r0) ** 3] for p in points], [p[1] for p in points])
    x = 0.0
    for _ in range(50):  # Newton on dE/dr = c1 + 2 c2 x + 3 c3 x^2
        x -= (c[1] + 2 * c[2] * x + 3 * c[3] * x * x) / (2 * c[2] + 6 * c[3] * x)
    k = 2 * c[2] + 6 * c[3] * x
    resid = max(abs(p[1] - sum(ci * (p[0] - r0) ** i for i, ci in enumerate(c))) for p in points)
    return {"re_a": r0 + x, "k_ev_a2": k, "we_cm1": AMU_EV_A2_TO_CM1 * math.sqrt(k / (FE_MASS / 2)) if k > 0 else None,
            "n_points": len(points), "fit_resid_max_ev": resid}


def breathing_derivative(forces, positions) -> float:
    """dE/ds (eV) for positions c + s (r - c) at s = 1, from the forces: -sum F.(r - c)."""
    c = [sum(p[k] for p in positions) / len(positions) for k in range(3)]
    return -sum(f[k] * (p[k] - c[k]) for f, p in zip(forces, positions) for k in range(3))


def breathing_shares(fs: dict, f_sv, f_sv_other, rel_dEds: float | None) -> dict | None:
    """Shares on the Fe8 breathing coordinate: gap = dE/ds(VASP Fe_pv) - dE/ds(6-311++G*). Relativity enters as
    d(E_DKH2 - E_NR)/ds from the breathing stretch, so this is the Fe8 measure that includes it."""
    b, pos = fs["bases"], fs["positions"]
    lim = next((x for x in G_LIMIT if x in b), None)
    if not (b.get(B0) and fs["vasp"] and lim):
        return None
    D = {k: breathing_derivative(f, pos) for k, f in b.items()}
    d_b0, d_v = D[B0], breathing_derivative(fs["vasp"], pos)
    gap = d_v - d_b0
    out = {"g_limit": lim, "gap_ev": gap, "basis": (D[lim] - d_b0) / gap}
    if all(x in D for x in FAMILIES):
        out["family_diff"] = abs(D[FAMILIES[1]] - D[FAMILIES[0]]) / abs(gap)
    if "pcseg-3" in D and "pcseg-4(-i)" in D:
        out["last_step"] = abs(D["pcseg-4(-i)"] - D["pcseg-3"]) / abs(gap)
    best = D[lim]
    if rel_dEds is not None:
        out["relativity"] = rel_dEds / gap
        best += rel_dEds
    if f_sv:
        d_sv = breathing_derivative(f_sv, pos)
        out["vasp_paw"] = (d_sv - d_v) / (d_b0 - d_v)
        out["residual"] = abs(best - d_sv) / abs(gap)
        if f_sv_other:
            out["sv_encut_shift"] = abs(breathing_derivative(f_sv_other, pos) - d_sv) / abs(gap)
    return out


def re_shares(g_fit: dict, v_fit: dict, e_fit: dict | None = None) -> dict | None:
    """The same shares on the Fe2 bond length: gap = re(6-311++G*) - re(VASP Fe_pv), both from scans. The
    relativity shift comes from e_fit (E(r) fits; the DKH2 links are energy-only)."""
    re_g = {k: f["re_a"] for k, f in g_fit.items() if f}
    e_fit = e_fit or {}
    re_v = {k: f["re_a"] for k, f in v_fit.items() if f}
    lim = next((x for x in G_LIMIT if x in re_g), None)
    if B0 not in re_g or VASP_PV not in re_v or not lim:
        return None
    gap = re_g[B0] - re_v[VASP_PV]
    out = {"g_limit": lim, "gap_a": gap, "basis": (re_g[B0] - re_g[lim]) / gap}
    if all(x in re_g for x in FAMILIES):
        out["family_diff"] = abs(re_g[FAMILIES[1]] - re_g[FAMILIES[0]]) / abs(gap)
    if "pcseg-3" in re_g and "pcseg-4(-i)" in re_g:
        out["last_step"] = abs(re_g["pcseg-4(-i)"] - re_g["pcseg-3"]) / abs(gap)
    best = re_g[lim]
    if all(e_fit.get(x) for x in DKH):
        shift = e_fit[DKH[1]]["re_a"] - e_fit[DKH[0]]["re_a"]
        out["relativity"], out["relativity_shift_a"] = -shift / gap, shift
        best += shift
    sv = next((x for x in VASP_SV if x in re_v), None)
    if sv:
        out["vasp_sv_variant"] = sv
        out["vasp_paw"] = (re_v[sv] - re_v[VASP_PV]) / gap
        out["residual"] = abs(best - re_v[sv]) / abs(gap)
    if all(x in re_v for x in VASP_SV):
        out["sv_encut_shift"] = abs(re_v[VASP_SV[0]] - re_v[VASP_SV[1]]) / abs(gap)
    return out


def decide_attribution(sh: dict | None) -> str:
    """docs/gaussian-basis-check.md, "Beyond def2-QZVPP": decision on one set of shares."""
    if sh is None:
        return "no_data"
    if sh.get("family_diff") is None or sh.get("last_step") is None:
        return "gaussian_ladder_incomplete"
    if max(sh["family_diff"], sh["last_step"]) > CONV_MAX:
        return "gaussian_ladder_not_converged"
    if sh.get("residual") is None:
        return "vasp_scan_missing"
    if sh.get("sv_encut_shift") is not None and sh["sv_encut_shift"] > SV_ENCUT_MAX:
        return "vasp_sv_not_converged"
    if sh["residual"] > RESIDUAL_MAX:
        return "unexplained_residual"
    named = [name for key, name in (("basis", "gaussian_basis"), ("relativity", "relativity"), ("vasp_paw", "vasp_paw"))
             if sh.get(key) is not None and sh[key] >= SHARE_MIN]
    return "+".join(named) or "within_noise"


def anchor_vs_previous(previous: str, per_task: dict, args) -> dict:
    """This package's built-in def2-QZVPP links against package 75's, geometry by geometry."""
    prev = Path(previous)
    rows = {(t["state_id"], t["fe2_stretch_a"]): t for t in csv.DictReader((prev / "tasks.csv").open(encoding="utf-8"))}
    out = {}
    for name, d in per_task.items():
        mine = d["links"].get("PW91/def2-QZVPP")
        t = rows.get((d["row"]["state_id"], d["row"]["fe2_stretch_a"]))
        if not (mine and mine["usable"] and t) or "PW91/def2-QZVPP" not in t["links"].split(";"):
            continue
        links = ac.parse_gaussian_log(prev / "gaussian" / "outputs" / f"{t['task']}.log", len(d["symbols"]),
                                      args.max_cycles, args.scf_conv_max)
        theirs = links[t["links"].split(";").index("PW91/def2-QZVPP")]
        out[name] = {"previous_task": t["task"], "dE_mev": 1000 * (mine["energy_ev"] - theirs["energy_ev"]),
                     "max_dF_mev_a": max(1000 * math.dist(a, b) for a, b in zip(mine["forces"], theirs["forces"]))}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", nargs="+", help="basis-check package results directories (inputs + outputs); several "
                    "are merged, e.g. package 83 and its revision")
    ap.add_argument("--calibration", required=True, help="package-62 results directory")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--cluster-mlip-src", required=True, help="src/ of a checkout that has cluster_mlip/vasp_ingest.py")
    ap.add_argument("--max-cycles", type=int, default=200)
    ap.add_argument("--scf-conv-max", type=float, default=1e-6)
    ap.add_argument("--s2-tol", type=float, default=0.1, help="largest <S**2> change from the 6-311++G* link")
    ap.add_argument("--previous", help="package-75 results: its def2-QZVPP links must equal this package's (beyond-qz)")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(args.cluster_mlip_src).resolve()))
    from cluster_mlip import vasp_ingest  # noqa: E402  (branch-only module)

    dirs, cal, out = [Path(r) for r in args.results], Path(args.calibration), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [{**t, "_res": res} for res in dirs for t in csv.DictReader((res / "tasks.csv").open(encoding="utf-8"))]
    flat, per_task, missing = [], {}, []
    for t in tasks:
        res = t["_res"]
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
            link["usable"] = link["scf_converged"] and same and (bool(link["forces"]) or label.endswith(SP_SUFFIX))
            flat.append({"task": t["task"], "state_id": t["state_id"], "link": label, "energy_ev": f"{link['energy_ev']:.6f}",
                         "scf_converged": link["scf_converged"], "scf_flags": link["scf_flags"], "cycles": link["cycles"],
                         "conv": link["conv"], "s2": link["s2"], "fe_pattern": link["pattern"], "nbasis": link["nbasis"],
                         "same_state": same, "usable": link["usable"]})
        per_task[t["task"]] = {"row": t, "symbols": symbols, "positions": positions, "links": lv,
                               "n_normal": n_normal, "n_links_expected": int(t["n_links"]),
                               "seconds": (res / "gaussian" / "outputs" / f"{t['task']}.seconds").read_text().strip()
                               if (res / "gaussian" / "outputs" / f"{t['task']}.seconds").is_file() else None}

    # --- control and per-state force comparison against package 62 --------------------------------
    # A state can span several tasks (beyond-qz splits Fe8 by basis): each task's link 2 is its own control,
    # and the state's bases are merged (the 6-311++G* entry from the first task).
    control, states, forces, gen_control = {}, {}, {}, {}
    for name, d in per_task.items():
        row = d["row"]
        if not row["pkg62_frame"]:
            continue
        sid, syms = row["state_id"], d["symbols"]
        g62 = ac.parse_gaussian_log(cal / "gaussian" / "outputs" / f"{row['pkg62_frame']}.log", len(syms),
                                    args.max_cycles, args.scf_conv_max)[1]
        b0 = d["links"].get(f"PW91/{B0}")
        if b0:
            control[name] = {"dE_mev": 1000 * (b0["energy_ev"] - g62["energy_ev"]),
                             "max_dF_mev_a": max(1000 * math.dist(a, b) for a, b in zip(b0["forces"], g62["forces"])),
                             "scf_converged": b0["scf_converged"]}
            control[name]["ok"] = (abs(control[name]["dE_mev"]) < 1 and control[name]["max_dF_mev_a"] < 1
                                   and b0["scf_converged"])
        v = ac.read_vasp(cal, "prod_pw91", sid, vasp_ingest)
        if not v or "forces" not in v:
            continue
        v_pattern = ac.pattern(syms, v["moments"]) if v["moments"] else ""
        st = states.setdefault(sid, {"formula": row["formula"], "vasp_scf_converged": v["scf_converged"],
                                     "vasp_pattern": v_pattern, "vasp_energy_ev": v["energy_ev"],
                                     "radial_vasp_mev_a": radial(v["forces"], d["positions"]), "bases": {}})
        fs = forces.setdefault(sid, {"symbols": syms, "positions": d["positions"], "vasp": v["forces"]
                                     if v["scf_converged"] else None, "bases": {}})
        for label, link in d["links"].items():
            if not label.startswith("PW91/"):
                continue
            basis = label.split("/", 1)[1]
            if basis in st["bases"]:
                continue
            e = {"usable": link["usable"], "energy_ev": link["energy_ev"], "nbasis": link["nbasis"],
                 "pattern_matches_vasp": link["pattern"] == v_pattern}
            if link["usable"] and link["forces"] and b0 and b0["usable"] and v["scf_converged"]:
                e.update({"force_rmse_vs_vasp_mev_a": {el: diff_rms(link["forces"], v["forces"], syms, el)
                                                       for el in sorted(set(syms))},
                          "radial_mev_a": radial(link["forces"], d["positions"]),
                          "alpha": alpha(link["forces"], b0["forces"], v["forces"]),
                          "shift_from_b0_mev_a": diff_rms(link["forces"], b0["forces"], syms),
                          "dE_from_b0_mev_atom": 1000 * (link["energy_ev"] - b0["energy_ev"]) / len(syms)})
                fs["bases"][basis] = link["forces"]
            st["bases"][basis] = e
    for d in per_task.values():  # Gen transcription control, on every geometry that has both links
        a, b = (d["links"].get("PW91/" + x) for x in GEN_CONTROL)
        if a and b and a["usable"] and b["usable"]:
            gen_control[d["row"]["task"]] = {
                "dE_mev": 1000 * (b["energy_ev"] - a["energy_ev"]), "nbasis": [a["nbasis"], b["nbasis"]],
                "max_dF_mev_a": max(1000 * math.dist(p, q) for p, q in zip(a["forces"], b["forces"]))}
            gen_control[d["row"]["task"]]["ok"] = (abs(gen_control[d["row"]["task"]]["dE_mev"]) < 0.1
                                                   and gen_control[d["row"]["task"]]["max_dF_mev_a"] < 0.1)
    for sid, st in states.items():  # ladder steps as shares of the package-62 gap
        fs = forces[sid]
        b0f = fs["bases"].get(B0)
        gap = diff_rms(fs["vasp"], b0f, fs["symbols"]) if b0f and fs["vasp"] else None
        st["ladder_steps"] = {f"{x}->{y}": diff_rms(fs["bases"][y], fs["bases"][x], fs["symbols"]) / gap
                              for x, y in LADDER_STEPS if gap and x in fs["bases"] and y in fs["bases"]}
        if "def2-TZVPP->def2-QZVPP" in st["ladder_steps"]:
            st["ladder_step_tz_to_qz"] = st["ladder_steps"]["def2-TZVPP->def2-QZVPP"]

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
    fe2, fe2_e, breath = {}, {}, {}
    for d in per_task.values():
        for label, link in d["links"].items():
            if not (label.startswith("PW91/") and link["usable"]):
                continue
            basis = label.split("/", 1)[1]
            if d["row"]["formula"] == "Fe2":
                r = math.dist(*d["positions"])
                fe2_e.setdefault(basis, {})[round(r, 5)] = link["energy_ev"]  # one energy per bond length
                if link["forces"]:
                    fe2.setdefault(basis, []).append((*bond_derivative(link["forces"], d["positions"]), link["energy_ev"]))
            elif d["row"].get("breathing_scale"):
                breath.setdefault(basis, {})[float(d["row"]["breathing_scale"])] = link["energy_ev"]
    fe2_fit = {basis: fit_re(sorted(p)) for basis, p in fe2.items()}
    fe2_efit = {basis: fit_re_energy(sorted(p.items())) for basis, p in fe2_e.items()}
    rel_dEds = None  # Fe8: d(E_DKH2 - E_NR)/ds by central difference over the breathing stretch
    if all(x in breath and {0.99, 1.01} <= set(breath[x]) for x in DKH):
        d_rel = {sc: breath[DKH[1]][sc] - breath[DKH[0]][sc] for sc in (0.99, 1.01)}
        rel_dEds = (d_rel[1.01] - d_rel[0.99]) / 0.02
    v_fe2 = ac.read_vasp(cal, "prod_pw91", "s000_Fe2_m7_minimum", vasp_ingest)
    fe2_vasp = None
    if v_fe2 and "forces" in v_fe2 and "s000_Fe2_m7_minimum" in states:
        d0 = next(d for d in per_task.values() if d["row"]["state_id"] == "s000_Fe2_m7_minimum" and d["row"]["pkg62_frame"])
        r0, g_v = bond_derivative(v_fe2["forces"], d0["positions"])
        best = next((fe2_fit[b] for b in (*G_LIMIT, "def2-QZVPP", "def2-TZVPP", B0) if fe2_fit.get(b)), None)
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

    # --- beyond def2-QZVPP: VASP Fe2 scan and attribution -----------------------------------------
    vasp_jobs, vasp_points, vasp_fe8, vasp_control = {}, {}, {}, None
    jobs = [(res, j) for res in dirs if (res / "vasp" / "jobs.csv").is_file()
            for j in csv.DictReader((res / "vasp" / "jobs.csv").open(encoding="utf-8"))]
    for res, j in jobs:
        v = ac.read_vasp(res, j["variant"], j["point"], vasp_ingest)
        usable = bool(v and "forces" in v and v["scf_converged"])
        vasp_jobs[j["job"]] = {"present": v is not None, "error": (v or {}).get("error"), "usable": usable,
                               "scf_steps": (v or {}).get("scf_steps"), "total_moment": (v or {}).get("total_moment")}
        if not usable:
            continue
        if j["state_id"] != FE2_STATE:
            vasp_fe8.setdefault(j["state_id"], {})[j["variant"]] = v["forces"]
            continue
        _, _, pos = mbc.read_poscar(res / "vasp" / "inputs" / j["variant"] / j["point"] / "POSCAR")
        r, g = bond_derivative(v["forces"], pos)
        vasp_points.setdefault(j["variant"], []).append((r, g, v["energy_ev"]))
        if j["point"] == FE2_STATE:
            vasp_fe8.setdefault(FE2_STATE, {})[j["variant"]] = v["forces"]
        if j["variant"] == VASP_PV and j["point"] == FE2_STATE:  # replays package 62's prod_pw91 Fe2
            v62 = ac.read_vasp(cal, "prod_pw91", FE2_STATE, vasp_ingest)
            vasp_control = {"dE_mev": 1000 * (v["energy_ev"] - v62["energy_ev"]),
                            "max_dF_mev_a": max(1000 * math.dist(a, b) for a, b in zip(v["forces"], v62["forces"]))}
            vasp_control["ok"] = abs(vasp_control["dE_mev"]) < 1 and vasp_control["max_dF_mev_a"] < 1
    vasp_fit = {variant: fit_re(sorted(p)) for variant, p in vasp_points.items()}
    fe2_shares = re_shares(fe2_fit, vasp_fit, fe2_efit)
    force_attr = {}
    for sid in ATTRIB_STATES:
        if sid not in forces:
            continue
        sv = vasp_fe8.get(sid, {})  # the Fe2 scan's r0 point is package 62's geometry
        f_sv = next((sv[x] for x in VASP_SV if x in sv), None)
        sh = force_shares(forces[sid], f_sv)
        if sh and all(x in sv for x in VASP_SV):
            sh["sv_encut_shift"] = diff_rms(sv[VASP_SV[0]], sv[VASP_SV[1]], forces[sid]["symbols"]) / sh["gap_mev_a"]
        force_attr[sid] = sh
    breath_attr = None
    if FE8_STATE in forces:
        sv = vasp_fe8.get(FE8_STATE, {})
        f_sv = next((sv[x] for x in VASP_SV if x in sv), None)
        other = next((sv[x] for x in VASP_SV if x in sv and sv[x] is not f_sv), None)
        breath_attr = breathing_shares(forces[FE8_STATE], f_sv, other, rel_dEds)
    attribution = None
    if any(b in st["bases"] for st in states.values() for b in G_LIMIT) or vasp_jobs:
        per = {"Fe2 r_e": fe2_shares, "Fe8 breathing": breath_attr,
               **{f"{sid} forces": sh for sid, sh in force_attr.items()}}
        outcomes = {k: decide_attribution(sh) for k, sh in per.items()}
        # Fe2 re and the Fe8 breathing coordinate are the measures that can carry relativity (energy-only DKH2)
        deciding = [outcomes[k] for k in ("Fe2 r_e", "Fe8 breathing") if k in outcomes]
        overall = deciding[0] if deciding and len(set(deciding)) == 1 else "state_dependent" if deciding else "no_data"
        if vasp_control and not vasp_control["ok"]:
            overall = "vasp_control_failed"
        if gen_control and not all(g["ok"] for g in gen_control.values()):
            overall = "gen_control_failed"
        if failed_controls:
            overall = "control_failed"
        attribution = {"overall": overall, "outcomes": outcomes, "shares": per, "vasp_control": vasp_control,
                       "vasp_jobs": vasp_jobs, "vasp_scan_fit": vasp_fit, "gen_control": gen_control,
                       "fe2_energy_fit": fe2_efit, "fe8_breathing_energies_ev": {k: {f"{sc:.2f}": e for sc, e in v.items()}
                                                                                 for k, v in breath.items()},
                       "fe8_relativity_dEds_ev": rel_dEds,
                       "anchor_vs_previous": anchor_vs_previous(args.previous, per_task, args) if args.previous else None}

    seconds = {k: d["seconds"] for k, d in per_task.items()}
    terminations = {k: f"{d['n_normal']}/{d['n_links_expected']}" for k, d in per_task.items()
                    if d["n_normal"] != d["n_links_expected"]}
    unconverged = [f"{r['task']} {r['link']} ({r['scf_flags']})" for r in flat if not r["scf_converged"]]
    state_changed = [f"{r['task']} {r['link']}" for r in flat if not r["same_state"]]
    report = {"results": [str(d) for d in dirs], "calibration": str(cal), "missing_logs": missing, "terminations_short": terminations,
              "scf_unconverged_links": unconverged, "state_changed_links": state_changed, "control": control,
              "states": states, "relative_energies": rel, "fe2_scan_fit": fe2_fit, "fe2_vasp_estimate": fe2_vasp,
              "fe2_experiment": FE2_EXP, "alpha_pure_fe": alphas, "alpha_median": med, "ladder_steps": steps,
              "verdict": {"outcome": verdict[0], "why": verdict[1]}, "attribution": attribution, "seconds": seconds,
              "thresholds": {"scf_conv_max": args.scf_conv_max, "max_cycles": args.max_cycles, "s2_tol": args.s2_tol,
                             "alpha_basis": ALPHA_BASIS, "alpha_not": ALPHA_NOT, "ladder_step_max": LADDER_STEP_MAX,
                             "conv_max": CONV_MAX, "share_min": SHARE_MIN, "residual_max": RESIDUAL_MAX,
                             "sv_encut_max": SV_ENCUT_MAX}}
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    if flat:
        with (out / "links.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0]), lineterminator="\n")
            w.writeheader(); w.writerows(flat)
    (out / "report.md").write_text(render(report), encoding="utf-8", newline="\n")
    print(json.dumps({"verdict": report["verdict"], "attribution": (attribution or {}).get("overall"), "unconverged": len(unconverged), "missing": missing}, indent=1))
    return 0


def _f(v, fmt="{:.0f}"):
    return "-" if v is None else fmt.format(v)


def render_attribution(a: dict | None) -> list[str]:
    if not a:
        return []
    L = [f"## Attribution beyond def2-QZVPP: **{a['overall']}**", "",
         "Shares of the package-62 gap (6-311++G* -> VASP Fe_pv), each projected on that gap. basis = the Gaussian "
         "limit stand-in; relativity = cc-pVQZ-DK+DKH2 minus cc-pVQZ (energy-only, so only on Fe2 r_e and the Fe8 "
         "breathing coordinate); vasp_paw = how far Fe_sv moves VASP back toward G09 (negative = further away); "
         "residual = |G limit (+DKH2) - VASP Fe_sv| / |gap|; family_diff = |def2-QZVPP - pcseg-4(-i)|; "
         "last_step = |pcseg-3 -> pcseg-4(-i)|. The overall outcome needs Fe2 r_e and Fe8 breathing to agree.", "",
         "| measure | outcome | G limit | gap | basis | relativity | vasp_paw | residual | family_diff | last_step | Fe_sv ENCUT shift |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, sh in a["shares"].items():
        if not sh:
            L.append(f"| {k} | {a['outcomes'][k]} | | | | | | | | | |")
            continue
        gap = (f"{sh['gap_a']:.4f} A" if "gap_a" in sh else f"{sh['gap_ev']:.3f} eV (dE/ds)" if "gap_ev" in sh
               else f"{sh['gap_mev_a']:.0f} meV/A")
        L.append(f"| {k} | {a['outcomes'][k]} | {sh['g_limit']} | {gap} | " + " | ".join(
            _f(sh.get(x), "{:.2f}") for x in ("basis", "relativity", "vasp_paw", "residual", "family_diff",
                                              "last_step", "sv_encut_shift")) + " |")
    L += ["", "VASP Fe2 M7 scans (fit of dE/dr from the forces, 7 points):", "",
          "| variant | re (A) | k (eV/A^2) | we (cm^-1) | points |", "|---|---:|---:|---:|---:|"]
    for v, fit in a["vasp_scan_fit"].items():
        if fit:
            L.append(f"| {v} | {fit['re_a']:.4f} | {fit['k_ev_a2']:.2f} | {_f(fit['we_cm1'])} | {fit['n_points']} |")
    L += ["", "Fe2 M7 from E(r) fits (cubic; the DKH2 links are energy-only):", "",
          "| basis | re (A) | k (eV/A^2) | points |", "|---|---:|---:|---:|"]
    for b, fit in (a.get("fe2_energy_fit") or {}).items():
        if fit:
            L.append(f"| {b} | {fit['re_a']:.4f} | {fit['k_ev_a2']:.2f} | {fit['n_points']} |")
    if a.get("fe8_relativity_dEds_ev") is not None:
        L += ["", f"Fe8 breathing: d(E_DKH2 - E_NR)/ds = {a['fe8_relativity_dEds_ev']:.4f} eV "
                  "(cc-pVQZ-DK+DKH2 minus cc-pVQZ, scales 0.99 and 1.01)."]
    vc = a["vasp_control"]
    L += ["", f"- VASP control (pw91_pv r0 vs package 62 prod_pw91): "
              + (f"dE {vc['dE_mev']:.3f} meV, max dF {vc['max_dF_mev_a']:.2f} meV/A, ok {vc['ok']}" if vc else "missing"),
          f"- Gen transcription control (def2-QZVPP built-in vs Gen): "
          + ("; ".join(f"{k} dE {g['dE_mev']:.4f} meV, max dF {g['max_dF_mev_a']:.3f}, NBasis {g['nbasis']}, ok {g['ok']}"
                       for k, g in a["gen_control"].items()) or "missing"),
          f"- def2-QZVPP vs package 75: " + ("; ".join(f"{k} dE {g['dE_mev']:.3f} meV, max dF {g['max_dF_mev_a']:.2f}"
                                                    for k, g in (a["anchor_vs_previous"] or {}).items()) or "not checked"),
          f"- VASP jobs not usable: " + (", ".join(k for k, j in a["vasp_jobs"].items() if not j["usable"]) or "none"), ""]
    return L


def render(r: dict) -> str:
    L = ["# G09 basis-set check on the package-62 force gap", "",
         f"Results: {', '.join(f'`{d}`' for d in r['results'])}; package 62: `{r['calibration']}`.", "",
         f"**Verdict: {r['verdict']['outcome']}** — {r['verdict']['why']}.", "",
         *render_attribution(r.get("attribution")),
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
