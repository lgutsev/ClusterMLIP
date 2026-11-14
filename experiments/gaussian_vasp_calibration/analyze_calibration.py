"""Metrics and the head decision for a finished Gaussian-vs-VASP calibration package.

Reads the package results directory (gaussian/outputs/frame_NNNN.log, vasp/outputs/<variant>/<state>/,
vasp/inputs/<variant>/<state>/job.json, states.csv) and writes, next to --output:

  states.csv     one row per (state, level): energy, SCF status, spins, matched or not, why
  report.json    every metric, per pair, plus the decision
  report.md      the same, readable

The rules are the ones in docs/gaussian-vasp-calibration.md. IOP(5/13=1) lets an unconverged SCF
run on to a Normal termination, so every Gaussian link is screened here: a link is unconverged if it
printed "Convergence criterion not met", or its final Conv= is above --scf-conv-max, or it used all
MaxCyc cycles. A VASP run is unconverged if its last SCF used NELM steps. Unconverged levels are
reported and never enter a metric.

    python experiments/gaussian_vasp_calibration/analyze_calibration.py \
        --cluster-mlip-src .claude/worktrees/periodic-vasp-ingest/src \
        D:/MLIP_Work_Folder/hpc_smoke_tests/62_clustermlip_gaussian_vasp_calibration \
        -o D:/MLIP_Work_Folder/Cluster_MLIP/gv_calibration_62
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

HARTREE_EV = 27.211386245988
HA_BOHR_TO_EV_A = 51.42208619083232
ELEMENTS = ("Fe", "N", "O")
G_LEVELS = ("G_BPW91", "G_PW91", "G_PBE")  # link order in every frame
V_LEVELS = {"V_PW91": "prod_pw91", "V_PBE": "prod_pbe"}
PAIRS = {  # name: (reference X, other Y); delta = E_Y - E_X
    "code_pw91": ("G_PW91", "V_PW91"), "code_pbe": ("G_PBE", "V_PBE"),
    "functional_pw91": ("G_BPW91", "G_PW91"), "functional_pbe": ("G_BPW91", "G_PBE"),
    "training_pw91": ("G_BPW91", "V_PW91"), "training_pbe": ("G_BPW91", "V_PBE"),
}
SPIN_SIGN_MIN = 0.5  # |moment| below this has no sign for the pattern
LARGE_FE = 8  # informational subset: clusters with at least this many Fe


# --- Gaussian ----------------------------------------------------------------------------------
def parse_gaussian_log(path: Path, n_atoms: int, max_cycles: int, conv_max: float) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    links = [s for s in re.split(r"\n Normal termination of Gaussian[^\n]*", text)]
    out = []
    for seg in links:
        if "SCF Done:" not in seg:
            continue
        done = re.findall(r"SCF Done:\s+E\((\S+)\)\s+=\s+(-?\d+\.\d+)\s+A\.U\. after\s+(\d+) cycles\s*\n\s*NFock=\s*\d+\s+Conv=(\S+)", seg)
        functional, energy, cycles, conv = done[-1]
        conv_value = float(conv.replace("D", "E"))
        not_met = "Convergence criterion not met" in seg or "Convergence failure" in seg
        reasons = [r for r, bad in (("criterion_not_met", not_met), (f"conv>{conv_max:g}", conv_value > conv_max),
                                     (f"cycles>={max_cycles}", int(cycles) >= max_cycles)) if bad]
        s2 = re.findall(r"S\*\*2 before annihilation\s+(\S+),\s+after\s+(\S+)", seg)
        out.append({
            "functional": functional, "energy_ev": float(energy) * HARTREE_EV, "cycles": int(cycles),
            "conv": conv_value, "scf_converged": not reasons, "scf_flags": ";".join(reasons),
            "s2": float(s2[-1][0]) if s2 else None,
            "forces": _table(seg, r"Forces \(Hartrees/Bohr\)\s*\n[^\n]*\n\s*-+\n", n_atoms, (2, 3, 4), HA_BOHR_TO_EV_A),
            "hirshfeld": [r[0] for r in _table(seg, r"Hirshfeld charges, spin densities, dipoles, and CM5 charges[^\n]*\n[^\n]*\n", n_atoms, (3,), 1.0)],
            "mulliken": [r[0] for r in _table(seg, r"Mulliken charges and spin densities:\s*\n[^\n]*\n", n_atoms, (3,), 1.0)],
        })
    return out


def _table(seg: str, header: str, n: int, cols: tuple[int, ...], scale: float) -> list[list[float]]:
    found = list(re.finditer(header, seg))
    if not found:
        return []
    lines = seg[found[-1].end():].splitlines()[:n]
    return [[float(line.split()[c]) * scale for c in cols] for line in lines]


# --- VASP --------------------------------------------------------------------------------------
def read_vasp(results: Path, variant: str, state: str, vasp_ingest) -> dict | None:
    job_dir = results / "vasp" / "outputs" / variant / state
    if not (job_dir / "vasprun.xml").is_file():
        return None
    try:
        run = vasp_ingest.read_vasp_job(job_dir, f"{variant}/{state}")
    except Exception as exc:  # an aborted run (probe without Libxc) has a vasprun.xml with no step
        return {"error": f"{type(exc).__name__}: {exc}"}
    if not run.steps:
        return {"error": "no ionic step"}
    order = json.loads((results / "vasp" / "inputs" / variant / state / "job.json").read_text(encoding="utf-8"))["order"]
    step = run.steps[-1]
    nelm = int(run.params.get("NELM", 60))

    def scatter(values):
        out = [None] * len(order)
        for k, i in enumerate(order):
            out[i] = values[k]
        return out

    return {
        "energy_ev": step.energy_free, "energy_sigma0_ev": step.energy_sigma0,
        "scf_steps": step.n_scsteps, "scf_converged": step.n_scsteps < nelm,
        "scf_flags": "" if step.n_scsteps < nelm else f"nelm={nelm}",
        "forces": scatter([list(f) for f in step.forces]),
        "moments": scatter(run.atomic_moments[-1]) if run.atomic_moments else [],
        "total_moment": run.total_moments[-1] if run.total_moments else None,
    }


# --- metrics -----------------------------------------------------------------------------------
def counts(formula: str) -> dict[str, int]:
    out = {el: 0 for el in ELEMENTS}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def pattern(symbols: list[str], moments: list[float]) -> str:
    return "".join("+" if m > SPIN_SIGN_MIN else "-" if m < -SPIN_SIGN_MIN else "0"
                   for el, m in zip(symbols, moments) if el == "Fe")


def lstsq(rows: list[list[float]], y: list[float]) -> list[float]:
    """Normal equations; columns with no data get 0 (tiny, well-conditioned systems only)."""
    k = len(rows[0])
    used = [j for j in range(k) if any(r[j] for r in rows)]
    a = [[sum(r[i] * r[j] for r in rows) for j in used] for i in used]
    b = [sum(r[i] * v for r, v in zip(rows, y)) for i in used]
    n = len(used)
    for c in range(n):  # Gauss-Jordan with partial pivoting
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        a[c], a[p], b[c], b[p] = a[p], a[c], b[p], b[c]
        for r in range(n):
            if r != c and a[c][c]:
                f = a[r][c] / a[c][c]
                a[r] = [x - f * z for x, z in zip(a[r], a[c])]
                b[r] -= f * b[c]
    sol = [0.0] * k
    for idx, j in enumerate(used):
        sol[j] = b[idx] / a[idx][idx]
    return sol


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else None


def mae(xs):
    return sum(abs(x) for x in xs) / len(xs) if xs else None


def pair_metrics(states: list[dict], x: str, y: str) -> dict:
    usable, excluded = [], defaultdict(list)
    for s in states:
        a, b = s["levels"].get(x), s["levels"].get(y)
        if not a or not b or "energy_ev" not in a or "energy_ev" not in b:
            excluded["missing"].append(s["state_id"]); continue
        if not a["scf_converged"] or not b["scf_converged"]:
            excluded["scf_unconverged"].append(s["state_id"]); continue
        if a["pattern"] != b["pattern"]:
            excluded["spin_pattern_mismatch"].append(s["state_id"]); continue
        usable.append(s)
    n_compared = len(usable) + len(excluded["spin_pattern_mismatch"])
    out: dict = {"n_matched": len(usable), "excluded": {k: v for k, v in excluded.items()},
                 "pattern_match_fraction": len(usable) / n_compared if n_compared else None}
    if len(usable) < 3:
        out["note"] = "fewer than 3 matched states"
        return out
    delta = {s["state_id"]: s["levels"][y]["energy_ev"] - s["levels"][x]["energy_ev"] for s in usable}
    rows = [[counts(s["formula"])[el] for el in ELEMENTS] for s in usable]
    mu = lstsq(rows, [delta[s["state_id"]] for s in usable])
    resid = {s["state_id"]: delta[s["state_id"]] - sum(m * c for m, c in zip(mu, r)) for s, r in zip(usable, rows)}
    per_atom = [1000 * resid[s["state_id"]] / s["n_atoms"] for s in usable]
    by_formula = defaultdict(list)
    for s in usable:
        by_formula[s["formula"]].append(1000 * delta[s["state_id"]] / s["n_atoms"])
    spread = {f: (max(v) - min(v)) for f, v in by_formula.items() if len(v) > 1}
    # relative energies: same formula and M (isomers/off-eq), same geometry different M (spin gaps)
    iso, gaps, ladders = [], [], {}
    for i, s in enumerate(usable):
        for t in usable[i + 1:]:
            dd = delta[t["state_id"]] - delta[s["state_id"]]
            if s["formula"] == t["formula"] and s["multiplicity"] == t["multiplicity"]:
                iso.append((dd, s["n_atoms"]))
            elif s["geometry_id"] == t["geometry_id"]:
                gaps.append((dd, s["n_atoms"]))
    by_geom = defaultdict(list)
    for s in usable:
        by_geom[s["geometry_id"]].append(s)
    for g, members in by_geom.items():
        if len(members) >= 2:
            low_x = min(members, key=lambda s: s["levels"][x]["energy_ev"])["multiplicity"]
            low_y = min(members, key=lambda s: s["levels"][y]["energy_ev"])["multiplicity"]
            ladders[f"{members[0]['formula']}:{g}"] = {"lowest_M_" + x: low_x, "lowest_M_" + y: low_y,
                                                       "agree": low_x == low_y, "members": len(members)}
    # forces
    by_el = defaultdict(list)
    cos_offeq = []
    fmax_diff = 0.0
    for s in usable:
        fa, fb = s["levels"][x]["forces"], s["levels"][y]["forces"]
        if not fa or not fb:
            continue
        for el, va, vb in zip(s["symbols"], fa, fb):
            d = [p - q for p, q in zip(vb, va)]
            by_el[el].extend(d)
            fmax_diff = max(fmax_diff, math.sqrt(sum(c * c for c in d)))
        if s["role"] != "minimum":
            dot = sum(p * q for va, vb in zip(fa, fb) for p, q in zip(va, vb))
            na = math.sqrt(sum(p * p for v in fa for p in v)); nb = math.sqrt(sum(q * q for v in fb for q in v))
            if na and nb:
                cos_offeq.append(dot / (na * nb))
    # moments (only meaningful when one side is VASP and the other Gaussian Hirshfeld)
    dm = []
    for s in usable:
        ma, mb = s["levels"][x].get("moments_cmp"), s["levels"][y].get("moments_cmp")
        if ma and mb:
            dm.extend(b - a for el, a, b in zip(s["symbols"], ma, mb) if el == "Fe")
    rel_all = [1000 * dd for dd, _ in iso + gaps]
    out.update({
        "mu_ev": dict(zip(ELEMENTS, mu)),
        "offset_residual_rms_mev_atom": rms(per_atom), "offset_residual_max_mev_atom": max(abs(v) for v in per_atom),
        "composition_spread_mev_atom": spread,
        "isomer_mae_mev": mae([1000 * d for d, _ in iso]), "isomer_mae_mev_atom": mae([1000 * d / n for d, n in iso]), "n_isomer_pairs": len(iso),
        "spin_gap_mae_mev": mae([1000 * d for d, _ in gaps]), "spin_gap_max_mev": max((abs(1000 * d) for d, _ in gaps), default=None),
        "spin_gap_mae_mev_atom": mae([1000 * d / n for d, n in gaps]), "n_spin_gap_pairs": len(gaps),
        "relative_mae_mev_atom": mae([1000 * d / n for d, n in iso + gaps]), "relative_mae_mev": mae(rel_all),
        "ladders": ladders, "ladders_disagreeing": sum(not v["agree"] for v in ladders.values()),
        "force_rmse_mev_a": {el: 1000 * rms(v) for el, v in by_el.items()},
        "force_max_diff_mev_a": 1000 * fmax_diff,
        "offeq_force_cosine_min": min(cos_offeq) if cos_offeq else None,
        "moment_diff_mean_mub": sum(dm) / len(dm) if dm else None,
        "moment_diff_std_mub": (math.sqrt(sum((v - sum(dm) / len(dm)) ** 2 for v in dm) / len(dm)) if dm else None),
        "residuals_mev_atom": {s["state_id"]: v for s, v in zip(usable, per_atom)},
    })
    return out


def decide(m: dict) -> tuple[str, list[str]]:
    """The doc's rule; returns (outcome, the conditions that failed for the next-better outcome)."""
    if m.get("n_matched", 0) < 3:
        return "undetermined", ["too few matched states"]
    f_max = max(m["force_rmse_mev_a"].values()) if m["force_rmse_mev_a"] else float("inf")
    pm = m["pattern_match_fraction"] or 0.0
    same = [
        ("residual RMS <= 2 meV/atom", m["offset_residual_rms_mev_atom"] <= 2),
        ("relative MAE <= 1 meV/atom", (m["relative_mae_mev_atom"] or 0) <= 1),
        ("spin-gap MAE <= 30 meV/cluster", (m["spin_gap_mae_mev"] or 0) <= 30),
        ("force RMSE <= 30 meV/A per element", f_max <= 30),
        ("pattern match >= 90 %", pm >= 0.90),
        ("lowest ladder M identical in every ladder", m["ladders_disagreeing"] == 0),
    ]
    sep = [
        ("residual RMS <= 10 meV/atom", m["offset_residual_rms_mev_atom"] <= 10),
        ("relative MAE <= 5 meV/atom", (m["relative_mae_mev_atom"] or 0) <= 5),
        ("force RMSE <= 100 meV/A", f_max <= 100),
        ("pattern match >= 75 %", pm >= 0.75),
        ("lowest ladder M differs in at most one ladder", m["ladders_disagreeing"] <= 1),
    ]
    if all(ok for _, ok in same):
        return "same_head", []
    if all(ok for _, ok in sep):
        return "separate_head", [c for c, ok in same if not ok]
    return "delta_only", [c for c, ok in sep if not ok]


def convergence_checks(results: Path, states_by_id: dict, vasp_ingest) -> dict:
    out = {}
    base = {}
    for sid in states_by_id:
        r = read_vasp(results, "prod_pbe", sid, vasp_ingest)
        if r and "energy_ev" in r:
            base[sid] = r
    for variant_dir in sorted((results / "vasp" / "outputs").iterdir()):
        variant = variant_dir.name
        if not variant.startswith("conv_"):
            continue
        rows = {}
        for job in sorted(variant_dir.iterdir()):
            r = read_vasp(results, variant, job.name, vasp_ingest)
            if not r or "energy_ev" not in r or job.name not in base:
                continue
            n = len(r["forces"])
            de = 1000 * (r["energy_ev"] - base[job.name]["energy_ev"]) / n
            dfs = [math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b))) for a, b in zip(r["forces"], base[job.name]["forces"])]
            rows[job.name] = {"dE_mev_atom": de, "max_dF_mev_a": 1000 * max(dfs), "scf_converged": r["scf_converged"]}
        des = [v["dE_mev_atom"] for v in rows.values()]
        out[variant] = {"jobs": rows, "relative_shift_mev_atom": (max(des) - min(des)) if des else None}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", help="the package results directory")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--cluster-mlip-src", required=True, help="src/ of a checkout that has cluster_mlip/vasp_ingest.py")
    ap.add_argument("--max-cycles", type=int, default=200, help="the route's SCF MaxCyc")
    ap.add_argument("--scf-conv-max", type=float, default=1e-6, help="largest acceptable final Conv= (Tight targets 1e-8)")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(args.cluster_mlip_src).resolve()))
    from cluster_mlip import periodic, vasp_ingest  # noqa: E402  (branch-only modules)

    results, out = Path(args.results), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    geoms = {s.info["structure_id"]: s for s in periodic.read_structures(results / "calibration_set.extxyz")}
    rows = list(csv.DictReader((results / "states.csv").open(encoding="utf-8")))
    states, flat = [], []
    for row in rows:
        sid = row["state_id"]
        symbols = list(geoms[sid].symbols)
        s = {**row, "multiplicity": int(row["multiplicity"]), "symbols": symbols, "n_atoms": len(symbols), "levels": {}}
        links = parse_gaussian_log(results / "gaussian" / "outputs" / f"{row['frame']}.log", len(symbols),
                                   args.max_cycles, args.scf_conv_max)
        for level, link in zip(G_LEVELS, links):
            link["pattern"] = pattern(symbols, link["hirshfeld"])
            link["moments_cmp"] = link["hirshfeld"]
            s["levels"][level] = link
        for level, variant in V_LEVELS.items():
            v = read_vasp(results, variant, sid, vasp_ingest)
            if v and "energy_ev" in v:
                v["pattern"] = pattern(symbols, v["moments"]) if v["moments"] else ""
                v["moments_cmp"] = v["moments"]
                s["levels"][level] = v
        states.append(s)
        for level, lv in s["levels"].items():
            flat.append({"state_id": sid, "frame": row["frame"], "formula": row["formula"], "M": row["multiplicity"],
                         "role": row["role"], "level": level, "energy_ev": f"{lv['energy_ev']:.6f}",
                         "scf_converged": lv["scf_converged"], "scf_flags": lv["scf_flags"],
                         "cycles_or_steps": lv.get("cycles", lv.get("scf_steps")), "conv": lv.get("conv", ""),
                         "s2": lv.get("s2", ""), "total_moment": lv.get("total_moment", ""), "fe_pattern": lv["pattern"]})
    # control: the G09 D.01 UBPW91 rerun against the archived dataset_spin_v0 label
    control = {}
    for s in states:
        if s["source"] == "dataset_spin_v0" and s["ref_energy_ev"] and "G_BPW91" in s["levels"]:
            g = s["levels"]["G_BPW91"]
            control[s["state_id"]] = {"dE_mev": 1000 * (g["energy_ev"] - float(s["ref_energy_ev"])),
                                      "scf_converged": g["scf_converged"],
                                      "fmax_rerun_ev_a": max(math.sqrt(sum(c * c for c in f)) for f in g["forces"]),
                                      "fmax_archived_ev_a": float(s["ref_fmax_ev_a"]) if s["ref_fmax_ev_a"] else None}
    pairs = {name: pair_metrics(states, x, y) for name, (x, y) in PAIRS.items()}
    decisions = {name: decide(pairs[name]) for name in ("training_pw91", "training_pbe")}
    # informational only: the same rule on the Fe >= LARGE_FE clusters (the Fe16 regime the models target)
    large = [s for s in states if int(s["n_fe"]) >= LARGE_FE]
    pairs_large = {name: pair_metrics(large, x, y) for name, (x, y) in PAIRS.items()}
    decisions_large = {name: decide(pairs_large[name]) for name in ("training_pw91", "training_pbe")}
    scf = {level: [s["state_id"] for s in states if level in s["levels"] and not s["levels"][level]["scf_converged"]]
           for level in (*G_LEVELS, *V_LEVELS)}
    report = {"results": str(results), "n_states": len(states), "scf_unconverged": scf, "control_g09_rerun": control,
              "pairs": pairs, "decision": {k: {"outcome": v[0], "failed": v[1]} for k, v in decisions.items()},
              "large_fe_min": LARGE_FE, "pairs_large": pairs_large,
              "decision_large": {k: {"outcome": v[0], "failed": v[1]} for k, v in decisions_large.items()},
              "vasp_convergence": convergence_checks(results, {s["state_id"]: s for s in states}, vasp_ingest),
              "thresholds": {"scf_conv_max": args.scf_conv_max, "max_cycles": args.max_cycles, "spin_sign_min": SPIN_SIGN_MIN}}
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    with (out / "states.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0]), lineterminator="\n")
        w.writeheader(); w.writerows(flat)
    (out / "report.md").write_text(render(report), encoding="utf-8", newline="\n")
    print(json.dumps({"decision": report["decision"], "scf_unconverged": {k: len(v) for k, v in scf.items()}}, indent=1))
    return 0


def _f(v, fmt="{:.2f}"):
    return "" if v is None else fmt.format(v)


def _pair_table(pairs: dict) -> list[str]:
    lines = [
              "| pair | matched | pattern match | mu Fe / N / O (eV) | resid RMS / max (meV/atom) | relative MAE (meV/atom) | spin-gap MAE / max (meV) | isomer MAE (meV) | ladders disagreeing | force RMSE Fe / N / O (meV/A) | dm Fe mean+-sd (muB) |",
              "|---|---:|---:|---|---|---:|---|---:|---:|---|---|"]
    for name, m in pairs.items():
        if "mu_ev" not in m:
            lines.append(f"| {name} | {m['n_matched']} | | | | | | | | | |"); continue
        mu = m["mu_ev"]; fr = m["force_rmse_mev_a"]
        lines.append(
            f"| {name} | {m['n_matched']} | {_f(m['pattern_match_fraction'], '{:.0%}')} | {mu['Fe']:.3f} / {mu['N']:.3f} / {mu['O']:.3f} "
            f"| {m['offset_residual_rms_mev_atom']:.2f} / {m['offset_residual_max_mev_atom']:.2f} | {_f(m['relative_mae_mev_atom'])} "
            f"| {_f(m['spin_gap_mae_mev'], '{:.1f}')} / {_f(m['spin_gap_max_mev'], '{:.1f}')} | {_f(m['isomer_mae_mev'], '{:.1f}')} | {m['ladders_disagreeing']}/{len(m['ladders'])} "
            f"| {' / '.join(_f(fr.get(el), '{:.0f}') for el in ELEMENTS)} | {_f(m['moment_diff_mean_mub'])} +- {_f(m['moment_diff_std_mub'])} |")
    return lines


def render(r: dict) -> str:
    lines = ["# Gaussian vs VASP calibration: results", "", f"Results: `{r['results']}`; {r['n_states']} states.", "",
             "## SCF screen (unconverged levels never enter a metric)", "", "| level | unconverged | states |", "|---|---:|---|"]
    for level, ids in r["scf_unconverged"].items():
        lines.append(f"| {level} | {len(ids)} | {', '.join(ids)} |")
    lines += ["", "## Control: G09 D.01 UBPW91 rerun vs archived label", "", "| state | dE (meV) | SCF ok | fmax rerun / archived (eV/A) |", "|---|---:|---|---|"]
    for sid, c in r["control_g09_rerun"].items():
        lines.append(f"| {sid} | {c['dE_mev']:.2f} | {c['scf_converged']} | {c['fmax_rerun_ev_a']:.4f} / {_f(c['fmax_archived_ev_a'], '{:.4f}')} |")
    lines += ["", "## Pairs, all states (delta = E_Y - E_X)", ""] + _pair_table(r["pairs"])
    lines += ["", "Exclusions per pair (state ids):", ""]
    for name, m in r["pairs"].items():
        ex = {k: len(v) for k, v in m["excluded"].items() if v}
        lines.append(f"- {name}: {ex or 'none'}")
    lines += ["", "## Decision (pair c)", ""]
    for name, d in r["decision"].items():
        lines.append(f"- **{name}: {d['outcome']}**" + (f" — failed: {'; '.join(d['failed'])}" if d["failed"] else ""))
    lines += ["", f"## Informational: Fe >= {r['large_fe_min']} clusters only", ""] + _pair_table(r["pairs_large"]) + [""]
    for name, d in r["decision_large"].items():
        lines.append(f"- {name} (Fe >= {r['large_fe_min']}): {d['outcome']}" + (f" — failed: {'; '.join(d['failed'])}" if d["failed"] else ""))
    lines += ["", "## VASP convergence (vs prod_pbe)", "", "| variant | relative shift (meV/atom) | per job dE (meV/atom), max dF (meV/A) |", "|---|---:|---|"]
    for v, c in r["vasp_convergence"].items():
        jobs = "; ".join(f"{k.split('_')[1]} {j['dE_mev_atom']:.2f}, {j['max_dF_mev_a']:.0f}" for k, j in c["jobs"].items())
        lines.append(f"| {v} | {_f(c['relative_shift_mev_atom'])} | {jobs} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
