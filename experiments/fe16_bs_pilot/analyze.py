"""Analyse returned broken-symmetry tandem logs.

    python analyze.py CAMPAIGN_DIR [LOG_DIR]

CAMPAIGN_DIR holds spin_jobs.csv and inputs/ (from `cluster-mlip prepare-broken-symmetry`
or make_validation_kit.py). LOG_DIR (default CAMPAIGN_DIR) is searched recursively for
the returned .log files. Writes bs_states.csv, bs_pairs.csv and bs_summary.md into
CAMPAIGN_DIR.

Per job, from cluster_mlip.fragment_tandem.parse_tandem_log:
  * link 0: fragment SCFs (count, cycles, unconverged), any supermolecule SCF;
  * link 1: was the guess read from the checkpoint, and does its <S**2> match the
    fragment pattern; every SCF energy/<S**2> and instability Stable=Opt met, i.e.
    the state before and after the stability optimization;
  * link 2: does the force SCF reproduce the stable state, and did the geometry stay fixed;
  * Mulliken and Hirshfeld spins, CM5 charges (charges, not spins), forces.

States: two usable jobs at one geometry are the SAME state only if energy (1 meV),
per-site Hirshfeld spin (0.2), <S**2> (0.05) and force RMS (5 meV/Å) all agree;
DISTINCT if the criteria disagree clearly (spin and energy both beyond tolerance, or
all criteria); otherwise AMBIGUOUS, which is reported and not counted. These are
reproducibility tolerances. A stable solution above the lowest one is a metastable
BS state, not a rejected one.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from cluster_mlip.fragment_tandem import compare_states, parse_tandem_log, plan_from_input


def fmt(values, digits=3):
    if values is None:
        return ""
    if isinstance(values, float):
        return f"{values:.{digits}f}"
    if isinstance(values, list):
        return " ".join(f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in values)
    return str(values)


def table_values(table, key):
    return [v[2] for v in (table or {}).get(key, [])]


def main() -> None:
    campaign = Path(sys.argv[1])
    search = Path(sys.argv[2]) if len(sys.argv) > 2 else campaign
    found = {path.name: path for path in search.rglob("*.log")}
    jobs: dict[str, dict] = {}
    for row in csv.DictReader((campaign / "spin_jobs.csv").open(encoding="utf-8")):
        jobs.setdefault(row["output"], row)
    rows, usable = [], defaultdict(dict)
    for output, meta in sorted(jobs.items()):
        plan, n_stages = plan_from_input((campaign / meta["input"]).read_text(encoding="utf-8"))
        force_stage = n_stages == (2 if plan is None else 3)
        base = {"record": meta["parent_record_id"], "pattern": meta.get("bs_pattern") or meta.get("fragment_label"),
                "log": output, "ideal_guess_s2": None if plan is None else plan.ideal_guess_s2}
        path = found.get(output)
        if path is None:
            rows.append({**base, "status": "missing"})
            continue
        r = parse_tandem_log(path.read_text(errors="replace"), plan, force_stage)
        rows.append({**base, **r})
        if r["status"] == "ok" and r.get("energy_ev") is not None:
            usable[base["record"]][base["pattern"]] = r

    cols = ["record", "pattern", "status", "issues", "s1_fragment_scf", "s1_unconverged_scf",
            "s1_supermolecule_scf", "ideal_guess_s2", "s2_guess_s2", "s2_scf_cycles", "s2_cycle_cap",
            "s2_instabilities", "stable_opt_changed_state", "stable_opt_dE_meV", "stable_opt_dS2",
            "s2_final_stable", "s3_scf_cycles", "s3_minus_s2_hartree", "geometry_max_shift_ang",
            "energy_ev", "s3_s2", "spins_hirshfeld", "spins_mulliken", "charges_hirshfeld", "charges_cm5", "log"]
    with (campaign / "bs_states.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["spins_hirshfeld"] = table_values(r.get("s3_hirshfeld"), "atomic_spins_hirshfeld")
            out["charges_hirshfeld"] = table_values(r.get("s3_hirshfeld"), "atomic_charges_hirshfeld")
            out["charges_cm5"] = table_values(r.get("s3_hirshfeld"), "atomic_charges_cm5")
            out["spins_mulliken"] = r.get("s3_mulliken_spins")
            out["issues"] = "; ".join(r.get("issues", []))
            w.writerow({k: fmt(out.get(k), 6 if k == "energy_ev" else 3) for k in cols})

    pairs, lines = [], ["# BS tandem summary", "",
                        "SAME needs energy (1 meV), per-site Hirshfeld spin (0.2), <S^2> (0.05) and force RMS "
                        "(5 meV/Å) to agree; these are reproducibility tolerances, not physical thresholds. "
                        "CM5 values are charges, not spins.", ""]
    n_geom_multi = 0
    for record, states in sorted(usable.items()):
        groups: list[list[str]] = []
        for pattern in sorted(states):
            for g in groups:
                if compare_states(states[g[0]], states[pattern])["verdict"] == "same":
                    g.append(pattern)
                    break
            else:
                groups.append([pattern])
        verdicts = []
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                c = compare_states(states[g1[0]], states[g2[0]])
                verdicts.append(c["verdict"])
                pairs.append({"record": record, "state_a": "+".join(g1), "state_b": "+".join(g2), **c})
        distinct = 1 + sum(1 for v in verdicts if v == "distinct") if groups else 0
        n_geom_multi += distinct >= 2
        e0 = min(states[g[0]]["energy_ev"] for g in groups)
        lines.append(f"- `{record}`: {len(states)} usable jobs -> {len(groups)} group(s): "
                     + "; ".join(f"{'+'.join(g)} ({1000 * (states[g[0]]['energy_ev'] - e0):+.1f} meV)" for g in groups)
                     + (f"; ambiguous pairs: {verdicts.count('ambiguous')}" if "ambiguous" in verdicts else ""))
    with (campaign / "bs_pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["record", "state_a", "state_b", "verdict", "dE_meV", "max_dspin_hirshfeld",
                  "hirshfeld_sign_flips", "dS2", "force_rms_diff_meV_ang"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows({k: fmt(v) for k, v in p.items()} for p in pairs)

    changed = [r for r in rows if r.get("stable_opt_changed_state")]
    lines += ["", f"Stable=Opt changed the read-in state in {len(changed)} job(s):",
              *[f"  - {r['log']}: dE {r['stable_opt_dE_meV']:+.1f} meV, d<S^2> {fmt(r.get('stable_opt_dS2'))}, "
                f"instabilities {r.get('s2_instabilities')}" for r in changed]]
    bad = [r for r in rows if r.get("status") != "ok"]
    lines += ["", f"Jobs not usable ({len(bad)}):",
              *[f"  - {r['log']}: {r.get('status')}: {'; '.join(r.get('issues', []))}" for r in bad], ""]
    if n_geom_multi >= 2:
        verdict = (f"Distinct stable solutions at fixed geometry, charge and M for {n_geom_multi} geometries. "
                   "Check bs_pairs.csv (dE, D_F) before any dataset decision.")
    elif n_geom_multi == 1:
        verdict = "Distinct stable solutions for one geometry only: below the pilot's bar (>= 2 geometries)."
    else:
        verdict = ("No alternative stable state was found for these geometries, this M and these guesses. "
                   "That does not show local spin is irrelevant in general.")
    lines.append("**Verdict:** " + verdict)
    (campaign / "bs_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
