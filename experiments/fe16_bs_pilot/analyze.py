"""Analyse returned broken-symmetry pilot logs.

    python analyze.py CAMPAIGN_DIR [LOG_DIR]

CAMPAIGN_DIR holds spin_jobs.csv (from `cluster-mlip prepare-broken-symmetry`);
LOG_DIR (default CAMPAIGN_DIR) is searched recursively for the returned .log
files (the Slurm worker writes them into slurm_batches/<batch>/). Writes
bs_states.csv, bs_pairs.csv and bs_summary.md into CAMPAIGN_DIR.

Per job: normal termination, last Stable=Opt verdict, stage-1 energy and forces,
<S^2> (supporting evidence only), Mulliken spins, Hirshfeld spin densities and
CM5 charges (charges only), all from the force stage's own SCF.

Two converged, stable solutions at one geometry are the SAME state when
|dE| <= 1 meV and max per-site |d s_Hirshfeld| <= 0.2; otherwise DISTINCT.
If every pattern returns the P0 state, the verdict says only that this pilot
found no alternative stable state for these geometries, M and guesses.
"""
from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from cluster_mlip.gaussian import parse_force_frames

SAME_DE_EV = 0.001
SAME_DS = 0.2


def stability(text: str) -> str:
    lower = text.lower()
    stable = lower.rfind("wavefunction is stable under the perturbations considered")
    unstable = max(lower.rfind("wavefunction has an internal instability"),
                   lower.rfind("wavefunction has an rhf -> uhf instability"),
                   lower.rfind("wavefunction is unstable"))
    if stable < 0 and unstable < 0:
        return "not_reported"
    return "stable" if stable > unstable else "unstable"


def values(table) -> list[float]:
    return [float(v[2]) for v in table] if table else []


def analyse_job(log: Path) -> dict:
    text = log.read_text(errors="replace")
    sections = text.split("Normal termination of Gaussian")
    row = {"log": log.name, "normal_terminations": len(sections) - 1,
           "error_termination": "Error termination" in text,
           "stability": stability(text.split("--Link1--")[0] if "--Link1--" in text else text)}
    frames = parse_force_frames(text, log)
    if not frames:
        row["status"] = "no_force_frame"
        return row
    f = frames[-1]
    md = f.record.metadata
    row.update({
        "status": "ok" if row["normal_terminations"] >= 2 and not row["error_termination"] else "incomplete",
        "energy_ev": f.energy_ev,
        "forces": f.forces_ev_ang,
        "s2_before": md.get("s2_before"), "s2_after": md.get("s2_after"),
        "spins_mulliken": values(md.get("atomic_spins")),
        "spins_hirshfeld": values(md.get("atomic_spins_hirshfeld")),
        "charges_cm5": values(md.get("atomic_charges_cm5")),
        "scf_convergence_warning": md.get("scf_convergence_warning"),
    })
    return row


def same_state(a: dict, b: dict) -> bool:
    if not a.get("spins_hirshfeld") or not b.get("spins_hirshfeld"):
        return abs(a["energy_ev"] - b["energy_ev"]) <= SAME_DE_EV
    ds = max(abs(x - y) for x, y in zip(a["spins_hirshfeld"], b["spins_hirshfeld"]))
    return abs(a["energy_ev"] - b["energy_ev"]) <= SAME_DE_EV and ds <= SAME_DS


def main() -> None:
    campaign = Path(sys.argv[1])
    search = Path(sys.argv[2]) if len(sys.argv) > 2 else campaign
    found = {path.name: path for path in search.rglob("*.log")}
    logs = campaign
    jobs = {}
    for r in csv.DictReader((campaign / "spin_jobs.csv").open(encoding="utf-8")):
        if r["stage_index"] == "0":
            jobs[r["output"]] = r
    by_geom: dict[str, dict[str, dict]] = defaultdict(dict)
    rows = []
    for output, meta in sorted(jobs.items()):
        path = found.get(output)
        if path is None:
            res = {"log": output, "status": "missing"}
        else:
            res = analyse_job(path)
        res.update({"record": meta["parent_record_id"], "pattern": meta["bs_pattern"],
                    "flipped_sites": meta["flipped_sites"]})
        rows.append(res)
        if res.get("status") == "ok" and res.get("stability") == "stable":
            by_geom[meta["parent_record_id"]][meta["bs_pattern"]] = res

    with (logs / "bs_states.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ["record", "pattern", "flipped_sites", "status", "stability", "energy_ev", "s2_before",
                "s2_after", "spins_hirshfeld", "spins_mulliken", "charges_cm5", "log"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (" ".join(f"{v:.3f}" for v in r[k]) if isinstance(r.get(k), list) else r.get(k))
                        for k in cols})

    pairs = []
    lines = ["# BS pilot summary", "",
             "Same state: |dE| <= 1 meV and max per-site |d s_Hirshfeld| <= 0.2. "
             "<S^2> is supporting evidence only. CM5 values are charges, not spins.", ""]
    n_geom_multi = 0
    for record, states in sorted(by_geom.items()):
        groups: list[list[str]] = []
        for pattern in sorted(states):
            for g in groups:
                if same_state(states[g[0]], states[pattern]):
                    g.append(pattern)
                    break
            else:
                groups.append([pattern])
        n_geom_multi += len(groups) >= 2
        lines.append(f"- `{record}`: {len(states)} stable converged jobs -> {len(groups)} distinct state(s): "
                     + "; ".join("+".join(g) for g in groups))
        for g1, g2 in ((a, b) for i, a in enumerate(groups) for b in groups[i + 1:]):
            a, b = states[g1[0]], states[g2[0]]
            df = math.sqrt(sum((x - y) ** 2 for fa, fb in zip(a["forces"], b["forces"])
                               for x, y in zip(fa, fb)) / (3 * len(a["forces"])))
            pairs.append({"record": record, "state_a": "+".join(g1), "state_b": "+".join(g2),
                          "dE_meV": 1000 * (b["energy_ev"] - a["energy_ev"]),
                          "D_F_meV_A": 1000 * df,
                          "max_ds_hirshfeld": max(abs(x - y) for x, y in zip(a["spins_hirshfeld"], b["spins_hirshfeld"]))
                          if a["spins_hirshfeld"] and b["spins_hirshfeld"] else None,
                          "ds2_before": (b["s2_before"] or 0) - (a["s2_before"] or 0)})
    with (logs / "bs_pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["record", "state_a", "state_b", "dE_meV", "D_F_meV_A",
                                           "max_ds_hirshfeld", "ds2_before"])
        w.writeheader()
        w.writerows(pairs)
    bad = [r for r in rows if r.get("status") != "ok" or r.get("stability") != "stable"]
    lines += ["", f"Jobs not usable (missing, incomplete, or not stable): {len(bad)}",
              *[f"  - {r['log']}: {r.get('status')}, {r.get('stability')}" for r in bad], ""]
    if n_geom_multi >= 2:
        verdict = (f"Distinct stable solutions at fixed geometry and M found for {n_geom_multi} geometries: "
                   "the roadmap gate for reopening local-spin tests is met; see bs_pairs.csv for dE and D_F.")
    elif n_geom_multi == 1:
        verdict = "Distinct stable solutions found for one geometry only: below the pilot's success bar (>= 2)."
    else:
        verdict = ("This pilot found no alternative stable state for these geometries, this M and these "
                   "guesses. It does not show that local spin is generally irrelevant.")
    lines += ["**Verdict:** " + verdict]
    (logs / "bs_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
