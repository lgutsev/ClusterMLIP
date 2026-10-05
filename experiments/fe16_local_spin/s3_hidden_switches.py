"""Step 8 (part): hidden electronic-state switches inside optimizations.

Only first/final frames carry Mulliken tables, so for all 1,251 frames we use two
proxies along each optimization stage (adjacent force frames, same q and M):

* <S^2> excess (after annihilation) jumps -- a site flipping between parallel and
  antiparallel changes the spin contamination by several units;
* energy-force consistency: dE + 0.5 (F_a + F_b) . dx  (trapezoid work).  On one
  smooth adiabatic surface it is small; a switch to another broken-symmetry
  solution shows up as an energy jump the forces do not account for.

Also: Mulliken stability between the first and final frame of each stage, and at
the M -> M-2 hand-offs (same geometry), including site-resolved sign changes.
"""
from __future__ import annotations

import collections
import json

import numpy as np
from ase.io import read

from common import COLLECT_ALL, RESULTS, SPIN_ALL, aligned_match, meta


def main() -> None:
    fr = read(COLLECT_ALL, ":")
    stages = collections.defaultdict(list)
    for a in fr:
        m = meta(a)
        stages[(m["job_id"], str(m.get("stage_index")))].append((int(m["force_frame_index"]), a, m))
    steps = []
    for key, lst in stages.items():
        lst.sort(key=lambda t: t[0])
        for (ia, a, ma), (ib, b, mb) in zip(lst, lst[1:]):
            if ib != ia + 1:
                continue
            dx = b.positions - a.positions
            work = 0.5 * ((a.arrays["REF_forces"] + b.arrays["REF_forces"]) * dx).sum()
            dE = b.info["REF_energy"] - a.info["REF_energy"]
            steps.append({
                "job": key[0], "stage": key[1], "frame": ib, "M": int(a.info["multiplicity"]),
                "dE_eV": dE, "resid_eV": dE + work, "step_A": float(np.sqrt((dx ** 2).sum(1).max())),
                "ds2": float(mb["s2_after_excess"]) - float(ma["s2_after_excess"]),
                "s2a": float(ma["s2_after_excess"]), "s2b": float(mb["s2_after_excess"]),
            })
    res = np.array([s["resid_eV"] for s in steps])
    ds2 = np.array([s["ds2"] for s in steps])
    big = [s for s in steps if abs(s["resid_eV"]) > 0.05 or abs(s["ds2"]) > 1.0]
    big.sort(key=lambda s: -abs(s["resid_eV"]))

    # Mulliken stability within a stage (first vs final) and across hand-offs.
    sp = read(SPIN_ALL, ":")
    by_stage = collections.defaultdict(dict)
    for a in sp:
        m = meta(a)
        by_stage[(m["job_id"], str(m.get("stage_index")))][m["stage_frame_position"]] = a
    within = []
    for key, d in by_stage.items():
        if "first" in d and "final" in d:
            A, B = d["first"], d["final"]
            rmsd, perm, R, *_ = aligned_match(A.positions, B.positions)
            sa, sb = A.arrays["local_moment"], B.arrays["local_moment"][perm]
            within.append({"stage": key, "M": int(A.info["multiplicity"]), "rmsd": rmsd,
                           "max_abs_ds": float(np.abs(sa - sb).max()),
                           "sign_changes": int(((sa < 0) != (sb < 0)).sum())})
    handoff = []
    for i, a in enumerate(sp):
        for b in sp[i + 1:]:
            Ma, Mb = int(a.info["multiplicity"]), int(b.info["multiplicity"])
            if abs(Ma - Mb) != 2:
                continue
            rmsd, perm, R, *_ = aligned_match(a.positions, b.positions)
            if rmsd > 0.01:
                continue
            hi, lo = (a, b.arrays["local_moment"][perm]) if Ma > Mb else (b, None)
            sa = a.arrays["local_moment"]
            sb = b.arrays["local_moment"][perm]
            s_hi, s_lo = (sa, sb) if Ma > Mb else (sb, sa)
            flipped = np.where((s_hi > 0) & (s_lo < 0))[0]
            handoff.append({"M_hi": max(Ma, Mb), "rmsd": rmsd, "new_negative_sites": len(flipped),
                            "lost_negative_sites": int(((s_hi < 0) & (s_lo > 0)).sum()),
                            "sum_drop": float(s_hi.sum() - s_lo.sum()),
                            "max_abs_ds": float(np.abs(s_hi - s_lo).max())})

    summary = {
        "adjacent_opt_steps": len(steps),
        "energy_force_residual_eV": {"median_abs": float(np.median(np.abs(res))),
                                     "p99_abs": float(np.percentile(np.abs(res), 99)),
                                     "max_abs": float(np.abs(res).max()),
                                     "n_gt_0.05": int((np.abs(res) > 0.05).sum())},
        "s2_excess_jump": {"median_abs": float(np.median(np.abs(ds2))),
                           "max_abs": float(np.abs(ds2).max()),
                           "n_gt_1": int((np.abs(ds2) > 1).sum())},
        "flagged_steps": big[:15],
        "first_vs_final_same_stage": {
            "n": len(within),
            "sign_changes": dict(collections.Counter(w["sign_changes"] for w in within)),
            "max_abs_ds_median": float(np.median([w["max_abs_ds"] for w in within])) if within else None,
            "max_abs_ds_max": float(max(w["max_abs_ds"] for w in within)) if within else None,
            "rmsd_median": float(np.median([w["rmsd"] for w in within])) if within else None,
            "examples_with_sign_change": [w for w in within if w["sign_changes"]][:8],
        },
        "handoffs_M_to_M-2_same_geometry": {
            "n": len(handoff),
            "new_negative_sites": dict(collections.Counter(h["new_negative_sites"] for h in handoff)),
            "lost_negative_sites": dict(collections.Counter(h["lost_negative_sites"] for h in handoff)),
            "sum_drop_mean": float(np.mean([h["sum_drop"] for h in handoff])) if handoff else None,
        },
    }
    (RESULTS / "hidden_switches.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
