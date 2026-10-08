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
            # Scale of the second-order work term: trapezoid is exact on a quadratic
            # surface, so on one smooth surface |resid| should be a small fraction of it.
            curv = abs(((b.arrays["REF_forces"] - a.arrays["REF_forces"]) * dx).sum())
            steps.append({
                "job": key[0], "stage": key[1], "frame": ib, "M": int(a.info["multiplicity"]),
                "dE_eV": dE, "work_eV": -work, "resid_eV": dE + work, "curv_term_eV": curv,
                "step_A": float(np.sqrt((dx ** 2).sum(1).max())),
                # before annihilation: contamination is not projected out
                "ds2_before": float(mb["s2_before_excess"]) - float(ma["s2_before_excess"]),
                "ds2_after": float(mb["s2_after_excess"]) - float(ma["s2_after_excess"]),
                "s2_before_a": float(ma["s2_before_excess"]), "s2_before_b": float(mb["s2_before_excess"]),
            })
    res = np.array([s["resid_eV"] for s in steps])
    curv = np.array([s["curv_term_eV"] for s in steps])
    step = np.array([s["step_A"] for s in steps])
    ds2 = np.array([s["ds2_before"] for s in steps])
    ds2_after = np.array([s["ds2_after"] for s in steps])
    big = sorted(steps, key=lambda s: -abs(s["resid_eV"]))[:10]
    from scipy.stats import spearmanr
    ratio = np.abs(res) / np.maximum(curv, 1e-12)
    # the stage that holds the largest residuals, in full
    worst = big[0]
    worst_stage = [s for s in steps if (s["job"], s["stage"]) == (worst["job"], worst["stage"])
                   and abs(s["frame"] - worst["frame"]) <= 6]

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
        "residual_vs_step": {
            "spearman_abs_resid_vs_step": round(float(spearmanr(np.abs(res), step)[0]), 3),
            "spearman_abs_resid_vs_curv_term": round(float(spearmanr(np.abs(res), curv)[0]), 3),
            "resid_over_curv_term_median": float(np.median(ratio)),
            "resid_over_curv_term_p99": float(np.percentile(ratio, 99)),
            "resid_over_curv_term_max": float(ratio.max()),
            "top10_steps_mean_step_A": float(np.mean([s["step_A"] for s in big])),
            "all_steps_median_step_A": float(np.median(step)),
        },
        "s2_excess_jump_before_annihilation": {"median_abs": float(np.median(np.abs(ds2))),
                                               "max_abs": float(np.abs(ds2).max()),
                                               "n_gt_0.5": int((np.abs(ds2) > 0.5).sum())},
        "s2_excess_jump_after_annihilation": {"max_abs": float(np.abs(ds2_after).max())},
        "s2_before_excess_range_all_frames": [float(min(float(meta(a)["s2_before_excess"]) for a in fr)),
                                              float(max(float(meta(a)["s2_before_excess"]) for a in fr))],
        "largest_residual_steps": big,
        "largest_residual_stage_window": worst_stage,
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
