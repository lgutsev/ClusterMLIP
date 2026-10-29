"""Bounded MagPT diagnostic on Fe16 spin states at fixed geometry.

    python experiments/fe16_magpt/run_magpt_diagnostic.py \
        --dataset D:/MLIP_Work_Folder/Cluster_MLIP/fe16_v1/02_fe16_dataset/all.extxyz \
        --checkpoint D:/MLIP_Work_Folder/external/magpt/MagPT/model/MagPT_399168.pth \
        --deepmd-src D:/MLIP_Work_Folder/external/magpt/deepmd-kit-magpt \
        --out experiments/fe16_magpt/report

Diagnostic only: no relaxation, nothing written into a training set. See README.md for the conventions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cluster_mlip.magpt_diagnostic import (  # noqa: E402
    DeepSpinEvaluator,
    SpinState,
    assign_ids,
    distinct_states,
    drop_twin_geometries,
    evaluate_state,
    force_metrics,
    group_states_by_geometry,
    load_states_from_extxyz,
    minimum_vacuum_gap,
    pairwise_ordering,
    relative_energies,
    spearman,
    spin_pattern_distance,
)

SCALES = (0.8, 0.9, 1.0, 1.1, 1.2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def provenance(args, evaluator: DeepSpinEvaluator) -> dict:
    import ase
    import deepmd
    import torch

    return {
        "model_repository": "https://github.com/MagTheoryLab/MagPT",
        "model_repository_commit": git_head(Path(args.checkpoint).resolve().parents[1]),
        "checkpoint_file": Path(args.checkpoint).name,
        "checkpoint_sha256": sha256(Path(args.checkpoint)),
        "checkpoint_training_set": "released 399,168-frame subset (not the full 912,492-frame MagPT)",
        "deepmd_repository": "https://github.com/MagTheoryLab/deepmd-kit (branch spin_virial_master_port)",
        "deepmd_commit": git_head(Path(args.deepmd_src)) if args.deepmd_src else "",
        "deepmd_version": getattr(deepmd, "__version__", ""),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "numpy_version": np.__version__,
        "ase_version": ase.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": "cpu",
        "type_map": evaluator.type_map,
        "rcut_ang": evaluator.rcut,
        "moment_input": "collinear m_i = (0, 0, s_i) mu_B, s_i = Gaussian Mulliken spin population (n_alpha - n_beta)",
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(Path(args.dataset)),
    }


def select_groups(states: list[SpinState], family_tol: float = 0.05, family_energy_tol: float = 0.01) -> list[list[SpinState]]:
    groups = group_states_by_geometry(states)
    groups = [distinct_states(g) for g in groups]
    groups = [g for g in groups if len({s.multiplicity for s in g}) > 1 or len(g) > 1]
    # one representative per structural family: reruns of the same chain land within ~1e-2 A and ~1 meV
    groups = drop_twin_geometries(groups, fingerprint_tol=family_tol, energy_tol=family_energy_tol)
    assign_ids(groups)
    return sorted(groups, key=lambda g: g[0].geometry_id)


def write_selection(groups: list[list[SpinState]], out: Path) -> None:
    """The selected geometry/state set and its Gaussian-side pair table (needs no MagPT)."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "selected_states.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geometry_id", "state_id", "frame_id", "parent_record_id", "multiplicity", "gaussian_energy_ev",
                    "total_moment_mu_b", "abs_moment_mu_b", "min_local_mu_b", "max_local_mu_b", "gaussian_force_rms_ev_ang",
                    "spin_method", "local_moments_mu_b", "source"])
        for g in groups:
            for s in g:
                frms = float(np.sqrt(np.mean(s.forces_ev_ang**2))) if s.forces_ev_ang is not None else ""
                w.writerow([s.geometry_id, s.state_id, s.frame_id, s.extra.get("parent_record_id", ""), s.multiplicity,
                            s.energy_ev, round(s.total_moment, 6), round(s.abs_moment, 6), float(s.local_spins.min()),
                            float(s.local_spins.max()), frms, s.spin_method,
                            json.dumps([round(float(x), 4) for x in s.local_spins]), s.source])
    with open(out / "selected_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geometry_id", "state_a", "state_b", "gaussian_dE_ev", "delta_total_moment_mu_b", "rms_local_moment_diff",
                    "max_local_moment_diff", "n_sign_changes_strong_sites", "gaussian_dF_rms_ev_ang"])
        for g in groups:
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    a, b = g[i], g[j]
                    pd = spin_pattern_distance(a.local_spins, b.local_spins)
                    dF = float(np.sqrt(np.mean((b.forces_ev_ang - a.forces_ev_ang) ** 2)))
                    w.writerow([a.geometry_id, a.state_id, b.state_id, b.energy_ev - a.energy_ev, b.total_moment - a.total_moment,
                                pd["rms_local_moment_diff"], pd["max_local_moment_diff"], pd["n_sign_changes_strong_sites"], dF])


def run(args) -> None:
    if args.select_only:
        write_selection(select_groups(load_states_from_extxyz(args.dataset), args.family_tol, args.family_energy_tol), Path(args.out))
        return
    import torch

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    groups = select_groups(load_states_from_extxyz(args.dataset), args.family_tol, args.family_energy_tol)
    evaluator = DeepSpinEvaluator(args.checkpoint)
    prov = provenance(args, evaluator)
    boxes = sorted(args.boxes)
    primary_box = boxes[-1]
    prov["boxes_ang"] = boxes
    prov["primary_box_ang"] = primary_box
    prov["moment_scales"] = list(SCALES)

    # nopbc is only a cross-check; record whether the spin model accepts it at all
    try:
        evaluate_state(evaluator, groups[0][0], None)
        nopbc_ok, nopbc_error = True, ""
    except Exception as exc:  # noqa: BLE001 - the error text is the finding
        nopbc_ok, nopbc_error = False, f"{type(exc).__name__}: {exc}"[:300]
    prov["nopbc_supported"] = nopbc_ok
    prov["nopbc_error"] = nopbc_error

    conditions: list[dict] = [{"condition": f"box{L:g}", "box": L, "scale": 1.0, "axis": "z", "control": ""} for L in boxes]
    conditions += [{"condition": f"scale{s:g}", "box": primary_box, "scale": s, "axis": "z", "control": ""} for s in SCALES if s != 1.0]
    conditions += [
        {"condition": "axis_x", "box": primary_box, "scale": 1.0, "axis": "x", "control": ""},
        {"condition": "repeat", "box": primary_box, "scale": 1.0, "axis": "z", "control": ""},
        {"condition": "control_time_reversed", "box": primary_box, "scale": 1.0, "axis": "z", "control": "reverse"},
        {"condition": "control_zero_moments", "box": primary_box, "scale": 1.0, "axis": "z", "control": "zero"},
    ]
    if nopbc_ok:
        conditions.append({"condition": "nopbc", "box": None, "scale": 1.0, "axis": "z", "control": ""})

    rows: list[dict] = []
    pred: dict[tuple[str, str], dict] = {}
    for group in groups:
        for state in group:
            for c in conditions:
                spins = state.local_spins
                if c["control"] == "reverse":
                    spins = -spins
                elif c["control"] == "zero":
                    spins = np.zeros_like(spins)
                r = evaluate_state(evaluator, state, c["box"], scale=c["scale"], axis=c["axis"], spins=spins)
                pred[(state.state_id, c["condition"])] = r
                fm = force_metrics(state.forces_ev_ang, r["forces"]) if state.forces_ev_ang is not None else {}
                rows.append(
                    {
                        "geometry_id": state.geometry_id,
                        "state_id": state.state_id,
                        "frame_id": state.frame_id,
                        "parent_record_id": state.extra.get("parent_record_id", ""),
                        "source": state.source,
                        "multiplicity": state.multiplicity,
                        "charge": state.charge,
                        "condition": c["condition"],
                        "box_ang": "" if c["box"] is None else c["box"],
                        "vacuum_gap_ang": "" if c["box"] is None else round(minimum_vacuum_gap(state.positions, c["box"]), 3),
                        "moment_scale": c["scale"],
                        "moment_axis": c["axis"],
                        "spin_method": state.spin_method,
                        "total_moment_mu_b": round(state.total_moment, 6),
                        "abs_moment_mu_b": round(state.abs_moment, 6),
                        "local_moments_mu_b": json.dumps([round(float(x), 4) for x in state.local_spins]),
                        "gaussian_energy_ev": state.energy_ev,
                        "magpt_energy_ev": r["energy_ev"],
                        "force_mae_vs_gaussian_ev_ang": fm.get("mae", ""),
                        "force_rmse_vs_gaussian_ev_ang": fm.get("rmse", ""),
                        "force_cosine_vs_gaussian": fm.get("cosine", ""),
                        "gaussian_force_rms_ev_ang": fm.get("ref_rms", ""),
                        "magpt_force_rms_ev_ang": fm.get("pred_rms", ""),
                        "magpt_net_force_ev_ang": r["net_force_norm"],
                        "magpt_mag_force_longitudinal_rms_ev_mu_b": r["mag_force_longitudinal_rms"],
                        "magpt_mag_force_longitudinal_max_ev_mu_b": r["mag_force_longitudinal_max"],
                        "magpt_mag_force_transverse_rms_ev_mu_b": r["mag_force_transverse_rms"],
                        "magpt_mag_forces_ev_mu_b": json.dumps([round(float(x), 5) for x in r["mag_forces"][:, 2 if c["axis"] == "z" else 0]]),
                        "model_checkpoint_sha256": prov["checkpoint_sha256"],
                        "deepmd_commit": prov["deepmd_commit"],
                    }
                )

    # relative energies: the lowest-Gaussian-energy state of each geometry is the reference on both sides
    by_geom: dict[str, list[SpinState]] = {g[0].geometry_id: g for g in groups}
    ref_index = {gid: int(np.argmin([s.energy_ev for s in g])) for gid, g in by_geom.items()}
    for row in rows:
        g = by_geom[row["geometry_id"]]
        ref = g[ref_index[row["geometry_id"]]]
        row["reference_state_id"] = ref.state_id
        row["gaussian_rel_ev"] = row["gaussian_energy_ev"] - ref.energy_ev
        row["magpt_rel_ev"] = row["magpt_energy_ev"] - pred[(ref.state_id, row["condition"])]["energy_ev"]

    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(groups, pred, conditions, primary_box, ref_index)
    summary["provenance"] = prov
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    write_markdown(out / "summary.md", summary)
    plot(groups, pred, primary_box, ref_index, out / "plots")
    plot_pooled(summary, out / "plots" / "pooled_dE.png")
    bulk_moment_scan(evaluator, out / "bulk_fe_moment_scan.csv")
    print(json.dumps(summary["verdict"], indent=2))


def summarize(groups, pred, conditions, primary_box, ref_index) -> dict:
    primary = f"box{primary_box:g}"
    geom_rows = []
    pooled_g, pooled_m = [], []
    for g in groups:
        gid = g[0].geometry_id
        ri = ref_index[gid]
        e_g = [s.energy_ev for s in g]
        rel_g = relative_energies(e_g, ri)
        rel = {c["condition"]: relative_energies([pred[(s.state_id, c["condition"])]["energy_ev"] for s in g], ri) for c in conditions}
        order = pairwise_ordering(rel_g, rel[primary])
        pairs = []
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g[i], g[j]
                dg = b.energy_ev - a.energy_ev
                dm = {k: float(v[j] - v[i]) for k, v in rel.items()}
                box_dm = [dm[c["condition"]] for c in conditions if c["condition"].startswith("box") or c["condition"] == "nopbc"]
                scale_dm = [dm[c] for c in ("scale0.9", "box%g" % primary_box, "scale1.1")]
                scale_dm_wide = [dm[c] for c in ("scale0.8", "scale0.9", "box%g" % primary_box, "scale1.1", "scale1.2")]
                fa = pred[(a.state_id, primary)]["forces"]
                fb = pred[(b.state_id, primary)]["forces"]
                dF_m = fb - fa
                dF_g = (b.forces_ev_ang - a.forces_ev_ang) if a.forces_ev_ang is not None and b.forces_ev_ang is not None else None
                pd = spin_pattern_distance(a.local_spins, b.local_spins)
                pairs.append(
                    {
                        "state_a": a.state_id,
                        "state_b": b.state_id,
                        "delta_moment_mu_b": b.total_moment - a.total_moment,
                        **pd,
                        "gaussian_dE_ev": dg,
                        "magpt_dE_ev": dm[primary],
                        "sign_agrees": bool(np.sign(dg) == np.sign(dm[primary])),
                        "magpt_dE_box_spread_ev": float(max(box_dm) - min(box_dm)),
                        "magpt_dE_repeat_diff_ev": abs(dm["repeat"] - dm[primary]),
                        "magpt_dE_axis_x_ev": dm["axis_x"],
                        "magpt_dE_by_scale_ev": {c: dm[c] for c in ("scale0.8", "scale0.9", primary, "scale1.1", "scale1.2")},
                        "sign_stable_scale_0.9_1.1": len({np.sign(x) for x in scale_dm}) == 1,
                        "sign_stable_scale_0.8_1.2": len({np.sign(x) for x in scale_dm_wide}) == 1,
                        "magpt_dE_time_reversed_ev": dm["control_time_reversed"],
                        "magpt_dE_zero_moments_ev": dm["control_zero_moments"],
                        "gaussian_dF_rms_ev_ang": float(np.sqrt(np.mean(dF_g**2))) if dF_g is not None else math.nan,
                        "magpt_dF_rms_ev_ang": float(np.sqrt(np.mean(dF_m**2))),
                        "dF_cosine": force_metrics(dF_g, dF_m)["cosine"] if dF_g is not None else math.nan,
                    }
                )
                pooled_g.append(dg)
                pooled_m.append(dm[primary])
        geom_rows.append(
            {
                "geometry_id": gid,
                "parent_record_id": g[0].extra.get("parent_record_id", ""),
                "states": [s.state_id for s in g],
                "reference_state": g[ri].state_id,
                "gaussian_rel_ev": [float(x) for x in rel_g],
                "magpt_rel_ev": [float(x) for x in rel[primary]],
                "ordering": order,
                "pairs": pairs,
                "time_reversal_max_abs_diff_ev": float(
                    max(abs(pred[(s.state_id, "control_time_reversed")]["energy_ev"] - pred[(s.state_id, primary)]["energy_ev"]) for s in g)
                ),
                "magnetic_energy_vs_zero_moments_ev": [
                    float(pred[(s.state_id, primary)]["energy_ev"] - pred[(s.state_id, "control_zero_moments")]["energy_ev"]) for s in g
                ],
            }
        )

    all_pairs = [p for gr in geom_rows for p in gr["pairs"]]
    noise = max(max(p["magpt_dE_box_spread_ev"], p["magpt_dE_repeat_diff_ev"]) for p in all_pairs)
    resolution = max(1e-3, 10 * noise)
    distinguished = [abs(p["magpt_dE_ev"]) > resolution for p in all_pairs]
    sign_agree = [p["sign_agrees"] for p in all_pairs]
    n_agree = sum(sign_agree)
    n = len(sign_agree)
    # one-sided binomial p-value for >= n_agree sign agreements under random signs
    p_binom = sum(math.comb(n, k) for k in range(n_agree, n + 1)) / 2**n
    stable = [p["sign_stable_scale_0.9_1.1"] for p in all_pairs]
    verdict = {
        "n_geometries": len(groups),
        "n_states": sum(len(g) for g in groups),
        "n_state_pairs": n,
        "resolution_ev": resolution,
        "max_box_or_repeat_noise_ev": noise,
        "pairs_distinguished": int(sum(distinguished)),
        "pairs_sign_agree": n_agree,
        "sign_agreement_binomial_p": p_binom,
        "pooled_spearman_dE": spearman(pooled_g, pooled_m),
        "pooled_pearson_dE": float(np.corrcoef(pooled_g, pooled_m)[0, 1]) if n > 2 else math.nan,
        "pairs_sign_stable_scale_0.9_1.1": int(sum(stable)),
        "pairs_sign_stable_scale_0.8_1.2": int(sum(p["sign_stable_scale_0.8_1.2"] for p in all_pairs)),
        "max_box_spread_ev": max(p["magpt_dE_box_spread_ev"] for p in all_pairs),
        "max_time_reversal_diff_ev": max(gr["time_reversal_max_abs_diff_ev"] for gr in geom_rows),
    }
    q1 = all(distinguished)
    geoms_agree = sum(1 for gr in geom_rows if all(p["sign_agrees"] for p in gr["pairs"]))
    q2 = geoms_agree >= 2 and p_binom < 0.05
    q3 = all(stable) and verdict["max_box_spread_ev"] < resolution
    verdict.update(
        {
            "q1_distinguishes_patterns": q1,
            "q2_reproduces_ordering": q2,
            "geometries_with_full_sign_agreement": geoms_agree,
            "q3_stable_box_and_magnitude": q3,
            "q4_larger_pilot_justified": bool(q1 and q2 and q3),
            "gate_rule": (
                "Q1: every pair |dE_MagPT| > max(1 meV, 10 x box/repeat noise). "
                "Q2: dE sign agrees with Gaussian in >= 2 whole geometries AND one-sided binomial p < 0.05 over all pairs. "
                "Q3: dE sign unchanged for moment scales 0.9-1.1 for every pair AND box spread < resolution. "
                "Q4: Q1 and Q2 and Q3."
            ),
        }
    )
    return {"geometries": geom_rows, "verdict": verdict}


def write_markdown(path: Path, summary: dict) -> None:
    v = summary["verdict"]
    lines = ["# MagPT Fe16 diagnostic — machine summary", "", "Generated by run_magpt_diagnostic.py; see README.md for interpretation.", ""]
    lines += ["| key | value |", "|---|---|"] + [f"| {k} | {v[k]} |" for k in v if k != "gate_rule"]
    lines += ["", f"Gate rule: {v['gate_rule']}", "", "## Per pair (primary box)", ""]
    lines += [
        "| geometry | pair | dM (μB) | rms Δm (μB) | sign flips | ΔE Gaussian (meV) | ΔE MagPT (meV) | sign | box spread (meV) | ΔE at 0.8/0.9/1.0/1.1/1.2 (meV) | ΔF Gaussian rms | ΔF MagPT rms | ΔF cos |",
        "|---|---|---:|---:|---:|---:|---:|:-:|---:|---|---:|---:|---:|",
    ]
    for gr in summary["geometries"]:
        for p in gr["pairs"]:
            sc = "/".join(f"{1000 * x:.1f}" for x in p["magpt_dE_by_scale_ev"].values())
            lines.append(
                f"| {gr['geometry_id']} | {p['state_a'].split(':')[1]}→{p['state_b'].split(':')[1]} | {p['delta_moment_mu_b']:.2f} | "
                f"{p['rms_local_moment_diff']:.3f} | {p['n_sign_changes_strong_sites']} | {1000 * p['gaussian_dE_ev']:.1f} | "
                f"{1000 * p['magpt_dE_ev']:.1f} | {'✓' if p['sign_agrees'] else '✗'} | {1000 * p['magpt_dE_box_spread_ev']:.3f} | {sc} | "
                f"{p['gaussian_dF_rms_ev_ang']:.3f} | {p['magpt_dF_rms_ev_ang']:.3f} | {p['dF_cosine']:.2f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(groups, pred, primary_box, ref_index, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = f"box{primary_box:g}"
    for g in groups:
        gid = g[0].geometry_id
        ri = ref_index[gid]
        rel_g = 1000 * relative_energies([s.energy_ev for s in g], ri)
        rel_m = 1000 * relative_energies([pred[(s.state_id, primary)]["energy_ev"] for s in g], ri)
        lo = 1000 * relative_energies([pred[(s.state_id, "scale0.8")]["energy_ev"] for s in g], ri)
        hi = 1000 * relative_energies([pred[(s.state_id, "scale1.2")]["energy_ev"] for s in g], ri)
        x = np.arange(len(g))
        fig, ax = plt.subplots(figsize=(3.6, 3.0), dpi=150)
        ax.axhline(0, color="#999", lw=0.6)
        ax.plot(x - 0.08, rel_g, "o", color="#1f5fa8", label="Gaussian UBPW91")
        ax.plot(x + 0.08, rel_m, "s", color="#c8501e", label=f"MagPT ({primary_box:g} Å box)")
        for xi, a, b in zip(x, lo, hi):
            ax.plot([xi + 0.08, xi + 0.08], [min(a, b), max(a, b)], color="#c8501e", lw=1, alpha=0.6)
        ax.set_xticks(x, [f"M={s.multiplicity}" for s in g])
        ax.set_ylabel("E − E(ref) (meV)")
        ax.set_title(f"Fe16 {gid}", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / f"{gid}.png")
        plt.close(fig)


def plot_pooled(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [p for gr in summary["geometries"] for p in gr["pairs"]]
    g = np.array([1000 * p["gaussian_dE_ev"] for p in pairs])
    m = np.array([1000 * p["magpt_dE_ev"] for p in pairs])
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=150)
    ax.axhline(0, color="#999", lw=0.6)
    ax.axvline(0, color="#999", lw=0.6)
    for p, x, y in zip(pairs, g, m):
        ax.plot(x, y, "o", color="#1f5fa8" if p["state_a"].endswith("M53") else "#c8501e")
    ax.plot([], [], "o", color="#1f5fa8", label="M53→M51")
    ax.plot([], [], "o", color="#c8501e", label="M51→M49")
    ax.set_xlabel("ΔE Gaussian (meV)")
    ax.set_ylabel("ΔE MagPT (meV)")
    ax.set_title("Fe16 same-geometry pairs (ΔE = E(M−2) − E(M))", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def bulk_moment_scan(evaluator, path: Path) -> None:
    """Reference: MagPT energy vs uniform moment in ferromagnetic bcc Fe (a = 2.83 A), its home domain."""
    from cluster_mlip.magpt_diagnostic import moment_vectors

    a = 2.83
    frac = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)], dtype=float)
    pos = np.concatenate([frac * a, (frac + 0.5) * a])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["moment_mu_b", "energy_per_atom_ev", "mag_force_z_mean_ev_mu_b"])
        for mu in (0.0, 1.0, 1.5, 2.0, 2.2, 2.4, 2.6, 3.0, 3.5, 4.0, 4.5, 5.0):
            e, _f, fm = evaluator(pos, np.eye(3) * 2 * a, moment_vectors([mu] * 16), ["Fe"] * 16)
            w.writerow([mu, e / 16, float(fm[:, 2].mean())])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--select-only", action="store_true", help="write the selected state set and stop (no MagPT needed)")
    p.add_argument("--deepmd-src", default="", help="checkout of the DeePMD-kit branch used, for the commit record")
    p.add_argument("--out", required=True)
    p.add_argument("--boxes", type=float, nargs="+", default=[20.0, 25.0, 30.0])
    p.add_argument("--family-tol", type=float, default=0.05, help="sorted-distance RMS (A) below which geometries are one family")
    p.add_argument("--family-energy-tol", type=float, default=0.01, help="per-state energy match (eV) for family twins")
    p.add_argument("--threads", type=int, default=1)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
