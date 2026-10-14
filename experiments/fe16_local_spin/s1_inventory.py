"""Step 1: which Fe16 frames carry a complete, correctly ordered local-spin vector.

Reads the full collected Fe16 set and the population-labelled subset; never fills a
missing value. Writes results/inventory.json and results/frames.csv.
"""
from __future__ import annotations

import collections
import csv
import json

import numpy as np
from ase.io import read

from common import COLLECT_ALL, RESULTS, SPIN_ALL, meta


def frame_key(a) -> str:
    return a.info["record_id"]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    allf = read(COLLECT_ALL, ":")
    spin = read(SPIN_ALL, ":")
    by_id = {frame_key(a): a for a in allf}

    reject = collections.Counter()
    for a in allf:
        m = meta(a)
        tab = m.get("atomic_spins") or []
        if not tab:
            reject["no Mulliken table associated with this force frame"] += 1
        elif len(tab) != len(a):
            reject["table does not cover every atom"] += 1

    checks = collections.Counter()
    rows = []
    for a in spin:
        m = meta(a)
        tab = m["atomic_spins"]
        idx = [int(t[0]) for t in tab]
        sym = [t[1] for t in tab]
        val = np.array([float(t[2]) for t in tab])
        s = a.arrays["local_moment"]
        ok_order = idx == list(range(1, len(a) + 1)) and sym == a.get_chemical_symbols()
        ok_column = np.allclose(val, s, atol=1e-6)
        src = by_id.get(frame_key(a))
        ok_geom = src is not None and np.allclose(src.positions, a.positions, atol=1e-8) and np.allclose(
            src.arrays["REF_forces"], a.arrays["REF_forces"], atol=1e-10)
        M = int(a.info["multiplicity"])
        ssum = float(s.sum())
        checks["order_and_elements_match"] += ok_order
        checks["column_equals_metadata_table"] += ok_column
        checks["geometry_forces_identical_to_collect"] += ok_geom
        checks["spin_sum_within_0.01_of_M-1"] += abs(ssum - (M - 1)) < 0.01
        Fn = np.linalg.norm(a.arrays["REF_forces"], axis=1)
        rows.append({
            "record_id": frame_key(a), "job_id": m.get("job_id"), "stage_index": m.get("stage_index"),
            "position": m.get("stage_frame_position"), "association": m.get("atomic_spin_association"),
            "split_group": m.get("split_group"), "charge": int(a.info["charge"]), "M": M,
            "spin_sum": round(ssum, 6), "abs_spin_sum": round(float(np.abs(s).sum()), 4),
            "n_negative": int((s < 0).sum()), "n_weak_lt1": int((np.abs(s) < 1).sum()),
            "s_min": round(float(s.min()), 4), "s_max": round(float(s.max()), 4),
            "pattern": m.get("fe_moment_pattern"), "E_eV": a.info["REF_energy"],
            "F_rms": round(float(np.sqrt((a.arrays["REF_forces"] ** 2).mean())), 5),
            "F_max": round(float(Fn.max()), 5),
            "ok_order": ok_order, "ok_column": ok_column, "ok_geom": ok_geom,
        })

    inv = {
        "collected_fe16_frames": len(allf),
        "usable_complete_local_spin_frames": len(spin),
        "rejected_frames": len(allf) - len(spin),
        "rejection_reasons": dict(reject),
        "independent_source_geometries_usable": len({r["split_group"] for r in rows}),
        "jobs_usable": len({r["job_id"] for r in rows}),
        "multiplicity_distribution": dict(sorted(collections.Counter(r["M"] for r in rows).items())),
        "charge_distribution": dict(collections.Counter(r["charge"] for r in rows)),
        "frame_position": dict(collections.Counter(r["position"] for r in rows)),
        "association": dict(collections.Counter(r["association"] for r in rows)),
        "pattern": dict(collections.Counter(r["pattern"] for r in rows)),
        "n_negative_sites": dict(sorted(collections.Counter(r["n_negative"] for r in rows).items())),
        "checks_passed_of_usable": dict(checks),
        "max_abs_spin_sum_error": max(abs(r["spin_sum"] - (r["M"] - 1)) for r in rows),
    }
    (RESULTS / "inventory.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")
    with open(RESULTS / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(inv, indent=2))


if __name__ == "__main__":
    main()
