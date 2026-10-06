"""Verify that each converged stage's checkpoint still holds that stage's solution (read-only).

usage: python chk_inventory.py COLLECT_DATASET CAMPAIGN_DIR OUT_DIR [--formchk CMD]

For every stage endpoint (stage_frame_position "final" or "only") in COLLECT_DATASET/all.extxyz:
  * find the stage's `checkpoint` file under CAMPAIGN_DIR (by name; several hits are reported);
  * convert it with formchk into OUT_DIR/fchk/ (the .chk itself is never written);
  * compare the fchk's multiplicity, charge, SCF energy and geometry with the converged frame.
A checkpoint is usable as a rattle parent only if all four match: multiplicity and charge exactly,
energy within 1e-6 Ha, geometry within 1e-4 A after optimal rotation (checkpoints store Bohr).
Writes chk_inventory.csv and summary.json.
"""
import argparse, csv, json, subprocess
from collections import Counter
from pathlib import Path
import numpy as np
from ase.io import read
from scipy.spatial.transform import Rotation

HARTREE_EV = 27.211386245988
BOHR_A = 0.529177210903

ap = argparse.ArgumentParser()
ap.add_argument("dataset"); ap.add_argument("campaign"); ap.add_argument("out")
ap.add_argument("--formchk", default="formchk", help="formchk command (e.g. full path after module load)")
args = ap.parse_args()
out = Path(args.out); (out / "fchk").mkdir(parents=True, exist_ok=True)


def fchk_fields(path):
    lines = path.read_text(errors="replace").splitlines()
    vals = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        key = line[:40].strip()
        if key in ("Multiplicity", "Charge"):
            vals[key] = int(line.split()[-1])
        elif key == "Total Energy":
            vals[key] = float(line.split()[-1])
        elif key == "Current cartesian coordinates":
            n = int(line.split()[-1]); nums = []
            while len(nums) < n:
                i += 1; nums += [float(x) for x in lines[i].split()]
            vals["coords_A"] = np.array(nums).reshape(-1, 3) * BOHR_A
        i += 1
    return vals


def aligned_max(x, y):
    x0, y0 = x - x.mean(0), y - y.mean(0)
    R, _ = Rotation.align_vectors(x0, y0)
    return float(np.linalg.norm(x0 - y0 @ R.as_matrix().T, axis=1).max())


campaign = Path(args.campaign)
index = {}
for p in campaign.rglob("*.chk"):
    index.setdefault(p.name, []).append(p)
rows = []
for a in read(Path(args.dataset) / "all.extxyz", ":"):
    md = json.loads(a.info["metadata"]) if isinstance(a.info.get("metadata"), str) else {}
    if md.get("stage_frame_position") not in ("final", "only"):
        continue
    name = md.get("checkpoint", "")
    hits = index.get(name, [])
    row = {"job_id": md.get("job_id"), "M": int(a.info["spin"]), "charge": int(a.info.get("charge", 0)),
           "checkpoint": name, "found": len(hits), "path": str(hits[0]) if hits else "",
           "multiplicity_ok": "", "charge_ok": "", "energy_diff_Ha": "", "geometry_max_dev_A": "", "verdict": ""}
    if len(hits) != 1:
        row["verdict"] = "missing" if not hits else "ambiguous_name"
        rows.append(row); continue
    fchk = out / "fchk" / (Path(name).stem + ".fchk")
    if not fchk.exists():
        r = subprocess.run(args.formchk.split() + [str(hits[0]), str(fchk)], capture_output=True, text=True)
        if r.returncode != 0 or not fchk.exists():
            row["verdict"] = "formchk_failed"; rows.append(row); continue
    v = fchk_fields(fchk)
    de = v.get("Total Energy", float("nan")) - a.info["REF_energy"] / HARTREE_EV
    dg = aligned_max(v["coords_A"], a.positions) if "coords_A" in v and len(v["coords_A"]) == len(a) else float("inf")
    row.update(multiplicity_ok=v.get("Multiplicity") == row["M"], charge_ok=v.get("Charge") == row["charge"],
               energy_diff_Ha=de, geometry_max_dev_A=dg)
    row["verdict"] = "usable" if (row["multiplicity_ok"] and row["charge_ok"] and abs(de) <= 1e-6 and dg <= 1e-4) \
        else "mismatch"
    rows.append(row)

with (out / "chk_inventory.csv").open("w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
summary = {"stage_endpoints": len(rows), "verdicts": dict(Counter(r["verdict"] for r in rows)),
           "usable_by_M": dict(Counter(r["M"] for r in rows if r["verdict"] == "usable"))}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
