#!/usr/bin/env python3
"""Package 35 (proposed): electronic-root check on package 26's geometry matches.

Package 26 found geometry matches (<= 0.02 Å) between the legacy Gaussian collection
and the warehouses, and inside the legacy collection. Same charge/multiplicity and the
same route do not prove the same SCF solution. This script attaches each record's own
SCF energy to every same-(q, M) match and asks whether the energy difference is too
large for the geometric difference -- the signature of a different electronic root
(another broken-symmetry solution) or of a different method hidden behind a route
string.

Read-only. Inputs:
  --run       package 26 run directory (overlap.csv, extracted/seeds.extxyz)
  --campaigns warehouse root with <Warehouse>/extracted/seeds.extxyz
  --source    legacy .out root (for <S^2> of flagged legacy records)
Outputs in --output: summary.json, report.md, pairs_flagged.csv, bins.csv.

Pairs examined: same charge and multiplicity, and either (a) legacy vs a warehouse
record, or (b) legacy vs legacy from DIFFERENT source files (re-runs and copies;
neighbouring steps of one optimization are skipped).

Flag rule (energies in eV, per pair):
  candidate_distinct_root  |dE| >= 0.10 eV and aligned RMSD <= 0.005 Å
  large_dE_at_match        |dE| >= 0.50 eV at any matched RMSD (<= 0.02 Å)
A 0.005 Å RMSD move changes the energy by ~F*dx; reaching 0.1 eV would need coherent
forces of order 1 eV/Å on every atom, so such pairs are unlikely to be geometric.
Thresholds are reported, not tuned; bins.csv gives the full distribution.

<S^2> (before/after annihilation) is read for flagged legacy records by locating the
SCF Done line with that record's energy in its source file. It is supporting
evidence only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

HARTREE_EV = 27.211386245988
KV = re.compile(r'(\w+)=("((?:[^"\\]|\\.)*)"|\S+)')
SCF_DONE = re.compile(r"SCF Done:\s+E\(\S+\)\s+=\s+([-\d.DE+]+)")
S2 = re.compile(r"S\*\*2 before annihilation\s+([-\d.]+),\s+after\s+([-\d.]+)")
CAND_DE, CAND_RMSD, LARGE_DE = 0.10, 0.005, 0.50
RMSD_BINS = (0.001, 0.002, 0.005, 0.01, 0.02)
DE_BINS = (0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
MAX_S2_PAIRS = 5000


def read_headers(path: Path) -> dict[str, tuple]:
    """record_id -> (energy_eV|None, n_atoms, formula, charge, mult, source, route)."""
    out = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        while True:
            first = fh.readline()
            if not first:
                break
            first = first.strip()
            if not first:
                continue
            n = int(first)
            info = {m.group(1): (m.group(3) if m.group(3) is not None else m.group(2))
                    for m in KV.finditer(fh.readline())}
            for _ in range(n):
                fh.readline()
            e = info.get("legacy_energy_hartree")
            try:
                ev = float(e) * HARTREE_EV if e not in (None, "", "None", "nan") else None
            except ValueError:
                ev = None
            out[info["record_id"]] = (ev, n, info.get("formula", ""), info.get("charge"),
                                      info.get("multiplicity", info.get("spin")), info.get("source", ""),
                                      info.get("legacy_route", ""))
    return out


def bin_label(value: float, edges: tuple) -> str:
    for edge in edges:
        if value <= edge:
            return f"<={edge}"
    return f">{edges[-1]}"


def s2_for(source_root: Path, rel: str, energy_ev: float, cache: dict) -> tuple:
    if rel not in cache:
        path = source_root / rel
        try:
            cache.clear() if len(cache) > 64 else None
            cache[rel] = path.read_text(errors="replace")
        except OSError:
            cache[rel] = None
    text = cache[rel]
    if text is None:
        return (None, None, "source_unreadable")
    target = energy_ev / HARTREE_EV
    for m in SCF_DONE.finditer(text):
        if abs(float(m.group(1).replace("D", "E")) - target) < 2e-7:
            s = S2.search(text, m.end(), m.end() + 20000)
            return (float(s.group(1)), float(s.group(2)), "ok") if s else (None, None, "no_s2_after_scf")
    return (None, None, "energy_not_found")


def run(args) -> int:
    t0 = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    status = {"phase": "reading_seeds"}
    (out / "RUN_STATUS.json").write_text(json.dumps(status))
    stores = {"incoming": read_headers(Path(args.run) / "extracted" / "seeds.extxyz")}
    for wh in ("FenOm_Warehouse", "FenOm_Warehouse2", "General_Warehouse"):
        p = Path(args.campaigns) / wh / "extracted" / "seeds.extxyz"
        if p.is_file():
            stores[wh] = read_headers(p)

    counts = collections.Counter()
    bins = collections.Counter()
    flagged = []
    with (Path(args.run) / "overlap.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            counts["overlap_rows"] += 1
            if row["same_charge_multiplicity"] != "True" or row["match_type"] != "geometry_match":
                continue
            other_wh = row["other_warehouse"]
            if other_wh == "incoming" and row["incoming_source"] == row["other_source"]:
                counts["skipped_same_document"] += 1
                continue
            a = stores["incoming"].get(row["incoming_record"])
            b = stores.get(other_wh, {}).get(row["other_record"])
            kind = ("legacy_vs_legacy" if other_wh == "incoming" else f"legacy_vs_{other_wh}") + \
                   ("|same_route" if row["identical_raw_route"] == "True" else "|different_route")
            if a is None or b is None or a[0] is None or b[0] is None:
                counts[f"{kind}|no_energy"] += 1
                continue
            counts[kind] += 1
            de = abs(a[0] - b[0])
            rmsd = float(row["aligned_rmsd_A"])
            bins[(kind, bin_label(rmsd, RMSD_BINS), bin_label(de, DE_BINS))] += 1
            flag = ("candidate_distinct_root" if de >= CAND_DE and rmsd <= CAND_RMSD else
                    "large_dE_at_match" if de >= LARGE_DE else "")
            if flag:
                counts[f"{kind}|{flag}"] += 1
                flagged.append({
                    "kind": kind, "flag": flag, "formula": a[2], "charge": a[3], "multiplicity": a[4],
                    "n_atoms": a[1], "dE_eV": round(a[0] - b[0], 6), "dE_meV_atom": round(1000 * de / a[1], 3),
                    "aligned_rmsd_A": rmsd, "aligned_max_dev_A": float(row["aligned_max_deviation_A"]),
                    "incoming_record": row["incoming_record"], "incoming_source": row["incoming_source"],
                    "incoming_energy_eV": a[0], "other_warehouse": other_wh,
                    "other_record": row["other_record"], "other_source": row["other_source"],
                    "other_energy_eV": b[0], "incoming_route": a[6][:200], "other_route": b[6][:200],
                })
            if counts["overlap_rows"] % 1_000_000 == 0:
                status = {"phase": "scanning_overlap", "rows": counts["overlap_rows"], "flagged": len(flagged)}
                (out / "RUN_STATUS.json").write_text(json.dumps(status))

    # <S^2> for flagged pairs with a readable legacy side, largest |dE| first.
    flagged.sort(key=lambda r: -abs(r["dE_eV"]))
    cache: dict = {}
    src = Path(args.source)
    for r in flagged[:MAX_S2_PAIRS]:
        r["incoming_s2_before"], r["incoming_s2_after"], r["incoming_s2_status"] = \
            s2_for(src, r["incoming_source"], r["incoming_energy_eV"], cache)
        if r["other_warehouse"] == "incoming":
            r["other_s2_before"], r["other_s2_after"], r["other_s2_status"] = \
                s2_for(src, r["other_source"], r["other_energy_eV"], cache)
        else:
            r["other_s2_status"] = "warehouse_source_not_read"

    cols = ["kind", "flag", "formula", "charge", "multiplicity", "n_atoms", "dE_eV", "dE_meV_atom",
            "aligned_rmsd_A", "aligned_max_dev_A", "incoming_s2_before", "incoming_s2_after",
            "incoming_s2_status", "other_s2_before", "other_s2_after", "other_s2_status",
            "incoming_record", "incoming_source", "incoming_energy_eV", "other_warehouse",
            "other_record", "other_source", "other_energy_eV", "incoming_route", "other_route"]
    with (out / "pairs_flagged.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(flagged)
    with (out / "bins.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "rmsd_bin_A", "abs_dE_bin_eV", "pairs"])
        for (k, rb, eb), n in sorted(bins.items()):
            w.writerow([k, rb, eb, n])
    fe16 = [r for r in flagged if r["formula"] == "Fe16"]
    by_formula = collections.Counter(r["formula"] for r in flagged)
    summary = {
        "thresholds": {"candidate_dE_eV": CAND_DE, "candidate_rmsd_A": CAND_RMSD, "large_dE_eV": LARGE_DE},
        "records": {k: len(v) for k, v in stores.items()},
        "counts": dict(counts), "flagged": len(flagged), "flagged_fe16": len(fe16),
        "flagged_top_formulas": by_formula.most_common(25),
        "s2_status": dict(collections.Counter(r.get("incoming_s2_status", "not_checked") for r in flagged)),
        "runtime_s": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Electronic-root check on package 26 geometry matches", "",
             f"Same-(q, M) geometry matches examined, by kind: "
             + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()) if "|" in k and "flag" not in k
                         and "candidate" not in k and "large" not in k), "",
             f"Flagged: {len(flagged)} (Fe16: {len(fe16)}). Rule: |dE| >= {CAND_DE} eV at RMSD <= {CAND_RMSD} Å "
             f"(candidate distinct root), or |dE| >= {LARGE_DE} eV at any match.", "",
             "A flag is a candidate, not a verdict: a different method behind a different route string, "
             "an unconverged SCF (IOP(5/13=1) lets it continue), or a wrong energy-geometry pairing in the "
             "legacy parse can also produce it. <S^2> is supporting evidence only.", "",
             "Top formulas among flagged pairs: " + ", ".join(f"{f} {n}" for f, n in by_formula.most_common(15))]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "RUN_STATUS.json").write_text(json.dumps({"phase": "done", "flagged": len(flagged)}))
    print(json.dumps(summary, indent=2))
    return 0


def self_test() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "run" / "extracted").mkdir(parents=True)
        (d / "camp" / "FenOm_Warehouse2" / "extracted").mkdir(parents=True)
        (d / "src").mkdir()

        def xyz(rid, e, src):
            return (f'2\nProperties=species:S:1:pos:R:3 record_id="{rid}" source="{src}" formula="Fe2" '
                    f'charge=0 multiplicity=7 legacy_energy_hartree={e} legacy_route="# UBPW91" '
                    f'metadata="{{\\"orientation\\": \\"input\\"}}"\nFe 0 0 0\nFe 0 0 2\n')
        (d / "run" / "extracted" / "seeds.extxyz").write_text(
            xyz("a", -2524.0, "x/a.out") + xyz("b", -2524.0, "x/b.out") + xyz("c", -2524.01, "x/c.out"))
        (d / "camp" / "FenOm_Warehouse2" / "extracted" / "seeds.extxyz").write_text(xyz("w", -2524.00001, "w.out"))
        (d / "src" / "x").mkdir()
        for name, e in (("a", -2524.0), ("c", -2524.01)):
            (d / "src" / "x" / f"{name}.out").write_text(
                f" SCF Done:  E(UB-PW91) =  {e}     A.U. after   20 cycles\n"
                " S**2 before annihilation    12.0500,   after    12.0001\n")
        hdr = ("incoming_record,other_warehouse,other_record,incoming_source,other_source,match_type,"
               "aligned_rmsd_A,aligned_max_deviation_A,same_charge_multiplicity,identical_raw_route,decision\n")
        rows = ["a,incoming,c,x/a.out,x/c.out,geometry_match,0.001,0.002,True,True,r",   # 0.27 eV -> candidate
                "a,incoming,b,x/a.out,x/b.out,geometry_match,0.001,0.002,True,True,r",   # same energy
                "a,FenOm_Warehouse2,w,x/a.out,w.out,geometry_match,0.001,0.002,True,True,r",
                "a,incoming,b,x/a.out,x/a.out,geometry_match,0.001,0.002,True,True,r"]   # same document
        (d / "run" / "overlap.csv").write_text(hdr + "\n".join(rows) + "\n")
        ns = argparse.Namespace(run=str(d / "run"), campaigns=str(d / "camp"), source=str(d / "src"),
                                output=str(d / "out"))
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            run(ns)
        s = json.loads((d / "out" / "summary.json").read_text())
        flagged = list(csv.DictReader((d / "out" / "pairs_flagged.csv").open()))
        assert s["flagged"] == 1, s
        assert flagged[0]["flag"] == "candidate_distinct_root"
        assert flagged[0]["incoming_s2_status"] == "ok" and float(flagged[0]["incoming_s2_before"]) == 12.05
        assert s["counts"]["skipped_same_document"] == 1
        assert s["counts"]["legacy_vs_FenOm_Warehouse2|same_route"] == 1
    print("self-test passed")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--run")
    p.add_argument("--campaigns")
    p.add_argument("--source")
    p.add_argument("--output")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if not all((a.run, a.campaigns, a.source, a.output)):
        p.error("--run, --campaigns, --source and --output are required")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
