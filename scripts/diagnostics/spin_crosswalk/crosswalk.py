"""Frame-level crosswalk of Mulliken spin tables: raw-log inventory vs two parsers.

usage: python crosswalk.py RAW_ROOT AUDIT_FRAMES.csv KIT_PROVENANCE.csv OUT_DIR

RAW_ROOT holds the Gaussian logs (slurm_batches/...). AUDIT_FRAMES.csv is the
`spin-audit` frames.csv (commit 09d3461); KIT_PROVENANCE.csv is `collect`'s
frame_provenance.csv from the local-spin kit. Neither parser is trusted: every
full atom-resolved Mulliken table is located in the raw text and classified by
what separates it from the force tables:

  in_step       after SCF Done s and before s's force table (same SCF, same geometry)
  post_final    after the LAST force table, with no new SCF Done or Link1 boundary
                in between and any intervening geometry block identical (<= 1e-6 A)
                to the geometry that SCF used (G09's final population analysis after
                "Stationary point found": same density, same geometry)
  ambiguous     anything else (a new SCF or geometry intervenes, no force table, ...)

"hydrogens summed into heavy atoms" tables are counted separately, never used.
A section = text between "Initial command" (Link1) markers. Nothing is copied
backwards along a trajectory: a table attaches to at most one force frame.
"""
import collections, csv, json, re, sys
from pathlib import Path

raw_root, audit_csv, kit_csv, out_dir = map(Path, sys.argv[1:5])
out_dir.mkdir(parents=True, exist_ok=True)

EVENTS = [
    ("section", re.compile(r"^ Initial command:")),
    ("geometry", re.compile(r"^\s+(Input|Standard) orientation:")),
    ("scf", re.compile(r"^ SCF Done:")),
    ("mulliken_summed", re.compile(r"^ Mulliken charges and spin densities with hydrogens summed")),
    ("mulliken", re.compile(r"^ Mulliken charges and spin densities:")),
    ("forces", re.compile(r"^ Center\s+Atomic\s+Forces \(Hartrees/Bohr\)")),
    ("stationary", re.compile(r"-- Stationary point found")),
    ("normal_end", re.compile(r"^ Normal termination of Gaussian")),
]


def parse_table(lines, start):
    """Rows of a full Mulliken table: (index, symbol, charge, spin)."""
    rows, i = [], start + 2  # header line + column line
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) != 4 or not parts[0].isdigit():
            break
        rows.append((int(parts[0]), parts[1], float(parts[2]), float(parts[3])))
        i += 1
    return rows


def coords(lines, n):
    out, i = [], n + 5
    while i < len(lines) and not lines[i].startswith(" ----"):
        p = lines[i].split(); out.append(tuple(float(x) for x in p[3:6])); i += 1
    return out


def inventory(path):
    lines = path.read_text(errors="replace").splitlines()
    events = []
    for n, line in enumerate(lines):
        for kind, rx in EVENTS:
            if rx.search(line):
                events.append((n, kind)); break
    frames = [k for k, (_, kind) in enumerate(events) if kind == "forces"]
    tables = []
    for k, (n, kind) in enumerate(events):
        if kind == "mulliken_summed":
            tables.append({"line": n + 1, "kind": "summed", "frame": None, "class": "summed_ignored"}); continue
        if kind != "mulliken":
            continue
        rows = parse_table(lines, n)
        before = [e for e in events[:k]]
        after = events[k + 1:]
        last_scf = max((j for j, (_, kd) in enumerate(before) if kd == "scf"), default=None)
        cls, frame = "ambiguous", None
        if last_scf is not None:
            between = [kd for _, kd in events[last_scf + 1:k]]
            nxt = next(((j, kd) for j, (_, kd) in enumerate(after) if kd in ("forces", "scf", "geometry", "section")), None)
            if "forces" not in between and "geometry" not in between and "section" not in between and nxt and nxt[1] == "forces":
                cls, frame = "in_step", frames.index(k + 1 + nxt[0])
            elif "forces" in between and "section" not in between:
                # this SCF's force table is already printed; any geometry printed since
                # must equal the geometry the SCF was computed at
                last_force_k = max(j for j, (_, kd) in enumerate(events[:k]) if kd == "forces")
                scf_geo = [events[j][0] for j in range(last_scf) if events[j][1] == "geometry"]
                later_geo = [events[j][0] for j in range(last_force_k + 1, k) if events[j][1] == "geometry"]
                ref = coords(lines, scf_geo[-1]) if scf_geo else None
                if ref and all(len(coords(lines, g)) == len(ref) and max(
                        abs(a - b) for u, v in zip(coords(lines, g), ref) for a, b in zip(u, v)) <= 1e-6
                        for g in later_geo):
                    cls, frame = "post_final", frames.index(last_force_k)
        tables.append({"line": n + 1, "kind": "full", "atoms": len(rows), "rows": rows,
                       "spin_sum": round(sum(r[3] for r in rows), 6), "frame": frame, "class": cls,
                       "after_stationary": any(kd == "stationary" for _, kd in before)})
    return {"force_frames": len(frames), "tables": tables,
            "normal_end": sum(kd == "normal_end" for _, kd in events),
            "sections": 1 + sum(kd == "section" for _, kd in events[1:])}


logs = sorted(raw_root.rglob("*.log"))
inv = {p.name: inventory(p) for p in logs}

audit = {}
with audit_csv.open(newline="") as h:
    for r in csv.DictReader(h):
        audit[(Path(r["gaussian_output"]).name, int(r["force_frame_index"]))] = r["spin_complete"].strip().lower() == "true"
kit = {}
with kit_csv.open(newline="", encoding="utf-8") as h:
    for r in csv.DictReader(h):
        kit[(Path(r["gaussian_output"]).name, int(r["force_frame_index"]))] = (
            r["local_spin_available"].strip().lower() == "true", r["atomic_spin_association"])

rows, reasons = [], collections.Counter()
for name, d in inv.items():
    by_frame = collections.defaultdict(list)
    for t in d["tables"]:
        if t["kind"] == "full" and t["frame"] is not None:
            by_frame[t["frame"]].append(t)
    for f in range(d["force_frames"]):
        ts = by_frame.get(f, [])
        a = audit.get((name, f)); kf = kit.get((name, f))
        row = {"log": name, "force_frame_index": f,
               "raw_in_step_tables": sum(t["class"] == "in_step" for t in ts),
               "raw_post_final_tables": sum(t["class"] == "post_final" for t in ts),
               "audit_spin_complete": a, "kit_in_dataset": kf is not None,
               "kit_spin_available": kf[0] if kf else None, "kit_association": kf[1] if kf else ""}
        raw_has = bool(ts)
        if a and kf and kf[0]: why = "both"
        elif a and not (kf and kf[0]): why = "audit_only:" + ("kit_dropped_frame" if not kf else "kit_no_spins")
        elif (kf and kf[0]) and not a: why = "kit_only:" + ("post_final" if row["raw_post_final_tables"] else "in_step" if row["raw_in_step_tables"] else "no_raw_table")
        elif raw_has: why = "raw_table_unused:" + ("kit_dropped_frame" if not kf else "neither_parser")
        else: why = "no_table"
        row["category"] = why; reasons[why] += 1
        rows.append(row)

with (out_dir / "raw_tables.csv").open("w", newline="") as h:
    w = csv.writer(h, lineterminator="\n")
    w.writerow(["log", "line", "class", "force_frame_index", "after_stationary", "atom_index", "symbol",
                "mulliken_charge", "mulliken_spin"])
    for name, d in inv.items():
        for t in d["tables"]:
            for idx, sym, q, sp in t.get("rows", []):
                w.writerow([name, t["line"], t["class"], t["frame"], t["after_stationary"], idx, sym, q, sp])
with (out_dir / "frame_crosswalk.csv").open("w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
table_classes = collections.Counter(t["class"] for d in inv.values() for t in d["tables"])
per_log = {n: {"force_frames": d["force_frames"], "sections": d["sections"],
               "full_tables": sum(t["kind"] == "full" for t in d["tables"]),
               "classes": dict(collections.Counter(t["class"] for t in d["tables"] if t["kind"] == "full"))}
           for n, d in inv.items()}
summary = {"logs": len(inv), "force_frames_raw": sum(d["force_frames"] for d in inv.values()),
           "audit_frames": len(audit), "audit_spin_complete": sum(audit.values()),
           "kit_frames": len(kit), "kit_spin_available": sum(v[0] for v in kit.values()),
           "raw_tables_by_class": dict(table_classes), "frames_by_category": dict(reasons)}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(out_dir / "per_log.json").write_text(json.dumps(per_log, indent=2) + "\n")
print(json.dumps(summary, indent=2))
