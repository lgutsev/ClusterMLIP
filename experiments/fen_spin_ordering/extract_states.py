"""Extract per-state spin labels for bare Fe_n (n = 2-25) from the Oct9 legacy documents.

    python experiments/fen_spin_ordering/extract_states.py \
        --source E:/Research_Archive/Research_June18_2025/2019/Oct9 \
        --data D:/MLIP_Work_Folder/Cluster_MLIP/fen_oct9_states \
        --report experiments/fen_spin_ordering/report

Source: the unpacked Oct9 archive (same bytes as General_Warehouse.zip, sha256 bd2d6d66...). Every parsed state
is written, at every level of theory, with its flags; ``clean`` marks the subset used for the spin-ordering test:
UBPW91/6-311+G* from the block's own route, block title or an unambiguous document title, spin sum = M - 1, no
disagreement with the warehouse's independent extraction, deduplicated.
Artifact sites, IOP(5/13=1), <S^2> excess and imaginary modes are recorded, not filtered, so the analysis can split
on them. Bulky outputs (extxyz, full CSV) go to --data, outside Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cluster_mlip.legacy_doc_states import deduplicate, document_text, parse_document_states  # noqa: E402

FE_FILE = re.compile(r"fe(\d+)(_[^.]*)?\.(doc|docx|log|out)$", re.I)
REFERENCE_LEVEL = "BPW91/6-311+G*"


def is_clean(s) -> bool:
    return (s.level == REFERENCE_LEVEL and s.level_source in ("route", "header", "document")
            and not {"spin_sum_mismatch", "convergence_failure_text", "warehouse_energy_mismatch"} & set(s.scf_flags))


def warehouse_cross_check(states, warehouse_extxyz: Path) -> dict:
    """Flag states whose geometry the General Warehouse extraction also has, at the same (q, M), but with an
    energy more than 5 meV away: the two independent parsers then disagree about which energy belongs where."""
    import numpy as np
    from ase.io import read

    def fp(pos):
        d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        return np.sort(d[np.triu_indices(len(d), 1)])

    ref: dict[int, list] = {}
    for a in read(str(warehouse_extxyz), ":"):
        if re.fullmatch(r"Fe\d*", a.get_chemical_formula()):
            ref.setdefault(len(a), []).append((fp(a.positions), int(a.info["multiplicity"]), int(a.info["charge"]),
                                               float(a.info["legacy_energy_hartree"]) * 27.211386245988))
    tally = Counter()
    for s in states:
        f = fp(s.positions)
        same = [r for r in ref.get(len(s.symbols), [])
                if np.abs(r[0] - f).max() < 2e-3 and r[1] == s.multiplicity and r[2] == s.charge]
        if not same:
            tally["not_in_warehouse_at_same_q_M"] += 1
        elif min(abs(r[3] - s.energy_ev) for r in same) <= 0.005:
            tally["agrees_within_5_meV"] += 1
        else:
            tally["energy_mismatch"] += 1
            s.scf_flags.append("warehouse_energy_mismatch")
    return dict(tally)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--warehouse", default="", help="General_Warehouse extracted seeds.extxyz, for the cross-check")
    args = p.parse_args(argv)
    from ase import Atoms
    from ase.io import write

    src, data, rep = Path(args.source), Path(args.data), Path(args.report)
    data.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    files = sorted((f for f in src.iterdir() if FE_FILE.fullmatch(f.name)),
                   key=lambda f: (int(FE_FILE.fullmatch(f.name).group(1)), f.name))
    states, skips, file_rows = [], Counter(), []
    for f in files:
        raw = f.read_bytes()
        report: dict = {}
        st = parse_document_states(document_text(raw), f.name, report)
        skips.update(report)
        states += st
        file_rows.append({"file": f.name, "sha256": hashlib.sha256(raw).hexdigest(), "states": len(st),
                          **{k: v for k, v in report.items() if k != "parsed"}})
    unique = deduplicate(states)
    cross = warehouse_cross_check(unique, Path(args.warehouse)) if args.warehouse else {}
    rows, frames = [], []
    for s in unique:
        clean = is_clean(s)
        rows.append({"state_id": s.state_id, "source": s.source, "block": s.block_index, "formula": s.formula,
                     "n": len(s.symbols), "charge": s.charge, "multiplicity": s.multiplicity,
                     "energy_ev": s.energy_ev, "s2": s.s2, "s2_excess": s.s2_excess, "level": s.level,
                     "level_source": s.level_source, "n_imaginary": s.n_imaginary,
                     "scf_flags": ";".join(s.scf_flags), "artifact_sites": ";".join(map(str, s.artifact_sites)),
                     "genuine_antiparallel": s.genuine_antiparallel, "spin_sum": round(float(s.spins.sum()), 5),
                     "max_abs_spin": round(float(abs(s.spins).max()), 4), "clean": clean,
                     "spins": json.dumps([round(float(x), 5) for x in s.spins]), "route": s.route})
        a = Atoms(s.symbols, positions=s.positions)
        a.arrays["mulliken_spin"] = s.spins
        a.info.update({"state_id": s.state_id, "source": s.source, "charge": s.charge, "multiplicity": s.multiplicity,
                       "REF_energy": s.energy_ev, "level": s.level, "level_source": s.level_source,
                       "clean": clean, "artifact": bool(s.artifact_sites), "scf_flags": ";".join(s.scf_flags) or "none",
                       "n_imaginary": -1 if s.n_imaginary is None else s.n_imaginary})
        frames.append(a)
    write(data / "states.extxyz", frames)
    for path, table in ((data / "states.csv", rows), (rep / "files.csv", file_rows)):
        keys = list(dict.fromkeys(k for r in table for k in r))
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(table)
    clean = [r for r in rows if r["clean"]]
    by_nq = Counter((r["n"], r["charge"]) for r in clean)
    summary = {
        "source": str(src), "files": len(files), "blocks": dict(skips), "states_parsed": len(states),
        "states_unique": len(unique), "clean_states": len(clean), "warehouse_cross_check": cross,
        "clean_by_n_charge": {f"{n}|{q}": c for (n, q), c in sorted(by_nq.items())},
        "levels": Counter(f"{r['level']}|{r['level_source']}" for r in rows).most_common(),
        "clean_flags": Counter(f for r in clean for f in r["scf_flags"].split(";") if f),
        "clean_with_artifact_site": sum(bool(r["artifact_sites"]) for r in clean),
        "clean_genuine_antiparallel": sum(r["genuine_antiparallel"] for r in clean),
        "clean_saddles": sum((r["n_imaginary"] or 0) > 0 for r in clean),
    }
    (rep / "extraction_summary.json").write_text(json.dumps(summary, indent=2, default=list), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("clean_by_n_charge", "levels")}, indent=2, default=list))


if __name__ == "__main__":
    main()
