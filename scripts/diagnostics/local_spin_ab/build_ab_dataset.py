"""Build the population-labelled A/B dataset from a `collect` dataset.

usage: python build_ab_dataset.py SRC_DATASET DEST_DATASET

Keeps only frames whose metadata carries a complete per-atom spin table
(`atomic_spins`: [index, symbol, value] for every atom, in atom order, symbols
matching), and writes the signed values exactly as supplied -- no clipping,
no sign flips -- as a per-atom `local_moment` column. Splits and split groups
are inherited from SRC, so model A (ignores the column) and model B
(`--local-moment-key local_moment`) train and test on identical frames.
Frames without a table are dropped, never zero-filled.
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from cluster_mlip.io import parse_extxyz_info_line  # noqa: E402

SPLITS = ("train.extxyz", "valid.extxyz", "test.extxyz", "all.extxyz")


def moments(info, atom_lines):
    rows = json.loads(info["metadata"]).get("atomic_spins") if "metadata" in info else None
    if not rows or len(rows) != len(atom_lines):
        return None, "no complete table"
    out = []
    for k, (row, line) in enumerate(zip(rows, atom_lines)):
        index, symbol, value = row
        if int(index) != k + 1 or str(symbol) != line.split()[0]:
            return None, f"atom {k + 1}: table row {row} does not match {line.split()[0]}"
        out.append(float(value))
    return out, None


def convert(src: Path, dest: Path) -> dict:
    lines = src.read_text(encoding="utf-8").splitlines()
    out, kept, dropped, reasons = [], [], 0, collections.Counter()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i]); header = lines[i + 1]; atom_lines = lines[i + 2:i + 2 + n]
        info = parse_extxyz_info_line(header)
        values, why = moments(info, atom_lines)
        if values is None:
            dropped += 1; reasons[why if why.startswith("no ") else "row mismatch"] += 1
        else:
            props = info.get("Properties", "")
            if "local_moment" in props or f"Properties={props}" not in header:
                raise ValueError(f"{src}: unexpected Properties in {header[:120]}")
            # The new column goes last, matching the value appended to each atom line.
            header = header.replace(f"Properties={props}", f"Properties={props}:local_moment:R:1", 1)
            out += [lines[i], header, *(f"{line} {v!r}" for line, v in zip(atom_lines, values))]
            meta = json.loads(info["metadata"])
            kept.append((meta.get("split_group"), int(info.get("multiplicity", info.get("spin"))),
                         meta.get("atomic_spin_association"), min(values), max(values)))
        i += n + 2
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {"kept": len(kept), "dropped": dropped, "drop_reasons": dict(reasons),
            "groups": dict(collections.Counter(k[0] for k in kept)),
            "multiplicities": dict(sorted(collections.Counter(k[1] for k in kept).items())),
            "association": dict(collections.Counter(k[2] for k in kept)),
            "moment_range": [min((k[3] for k in kept), default=None), max((k[4] for k in kept), default=None)]}


def main():
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    if dest.exists() and any(dest.iterdir()):
        sys.exit(f"{dest} is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    summary = {name: convert(src / name, dest / name) for name in SPLITS}
    summary["source"] = str(src.resolve())
    (dest / "ab_dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name in SPLITS[:3]:
        s = summary[name]
        print(f"{name:13} kept {s['kept']:4d} dropped {s['dropped']:4d} groups {len(s['groups'])} "
              f"M {s['multiplicities']} moments [{s['moment_range'][0]}, {s['moment_range'][1]}]")


if __name__ == "__main__":
    main()
