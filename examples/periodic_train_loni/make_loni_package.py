#!/usr/bin/env python3
"""Generate the periodic MACE training package for the LONI dispatch desk.

    python examples/periodic_train_loni/make_loni_package.py DEST --package 32_clustermlip_periodic_train

Copies the job script, driver, evaluation script and a pinned snapshot of the
cluster_mlip modules they need (runtime/, standard library only), so the job
does not depend on any ClusterMLIP checkout on LONI. No DFT data and no model
weights are copied: the job reads package 27's outputs and the desk's master
MACE-MP-0 small checkpoint on LONI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

RUNTIME_MODULES = ("io.py", "models.py", "periodic.py", "vasp_ingest.py", "training.py")


def _lf_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("destination", type=Path, help="new directory; its name should be the package name")
    p.add_argument("--package", required=True, help="package name, e.g. 32_clustermlip_periodic_train")
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    dest = args.destination
    dest.mkdir(parents=True, exist_ok=False)

    for name in ("periodic_train.py", "run_periodic_train.slurm", "reactions_A3.json", "README.md"):
        _lf_copy(here / name, dest / name)
    _lf_copy(repo / "examples" / "periodic_zero_shot" / "zero_shot.py", dest / "zero_shot.py")
    runtime = dest / "runtime" / "cluster_mlip"
    runtime.mkdir(parents=True)
    (runtime / "__init__.py").write_text("", encoding="utf-8", newline="\n")
    for name in RUNTIME_MODULES:
        _lf_copy(repo / "src" / "cluster_mlip" / name, runtime / name)
    (dest / "logs").mkdir()
    (dest / "logs" / "README.txt").write_text("Slurm logs land here.\n", encoding="utf-8", newline="\n")
    (dest / ".gitignore").write_text("outputs/\nwork/\nlogs/*\n!logs/README.txt\n", encoding="utf-8", newline="\n")
    (dest / "export.list").write_text("outputs/*\n", encoding="utf-8", newline="\n")

    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "src", "examples"],
                           capture_output=True, text=True, check=False).stdout.strip()
    hashes = {f.relative_to(dest).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(dest.rglob("*")) if f.is_file()}
    (dest / "package.json").write_text(json.dumps({
        "package": args.package,
        "owner": "ClusterMLIP",
        "generator": "examples/periodic_train_loni/make_loni_package.py",
        "commit": commit + ("-dirty" if dirty else ""),
        "cluster": "LONI QB4",
        "partition": "gpu2",
        "array": ["scratch_s1", "scratch_s2", "ft_mp0small"],
        "data": "package 27 outputs A3_slab_fix, A3_fe3_M11, A3_fe3_M9, A3_fe3_free, A3_fe3n2_side, "
                "A3_fe3n2_end (read on LONI, not copied)",
        "model": {"foundation": "MACE-MP-0 small via model_paths.sh resolve_smoke_mace", "copied": False},
        "outputs": ["outputs/<variant>/"],
        "files_sha256": hashes,
        "submitted": False,
    }, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {dest} ({len(hashes)} files, commit {commit}{'-dirty' if dirty else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
