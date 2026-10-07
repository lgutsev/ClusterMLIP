#!/usr/bin/env python3
"""Build one periodic MACE training variant from finished VASP jobs (runs on LONI).

    python periodic_train.py --outputs PKG27/outputs --jobs A3_slab_fix,... \
        --valid-jobs A3_fe3_free --test-jobs A3_fe3n2_end --variant scratch --seed 1 --work work/scratch_s1

Steps: `vasp-ingest` of exactly the named jobs into <work>/dataset (split by job,
fractional-spin frames dropped), refusal unless every job is clean and the jobs
share one level of theory, then `cluster-mlip train` into <work>/campaign.
Prints the generated run.sh path on the last line. Uses the pinned
cluster_mlip snapshot in runtime/ (PYTHONPATH), standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cluster_mlip.training import TrainingConfig, write_training_campaign
from cluster_mlip.vasp_ingest import JobSplit, ingest_vasp_runs


def _names(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outputs", type=Path, required=True, help="package 27 outputs/ directory")
    p.add_argument("--jobs", required=True, help="comma list of job directories to use")
    p.add_argument("--valid-jobs", required=True)
    p.add_argument("--test-jobs", required=True)
    p.add_argument("--variant", choices=["scratch", "finetune"], required=True)
    p.add_argument("--foundation", help="foundation checkpoint path (finetune)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--max-epochs", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--forces-weight", type=float, default=10.0)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    jobs = _names(args.jobs)
    dirs = [args.outputs / job for job in jobs]
    missing = [str(d) for d in dirs if not (d / "vasprun.xml").is_file() or not (d / "DONE").is_file()]
    if missing:
        print(f"REFUSED: not finished (no vasprun.xml or DONE): {missing}", file=sys.stderr)
        return 2

    dataset = args.work / "dataset"
    summary = ingest_vasp_runs(
        dirs, dataset, cluster_elements={"Fe"}, drop_fractional_spin=True,
        split=JobSplit(valid_jobs=_names(args.valid_jobs), test_jobs=_names(args.test_jobs)),
    )
    print(json.dumps({k: summary[k] for k in ("jobs", "frames", "jobs_flagged", "splits")}, indent=1))
    if summary["jobs_flagged"]:
        print(f"REFUSED: flagged jobs {summary['jobs_flagged']}", file=sys.stderr)
        return 2
    if summary["mixed_levels"] or summary["potcar_conflicts"]:
        print(f"REFUSED: mixed levels {list(summary['levels_of_theory'])} or POTCARs "
              f"{summary['potcar_conflicts']}", file=sys.stderr)
        return 2

    extra = (
        f"--lr={args.lr}", f"--batch_size={args.batch_size}", f"--valid_batch_size={args.batch_size}",
        "--restart_latest",  # a resubmitted array task resumes from its last checkpoint
        "--save_cpu",
    )
    config = TrainingConfig(
        dataset_dir=dataset, output_dir=args.work / "campaign", run_name=f"periodic_{args.work.name}",
        mode=args.variant, seeds=(args.seed,), foundation_model=args.foundation or "small",
        forces_weight=args.forces_weight, max_num_epochs=args.max_epochs,
        patience=max(20, args.max_epochs // 5), extra_args=extra, force=True,
    )
    plan = write_training_campaign(config)
    print(json.dumps({k: plan[k] for k in ("mode", "periodic", "multiplicity_range", "label_routes")}, indent=1))
    print(args.work / "campaign" / plan["seed_runs"][0]["script"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
