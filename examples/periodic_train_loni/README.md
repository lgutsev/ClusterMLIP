# Periodic MACE on Fe3/θ-Al2O3(010) ± N2, stage A3 (ClusterMLIP)

Owner: ClusterMLIP. Generator: `examples/periodic_train_loni/make_loni_package.py` on branch
`feat/periodic-vasp-ingest`. Regenerate there, not here; `package.json` records the commit and every
file's SHA-256.

This is the first MACE trained on periodic DFT in ClusterMLIP. It tests whether the workflow works
on a supported cluster: periodic ingest, a job-level split, training conditioned on total spin and
charge, and evaluation. It is not a production model. The six A3 relaxations are one system at one
level of theory.

## What it runs

One `gpu2` array, one GPU per task, 12 h cap:

| task | variant | start | epochs | lr |
|---|---|---|---|---|
| 0 | `scratch_s1` | from scratch, seed 1 | 300 | 0.001 |
| 1 | `scratch_s2` | from scratch, seed 2 | 300 | 0.001 |
| 2 | `ft_mp0small` | fine-tune MACE-MP-0 small | 100 | 0.0001 |

Every task carries out these steps:

1. Re-reads package 27's six **A3** VASP jobs on LONI (`A3_slab_fix`, `A3_fe3_M11`, `A3_fe3_M9`,
   `A3_fe3_free`, `A3_fe3n2_side`, `A3_fe3n2_end`) with the pinned `cluster_mlip` snapshot in
   `runtime/`. Nothing from package 27 goes through git.
2. Splits by whole job: `A3_fe3_free` for validation, `A3_fe3n2_end` held out as test, and the rest
   for training. Released-spin frames whose moment is not near an integer are left out and counted
   in `jobs.csv`.
3. Refuses to train if any job is unfinished or flagged, or if the jobs do not share one level of
   theory. The level includes the A3 dipole correction, so the earlier A1/A2 frames (frozen-bottom
   slab, no dipole correction) cannot slip in.
4. Writes the `cluster-mlip train` campaign: ScaleShiftMACE, float64, total-spin and total-charge
   embeddings, force weight 10, stress weight 0. Runs it with `--restart_latest`.
5. Evaluates the trained model with `zero_shot.py` on all frames and on the test job, against the
   DFT labels.

Task 2 also tests whether mace-torch accepts new spin/charge embedding modules on a foundation
checkpoint pretrained without them. If it does not, only task 2 fails.

## Needs

- **Package 27 stage A3 finished.** All six `outputs/A3_*/DONE` must exist; otherwise the task exits
  2 at once. The script finds `../27_doe2_fe3_stageA/outputs` or `../batch*/27_doe2_fe3_stageA/outputs`;
  `export PKG27_OUTPUTS=/abs/path` overrides.
- **Env:** `/project/lgutsev/env/mace_env` (as packages 03/04); `PERIODIC_MACE_ENV` overrides. The
  log prints the mace, torch and CUDA versions. The desktop runs used mace-torch 0.3.16.
- **Foundation (task 2):** MACE-MP-0 small through `../model_paths.sh` (`resolve_smoke_mace`, master copy
  under `MLIP_PROJECT_STORAGE/MLIP_Foundational_Models/mace/`, hash-checked). Not copied.

## Run

```bash
cd /work/lgutsev/loni_smoke_tests && git pull
cd 32_clustermlip_periodic_train && sbatch run_periodic_train.slurm
```

`submit_smokes.sh` dispatches by script name and does not know `run_periodic_train.slurm` (as with 27's
`run_stageA.slurm`); the desk may add a case for it. `export.list` names `outputs/*` for the export.

If a task is preempted or runs out of time, submit it again; it resumes from its last checkpoint.
Expect hours, not minutes. On an 8 GB laptop GPU, one epoch over the A1/A2 frames took more than
4 minutes at batch size 2.

## Outputs to export

`outputs/<variant>/`:

- `DONE`, `exit_code` and `source.txt` (which package 27 directory was used, and the model path);
- `prepare.log`, `ingest_summary.json`, `jobs.csv` and `train_manifest.json`;
- the trained `periodic_*.model`;
- MACE `logs/` and `results/`;
- `report_all/` and `report_test/` (`zero_shot_report.json` and `zero_shot_table.md`).

The `preds_*` directories hold about 1 MB of predictions each. `work/`, which has the dataset
copies, checkpoints and compiled models, stays on LONI.

## Pass criteria (desk)

- All three variants have `DONE` and `exit_code` 0. Task 2 failing on the embedding question is a
  recorded result, not a package failure.
- MACE's log shows `total_spin` and `total_charge` counts equal to the frame counts. The generated
  `run.sh` guard stops a run that would train spin-blind.
- `report_test/zero_shot_table.md` exists for every finished variant.

## Hand-off

Route: agent `ClusterMLIP`, source `examples/periodic_train_loni/make_loni_package.py`. Check: read
`outputs/*/report_test/zero_shot_report.json` and `report_all/` in the ClusterMLIP repo.
Next step: the ClusterMLIP agent compares the two scratch seeds and the fine-tune against the zero-shot
foundation models on the held-out end-on N2 job.
