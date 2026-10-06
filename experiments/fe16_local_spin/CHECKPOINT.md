# Checkpoint 2026-10-05 ~19:10 — Fe16 local-spin hypothesis test

Branch `test/fe16-local-spin-message-passing` (worktree `.claude/worktrees/fe16-local-spin`, from origin/main 23fa604;
NOT pushed, do not merge). Commits: 86c0c9a cherry-pick of `--local-moment-key` (c1c8c25), a2def6f experiment scripts.
Python: `C:/Users/lguts/micromamba/envs/defects/python.exe`. Outputs: `D:/MLIP_Work_Folder/Cluster_MLIP/fe16_v1/12_local_spin_mp/`
(data/train.extxyz 60 frames = train = valid, data/heldout.extxyz 90; runs/<regime>_<A|B|C>_s<seed>/).

## Running (background, sequential: `runs/run_all.sh`, resumable — skips runs with `exit_code`)
fixed_{A,B,C}_s{1,2}: lr 1e-3, fw 10, 400 epochs, ~6 s/epoch (GPU shared with the other session's LOFO fold3 run).
If killed: rerun `bash D:/.../12_local_spin_mp/runs/run_all.sh` (delete the partial run dir's logs/results/checkpoints first).

## Done
- s1 inventory: 150/1251 frames usable (1101: no Mulliken table; only first/final opt frames are printed). q=0 only;
  M 47:5, 49:30, 51:55, 53:60; 10 source geometries, 91 jobs. All 150 pass order/element/column/geometry checks;
  sum s_i = M-1 within 2e-6 (trivially, Mulliken). 125 frames have exactly 1 antiparallel Fe, 25 none.
- s2 pairs (11,175 pairs, permutation+rotation aligned): 563 same-(q,M) pairs with RMSD<0.1 Å — ALL same spin pattern
  (D_s<0.25). Closest different-pattern same-M pair: 0.79 Å. Same geometry, M vs M-2: D_F 200–370 meV/Å (B sees M).
  => criterion 1 (near-identical geometry, same M, different spin, different force) NOT met: no such pairs exist.
- s3 hidden switches: 1158 adjacent opt steps, |dE + work| max 0.028 eV, <S^2> jumps max 0.004 => no hidden state
  switches. First vs final of a stage: 12/57 have one sign change, all weak sites (|ds| <= 0.68).
- s8 examples -> results/examples.md.
- ref regime (old failing settings lr 5e-3, fw 100, 60 ep), train-set final: E MAE / F MAE
  A 41.2 / 56.4, B 42.8 / 54.6, C 19.0 / 54.3. All stuck on the zero-force plateau; C's energy gain is likely the
  geometry-independent embedding-readout shortcut (per-frame offsets keyed on s_i) — check with the shifted-s control.

## Next
1. When all fixed runs have exit_code: `python s6_evaluate.py` (predictions + shifted-s control + pair diffs), then
   `python s7_report.py` (curves, comparison.md) — set PYTHONIOENCODING=utf-8.
2. Copy nothing large into git; results/*.md/json/csv/png are small and go in the commit.
3. Write README.md (hypothesis, implementation: MACE 0.3.16 adds the per-atom continuous embedding to node_feats
   before the first interaction -> s_i is in h_i^(0) and in every message; inventory; diagnostic; table; examples;
   verdict). Leaning NOT SUPPORTED for forces unless fixed C memorizes forces qualitatively better than fixed B.
   State whether first-class local spin is justified (likely: not on current data; would need same-geometry,
   same-M alternative broken-symmetry SCF labels).
4. Commit; do not push or merge without asking.

## Update 2026-10-06
- The session-bound runner was killed by the 2 h background limit. fixed_A_s1 finished; fixed_B_s1 was cut at epoch 120
  (kept as runs/aborted_fixed_B_s1_ep120). Runner relaunched DETACHED (PowerShell Start-Process of Git bash running
  run_all.sh, appending to run_all.log); it resumes from fixed_B_s1. Check: `ls runs/*/exit_code`.
- Interim (train set, EMA): fixed_A_s1 final ep395: E MAE 19.5 meV/atom, F MAE 55.4 meV/Å (no force learning in
  400 epochs: A cannot separate same-geometry frames at different M, whose forces differ by 200-370 meV/Å).
  aborted fixed_B_s1 at ep120: E 18.3, F 33.3 and still falling. Global M clearly matters; B vs C still open.
