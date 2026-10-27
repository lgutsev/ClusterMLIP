# Status: GRACE seminar follow-up (2026-10-27)

## What changed

- `evaluate` now writes `by_element.csv` and a "Forces by element" report section. Each element's
  force MAE/RMSE sits beside the RMSE of predicting zero force for that element's atoms
  (`element_force_errors` in `evaluate.py`).
- New `cluster-mlip calibrate-committee`. It compares committee disagreement with the
  committee-mean error on labeled held-out frames and reports Spearman ρ, top-fraction error recall
  and confident failures (`committee_error_calibration` in `active_learning.py`).
- New docs:
  - `docs/grace-seminar-assessment-2026-10-27.md`: the seminar-to-repository matrix, ranked
    actions and a verified-claims table.
  - `experiments/foundation_transfer_pilot/SPEC_2026-10-27.md`: the pilot specification, which is
    proposed only.
- README sections 6 and 7 document both additions.
- The proposal hand-off was written outside the repo, at
  the hand-off file in `D:\MLIP_Work_Folder\Cluster_MLIP\grace_seminar_2026-10-27\`. It is
  **prepared, not delivered**, because no proposal session was running.

Everything is on branch `docs/grace-seminar-assessment` (worktree `.claude/worktrees/grace-seminar`,
from `main` ae1a7e7). Nothing was committed or pushed. The dirty `feat/irc-force-labels` checkout was
not touched.

## What was tested

- Unit tests: `tests/test_uncertainty_and_element_errors.py` (9 tests). Full suite: 314 passed,
  2 skipped.
- Real run: `calibrate-committee` and `evaluate` on existing Fe16 checkpoints (`long_B_s1/s2`, 90
  held-out frames), CPU, 46 s. The single-model force MAE (19.3 meV/Å) matches the experiment's own
  per-class numbers. Results are in assessment §3; outputs are in
  `D:\MLIP_Work_Folder\Cluster_MLIP\fe16_v1\13_seminar_checks\`.

## Still unverified

- Any transfer-learning benefit for our chemistry. The pilot has not run, and its dataset and
  independent DFT test set do not exist yet.
- Whether a MACE fine-tune keeps the charge/spin embeddings. That needs the guard on
  `fix/train-embedding-model-check`, which is unmerged.
- GRACE on our data: no checkpoint was downloaded (ASL licence) and nothing was run zero-shot.
- A label-noise floor for UBPW91 forces. The acceptance thresholds in the spec are proposed.
- Seminar-only claims: GRACE warp at 170× / 1.9×, the B–W table, and the Fe–O–H Adv. Sci. 2026 paper.

## Next most useful action

Merge the saved-model embedding guard and the IRC/local-spin collection code (both waiting on the
user). Then collect the Fe3–Fe6 (N2, H) UBPW91 pilot dataset on qbd with unique-geometry splits.
Every later comparison depends on that dataset.
