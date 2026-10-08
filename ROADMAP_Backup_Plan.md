# ROADMAP: Fe16 Backup Plan

Updated: 2026-10-06.

## Decision and scope

Keep the local-spin experiment as a recoverable experimental branch. Continue with the standard charge/multiplicity-conditioned training strategy for current work. This document preserves contingency ideas; it does not authorize new training, Gaussian submissions, a production architecture change, or merging the experimental branch.

Do not launch the diagnostics below merely because they are listed. Reopen them when a specific, reproducible failure justifies the cost.

## Current evidence (provisional, reported by the local worker)

The local experiment uses 60 training frames, repaired settings including learning rate 0.001, and 400 epochs. Reported seed-1 final-epoch training errors:

| Model | Inputs | Energy MAE (meV/atom) | Force MAE (meV/Å) |
| --- | --- | ---: | ---: |
| A | Geometry and elements | 19.5 | 55.4 |
| B | A + total charge and multiplicity | 5.1 | 4.3 |
| C | B + per-Fe signed local moments | 4.3 | 5.0 |

B's reported force MAE falls from 39.9 at epoch 100 to 16.7 at 200, 6.9 at 300, and 4.3 at 400. C follows a similar curve and does not improve seed-1 force memorization.

These figures come from the user's worker report, not an independent checkpoint evaluation in this commit. Seed 2, the 90-frame held-out evaluation, and the shuffled-moment control were still pending in the latest supplied report.

Interpretation:
- The tested global-state model can memorize this subset. Missing local spin is therefore not required to explain the original failure to memorize it.
- Repaired optimization settings and adequate training duration are the leading explanation for the earlier stall. Their individual contributions have not been isolated; settings beyond learning rate must also be recorded.
- This does not establish generalization, eliminate magnetic effects, or prove that geometry and multiplicity uniquely determine local spin.
- Same-geometry examples at different multiplicity reportedly differ in forces by 200–370 meV/Å. The labelled subset has no near-identical same-charge/same-multiplicity examples with different spin patterns; closest mismatched patterns reportedly occur around 0.79 Å RMSD.
- Stable spin expectation values and small energy–force residuals are consistency checks, not proof that electronic-state switches never occur.

Finish and archive existing results when available. Do not expand the training campaign for this roadmap.

## Preserve the actual experimental branch

Reported branch: `test/fe16-local-spin-message-passing`.

Reported local worktree: `.claude/worktrees/fe16-local-spin`.

Reported checkpoint: `experiments/fe16_local_spin/CHECKPOINT.md`, with a copy at
`D:/MLIP_Work_Folder/Cluster_MLIP/fe16_v1/12_local_spin_mp/CHECKPOINT.md`.

Reported commits include `95520c1` and the later checkpoint commit `aa884fd`. At roadmap creation, neither the experimental branch nor `aa884fd` was available through GitHub. The existing remote `diag/fe16-baseline-training` branch is not a substitute for the local experiment.

From the actual Windows worktree, inspect status, branch, latest commits and remote, then push the real branch without forcing:

```bash
git status --short
git branch --show-current
git log -5 --oneline
git remote -v
git push -u origin test/fe16-local-spin-message-passing
```

Confirm the branch points to the intended latest experiment commit. Do not stage unrelated changes, reset the dirty `feat/irc-force-labels` checkout, or manufacture an empty branch from main under the experiment's name. This roadmap commit does not preserve unpushed implementation code.

Keep large checkpoints out of git. Record durable locations, hashes, dataset inventories, environments, configurations and exact resume commands in the experiment checkpoint. Preserve optimizer and scheduler state as well as weights.

## Reopening gates

| Observed snag | First useful action |
| --- | --- |
| Force training stalls again | Check training mechanics, update count, LR history and resume integrity |
| Training fits, held-out performance is poor | Audit provenance-separated splits, coverage and state-specific errors |
| Mixed-state data fit poorly but individual states fit | Run matched single-state/mixed-state controls |
| Reliable same-(Q,M) states have distinct local spin and forces | Reopen explicit local-spin representation tests |
| Graph coverage may exclude relevant interactions | Measure receptive field, then test whole-cluster coverage |
| MACE cannot memorize an audited smooth subset | Try an independent global force learner |
| Spin-only descriptors fail and reliable local charges exist | Test charge/spin descriptors on a common labelled subset |

## 1. Training mechanics and optional LONI migration

Before modifying the architecture:
- Reproduce checkpoint metrics with the evaluator; use the same frames and checkpoint-selection rule.
- Verify units, force sign, atom correspondence, label keys, loss normalization and geometry-dependent gradients.
- Report the zero-force baseline on the exact same subset, force magnitudes, energy range and state distribution.
- Record seed, initialization, dataset hashes, software versions, parameter count, batch size, epochs and optimizer updates.
- Inspect optimizer/scheduler state after resume and verify a short save/resume run.
- Compare train and validation curves; distinguish insufficient optimization from overfitting or unseen-state failure.

For future sustained training, prefer a scheduled LONI GPU job if the laptop's sleep/session interruptions remain limiting. This is a future infrastructure task, not a submission instruction. Inspect current dispatch procedures and GPU queue constraints before generating a job; do not invent scheduler directives. Use an isolated working runtime, a short train/load/evaluate smoke test, measured time per update, periodic restartable checkpoints and suitable walltime. Heavy work belongs on compute nodes.

Do not transplant “400 epochs” from 60 frames to a much larger dataset as a convergence guarantee. Compare update counts and validation behavior, and retain the best validation checkpoint.

## 2. Complete the local-spin comparison if it is reopened

Use identical data, architecture, loss settings, seeds and training budgets:
- A: geometry/elements.
- B: A + global charge/multiplicity.
- C-real: B + correctly assigned continuous signed moments.
- C-shuffled: C with a fixed reproducible permutation of moment values among Fe atoms in each frame.

Shuffling preserves the per-frame moment distribution but breaks site assignment. It is a useful control, not a definitive exclusion of every fingerprint or geometry-correlated shortcut.

Report force MAE/RMSE, energy errors and convergence for both seeds, separately for training and held-out data. Verify moments actually enter message passing.

Conclusions should remain narrow:
- C-real substantially improves forces over B and shuffled C: evidence for useful site-resolved information.
- Energy alone improves: insufficient evidence of magnetic force learning.
- B and C both memorize equally well: local moments do not resolve a memorization limitation on this subset.
- Lack of distinct same-(Q,M) near-geometry states: ambiguity is not demonstrated by this dataset.

## 3. Explicit spin–distance interactions

Trigger: a reproducible magnetic-state-specific failure remains after sound baseline training.

Inspect MACE implementation and verify the magnetic HDNNP/sACSF and SpinGNN papers before coding. Published methods are motivation; a MACE adaptation is our proposal.

Try minimal Fe–Fe edge features such as `s_i*s_j`, `abs(s_i-s_j)`, and `abs(s_i+s_j)`, coupled to distance-dependent messages or interaction terms. Do not manually enumerate spin chains. Specify non-Fe moment handling and never silently fill missing values.

Compare B, node-moment C, correct spin edges and shuffled spin edges on identical data and seeds. Record added parameters and unavoidable architectural differences.

Verify force sensitivity to edge features, atom permutation equivariance, spatial rotation behavior, global spin-reversal invariance for the collinear no-field/no-SOC case, and force agreement with finite differences at fixed supplied moments. Improvement supports a representation limitation; it does not prove hidden electronic-state ambiguity.

## 4. Single-state/single-trajectory overfit

Trigger: mixed-state force fitting remains difficult.

Choose approximately 50–100 audited frames from one composition, charge and multiplicity, preferably one continuous trajectory with consistent reference route. Fit the repaired baseline and report force scales, geometric range and zero-force error. Include a matched-size mixed-state comparison if practical.

One easier trajectory fitting better is not sufficient to blame state mixing. Charge/multiplicity do not uniquely identify broken-symmetry states. If controlled comparisons support interference, consider a small shared-network/state-head diagnostic; defer production redesign.

## 5. Whole-cluster coverage

Trigger: measured graph connectivity or effective message-passing coverage is insufficient.

Measure the existing receptive field first. Compare against a cutoff connecting every atom in each chosen cluster, on the same dataset and seeds. Record edge counts, runtime and other differences.

Improvement supports a coverage limitation in the implementation; it does not uniquely identify long-range magnetism. If the graph already provides whole-cluster communication, explain what changing cutoff actually changes before running it.

## 6. Independent global force learner

Trigger: MACE still cannot memorize an audited, smooth fixed-state subset.

Try GDML/sGDML after checking package suitability and permutation handling. Keep composition, charge and multiplicity fixed and check electronic-solution consistency. Begin with training-set memorization, then trajectory-separated evaluation where possible.

A successful independent fit demonstrates learnability on that subset and motivates examining MACE formulation. Failure of both models calls for checking conditioning, sampling, implementations and labels; it does not prove Gaussian labels are wrong.

## 7. Local charge and spin together

Trigger: previous tests remain inconclusive and reliable atom-resolved populations already exist.

On one common fully labelled subset compare B, B+spin, B+charge, and B+charge+spin. Use populations from the same geometry and electronic solution as energies/forces; do not mix definitions or impute missing labels.

These are supplied electronic descriptors. Successful fitting is not yet a geometry-only MD potential: inference needs a population predictor or a self-consistent electronic-variable model.

State the force convention explicitly. With supplied populations, forces are position derivatives at fixed populations. If populations subsequently depend on geometry, the chain-rule term need not vanish unless the learned energy is stationary with respect to those variables. Do not assume fixed-population and total derivatives are equivalent.

## 8. Targeted reference data, only after separate authorization

If existing data cannot identify magnetic ambiguity, propose a small fixed-geometry study: roughly 5–10 geometries and 5–10 alternative broken-symmetry guesses per geometry, holding charge and multiplicity fixed.

Alternative guesses may converge to the same final electronic state. Verify distinct solutions and atom-resolved patterns before comparing energies/forces. Prepare inputs and a scientific rationale only under an appropriately scoped subsequent task; do not submit jobs from this roadmap.

Constrained magnetic datasets and magnetic derivative targets may ultimately be useful, but equilibrium Mulliken labels alone are not such a dataset. Do not infer magnetic forces from nuclear force tables.

## Literature leads to verify before implementation

These links are inherited research leads, not independently verified citations for this roadmap commit. Read the primary papers and record their exact scope and assumptions before adapting them.

- Magnetic HDNNP/sACSF: https://doi.org/10.1038/s41524-021-00636-z
- SpinGNN: https://doi.org/10.1103/PhysRevB.110.104427
- Locality discussion: https://doi.org/10.1140/epjb/s10051-021-00156-1
- Electronic-population model lead: https://pubmed.ncbi.nlm.nih.gov/32502350/
- Charge redistribution model lead: https://pubmed.ncbi.nlm.nih.gov/40856241/
- sGDML lead: https://pubmed.ncbi.nlm.nih.gov/30901990/
- Magnetic MTP lead: https://doi.org/10.1038/s41524-022-00696-9
- Constrained magnetic/magnetic-force lead: https://doi.org/10.1016/j.commatsci.2024.113331
- MagNet lead: https://doi.org/10.1007/s44214-024-00055-3

## Deliverables and stopping rule for any reopened diagnostic

Work on an isolated experimental branch, preserving unrelated worktrees. Record reproducible commands, hashes, inventories, curves, a comparison table, runtime, seed variation, concrete prediction examples, verified literature and the smallest justified next action.

Use conclusions “supported by this diagnostic”, “not demonstrated”, or “inconclusive”, attached to a specific tested claim. Stop when a result identifies the next useful action. Avoid broad hyperparameter sweeps and do not merge experimental architectures into main without a separate decision.
