# Revised bounded transfer pilot, 2026-10-27

Status: reproducible specification; **not a training authorization or completed benchmark**. Supersedes the methodological recommendations of the earlier `SPEC_2026-10-27.md` in `.claude/worktrees/grace-seminar` for this audit. Preserve that record. Its conjunction of negative performance and an invalid embedding as a falsification rule is incorrect: invalid conditioning invalidates an arm; it is not evidence against transfer. Cross-functional disagreement is not reference noise.

## Scope and arms

The first physics comparison is gas-phase Fe3–Fe6, bare and N2-bound, plus H-bearing configurations **only where same-level labels and independent groups exist**. Include q=0 and observed +/-1 states only with verified electron parity, state labels and coverage. No claim that existing Fe16 tests cover this domain. O/H2O/OH transfer is a later separately validated extension, relevant to current proposal Aim 2. Supported delta-interaction data remain separate.

S: compact two-interaction, charge/multiplicity-conditioned scratch MACE, using the existing reference architecture. F: native-conditioned **MACE-OMOL-0**, recommended release asset `MACE-omol-0-extra-large-1024.model`, release `mace_omol_0`. The native channel and molecular/transition-metal remit justify this candidate over generic state-blind materials models; its size makes it a stringent cost comparison. Both pretraining and architecture differ, so the result tests practical transfer, not a pure causal effect of weights. A reset-weight copy of F would be a later ablation, not an extra pilot arm.

Target both arms to the audited Gaussian UBPW91/6-311++G* campaign level, same grid, spherical/cartesian convention and method extras. Public OMOL pretraining is a different hybrid-DFT reference (official MACE model table lists wB97M-VV10). Do not retain its energy zero or silently mix methods. Use identical training-only fitted E0s for the pilot and report their limitation; same-level isolated-atom anchors, if available, are preferable for later production stability. Do not estimate offsets from validation/test.

MACE-POLAR is a recorded alternative, not a second foundation arm: local records report Fe16 coordination failures. Smaller clusters may behave differently, but that must be measured. GRACE is deferred: a single-state arm is only sensible after verifying the exact checkpoint, all target elements, label method and input signature. Never pool multiplicities under a state-blind checkpoint.

## Gates before any allocation

1. Select one source snapshot that contains the current collection/spin-pair work **and** reviewed embedding fixes. Root `training.py` is not ready unchanged: explicit info-key mapping and accepted Boolean syntax matter. Do not switch the active environment or cherry-pick automatically.
2. Pin Python, MACE, torch, ASE, CUDA, source revision/dirty patch hash, hardware and checkpoint asset SHA256. An alias alone does not pin weights. Confirm the loader resolves the intended 1024-channel asset. Current `_foundation_family` recognizes `mace-omol`, while an arbitrary local checkpoint path is treated as generic: passing a path is not an equivalent shortcut.
3. Run two epochs on a tiny **training-only** subset, reload the saved model, confirm the native embeddings survive F, confirm S uses `charge` and `spin`, and inspect dataset feature counts. In this repository `spin` stores M, not S or M-1. Native checkpoint indexing must be traced in the pinned loader. Include same-geometry M/M-2 and q/q+1 checks plus finite forces and a finite-difference force check. State responsiveness is necessary, not proof of accuracy. Never change both q/M and geometry in this check.
4. Run the split/provenance audit below. If independent data are missing, stop at a negative coverage report. No pilot accuracy claim is available today.

## Data lock and leakage control

Build a manifest with frame ID/hash, parent, trajectory/campaign lineage, geometry family, charge, M, method/basis/grid, units, force-label geometry, convergence and label origin. A group is the **connected component** of parent/trajectory ancestry, geometry relatives across M/q, verified spin-pair links and future teacher-sibling ancestry. Do not substitute rounded coordinates alone for family grouping. Compare permutation/rotation-aligned geometry similarity across proposed splits (audit sensitivity at 0.01 and 0.05 Ang); review collisions manually. Keep all transitive relatives together even if splitting strata would be more balanced.

Freeze validation and test groups before choosing learning subsets. Use nested 25/50/100% subsets of training **groups**, seeds 11/23/37. Do not derive subset selection from test error. Hold out at least one size/composition/environment where coverage permits; report missing classes rather than promising full Cartesian coverage.

Create a genuinely independent, sealed DFT test set: proposed initial 60 labels on geometries from withheld parent families and separately generated state-fixed distortions/approach scans, not siblings of training trajectories. Reserve a separate 30-frame DFT calibration set for uncertainty-threshold choices. Final test is opened once; no selection, stopping or thresholds use it. No new DFT was run. If existing DFT provenance can establish this independence, reuse it; otherwise these labels are prerequisites.

For adsorption triples include consistently referenced Fe_nX, Fe_n and X endpoints with declared states; distinguish relaxed adsorption from vertical interaction energy. Barrier tests need verified reactant/TS/product and reaction-family mapping. Gas references must share the same method and force conventions. A naked H-bearing composition count does not establish H2 activation coverage.

## Matched budget and executable plan generation

Configuration: `pilot_config_2026-10-27.json`. For each fraction, identical split hashes, batches of 8, energy/force weights 1/10, float64, no stress. Use 100 fixed epochs and patience 101 for **both** arms initially, so optimizer-update counts match for equal datasets. Record actual updates, dropped batches and batch semantics; rerun or cap if they differ. Best checkpoint selection uses validation only. Prespecify lr S=0.001, F=0.0001; any tuning gets the same two-value search budget per arm. Report GPU-hours separately because equal updates are not equal compute. A later equal-wall-time analysis is secondary.

Run these PowerShell commands in a reviewed source snapshot; variables are explicit inputs. They generate plans and do not launch training:

```powershell
$py = 'C:\Users\lguts\micromamba\envs\defects\python.exe'
$env:PYTHONPATH = (Resolve-Path 'src').Path
$pilotDataset = 'D:\MLIP_Work_Folder\Cluster_MLIP\pilot_small_clusters\f100'
$pilotRuns = 'D:\MLIP_Work_Folder\Cluster_MLIP\pilot_small_clusters\plans'
& $py -m cluster_mlip.cli train $pilotDataset -o "$pilotRuns\S_f100" --run-name S_f100 --seed 11 --seed 23 --seed 37 --forces-weight 10 --max-num-epochs 100 --extra-arg=--lr=0.001 --extra-arg=--patience=101 --extra-arg=--ema_decay=0.99
& $py -m cluster_mlip.cli train $pilotDataset -o "$pilotRuns\F_f100" --run-name F_f100 --finetune --foundation-model mace-omol --seed 11 --seed 23 --seed 37 --forces-weight 10 --max-num-epochs 100 --extra-arg=--lr=0.0001 --extra-arg=--patience=101 --extra-arg=--ema_decay=0.99
```

Repeat with f025/f050 datasets and fresh output directories. These CLI options were checked against the current parser; the native model load and repaired renderer remain gates. No ellipsis is an executable argument. `mace_run_train` is invoked only by each generated `seed_N/run.sh`, using Bash on the authorized training host after gate review; do not execute unrepaired root scripts. Pin the alias resolution before those scripts run.

Evaluation, once a specific saved checkpoint and independent DFT file exist:

```powershell
$pilotTest = 'D:\MLIP_Work_Folder\Cluster_MLIP\pilot_small_clusters\independent_test.extxyz'
$pilotModel = 'D:\MLIP_Work_Folder\Cluster_MLIP\pilot_small_clusters\accepted_model.model'
& $py -m cluster_mlip.cli evaluate $pilotTest --model $pilotModel -o 'D:\MLIP_Work_Folder\Cluster_MLIP\pilot_small_clusters\evaluation' --device cpu
```

The current evaluator does not produce every metric below. Reuse the separate seminar worktree's element report/calibration functions after reviewed integration; endpoint/site reports require curated annotations. No fabricated CLI for those missing features is supplied.

## Metrics and proposed decision rules

Report counts of independent groups, frames, state classes and missing labels beside every metric. Bootstrap **groups**, with seed spread separately; three seeds are not three independent datasets.

| Metric | Proposed target; not an established tolerance |
|---|---|
| Same-composition relative total energies | MAE <=0.03 eV/cluster, plus RMSE/max and ordering. At 700 K, 0.03 eV changes a Boltzmann ratio by about 1.64; propagate measured errors to ensemble weights. |
| Vertical M-pair gaps | MAE <=0.05 eV/cluster; ordering only adjudicated for gaps exceeding the matched reference uncertainty interval. Near degeneracies remain unresolved. |
| Adsorption/reaction/barrier energies | MAE <=0.05 eV when endpoint labels exist; report per-cluster eV. At 700 K a 0.05 eV barrier error changes a simple rate factor by about 2.3. Missing endpoint mappings => unavailable. |
| Forces | Component MAE/RMSE, worst-atom vector error and tails by Fe/N/H/O and atom-level active-site/coordination class. Provisional RMSE <=0.1 eV/Ang and improvement over zero force for each populated element; near-zero baselines require absolute errors, not unstable ratios. |
| Regions and states | Minima, IRC/saddle, stretched bonds, dissociation/approach, high force, support vacancies/OH only if independently labeled; fine q/M and coverage, not just an aggregate. |

Measure method-consistent reference sensitivity on a proposed 12 matched geometries (representative states/regions), changing numerical convergence settings at fixed reference definition. Basis/functional changes are separate model-discrepancy tests. If reference uncertainty exceeds an acceptance tolerance, that property is unresolved; do not simply loosen the tolerance. Spin populations/oxidation states remain electronic-structure diagnostics.

**Transfer benefit:** proposed >=20% lower primary independent-test relative-energy MAE at a data fraction, with paired group-bootstrap evidence of improvement and no material loss of resolved spin ordering, adsorption or element/site force accuracy. Data-efficiency-only improvement is reported as such. **No demonstrated benefit:** improvements stay within reference/seed/group uncertainty, or chemistry-critical metrics degrade despite lower global error. **Invalid:** conditioning lost, methods mixed, leakage or incompatible checkpoint; repair before comparing. Failure at one data fraction is not a universal theorem about transfer.

## Distillation follow-up, gated

Teacher = accepted S or F checkpoint, validated only in named composition/q/M/PES domains. Student = smaller one-interaction MACE with the same verified state semantics. Generate short state-fixed MD, bounded rattles and N2/H approach scans within validated temperatures/bond ranges; reserve H2O/OH for later teacher validation. Select diverse parent groups using invariant features and state/environment quotas; include low-disagreement candidates in a DFT audit so common errors are observable. Unsupported states are outside the synthetic domain.

Retain **all original teacher-training DFT anchors**, with explicit loss weights and capped synthetic-to-DFT weight (proposed at most 4:1 total effective weight). Keep DFT validation/test anchors out of student training. Each synthetic frame records `label_origin=teacher`, teacher SHA256, dataset/sampling parents, q/M, energy/force units and `teacher:<sha>` method identity. Never copy a Gaussian route as synthetic label provenance. Current route-based checks alone do not enforce this contract; a DFT/synthetic mixing guard is a prerequisite, not a completed feature.

Report approximation error student-teacher, teacher error versus independent DFT, and student error versus independent DFT separately. Reject a faster student if it amplifies unsupported chemistry or changes ensemble/reaction conclusions beyond the preregistered uncertainty.

## Uncertainty follow-up

CALM in GRACE is not a drop-in MACE estimator. A practical **CALM-inspired** prototype can use frozen `MACECalculator.get_descriptors` invariants on ScaleShiftMACE, after checking descriptor sensitivity to q/M, rotations and layer selection. Fit per-element clusters/regularized covariance on training environments only; insufficient H/N sample sizes require reduced dimensions or a documented abstention. Fit calibration thresholds on the separate DFT calibration set. Do not claim the GRACE feature map or few-percent overhead carries over.

Compare against the existing worst-atom committee score and random/state-quota selection on independent DFT. Withhold coordination classes, entire composition/size groups, stretched bonds and same-geometry other-M states; include numerical/reference failures shared by all models. Report energy and force-error rank correlations, precision/recall of chemistry-relevant error exceedances, risk versus accepted coverage, false negatives and selection yield per DFT call. Thresholds and test set stay separate. The 94-element seminar simulation supplies no validation for these clusters.

## Resource estimates and timing

Proposed first stage: 2 arms x 3 seeds x 100 epochs at f100; stop for review before the 18-run learning curve. For N training frames the update budget per arm/seed is `100*ceil(N/8)` (verify actual loader behavior). Measure 10 steps and a two-epoch load test first. Extrapolate `steps * measured_seconds_per_step`, adding measured validation/checkpoint cost. Reserve an initial 0.5 GPU-hour smoke budget and up to 4 GPU-hours per run provisionally (24 GPU-hours f100; 72 for all fractions), not a measured prediction. Large OMOL may exceed memory; reduce both arms' batch size consistently or declare the run infeasible, recording the change.

Independent DFT: 60 test + 30 calibration + 12 repeats = 102 labels before extra adsorption endpoints. Provisional 0.3–1 node-hour/label gives 31–102 node-hours; magnetically difficult SCF cases can exceed that. Calibrate on a few representative jobs before requesting production time. Do not use earlier periodic-slab timings for these Gaussian labels. No HPC jobs were submitted.

Benchmark model loading, first call and warm-up separately. Then 10 warm-ups and >=100 timed force+energy calls, repeated over target sizes/states; GPU synchronize around timings. Record CPU/GPU, threads, model hash, dtype, requested/effective backend, accelerator fallback, graph/rebuild settings, batch size, memory and median/p95 latency. No webinar timing/error ratio is a forecast.
