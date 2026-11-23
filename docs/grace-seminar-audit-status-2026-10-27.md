# GRACE audit status, 2026-10-27

Assessment: `docs/grace-seminar-audit-2026-10-27.md`.
Experiment: `experiments/foundation_transfer_pilot/PILOT_AUDIT_2026-10-27.md` and `pilot_config_2026-10-27.json`.
Technical handoff: `docs/grace-proposal-handoff-2026-10-27.md`.

## Changes

Added dated audit artifacts and a read-only evidence verifier. Existing seminar assessment/reporting code in `.claude/worktrees/grace-seminar` was reused as evidence, not duplicated or merged. Current dirty source changes remain untouched. No training, DFT/HPC, environment upgrade, checkpoint download, commit, push or proposal-draft edit occurred.

The revised pilot selects native MACE-OMOL as one conditional candidate beside the scratch control; it corrects the earlier falsification logic, separates reference uncertainty from method mismatch, specifies transitive split protection and separates DFT calibration from the sealed final test. The Fe–O–H 2026 primary reference was located and qualified. All thresholds/resource numbers are proposed.

## Verification

* `verify_archive.py --archive D:/MLIP_Work_Folder/Cluster_MLIP` completed. Aggregate results and source hashes: `experiments/foundation_transfer_pilot/archive_checks_2026-10-27.json`. Ninety archived frames, two models, rho=0.954959 from rounded CSV (recorded unrounded rho=0.954875), top-nine recall=3/9. All four archived dataset hashes match the training manifest. This is output reanalysis, not a new inference run or certified independent split.
* Separate worktree's `test_uncertainty_and_element_errors.py`: **9 passed** in 0.015 s. It tests existing unmerged reporting additions.
* New `tests/test_grace_archive_checks.py`: **2 passed** in 0.014 s, checking tied ranks, a known inverse-ranking example and detection of changed dataset hashes.
* Current-root full suite before new verifier tests: **324 tests run; 4 failures, 5 errors, 2 skipped** after supplying Git Bash and a writable temporary directory. Failures/errors are concentrated in Gaussian/Slurm shell integration. Bash invocations returned Windows process status 3221225794 instead of expected script codes; one completion assertion also failed. The shell environment prevents a clean verification, and these results do not establish a production-code regression. Do not call the full suite green. No production code changed in this task. Earlier attempts with Windows' unavailable Python alias/default temp and without Bash are superseded by this more informative result.
* Current train/evaluate CLI flags were inspected; actual native-checkpoint training/load remains unverified. The root renderer has known conditioning/key/Boolean issues, with repairs present in another worktree. Pilot commands remain gated on a reviewed source snapshot and a model smoke check.

## Handoff

**Delivered to the existing proposal chat through the authorized coordination tool.** `send_message_to_thread` returned thread ID `01a17869-cbf9-7150-9241-3a7bdb279543` without error. The message includes exact artifact paths, verified findings, limitations and the request to review method-consistency wording. This confirms acceptance by the messaging mechanism, **not** that the agent has read it, accepted recommendations or edited the proposal. No such acknowledgment is available: thread-history and status reads returned `thread/items/list is not supported yet`. No email or other outreach to people was sent.

## Next action

Review/reconcile the existing embedding guard and collection changes, then audit small-cluster data coverage and split independence. Establish a same-method reference sensitivity estimate and independent DFT labels before any claim of transfer or catalytic validation. Distillation/CALM-inspired work remains gated; delta-only, water/OH scope, microkinetics and the bounded nonadiabatic task are preserved.
