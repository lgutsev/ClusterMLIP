# Gaussian reference workflow after the September review

The target is adsorption and reaction chemistry on Fe/Fe–O clusters. Keep
spin-polarized reference calculations and their charge/multiplicity labels.
Local spin populations are diagnostics; exact reproduction of each archived
AFM solution is not a prerequisite for launching the campaign. Compare
alternative starts on a representative set of active-site geometries before
considering an explicit local-spin model. Compare reaction energies and local
forces, rather than only total energy errors divided by cluster size.

The magnetite/water work provides a useful precedent for a structural potential
with electronic-state data curation:
https://arxiv.org/html/2408.11538v2 (particularly section II.2).

## Current preparation

The latest default spin route keeps UBPW91/6-311++G*, NoSymm, ordinary Opt,
UltraFine, and IOP(5/13=1,5/36=1,8/11=1). Successor stages use Guess=Read.
Freq is optional. Ordinary Opt applies only to minima and to records whose
curvature was never established: a saddle-labeled seed needs
Opt=(TS,CalcFC,NoEigenTest) and is now refused with a minimizing route (see
"Transition states launched as ordinary Opt"). Stable=Opt is not combined with
Opt; Pop=Full is not required. The inventory does not need to be rerun for
launcher/collector updates.

`run_one.sh` in spin campaigns now defaults to g09, accepts a
GAUSSIAN_COMMAND override, and runs beside the supplied input so relative
checkpoint paths resolve consistently. Load the Gaussian module first.

## Updating existing batch scripts

Run preparation where the Python environment is installed (QB4). Gaussian
submission on QB3 needs only the generated scripts and the Gaussian modules.
For an existing campaign, read `slurm_plan.json` and retain its jobs_per_batch
value and input ordering. Regenerating with the same batch map updates scripts
and input links while preserving Gaussian logs and checkpoints. Do not regenerate
scripts while those batch jobs are running. Use a fresh campaign to change the
batch map after outputs exist.

For a new QB3 campaign, an example allocation is four Gaussian jobs per node,
12 CPUs per job, 48 CPUs total:

```bash
cluster-mlip prepare-slurm campaigns/FenOm_Warehouse2/gaussian_spin_qb3_g09_v1 \
  --jobs-per-batch 30 --concurrent-jobs 4 --cpus-per-job 12 \
  --partition workq --time 48:00:00 --account loni_perovsk27 \
  --gaussian-command g09 --gaussian-module 'mvapich2 gaussian/g09-d01' \
  --job-name W2_g09 --scratch-root '/work/$USER/g09-scr'
```

Inputs must already declare `%nprocshared=12`. Memory per Gaussian process
must fit the node when multiplied by the concurrent job count.

On QB3, inside that campaign:

```bash
bash submit_gaussian_batches.sh --start 1 --end 1
bash gaussian_batch_status.sh
```

The head launcher is executed with bash, not submitted with sbatch. Resume,
worker completion, and batch status now require normal termination for every
explicit Link1 stage, reject error termination and nonzero recorded exit codes,
and detect a later calculation started after an earlier normal termination.
A failed later stage cannot make the whole ladder appear complete.

## Collecting labels

From the machine with ClusterMLIP installed:

```bash
cluster-mlip collect campaigns/FenOm_Warehouse2/gaussian_spin_qb3_g09_v1 \
  -o campaigns/FenOm_Warehouse2/dataset --frames all
```

The default `--frames final` collects one final force frame per manifest job
(or per spin stage). `--frames all` includes every parseable force-bearing
step in completed outputs, including optimization and IRC steps when their
energy, geometry, and force tables are present. It does not infer forces from
geometry-only archive records. `--frames converged` retains only the last force
frame from a stage that both completed its optimization and terminated
normally. Combine it with `--allow-partial` to recover those completed stages
from an interrupted ladder; that combination rejects all force tables from the
unfinished optimization. Retained frames record `source_job_complete`,
`spin_stage_normal_termination`, and `spin_stage_optimized` in their metadata.

Collection reads spin_jobs.csv, maps observed charge/multiplicity to an
unambiguous stage, and retains checkpoint/plan/source provenance plus available
S² values and atom-indexed Mulliken spin populations for each force frame. Frames from
one parent remain in one split. Duplicate output names are reported instead
of silently duplicating training labels. Unrelated scheduler logs are ignored
when a manifest is available. Inspect failed_outputs.tsv and label_report.md.
An empty collection exits nonzero and refreshes the report to zero frames.
SCF convergence warnings are retained as frame metadata, not silently discarded
or treated as proof of physically bad labels. IOP(5/13=1) remains unchanged.

`validate-spins --strict` continues to test archived-root coverage and lineage.
It no longer requires a stability calculation by default; add
`--require-stability` only when explicitly auditing stability-tested jobs.

## Validation limits

Regression tests execute generated shell checks and submission wrappers with
synthetic Gaussian logs, and exercise actual-format force headers, partial
ladders, manifest mapping, and grouped splitting. Gaussian09 and Slurm have
not been executed on QB3 in this review. Intermediate frames should be curated
for redundancy and chemical coverage before large training runs; the new
collector does not implement ground-state selection or local-spin dynamics.

## Per-batch progress and audit

With the ClusterMLIP environment active on QB4 (the shared files are readable
there), run:

```bash
W2=/ddnB/work/lgutsev/ClusterMLIP/campaigns/FenOm_Warehouse2/gaussian_spin_qb3_g09_v3
cluster-mlip campaign-status "$W2" --by-batch
cluster-mlip campaign-status "$W2" --audit --start 1 --end 10
```

The batch table counts Gaussian input files separately from their spin stages.
It shows completed inputs, failures, incomplete logs, unconfirmed activity,
unstarted inputs, and completed/planned stages. Unstarted inputs are mapped
from each batch's inputs.txt, so they remain visible before any log exists.
The job report includes the latest observed multiplicity, energy, optimization
step, output size and timestamp, and log path for manual inspection.

Reports are written under `monitoring/`: `batch_progress.csv`,
`job_progress.csv`, `audit_issues.csv`, and `summary.json`. Each invocation
replaces these reports for the selected range; use `-o DIRECTORY` to keep a
separate snapshot. Inspection does not modify inputs, checkpoints, or results.
No regeneration of the campaign is required to use these commands.

`--audit` additionally checks manifest hashes, CPU settings, contradictory
Guess=(Read,Always), failed calculations, availability of final force labels,
and whether each job's route searches for the stationary point its label
claims. Audit errors produce exit code 2. SCF convergence warnings are
reported without changing IOP settings or automatically rejecting labels.

These are filesystem snapshots, not scheduler queries: `Active?` means a
start marker without a matching finish; a killed job may leave the same
markers. `Waiting` includes both queued and unsubmitted calculations. Use
`squeue -u "$USER"` on QB3 to establish the live allocation state. Running
`squeue` on QB4 does not establish QB3 job status. Output files can change
while a snapshot is being read; repeat an audit after completion before
making a final training-data decision.

## Transition states launched as ordinary Opt

The completion-oriented audit above cannot see this class of error. A saddle
point launched with a plain `Opt` follows the energy downhill to the nearest
minimum: the job terminates normally, converges, and produces a clean final
force frame, and the geometry it reports is simply not the stationary point the
record exists for. Only the route says so. The same applies to the archived
`first_order_saddle` and `higher_order_saddle` records.

Both default routes carried a bare `Opt` for every seed regardless of
`config_type`, so any saddle-labeled record prepared before this change was
launched as a minimization. Diagnose it per campaign:

```bash
W2=/ddnB/work/lgutsev/ClusterMLIP/campaigns/FenOm_Warehouse2/gaussian_spin_qb3_g09_v3
cluster-mlip audit-routes "$W2"
```

Exit code 2 means at least one job must be relaunched. The console summary
counts each finding; `monitoring/route_audit.csv` has one row per input with its
label, the search its route requests, the search Gaussian actually executed,
the job state, and the imaginary-mode count of its last frequency analysis.
`monitoring/route_relaunch_candidates.csv` is just the rows that must be redone,
and `monitoring/route_audit.md` is the readable version of the same.

The structural label is read from the manifest's `config_type` column, falling
back to the label encoded in the generated filename. Spin campaigns prepared
before this change have no such column, so that fallback is the only label they
have; the audit reports which source it used per row in `config_type_source`.
Records whose curvature was never established (`warehouse_structure`,
`optimized_unverified`, `checkpoint_geometry`, IRC points), and every rattled
variant, are `unconstrained` and are never flagged.

Preparation now refuses to create this state: `prepare` gives saddle-labeled
seeds `Opt=(TS,CalcFC,NoEigenTest) Freq` and rejects a `--saddle-route` that is
not a saddle search, and `prepare-spins` rejects a minimizing `--route` while
any saddle-labeled seed is selected. `collect` refuses labels from a
mislaunched job unless `--allow-route-mismatch` is given, so an affected
campaign cannot quietly contribute a minimum labeled as a saddle to a dataset.

## Relaunching those jobs

First confirm on QB3 that the affected allocations have stopped. Then:

```bash
cluster-mlip relaunch-routes "$W2" --start 1 --end 30 --dry-run
cluster-mlip relaunch-routes "$W2" --start 1 --end 30 --assume-stopped
```

This is not a checkpoint restart, and that difference is the point. The
converged geometry and every checkpoint written from a collapsed run hold the
wrong stationary point, so `relaunch-routes` rebuilds each input from its
**original guess geometry** and renames every `%chk`/`%oldchk` to a fresh
`-routefixNN` name. The old checkpoints stay on disk, unread and unoverwritten.
Only optimizing stages are rewritten: a `Force` label stage keeps its route, and
for a spin ladder every stage becomes a TS search, so the corrected geometry
propagates down the chain rather than a collapsed one.

Nothing is deleted. Each invalid log and its `.rc`/`.status`/`.started`/
`.finished` markers become `NAME__before-routefixNN.*` in the same batch folder,
the replacement input is activated in that batch's existing `inputs.txt`, and
the superseded manifest rows are marked `submission_active=false` with a
`route_invalidated` reason and a `superseded_by_job_id` pointer. Inspect
`route_fix_plan.csv` for each job's before/after route, previous state, renamed
checkpoints, and rebuilt-geometry hash, and `skipped_route_fixes.csv` for
anything left alone. Manifest and `inputs.txt` snapshots go to
`route_fix_backups/`.

`--assume-stopped` is required for inputs with an unmatched `.started` marker;
do not pass it until `squeue` on QB3 confirms those jobs are gone.
`higher_order_saddle` records are skipped unless `--saddle-order N` is supplied,
because their intended order is recorded nowhere and defaulting them to a
first-order TS search would be a guess rather than a correction.

Do not run `prepare-slurm` again. Resubmit the same range with the existing head
launcher, then re-audit before collecting:

```bash
bash "$W2/submit_gaussian_batches.sh" --start 1 --end 30
cluster-mlip audit-routes "$W2"
```

## Restarting interrupted spin ladders

First confirm on QB3 that the original allocations have stopped. The restart
command works in place: it finds the first stage lacking both optimization
completion and normal termination, archives that attempt's log and marker
files in the same batch folder, copies its checkpoint there, and activates a
new shortened input in that batch's existing `inputs.txt`.

```bash
W2=/ddnB/work/lgutsev/ClusterMLIP/campaigns/FenOm_Warehouse2/gaussian_spin_qb3_g09_v3

cluster-mlip prepare-spin-restarts "$W2" \
  --start 1 --end 30 --assume-stopped --dry-run
cluster-mlip prepare-spin-restarts "$W2" \
  --start 1 --end 30 --assume-stopped
```

`--assume-stopped` is required when a killed Slurm allocation left an unmatched
`.started` marker. Do not pass it until `squeue` on QB3 confirms those jobs are
gone. The original input remains under `inputs/`; its interrupted log becomes
`NAME__before-restartNN.log` beside the replacement run. Each seed copy is
hashed in `spin_jobs.csv`; the new first stage uses `%oldchk` plus
`Geom=Checkpoint Guess=Read` and writes a distinct `%chk`. Inspect
`restart_plan.csv` for each original batch, first unfinished multiplicity,
source checkpoint, and number of remaining stages. If the unfinished stage's
checkpoint is absent, the tool can use the immediately preceding completed
stage checkpoint; it reports that exact choice in the plan. Manifest and
`inputs.txt` snapshots are kept together under the single `restart_backups/`
directory.

Do not run `prepare-slurm` again. The existing batch map and QB3 resource
directives remain active. Submit the same batch range with the existing head
launcher:

```bash
bash "$W2/submit_gaussian_batches.sh" --start 1 --end 30
```

After the retries finish, the one campaign manifest maps both archived and
current outputs. Collect its converged stages directly:

```bash
cluster-mlip collect "$W2" \
  -o /ddnB/work/lgutsev/ClusterMLIP/campaigns/FenOm_Warehouse2/dataset_spin_v1 \
  --frames converged --allow-partial
```
