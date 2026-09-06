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
Freq is optional. Stable=Opt is not combined with Opt; Pop=Full is not required.
The inventory does not need to be rerun for launcher/collector updates.

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
geometry-only archive records, and does not recover partial failed ladders.

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
Guess=(Read,Always), failed calculations, and availability of final force
labels. Audit errors produce exit code 2. SCF convergence warnings are
reported without changing IOP settings or automatically rejecting labels.

These are filesystem snapshots, not scheduler queries: `Active?` means a
start marker without a matching finish; a killed job may leave the same
markers. `Waiting` includes both queued and unsubmitted calculations. Use
`squeue -u "$USER"` on QB3 to establish the live allocation state. Running
`squeue` on QB4 does not establish QB3 job status. Output files can change
while a snapshot is being read; repeat an audit after completion before
making a final training-data decision.
