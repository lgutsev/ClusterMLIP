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
normally. A fixed-geometry `Force` stage (an IRC frame or higher-order
candidate, see below) has no optimization to complete: it is retained when it
terminated normally with an `SCF Done` energy and a force table whose geometry
matches the archived input coordinates (rotation- and translation-invariant,
1e-3 A RMS); a Force stage whose atoms moved is rejected. Combine
`--frames converged` with `--allow-partial` to recover completed stages from
an interrupted *active* ladder; that combination rejects all force tables from
the unfinished optimization. Completed stages of an *archived* attempt
(inactive manifest rows left by `prepare-spin-restarts`, not invalidated) are
recovered under `--frames converged` alone, and when an archived attempt and
its rerun both finished the same stage only the active attempt's label is kept
(the dropped duplicates are listed in `superseded_labels.tsv`). Outputs that
`relaunch-routes` archived as `route_invalidated` are always rejected, even
with `--allow-route-mismatch`. Retained frames record `source_job_complete`,
`spin_stage_normal_termination`, `spin_stage_optimized`, `force_label_kind`
(`optimization` or `fixed_geometry_force`) and, for Force stages,
`fixed_geometry_check` in their metadata.

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

For a large range, submit the inspection itself to a compute node so the login
node's process-duration policy cannot kill it:

```bash
cluster-mlip campaign-status "$W2" --audit --start 1 --end 130 --sbatch
```

This requests one CPU for four hours on `single` under `loni_perovsk27` and
writes both the generated sbatch file and `campaign-status-JOBID.stdout` /
`.stderr` under `monitoring/`. Override those defaults with `--sbatch-time`,
`--sbatch-partition`, and `--sbatch-account` when needed. The submitted script
uses the exact `cluster-mlip` executable found in the active environment.

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
claims. Audit errors produce exit code 2. Stage counts and input hashes are
taken only from manifest rows whose `submission_active` is not false: an input
rerun in place keeps its archived attempt's rows for provenance and label
collection, and counting those as stages of the current input produced false
"Number of Link1 stages differs from spin manifest" errors. A listed input
whose rows are *all* inactive is reported as an error, since the batch would
rerun an attempt the manifest no longer describes. SCF convergence warnings are
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
`optimized_unverified`, `checkpoint_geometry`), and every rattled variant, are
`unconstrained` and are never flagged. IRC frames and higher-order saddle
candidates are *not* unconstrained: see "IRC frames and higher-order saddle
candidates" below.

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
Only optimizing stages are rewritten: a `Force` label stage keeps its route.

### Two kinds of job, and what happens to the ladders

The audit's `geometry_source` column says which case each job is, and the
console summary counts them, so the split is visible before anything is
rebuilt.

`input_coordinates` -- a ladder generated by `prepare-spins`. It carries its own
coordinates, so it is corrected stage by stage and **the spin-flip chain is
preserved exactly**: every stage becomes a TS search, and because every
`%chk`/`%oldchk` is renamed the same way, stage k still reads stage k-1's
checkpoint. The one-spin-flip-at-a-time pathway, and its `Geom=Checkpoint`
initialization at each later multiplicity, are unchanged; only the search type
and the checkpoint names differ.

`checkpoint` -- a continuation produced by `prepare-spin-restarts`. These have
**no coordinates at all**: the first stage is `%oldchk` plus `Geom=Checkpoint`,
reading a seed copied out of the interrupted run. Such an input cannot be
corrected in place, for two independent reasons -- there is no geometry in it to
correct, and the seed checkpoint holds the collapsed minimum the plain `Opt`
produced, so continuing from it is wrong however good the route is. For these,
the tool retires the **whole restart lineage** (the root ladder's rows and every
restart attempt's rows) and rebuilds the root ladder from its real coordinates,
restoring the complete high-to-low pathway rather than the truncated tail the
restart had been reduced to. `ladder_stages` in `route_fix_plan.csv` records how
many multiplicities the rebuild covers, `rebuilt_from` names the root input, and
`retired_inputs` lists everything superseded. The restart seed checkpoints are
left untouched and unreferenced.

A restart whose manifest row has no `restart_root_input`, or whose root input is
missing from `inputs/`, is skipped with that reason rather than guessed at.

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
### higher_order_saddle records

`transition_state` and `first_order_saddle` get `Opt=(TS,...)`. A
`higher_order_saddle` label only says the archived log reported more than one
imaginary mode -- often an artifact rather than a target: an IRC path point is
not a stationary point at all, and a fragmenting structure has soft modes that
register as imaginary. Such a record is a *higher-order candidate*, and by
default it is relaunched as a fixed-geometry `Force` label at its archived
geometry (next section). An `Opt=(Saddle=N)` search happens only on explicit
request with a known N:

```bash
cluster-mlip relaunch-routes "$W2" --saddle-order-from extracted/seeds.extxyz
```

`extract` writes each record's `imaginary_frequencies` into the seeds extxyz,
so this gives every record **its own** order, matched through the manifest's
`parent_record_id`. `route_fix_plan.csv` records the `saddle_order` used and
the `saddle_order_source` it came from, and the rebuilt rows record it as
`requested_saddle_order`, which is what later audits require before they
accept a `Saddle=N` search on a higher-order candidate. A later
`relaunch-routes` honours a recorded `requested_saddle_order` (plan
`saddle_order_source=recorded_request`) unless `--higher-order-policy force` is
given explicitly, in which case the override is stated in `repair_reason`. A record whose seed
reports only one imaginary mode while the label says higher-order is skipped
as a disagreement rather than silently resolved. `--saddle-order N` forces one
order for everything, which is only appropriate when they genuinely share it.
`--higher-order-policy force` together with either flag is refused as
contradictory.

## IRC frames and higher-order saddle candidates

An IRC frame is a fixed point along a reaction path, not a stationary point.
Optimizing it -- `Opt`, `Opt=TS`, `Opt=(Saddle=N)` -- moves the atoms away from
the archived geometry, so the resulting label no longer describes the frame;
`Freq` there reports curvature that means nothing. What the MLIP needs is the
DFT energy and forces at exactly the archived coordinates, which is a
fixed-geometry `Force` job. A higher-order candidate is labeled the same way
unless a saddle search is explicitly requested (above).

**Geometry role.** Every record and manifest row now carries a
`geometry_role` next to its `config_type`: `stationary_minimum`,
`transition_state`, `irc_point`, `reaction_path_endpoint`,
`higher_order_candidate` or `unconstrained_geometry`, with
`geometry_role_source` saying where it came from. `extract` derives it from the
IRC route and Gaussian's `Point Number/Path Number` markers *before* looking at
any frequency count, so an IRC frame whose frequency analysis shows several
imaginary modes is no longer classified `higher_order_saddle`. IRC frames also
record `irc_direction`, the point index, `irc_path_position`
(`ts`/`intermediate`/`endpoint`; the last point of a path is an endpoint only
when Gaussian reports that path complete), `irc_parent_record_id` (the TS the
path started from), `source_calculation_type=irc`, and the original charge and
multiplicity.

Gaussian 09 prints each `Point Number: N  Path Number: M` summary *after*
point N has converged, followed by `# OF POINTS ALONG THE PATH` and
`# OF STEPS` lines (the Warehouse 2 seeds carry those lines as the "route" of
their IRC frames, which is how the old parser lost the IRC route). A geometry
is therefore numbered by the marker that *closes* it: `irc_frame_kind` is
`converged_point` for the geometry printed just before a marker,
`optimization_step` for the constrained-optimization steps before it, `ts`
for the first geometry of the calculation, and `unconverged_step` for steps of
a point the job never finished. `# OF ...` lines are no longer read as
routes. Logs without Gaussian's summary lines are read the other way round
(marker introduces the geometry after it); `irc_marker_layout` records which
layout a frame was read with. For legacy records the only surviving evidence is often the
source file or folder name; `irc` in a name is used **only as an explicit
fallback** (`geometry_role_source=filename_fallback`), never overrides a
`transition_state` label or an established minimum, and is counted in the
`extract`, `prepare`, `audit-routes` output and listed in `route_audit.md`.
A `first_order_saddle` (or requested higher-order) row whose job already runs a
saddle search -- for instance from the transition-state relaunch -- is kept as
that search even when its source name mentions an IRC
(`geometry_role_source=running_saddle_search`); those rows are counted and
listed separately in `route_audit.md` so the decision stays visible.
Pass the campaign's seeds file with `--seeds` to `audit-routes` or
`relaunch-routes` to replace that fallback with the archived route where the
seed record has one.

**Routes.** A force-only root stage keeps the campaign protocol and replaces
`Opt` with `Force` in place:

```text
#p UBPW91/6-311++G* SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm Force IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine Pop=Regular
```

and every later ladder stage reads the same geometry and its predecessor's
orbitals from the checkpoint:

```text
#p UBPW91/6-311++G* SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm Force IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine Geom=Checkpoint Guess=Read Pop=Regular
```

Stage 0 reads the archived coordinates; stage k has `%oldchk` = stage k-1's
`%chk`. `Freq`, `Stable`, `IRC` and `Guess=Always` are removed and never
introduced (a fragment guess on a Force job is `Guess=(Fragment=N)` without
`Always`). `prepare` and `prepare-spins` generate these routes automatically
for IRC frames and, unless `--higher-order-policy saddle-search`, for
higher-order candidates.

**Audit rules** (`audit-routes`, `campaign-status --audit`, `collect`):

| Geometry | Route | Verdict |
|---|---|---|
| IRC point / path endpoint | any `Opt` | `path_point_launched_as_optimization`: error, relaunch required |
| IRC point / path endpoint | `Force` | valid |
| IRC point / path endpoint | `Force` + `Freq`/`Stable`/`Guess=Always` | `path_point_route_has_extra_keywords`: warning, label usable |
| higher-order candidate | ordinary `Opt` | `higher_order_launched_as_minimum_search`: error, relaunch required |
| higher-order candidate | `Force` | valid MLIP label |
| higher-order candidate | `Opt=(Saddle=N)` / `Opt=TS` | valid only with `requested_saddle_order` = N recorded; otherwise `higher_order_saddle_search_unrequested` |

**Collection.** A Force label is accepted only if it can be checked against
the archived input: coordinates must agree (1e-3 A RMS pair distance), or, for
a Z-matrix input, the atom sequence. A checkpoint-seeded restart whose
coordinate-bearing `restart_root_input` is missing is rejected rather than
accepted unchecked.

**Completion.** A normally terminated Force stage is complete: restarts
(`prepare-spin-restarts`), `campaign-status`, the progress report and
`validate-spins` no longer wait for an optimization-convergence marker that a
Force job never prints. Optimization stages keep the converged-only rule.

### Repairing an existing campaign (Warehouse 2)

The audit of `gaussian_spin_qb3_g09_v3` found 13 active inputs labeled
`higher_order_saddle` generated with ordinary `Opt`, six of them derived from
IRC calculations. The repair runs in place with the same machinery as the TS
relaunch above -- same batch folders, same `inputs.txt` slots, no new batch
map:

```bash
W2=/ddnB/work/lgutsev/ClusterMLIP/campaigns/FenOm_Warehouse2/gaussian_spin_qb3_g09_v3
SEEDS=/path/to/the/seeds.extxyz        # optional: archived routes instead of filename evidence

# QB4 -- read-only picture first
cluster-mlip audit-routes "$W2" --seeds "$SEEDS"
cluster-mlip campaign-status "$W2" --audit --start 6 --end 130 --sbatch

# QB3 -- nothing in 6-130 may still be running
squeue -u "$USER"

# QB4 -- plan (writes nothing but the plan CSV you point it at), then apply
cluster-mlip relaunch-routes "$W2" --start 6 --end 130   --path-point-policy force --higher-order-policy force   --seeds "$SEEDS" --assume-stopped --dry-run   --plan-output "$HOME/w2_force_label_plan.csv"
cluster-mlip relaunch-routes "$W2" --start 6 --end 130   --path-point-policy force --higher-order-policy force   --seeds "$SEEDS" --assume-stopped

# QB4 -- re-audit; expect no route errors and no Link1-stage mismatches
cluster-mlip campaign-status "$W2" --audit --start 6 --end 130 --sbatch

# QB3 -- resubmit the same range with the existing head launcher
bash "$W2/submit_gaussian_batches.sh" --start 6 --end 130
```

Omit `--seeds` if the seeds file is not at hand; the filename fallback then
applies and every such row is flagged in the output. Read the dry run before
applying: it relaunches **every** must-relaunch job in the range, not only
these 13 -- including any IRC-typed input (`irc_forward`/`irc_reverse`) that
was launched with `Opt`, which the old audit treated as unconstrained. The
plan CSV (and `route_fix_plan.csv` after the real run) records per input the
original and replacement input, `geometry_role` and its source, the `repair`,
old and new route (`route_before`/`route_after`, plus every stage in
`stage_routes_after`), the archived log, `ladder_stages`/`input_stages`, and a
`repair_reason`.

What the real run does, per input: archives a finished or partial log and its
`.rc`/`.status`/`.started`/`.finished` markers in the same batch folder as
`NAME__before-routefixNN.*`; keeps the old rows in the manifest as inactive
provenance with `route_invalidated` set, so `collect` never ingests the
ordinary-`Opt` results; rebuilds a checkpoint-seeded restart lineage from its
coordinate-bearing root; renames every `%chk`/`%oldchk` consistently so the
spin multiplicities and checkpoint chain are unchanged; and replaces the
active entry in the existing `inputs.txt`. An input with an unmatched
`.started` marker is refused unless `--assume-stopped` is given, and every
override is reported. Completed valid jobs (minima, correct TS searches) are
not touched. Afterwards, collect with:

```bash
cluster-mlip collect "$W2" -o "$(dirname "$W2")/dataset" --frames converged
```

### Internal-coordinate failures (FormBX / Tors failed)

A different failure, unrelated to the route. Gaussian's Berny optimizer works
in redundant internal coordinates; `FormBX` builds the Wilson B matrix for that
transformation, and a torsion whose three defining atoms are near-collinear has
no well-defined value, which makes the matrix singular:

```text
 Tors failed for dihedral     1 -     2 -     3 -     4
 FormBX had a problem.
```

The job dies before taking a step. This is not affected by the GEDIIS/GDIIS
optimizer choice, since the breakdown is in forming the coordinates rather than
in stepping. `audit-routes` reports it as `internal_coordinate_failure`, and a
relaunch adds `Cartesian` to the existing `Opt` options, optimizing in
Cartesians and bypassing the transformation entirely. That converges in more
steps than the redundant-internal default but cannot hit a degenerate torsion,
and it needs no change to the geometry.

A Z-matrix with dummy atoms is the other valid fix, and often the better one
for a genuinely linear fragment: it keeps internal-coordinate convergence and
defines the torsion explicitly. A plain Z-matrix does not help by itself -- a
Z-matrix dihedral through a ~180 degree angle is undefined for exactly the same
reason -- which is why a builder such as ChemCraft inserts the dummy atom. The
tradeoff is per-structure manual work, so `Opt=Cartesian` is the automatic
default and a hand-built Z-matrix is worth it for a stubborn few.

Inputs carrying a Z-matrix are recognized and **kept in internal
coordinates**: `Cartesian` is never added to one, since that would discard the
dummy-atom construction that repaired it. Such a job still receives its route
correction, and `route_fix_plan.csv` records `coordinate_system` as
`zmatrix_preserved` rather than `cartesian`. Dummy atoms are already dropped
when outputs are parsed for labels, so a Z-matrix job collects normally.

If a job fails this way *even in Cartesians*, it is reported as
`internal_coordinate_failure_in_cartesian` and deliberately not relaunched: no
route change can fix it, and the geometry itself needs attention -- usually a
nudged starting structure or the dummy-atom Z-matrix.

### Launcher safety

The generated batch scripts are driven strictly by each batch's `inputs.txt`
(`mapfile -t inputs < "$batch_file"`) and never glob for `*.gjf`, so a
superseded input left in a batch folder is inert -- it stops being run the
moment it leaves that list. `RUN_POLICY=resume` then decides per listed input
whether its `<stem>.log` is a complete Gaussian job, so a batch containing a
freshly activated replacement is resubmitted while its completed neighbours are
skipped.

Before writing anything, `relaunch-routes` checks the end state that policy
depends on and **refuses the whole run** if any of it would break, because a
half-applied relaunch is worse than none:

- no two rebuilds writing the same input name;
- every batch's resulting `inputs.txt` non-empty, free of duplicates, with
  unique output stems so no two jobs can write the same `.log`;
- every replacement actually listed, and every retired lineage input removed;
- no pre-existing `.log`/`.chk`/`.rc`/`.status`/`.started`/`.finished` for the
  new stem, and none of the checkpoint names the rebuild will write already on
  disk -- one of those could belong to a job that is still running;
- `%nprocshared` still matching `slurm_plan.json`'s `cpus_per_job`;
- for a spin ladder, Link1 stage count equal to the number of manifest rows.

`--dry-run` reports the same checks under `launcher_problems` without touching
the campaign. After applying, the tool reads the result back off disk and fails
loudly if a manifest hash, an archived output, or a batch listing does not match
what it intended to write. The batch map itself is never regenerated, so
`slurm_plan.json` and the QB3 resource directives stay exactly as submitted.

### Which machine runs what

`relaunch-routes` changes only data: the files under `inputs/`, each batch's
`inputs.txt`, the manifest, and the plan CSVs. It regenerates no script and
never alters the batch map, so nothing new is needed on QB3. The generated
launchers are pure bash/awk/Slurm with no reference to the Python package;
`run_batch.sbatch` reads its input list at run time (`mapfile -t inputs <
"$batch_file"`) rather than globbing, so replacing an entry is enough; and
`gaussian_complete` derives the expected Link1 stage count from the input file
itself, so a rebuilt three-stage ladder is automatically required to produce
three normal terminations. `submit_gaussian_batches.sh` and
`gaussian_batch_status.sh` hard-code only the batch count, which a relaunch
never changes.

Prepare on QB4, submit on QB3. `gaussian_batch_status.sh` only reports;
`submit_gaussian_batches.sh` is what launches.

**Match the ranges.** `relaunch-routes` defaults to every batch, so scope it to
the range you are about to submit, and only after `squeue` shows that range
idle:

```bash
# QB4
cluster-mlip relaunch-routes "$W2" --start 31 --end 50 --dry-run
cluster-mlip relaunch-routes "$W2" --start 31 --end 50 --assume-stopped
```

```bash
# QB3
bash "$W2/submit_gaussian_batches.sh" --start 31 --end 50
```

`--assume-stopped` is an assertion about a scheduler QB4 cannot see. Every
input it overrides -- one whose `.started` marker has no matching `.finished`
-- is counted, printed as a warning naming the input, and written to
`route_fix_overrides.csv`. If any of those is still running, its log is being
archived out from under a live process and a duplicate will be submitted, so
treat a nonzero override count as a reason to stop and check the queue.

Do not run `prepare-slurm` again. Resubmit the same range with the existing head
launcher, then re-audit before collecting:

```bash
cluster-mlip audit-routes "$W2"
```

## Restarting interrupted spin ladders

First confirm on QB3 that the original allocations have stopped. The restart
command works in place: it finds the first stage lacking both optimization
completion and normal termination (a fixed-geometry Force stage needs only the
normal termination), archives that attempt's log and marker
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
`Geom=Checkpoint Guess=Read` and writes a distinct `%chk`. Its molecule
specification is the charge/multiplicity line alone: when the interrupted
stage is stage 0, the only stage with coordinates, those coordinates are
removed, since Gaussian reads a `Geom=Checkpoint` stage's leftover atom lines
as the next input section. The tool refuses to write any restart stage that
reads `Geom=Checkpoint` and still carries coordinates, and
`campaign-status --audit` reports an existing one as an error. Inspect
`restart_plan.csv` for each original batch, first unfinished multiplicity,
source checkpoint, and number of remaining stages. If the unfinished stage's
checkpoint is absent, the tool can use the immediately preceding completed
stage checkpoint; it reports that exact choice in the plan. Manifest and
`inputs.txt` snapshots are kept together under the single `restart_backups/`
directory.

If neither the unfinished stage nor its completed predecessor has a usable
checkpoint, the default remains to skip that input. After confirming the old
allocation is stopped, archive its partial log and reactivate the unchanged
input from scratch with:

```bash
cluster-mlip prepare-spin-restarts "$W2" \
  --start 1 --end 30 --assume-stopped --rerun-missing-checkpoints --dry-run
cluster-mlip prepare-spin-restarts "$W2" \
  --start 1 --end 30 --assume-stopped --rerun-missing-checkpoints
```

The archived log remains represented by inactive rows in `spin_jobs.csv`, so
its converged stages remain available to `collect --frames converged` (without
`--allow-partial`); once the rerun finishes the same stage, only the rerun's
label is kept. The
active cloned rows reuse the original input and output names; marker files are
archived beside the old log before the rerun begins.

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
