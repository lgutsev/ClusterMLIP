# ClusterMLIP

Reproducible tooling for turning heterogeneous cluster-DFT warehouses into a
charge/spin-conditioned MACE potential trained from scratch.

The first milestone deliberately covers one model strategy:

- a compact MACE trained from scratch;
- graph-level integer charge and spin-multiplicity embeddings;
- isolated-cluster energy and force labels;
- provenance-aware splits that keep sibling perturbations together.

Foundation-model fine-tuning is deferred until the scratch model provides a
clean control; `configs/finetune_foundation.sh` is an unverified starting
point for that milestone, not a claim that it is ready (see below).

## What it handles

- Gaussian `.log`, `.out`, `.com`, and `.gjf` files;
- formatted checkpoints (`.fchk`/`.fch`);
- binary `.chk` files when Gaussian `formchk` is available;
- native `TITLE`/`MOLDAT` warehouse coordinate files;
- legacy `.doc` and `.docx` records;
- ZIP archives and nested ZIP archives (four levels by default).

Explicit Gaussian IRC point records are split into separate configurations.
TS/QST routes and filenames are recognized, while a structure with exactly one
imaginary frequency is conservatively called `first_order_saddle`. The scanner
does not invent reaction labels from chemical intuition.

## Install

Python 3.10+ is required. The Unix `strings` command is used for old binary Word
documents, and `formchk` is needed only for binary Gaussian checkpoints.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

MACE is needed only for training, fine-tuning, or `select-next-batch`:

```bash
python -m pip install -e '.[train]'
```

Run `cluster-mlip doctor` after installing to check which of these optional
tools (`strings`, `formchk`, `mace-torch`) are actually on `PATH`/importable
before starting a large extraction or training run.

On LONI, do not upgrade a shared InterfaceForge/OpenMM runtime in place merely
to satisfy ClusterMLIP's `mace-torch>=0.3.16` requirement. Clone it and update
the clone so the known-working MACE 0.3.15 environment remains recoverable:

```bash
conda create -y \
  --prefix /project/lgutsev/env/cluster_mlip_runtime \
  --clone /project/lgutsev/env/iface_mace_runtime
conda activate /project/lgutsev/env/cluster_mlip_runtime
python -m pip install --upgrade 'mace-torch>=0.3.16'
python -m pip install -e '.[train]'
cluster-mlip doctor
```

Inspect the packages pip proposes to replace if it needs to change PyTorch or
CUDA-facing dependencies. Keep `iface_mace_runtime` and `lgutsev_dev`
unchanged until the cloned environment passes a short train/load/evaluate
smoke test.

## 1. Analyze a database first

```bash
cluster-mlip analyze Warehouse2.zip -o reports/warehouse2
```

The analyzer traverses nested ZIPs safely and writes:

- `report.md` — readable database summary;
- `summary.json` — aggregate machine-readable statistics;
- `files.csv` — every file, format, parse status, and error;
- `records.csv` — structure/state provenance and geometry hashes.

It reports file coverage, parse failures, duplicate structures, elements and
formulas, cluster-size range, charge/multiplicity states, stationary-point and
IRC classes, and legacy Gaussian routes. Unsupported files remain visible in
the inventory. Binary `.chk` files are reported as errors if `formchk` is not
available.

Add `-j N`/`--jobs N` to parse files in `N` worker processes for a large
(LONI-scale) warehouse; the default of 1 runs the original single-process
scan unchanged. `analyze` and `audit` both accept it.

For a private LONI audit with both a complete inventory and the initial Fe/N/O
pilot selection, use the first-class audit command:

```bash
cluster-mlip audit /path/to/Oct9.zip \
  --elements Fe,N,O \
  --require-elements Fe \
  --max-atoms 20
```

This writes `full/`, optional `selection/`, and `provenance.json` under
`private_audits/Oct9/`, which is git-ignored. Provenance includes the LONI host,
timestamp, source path, byte size, SHA-256 checksum, and applied filters. The
generated report, record tables, and warehouse-specific conclusions must remain
on LONI or another controlled storage system; they are not repository
documentation. `scripts/generate_private_audit.sh` remains as a thin wrapper
using the same Fe/N/O defaults.

## 2. Extract structures

```bash
cluster-mlip extract Warehouse2.zip \
  -o extracted \
  --elements Fe,N,O \
  --require-elements Fe
```

Outputs are `seeds.extxyz`, `manifest.csv`, and `errors.tsv`. Identity includes
geometry (to 1e-6 Å), charge, multiplicity, configuration type, and IRC point.

## 3. Prepare consistent Gaussian labels

```bash
cluster-mlip prepare extracted/seeds.extxyz \
  -o gaussian_jobs \
  --rattles-per-seed 4 \
  --rattle-sigma 0.05 \
  --memory 32GB \
  --nproc 16
```

Every seed is retained and receives reproducible Cartesian perturbations so the
force set is not dominated by near-zero stationary-point forces. The default
route is:

```text
#p wB97M-V/def2TZVPP Force SCF=(XQC,Tight,MaxCycle=512) Integral=UltraFine NoSymm
```

Use `--route` to select the single reference method for a campaign. Do not mix
energies or forces from different electronic-structure methods in one target
head. `jobs.csv` preserves each job's parent structure; `run_one.sh` is a small
Gaussian runner to adapt to the local scheduler.

### Generate separately monitorable LONI Slurm batches

After `prepare`, split the campaign into physical batch directories and create
one independently submitted Slurm job per directory:

```bash
cluster-mlip prepare-slurm gaussian_jobs \
  --jobs-per-batch 30 \
  --concurrent-jobs 4 \
  --cpus-per-job 16 \
  --time 72:00:00 \
  --partition checkpt \
  --account loni_dspm_25 \
  --gaussian-module gaussian/g16-c01
```

This preserves the useful two-level batching strategy: every
`slurm_batches/batch_NNNN/` directory is a distinct Slurm job containing 30
input symlinks, while that job runs up to four independent 16-core Gaussian
calculations at once on its node. Each folder has its own `inputs.txt`,
`run_batch.sbatch`, `submit.sh`, scheduler stdout/stderr, Gaussian `.log`,
`.status`, `.rc`, and timestamp files. The original inputs remain unmoved and
unmodified, and `collect` scans the batch folders recursively.

Submit from anywhere with the generated wrapper:

```bash
./gaussian_jobs/submit_gaussian_batches.sh
./gaussian_jobs/gaussian_batch_status.sh
```

Each batch can also be watched or resubmitted on its own:

```bash
cd gaussian_jobs/slurm_batches/batch_0001
./submit.sh
tail -f scheduler-*.stdout
```

The wrappers export the actual campaign and batch directories rather than
relying on `$PWD` or the Slurm spool directory. A resubmission is safe by default:
`RUN_POLICY=resume` skips any `.log` containing a normal Gaussian termination
and reruns missing/incomplete calculations. The all-batch submitter skips a
fully completed batch instead of creating a new Slurm job for it. Force a
complete rerun only when deliberate:

```bash
RUN_POLICY=all ./gaussian_jobs/submit_gaussian_batches.sh
```

Gaussian scratch defaults to `/work/$USER/g16-scr`; override it at submission
time with `GAUSSIAN_SCRATCH_ROOT`. Scratch is removed per completed worker
unless `KEEP_GAUSSIAN_SCRATCH=1` is exported.

For Gaussian `External` calculations that need the xTB conda environment and
wrapper on `PATH`, copy and source the supplied optional initialization hook:

```bash
cluster-mlip prepare-slurm gaussian_jobs \
  --worker-init examples/gaussian_worker_init_xtb.sh
```

Its conda initialization path, environment name, and wrapper directory can be
overridden with `CLUSTER_MLIP_CONDA_SH`, `CLUSTER_MLIP_XTB_ENV`, and
`CLUSTER_MLIP_XTB_WRAPPER_DIR`. Ordinary wB97M-V Gaussian labeling does not
need this hook.

## Spin-safe Fe-cluster preparation

> **Experimental status:** this workflow is code-tested but has not yet been
> human-tested on a real Gaussian campaign. It still requires a one-case
> Gaussian smoke test, scientific review of the fragment definitions, and
> likely convergence/restart tuning before production use.

Do not optimize a legacy low-spin Fe-cluster entry directly from a generic
orbital guess. Its multiplicity may be numerically valid while its intended
broken-symmetry/AFM root is absent. The spin workflow keeps requested state,
converged SCF root, and local-spin pattern separate.

First extract a state inventory from the original warehouse. This records
electron-count parity, multiplicity, `<S^2>`, Mulliken atomic spin densities,
normal termination, optimization status, stability-test evidence, and a local-
spin root signature when the source contains them. Every record also carries
a `state_inference` provenance tag (`filename`, `electron_parity_fallback`,
`default_unmatched_singlet`, or blank for a charge/multiplicity read directly
off an explicit Gaussian `Charge =`/`Multiplicity =` line) in `manifest.csv`,
`records.csv`, and `spin_inventory.csv`, so a defaulted guess is never
indistinguishable from a validated state:

```bash
cluster-mlip spin-extract Warehouse2.zip -o spin_inventory
```

Then prepare a trusted high-spin reference and a one-spin-flip-at-a-time
multiplicity ladder. Missing intermediate multiplicities are inserted
automatically, so the following produces 29 -> 27 -> 25 -> 23 -> 21 even if
only 21 is requested:

```bash
cluster-mlip prepare-spins spin_inventory/seeds.extxyz \
  -o gaussian_spin_jobs \
  --high-spin 29 \
  --targets 21 \
  --memory 32GB \
  --nproc 16
```

Each Link1 stage reads the immediately preceding checkpoint with
`Geom=Checkpoint Guess=(Read,Always)`, writes a new checkpoint, and preserves
the previous root. This is a multiplicity ladder, not proof of AFM coupling;
the output must still be characterized from local spins and `<S^2>`.

The independent fragment pathway follows Gaussian's
[`Guess=Fragment`](https://gaussian.com/guess/) and
[fragment molecule-specification](https://gaussian.com/molspec/) conventions.
Atom membership, fragment charge, fragment multiplicity, and alpha/beta
orientation must be supplied explicitly; ClusterMLIP never guesses a magnetic
partition from geometry. The generated molecule-specification line is ordered
exactly as `total charge, total multiplicity, fragment 1 charge, fragment 1
multiplicity, ...`. A negative local multiplicity requests β-spin unpaired
orbitals for that fragment; it is not a negative total spin.

The preparer rejects fragment inputs unless all atoms are assigned exactly
once, fragment charges sum to the molecular charge, every local multiplicity
has valid electron parity, and
`sum[orientation * (local multiplicity - 1)] == total multiplicity - 1`.
`examples/spin_fragments.schema.json` documents the same shape as a JSON
Schema (most editors will validate a `--fragment-spec` file against it live
if it keeps the example's `$schema` pointer); `prepare-spins` also checks the
shape itself before rendering anything, so a missing/mistyped field is
reported all at once instead of as a bare traceback from deep inside the
fragment validator. For example:

```json
{
  "guesses": [
    {
      "record_id": "REPLACE_WITH_HIGH_SPIN_RECORD_ID",
      "name": "two-sublattice-afm",
      "target_multiplicity": 1,
      "fragments": [
        {"atoms": [1, 2], "charge": 0, "multiplicity": 5, "orientation": "alpha"},
        {"atoms": [3, 4], "charge": 0, "multiplicity": 5, "orientation": "beta"}
      ]
    }
  ]
}
```

Pass it alongside the ladder:

```bash
cluster-mlip prepare-spins spin_inventory/seeds.extxyz \
  -o gaussian_spin_jobs \
  --high-spin 29 \
  --targets 27,25,23,21 \
  --fragment-spec examples/spin_fragments.example.json
```

`spin_jobs.csv` records the pathway, parent state, predecessor multiplicity,
stage index, checkpoint, intended charge/multiplicity, and output file. Fragment
candidates are separate jobs, so convergence to a new root cannot overwrite a
ladder solution.

Finally compare the original warehouse against all new outputs:

```bash
cluster-mlip validate-spins Warehouse2.zip gaussian_spin_jobs \
  -o spin_validation --strict
```

The validator writes:

- `legacy_coverage.csv` — matched root, uncharacterized match, alternative root,
  incomplete calculation, or missing legacy state;
- `new_states.csv` — every new root, including novel/unmatched candidates;
- `planned_state_coverage.csv` — every requested ladder/fragment stage and
  whether it appeared in an output;
- `report.md`, `summary.json`, and `errors.tsv`.

Geometry matching uses a translation-, rotation-, and atom-order-invariant
element-pair distance fingerprint. Electronic-root comparison uses element-
resolved local-spin distributions when both old and new outputs contain them.
An `alternative_root` is retained as potentially useful training data rather
than silently discarded. `--strict` returns nonzero for missing states,
incomplete calculations, alternative roots, or planned stages without output.

## 4. Collect labels and split by parent

Put completed `.log`/`.out` files beside `jobs.csv`, then run:

```bash
cluster-mlip collect gaussian_jobs -o dataset
```

The collector converts Gaussian energies and forces to eV and eV/Å and writes
`all.extxyz`, `train.extxyz`, `valid.extxyz`, and `test.extxyz`. All rattles from
one parent remain in one split, preventing near-duplicate leakage.

The split is geometry-stratified by default (`--stratify-by
pes_region,charge_spin_class`): every parent-record group is classified (see
`cluster_mlip.stratify`) into which PES region it's from (minimum/saddle/irc/
other) and which charge/spin regime it's in, and each resulting stratum is cut
into train/valid/test at the requested proportions *independently*. Plain
per-group hashing (the old, still-available behavior via the Python API's
`stratify_by=None`) is unbiased in aggregate, but a small class -- say four
transition-state groups in the whole warehouse -- has a real chance of landing
entirely in `train` by chance alone with nothing in `valid`/`test` to catch a
model that's bad at exactly that class. `--stratify-by` (comma-separated,
choose from `pes_region`, `displacement_class`, `coordination_class`,
`compactness_class`, `charge_spin_class`, `provenance_tier`) controls which
axes are used; pass an empty string to fall back to one pseudo-stratum
containing everything.

`collect` also writes `label_report.md`/`label_report.json`: energy and
force-RMS statistics grouped by charge/multiplicity and by every stratify axis,
split coverage (how many groups per stratum landed in each split -- a stratum
showing 0 in valid/test didn't have enough groups to round up to one at the
requested fractions, and now you can see that instead of finding out later),
and every frame whose force RMS exceeds `--force-outlier-threshold` (default
5 eV/Å, adjust for your route and system). A non-converged SCF root or a
rattle that blew up a geometry shows up here before it ever reaches training.

## 5. Train from scratch

```bash
bash configs/train_from_scratch.sh dataset
```

The initial configuration is a two-interaction `ScaleShiftMACE` with graph-level
categorical embeddings for total charge and spin multiplicity. It uses energy
and force losses and no stress target because these are isolated clusters.
Hyperparameters are a documented baseline, not a claim of final convergence.

### Fine-tuning a foundation model (deferred, unverified)

`configs/finetune_foundation.sh` is a starting point for the fine-tuning
milestone the top of this README defers, written from MACE's documented
naive-fine-tuning flags (`--foundation_model`, low `--lr`, few epochs) rather
than from a run that has actually been executed against this project's data.
It carries the open question in a comment: whether `--foundation_model`
tolerates the project's `--embedding_specs`/`--use_embedding_readout`
charge/spin modules being attached to a checkpoint that was never trained
with them. Smoke-test on a handful of structures before a real allocation,
the same way the spin workflow needs a one-case check first.

```bash
FOUNDATION_MODEL=medium bash configs/finetune_foundation.sh dataset
```

## 6. Evaluate a trained model

```bash
cluster-mlip evaluate dataset/test.extxyz --model checkpoints/model.model -o evaluation
```

Runs the model over a labeled `extxyz` (from `collect`) and reports energy
(eV/atom) and force (eV/Å) MAE/RMSE overall, by charge/multiplicity, and by
every `stratify` axis (PES region, rattled vs. relaxed, coordination class,
compactness, charge/spin class, provenance tier) -- in `evaluation.json`, a
`by_*.csv` per axis, and `report.md`. A model that's fine on average but bad
at saddle points or under-coordinated atoms no longer hides behind one number.
This implements the first bullet of "Validation priorities" below.

It then runs five cheap, first-pass physical sanity checks, one per
structural class (`--skip-physical-checks` to opt out): near-zero predicted
force at labeled minima/saddles, force-direction agreement on rattled
displacements, force-error concentration on low-coordination atoms, net-force
direction agreement between the pieces of a fragmenting cluster, and whether
the model agrees with DFT on which spin state is the ground state for
matched-geometry multiplicity groups (reusing `spin.geometry_distance`).
Results go to `physical_checks.md`/`.json`. These are sanity checks, not a
certification -- thresholds are first-pass defaults (see
`cluster_mlip.physical_checks`) meant to be recalibrated once you have real
error distributions, and a check reports "n/a" rather than a number when
nothing in the dataset matches its class.

Needs the training extra (`pip install -e '.[train]'`).

## 7. Active learning: pick what to label next

```bash
cluster-mlip select-next-batch extracted/seeds.extxyz \
  --models checkpoints/seed1.model checkpoints/seed2.model checkpoints/seed3.model \
  --top-k 50 -o next_batch
```

Runs a committee of two or more independently trained checkpoints (different
seeds and/or data subsets) over an unlabeled candidate pool, scores each
structure by the worst-atom force disagreement across the committee, and
writes the `top-k` most-disagreed-upon structures to `next_batch.extxyz` (plus
`selection.csv` with the scores) -- these are the structures the current
dataset constrains the least, and so are the best next candidates for the
next (expensive) DFT labeling round rather than labeling the warehouse in
arbitrary order. Needs the training extra.

## Model and data storage

Keep code, small configs, manifests, and reports in Git. Keep raw warehouses,
Gaussian outputs, training datasets, checkpoints, and final model weights in
versioned object storage or an institutional data repository. Record their
checksums and immutable storage URIs in experiment manifests; do not commit
multi-gigabyte model or calculation artifacts to this repository.

```bash
cluster-mlip manifest dataset -o manifests/exp001.json \
  --config configs/train_from_scratch.sh --notes "first scratch run"
```

Bundles the dataset's file checksums, the training config's checksum, the
current git commit (and whether the checkout is dirty), and a timestamp into
one JSON record -- the "checksums ... in experiment manifests" half of the
paragraph above; the storage-URI half is still on you, since this tool has no
opinion on where your object storage lives.

## Validation priorities

- energy and force errors by charge and multiplicity -- implemented, see
  `cluster-mlip evaluate` above, now broken out by PES region, displacement
  class, coordination class, compactness, and provenance tier too (see
  `cluster_mlip.stratify`), not just charge/multiplicity;
- isomer ordering within composition/state groups -- partially covered: the
  `spin_state_ordering` physical check compares DFT vs. model agreement on
  which spin state is lowest-energy for matched-geometry multiplicity groups,
  which is one specific case of isomer ordering (same geometry, different
  electronic state); ordering across genuinely different isomer geometries at
  the same composition is not covered;
- held-out reaction families and cluster sizes -- not yet implemented (no
  reaction-family labels exist yet to hold out by; cluster-size stratification
  could be added as a `stratify` axis if it turns out to matter);
- barrier errors once genuine TS/IRC labels are available -- not yet
  implemented (needs TS-to-IRC-endpoint pairing this pipeline doesn't track);
- fragmentation curves and short trajectory stability -- the
  `fragmenting_force_direction` physical check is a static, single-frame
  proxy (does the model push/pull separating fragments the right way in one
  labeled geometry); an actual dissociation curve or MD trajectory stability
  check is not implemented;
- explicit checks for inconsistent broken-symmetry SCF roots -- not yet
  implemented as a validation-priority check (`validate-spins` already flags
  `alternative_root` states during spin-campaign preparation, which is a
  related but earlier-stage check).

## Before a large run

```bash
cluster-mlip doctor
```

Checks Python version plus whether `strings`, `formchk`, a Gaussian
executable, and the optional `mace-torch`/`torch` stack are actually
available, so a multi-thousand-file HPC run doesn't discover a missing tool
partway through. Everything but the Python-version check is advisory: most
commands don't need every tool on the list.

## Tests

```bash
python -m unittest discover -s tests -v
python -m pip install -e '.[dev]' && mypy src
```

The fixtures exercise minimum, TS, IRC, checkpoint, warehouse, force-table,
grouped job, nested-ZIP analysis, label-report, evaluation, active-learning
disagreement-ranking, fragment-spec shape-validation, parallel-scan,
geometry-stratification (bonding graph/coordination/compactness/PES-region/
charge-spin classification, stratified `grouped_split`), and physical-check
paths. CI (`.github/workflows/tests.yml`) runs the unit tests on Python
3.10/3.11/3.12 and `mypy` separately; the package is fully type-hinted. None
of this needs `mace-torch` installed -- the stratification, reporting, and
physical-check logic is plain Python over data already in hand; only the
CLI's actual model inference calls need the training extra, same as before.
