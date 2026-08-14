# ClusterMLIP

Reproducible tooling for turning heterogeneous cluster-DFT warehouses into a
charge/spin-conditioned MACE potential trained from scratch.

The first milestone deliberately covers one model strategy:

- a compact MACE trained from scratch;
- graph-level integer charge and spin-multiplicity embeddings;
- isolated-cluster energy and force labels;
- provenance-aware splits that keep sibling perturbations together.

Foundation-model fine-tuning is deferred until the scratch model provides a
clean control.

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

MACE is needed only when training:

```bash
python -m pip install 'mace-torch>=0.3.16'
```

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
spin root signature when the source contains them:

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
For example:

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

## 5. Train from scratch

```bash
bash configs/train_from_scratch.sh dataset
```

The initial configuration is a two-interaction `ScaleShiftMACE` with graph-level
categorical embeddings for total charge and spin multiplicity. It uses energy
and force losses and no stress target because these are isolated clusters.
Hyperparameters are a documented baseline, not a claim of final convergence.

## Model and data storage

Keep code, small configs, manifests, and reports in Git. Keep raw warehouses,
Gaussian outputs, training datasets, checkpoints, and final model weights in
versioned object storage or an institutional data repository. Record their
checksums and immutable storage URIs in experiment manifests; do not commit
multi-gigabyte model or calculation artifacts to this repository.

## Validation priorities

- energy and force errors by charge and multiplicity;
- isomer ordering within composition/state groups;
- held-out reaction families and cluster sizes;
- barrier errors once genuine TS/IRC labels are available;
- fragmentation curves and short trajectory stability;
- explicit checks for inconsistent broken-symmetry SCF roots.

## Tests

```bash
python -m unittest discover -s tests -v
```

The fixtures exercise minimum, TS, IRC, checkpoint, warehouse, force-table,
grouped job, and nested-ZIP analysis paths.
