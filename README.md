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
