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
  --require-elements Fe,O \
  --max-atoms 20
```

This writes `full/`, optional `selection/`, and `provenance.json` under
`private_audits/Oct9/`, which is git-ignored. Provenance includes the LONI host,
timestamp, source path, byte size, SHA-256 checksum, and applied filters. The
generated report, record tables, and warehouse-specific conclusions must remain
on LONI or another controlled storage system; they are not repository
documentation. `scripts/generate_private_audit.sh` remains as a thin wrapper
using the same Fe/N/O defaults.

## Inventory a folder of many deliveries, then audit against the literature

Two commands for a different situation than `analyze`/`audit` above: instead
of one warehouse ZIP, you have a *folder* of deliveries collected over time
(from one or more collaborators) and want to know what you actually have across
all of them, and what published work might still be missing.

```bash
cluster-mlip inventory /path/to/deliveries -o inventory
```

Finds every `*.zip` directly under the folder (`--recursive` to also search
subfolders), analyzes each one exactly like `analyze` does
(`inventory/by_source/<zip-name>/report.md`, one per ZIP), and additionally
writes one **merged master list**: every unique formula/charge/multiplicity/
structure-type combination found *anywhere*, with which ZIP(s) it came from --
`inventory/inventory.md` and `inventory/inventory.json`. `-j N`/`--jobs N`
parses each ZIP's files across `N` worker processes, same as `analyze`.

A folder of many/large deliveries can take hours to parse --
`scripts/run_inventory_slurm.sh` is a ready-to-submit single-node sbatch
script (16 CPUs, 8-hour default, LONI `checkpt`/`loni_perovsk27` defaults,
both overridable on the `sbatch` command line without editing the file):
`cd` into the folder of ZIPs, then `sbatch
/path/to/scripts/run_inventory_slurm.sh`. `literature-gap` is deliberately
not bundled into this job -- it needs live internet, which compute nodes
typically do not have; run it separately afterward from a login node.

```bash
cluster-mlip literature-gap inventory \
  --orcid 0000-0002-1825-0097 \
  --author-name "Researcher Name" \
  --contact-email you@example.edu \
  -o gap
```

Fetches every paper by a given author about iron/iron-oxide (or other
transition-metal) clusters from [OpenAlex](https://openalex.org) -- a free,
open scholarly database, not a search-engine scrape -- and checks which
formulas mentioned in each paper's title/abstract are and are not present in
the inventory above. `literature-gap`'s first argument can be an existing
`inventory` output directory (as above) or a raw folder of ZIPs, in which case
the inventory is built inline into `<output>/inventory/`.

**This is the one command in the whole pipeline that needs live internet
access** -- run it from your laptop or an HPC login node, not an offline
compute node, since nothing else in ClusterMLIP makes a network call.
`--contact-email` is optional and only improves OpenAlex's rate-limit
priority (their documented "polite pool"); nothing is sent anywhere else.

There is deliberately **no default author**. Identify the researcher with a
verified ORCID using `--orcid`, or with an OpenAlex author ID using
`--author-id`; both options are repeatable and may be combined. ORCID URLs and
bare hyphenated iDs are accepted, and their check digits are validated before
the inventory is scanned or any network request is made. `--author-name` is an
optional display label for the report and is never used for identity
resolution. This keeps author disambiguation explicit and makes the command
equally applicable to any researcher. For example, Gennady L. Gutsev's
verified primary OpenAlex profile is `A5029253658`; his smaller,
likely-duplicate profile is `A5140774346` and can be included with a second
`--author-id`.

`literature-gap` fetches the author's **entire** OpenAlex bibliography
(fully paginated -- a prolific author's papers are not silently truncated to
one page) and narrows it to relevant papers itself, client-side. It does
*not* restrict the OpenAlex query by keyword: an earlier version did, and it
silently dropped most of a cluster-science author's papers, because this
field's titles are almost always written as formulas ("Fe6O20", "Fe2O4-6+
Clusters") rather than the English phrases a query-level filter needs
("iron oxide cluster") -- confirmed against a real author query that
returned far fewer papers than the author's true output. Instead, a fetched
paper is kept if either (a) a formula extracted from its title/abstract
shares an element with the local warehouse's own inventory (the precise
signal), or (b) as a fallback when no formula could be extracted, its
title/abstract contains one of `--keywords` (default `cluster`, `iron`,
`oxide`) as a plain substring. Replace the keywords for another element/
literature domain, e.g. `--keywords "nickel" "cluster"`. When the two counts
differ, the report states the funnel plainly: "OpenAlex returned N papers by
this author; M of them mention a relevant formula or keyword."

The output, `gap/literature_gap.md`, is written to be read directly by a
person, not parsed: a short plain-English count at the top, then papers
grouped as **"please send these"** (a formula was mentioned that isn't in the
inventory) first, **"not sure, please check"** (no specific formula could be
picked out of the title/abstract -- still listed, never silently dropped)
second, and **"already have"** last -- each paper as one short numbered block
(title, year, formulas mentioned, link), not a dense table. When OpenAlex
reports a genuinely free copy of a paper (its own `open_access` tracking --
an institutional repository, a fully-OA journal, etc.), the block links that
first as "Free PDF"; otherwise the DOI is labeled plainly as the publisher's
page, which may be paywalled. Most literature from before open-access norms
existed has no free copy anywhere regardless of institutional subscriptions
-- this reports that honestly rather than pointing at a paywall labeled as
free. This is meant to be handed directly to the researcher or data provider as a plain reading list,
e.g. as-is or pasted into an email -- it deliberately doesn't assume the reader wants to parse
JSON or a spreadsheet. The optional `--author-name` makes the heading suitable
for handing directly to that author. Formula matching is a text-mining
heuristic (a paper's title/abstract rarely states charge or multiplicity, so
matching is by formula only, and general "Fe_n"-style series notation without
a specific number isn't extracted as a composition) -- treat every row as a
starting point for a human to confirm, not an authoritative claim.

### Cross-checking against the actual PDFs, not just OpenAlex's abstract

When you have the real papers on hand (e.g. a delivery of PDFs from a
collaborator), `literature-gap`'s formula matching can go deeper than the
title/abstract OpenAlex provides -- a composition discussed only in a paper's
body would otherwise be missed entirely.

```bash
cluster-mlip pdf-index /path/to/papers.zip -o pdf_index
cluster-mlip literature-gap inventory --orcid 0000-0002-1825-0097 \
  -o gap --pdf-index pdf_index/pdf_index.json
```

`pdf-index` accepts a ZIP of PDFs, a folder of already-unpacked PDFs, or a
single `.pdf` file; it extracts each PDF's full text, mines it for chemical
formulas the same way `literature-gap` does, and looks for a DOI in the text
to match each PDF back to the right OpenAlex work. Every PDF gets an entry in
`pdf_index.json` -- one that couldn't be read (`unreadable`) or had no DOI
findable in its text (`no_doi_found`, e.g. a scan with no text layer) is
reported with that status rather than silently dropped. Passing the resulting
`pdf_index.json` to `literature-gap --pdf-index` unions those full-text
formulas into a matched paper's own compositions, and the report notes when a
paper's formulas came from "full text" rather than "title/abstract".

`pdf-index` needs the optional `pdf` extra (`pip install -e '.[pdf]'`,
pulling in `pypdf`) -- the only third-party dependency anywhere in this
package, isolated to this one command. Parsing a large PDF corpus is slow and
CPU-bound but, unlike `literature-gap`, needs no network access, so it belongs
on an offline compute node rather than a login node.
`scripts/run_pdf_index_slurm.sh` is a ready-to-submit single-core sbatch
script for LONI's `single` (serial) queue: `sbatch
scripts/run_pdf_index_slurm.sh /path/to/papers.zip`.

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
force set is not dominated by near-zero stationary-point forces. Defaults match
the Gaussian 09-era unrestricted BPW91 protocol used for the legacy database,
with a mixed `Gen` basis (see `basis.py`): `6-31G*` for light/organic elements
(intended to be upgraded to `6-311G*` later; diffuse functions are not used)
and an explicit def2-TZVP contraction with the f-shell dropped for Fe. An
unperturbed seed first runs:

```text
#p UBPW91/Gen SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm Opt Freq IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine
```

Rattled structures instead use the same method as a non-optimizing SP so their
off-equilibrium displacements are not erased. Both paths then use `--Link1--`
and the existing checkpoint for the final force label, on the same Gen basis:

```text
#p UBPW91/Gen Force SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm Guess=Read Geom=Checkpoint IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine
```

This is a single-point gradient calculation rather than a strict energy-only
SP because MACE training requires forces. Override the three stages with
`--route`, `--rattle-route`, and `--link1-route`, respectively. Do not mix
energies or forces from different electronic-structure methods in one target
head. `jobs.csv` preserves each job's parent structure; `run_one.sh` is a small
Gaussian runner to adapt to the local scheduler.

Generated filenames are human-readable but remain collision-safe. For example:

```text
fe2o2-5-12345-lb__fe2o2__minimum__q0-m5__reference__1a2b3c4d5e.gjf
fe2o2-5-12345-lb__fe2o2__minimum__q0-m5__r01__6f7e8d9c0b.gjf
```

The final short suffix is the stable machine identity, not a random filename.
`jobs.csv` is the authoritative crosswalk and records the readable ID, machine
job ID, original archive path and record ID, parent ID, charge/multiplicity,
variant and deterministic rattle seed, input and parent geometry SHA-256,
legacy energy/route, exact new routes, filenames, and input-file SHA-256.
`campaign_manifest.json` fingerprints `jobs.csv` and records the campaign-wide
method and resource settings. Rattles are derived independently from the
campaign seed plus parent ID and variant, so filtering or reordering seeds no
longer changes a geometry while leaving its apparent identity unchanged.
`collect` copies the same crosswalk fields into each labeled extxyz frame's
metadata, so the training dataset remains traceable even when moved separately
from the raw Gaussian campaign.

For safety, `prepare` refuses to overwrite a campaign containing Gaussian
outputs, checkpoints, or status files. Use a new output directory when changing
the method, seed, rattle policy, or selection.

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
  --account loni_perovsk27 \
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

For a persistent, job-by-job audit table rather than only batch totals, run:

```bash
cluster-mlip campaign-status gaussian_jobs
```

This refreshes `gaussian_jobs/progress.csv` and
`gaussian_jobs/progress_summary.json`. Every row carries the original-source
crosswalk alongside batch, pending/running/failed/completed state, timestamps,
return code, normal-termination and force-parse checks, output SHA-256, output
geometry SHA-256, and final label energy. For reference structures it also
reports the raw new-minus-legacy energy difference. That raw difference must
not be interpreted as an error when the legacy and label routes/bases differ;
the two route columns remain beside it specifically to make that distinction
auditable.

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
`CLUSTER_MLIP_XTB_WRAPPER_DIR`. Ordinary BPW91 Gaussian labeling does not
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
spin root signature when the source contains them. It also records the number
of resolved positive, negative, and weak Fe moments; whether the Fe spin block
is complete; whether all resolved Fe moments are parallel; and the mean
absolute local Fe spin inferred from the actual Mulliken population. Every
record also carries
a `state_inference` provenance tag (`filename`, `electron_parity_fallback`,
`default_unmatched_singlet`, or blank for a charge/multiplicity read directly
off an explicit Gaussian `Charge =`/`Multiplicity =` line) in `manifest.csv`,
`records.csv`, and `spin_inventory.csv`, so a defaulted guess is never
indistinguishable from a validated state:

```bash
cluster-mlip spin-extract Warehouse2.zip -o spin_inventory
```

### Automatic data-derived oxide campaign

For hundreds of oxide structures, do not select record IDs or multiplicities
by hand. Prepare the whole archive with:

```bash
cluster-mlip prepare-spins spin_inventory/seeds.extxyz \
  -o gaussian_spin_jobs_auto_v1 \
  --auto-from-data \
  --elements Fe,O \
  --require-elements Fe,O \
  --memory 32GB \
  --nproc 16
```

`--elements Fe,O` excludes Ga-, Ti-, U-, and other mixed oxides, while
`--require-elements Fe,O` requires both elements to be present. Use only the
latter if the mixed-oxide entries are intentionally part of the campaign.

The planner groups real archive entries by formula and charge. For each group,
the highest reliably observed multiplicity is accepted when either (a) its
archived Mulliken data show all resolved Fe moments parallel or (b) the group
contains at least two reliably observed multiplicities. Case (b) is explicitly
labeled `highest_observed_group_multiplicity`, not presented as proven local-
spin coupling. A singleton group without parallel-spin evidence is skipped as
`insufficient_real_data_no_parallel_reference`.

No ionic or idealized `4*N(Fe)+1` multiplicity is generated. Thus an Fe10 oxide
group actually observed at multiplicities 29 and 17 produces a 29 -> 27 -> 25
-> 23 -> 21 -> 19 -> 17 ladder; the code does not invent multiplicity 41. Each
archived geometry remains its own parent and its archived multiplicity is the
target. The inferred high-spin state is initialized on that target geometry,
then every lower state reads the immediately preceding checkpoint.

Start with `spin_plan_summary.json`; it counts planned/skipped records by
inference, skip reason, formula/charge group, and high-to-target ladder. Drill
into `spin_plan.csv` only for exceptions. It contains one
row per archive record with `planned`/`skipped`, formula/charge group, archived
target, inferred high multiplicity, inference class, evidence record IDs, all
observed group multiplicities, generated input, and any skip reason.
`spin_campaign.json` hashes both this plan and `spin_jobs.csv`; every generated
stage carries the plan ID and evidence crosswalk.

### Manual single-case or fragment campaign

Then prepare a trusted high-spin reference and a one-spin-flip-at-a-time
multiplicity ladder. Missing intermediate multiplicities are inserted
automatically. For Fe10, the following produces
29 -> 27 -> 25 -> 23 -> 21 -> 19 -> 17 even though only 17 is requested:

```bash
cluster-mlip prepare-spins spin_inventory/seeds.extxyz \
  -o gaussian_spin_jobs \
  --record-id REPLACE_WITH_FE10_M29_RECORD_ID \
  --high-spin 29 \
  --targets 17 \
  --strategy ladder \
  --memory 32GB \
  --nproc 16
```

Multiplicity 29 is the only directly initialized state. Each lower-spin Link1
stage reads the immediately preceding checkpoint with `%oldchk`, writes a
different checkpoint with `%chk`, and uses
`Geom=Checkpoint Guess=(Read,Always)`. Thus the multiplicity-17 state cannot be
mistaken for an independently initialized low-spin calculation. This is a
multiplicity ladder, not proof of the desired AFM coupling; the output must
still be characterized from local spins and `<S^2>`.

The default route is the Gaussian 09-era protocol used for the original work:
unrestricted BPW91/6-311G*, the VShift/NoIncFock convergence controls, NoSymm,
Opt/Freq, the historical IOP settings, and UltraFine integration. `Stable=Opt`
and full spin-density population analysis are added so the electronic root can
be audited. It does not use wB97M-V/def2TZVPP.

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

Choose the manual fragment pathway by itself:

```bash
cluster-mlip prepare-spins spin_inventory/seeds.extxyz \
  -o gaussian_spin_jobs \
  --record-id REPLACE_WITH_FE10_M29_RECORD_ID \
  --high-spin 29 \
  --targets 17 \
  --strategy fragment \
  --fragment-spec my_fe10_fragments.json
```

Use `--strategy both` to prepare independent ladder and fragment candidates;
`auto` (the default) selects both when `--fragment-spec` is supplied and the
ladder otherwise. Fragment-only mode refuses to proceed unless every requested
record/multiplicity has an explicit fragment definition.
Copy the exact high-spin parent `record_id` from `spin_inventory.csv` and pass
it with `--record-id`; repeat the option only when you intentionally want more
than one parent in the same campaign. An unknown ID fails immediately.

`spin_jobs.csv` records, for every state, its initialization and audit
classification, parent state, stage, intended charge/multiplicity, predecessor
job and checkpoint, new checkpoint, complete checkpoint lineage, source
geometry hash, input hash, and (for fragment jobs) fragment-definition hash.
The original fragment JSON is copied to
`fragment_specifications.lock.json`; `spin_campaign.json` records the selected
strategy, trusted high spin, route, requested targets, and manifest hash.
Generated input filenames are human-readable and include the original source,
formula, charge/multiplicity, and pathway while retaining a short collision-safe
machine identity.
In manual mode, non-high-spin seed records are deliberately not submitted
directly; they are listed in `skipped_spin_seeds.csv`. In automatic data-derived
mode, lower-spin records instead receive their inferred high-to-target ladder,
while unsupported groups are listed with a precise evidence failure reason.
Preparation also refuses to overwrite a nonempty campaign directory; use a new
output directory for each attempt so checkpoints, inputs, and audit hashes
cannot be mixed across runs.

The ordinary Slurm generator accepts a spin campaign and deduplicates the
multiple Link1 manifest rows into one submitted Gaussian input. Live progress
still expands that shared output back into individual multiplicity stages:

```bash
cluster-mlip prepare-slurm gaussian_spin_jobs --jobs-per-batch 30 --concurrent-jobs 4
./gaussian_spin_jobs/submit_gaussian_batches.sh
cluster-mlip campaign-status gaussian_spin_jobs
```

For a 29 -> ... -> 17 ladder, `progress.csv` therefore shows m29, m27, ...,
m17 separately, including whether each stage was observed, optimized, advanced
to a successor, stable, and characterized by local spins and `<S^2>`. Scheduler
completion is not scientific root validation; run `validate-spins` below after
the outputs are available.

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
  whether it appeared in an output, plus its complete preparation lineage and
  a `verified`/failure status, normal termination/optimization/stability,
  `<S^2>`, local-spin root signature, and predecessor-completion status. For a
  fragment job it also compares the converged net Mulliken spin on each
  fragment with the requested alpha/beta orientation;
- `report.md`, `summary.json`, and `errors.tsv`.

Geometry matching uses a translation-, rotation-, and atom-order-invariant
element-pair distance fingerprint. Electronic-root comparison uses element-
resolved local-spin distributions when both old and new outputs contain them.
An `alternative_root` is retained as potentially useful training data rather
than silently discarded. `--strict` returns nonzero for missing states,
incomplete calculations, alternative roots, planned stages without output,
broken checkpoint/fragment lineage, or outputs not linked to `spin_jobs.csv`.

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
