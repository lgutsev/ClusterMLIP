# Package 26: ClusterMLIP legacy inventory and overlap audit

Owner: ClusterMLIP. Source: `examples/legacy_inventory/` in lgutsev/ClusterMLIP.
This package reads existing data only; no Gaussian calculations or training.

## Run on QB4

From the loni_smoke_tests checkout, after `git pull --ff-only`:

```bash
bash submit_smokes.sh --dry-run 26
bash submit_smokes.sh 26
```

The job uses `single`, one CPU, two hours, no `--mem`. A longer walltime may be
needed once collection size/timing is known; override with `sbatch --time` when
submitting manually from this package directory. Runtime defaults to
`/project/lgutsev/env/cluster_mlip_runtime`; it needs NumPy/SciPy, `strings`, and
Gaussian `formchk` for binary checkpoints. No installation happens in the job.
Parser modules are a pinned snapshot copied by the generator, so no branch switch
or modification of the running QB3 ClusterMLIP checkout is needed.

Source: `/ddnB/project/ramu/gutsev/glg/`.
References: `/ddnB/work/lgutsev/ClusterMLIP/campaigns/{FenOm_Warehouse,FenOm_Warehouse2,General_Warehouse}/extracted/{manifest.csv,seeds.extxyz}`.
Missing pairs fail explicitly; manifest IDs must correspond to geometry IDs.
Environment overrides: LEGACY_SOURCE_ROOT, CLUSTER_CAMPAIGNS_ROOT, CLUSTER_MLIP_ENV.

Every job gets a new, non-overwriting campaign `run_<jobid>` directory under
`Gutsev_Legacy_2026-10`. Original sources and existing warehouses are read-only.
The full manifest and geometry data stay there; compact tables/reports are copied
into package `outputs/` for normal export/intake. A failed attempt is retried with
`submit_smokes.sh --retry-failed 26`; it does not overwrite earlier campaign runs.

```bash
bash export_results.sh 26
```

## Review

`outputs/DONE` and exit code 0 mean the inventory process completed, not that all
records are accepted. Read `RUN_STATUS.json`, `summary.json`, `files.csv` and
`overlap.csv`: partial/unreadable files and no-record sources remain visible.
Unsupported files are hashed/listed. Zip contents are extracted only temporarily.
Every parsed seed is retained, with unique incoming ID and original-ID aliases.

The overlap filter compares element-pair sorted distances with a 0.02 Å maximum
aligned-deviation target, then performs element-preserving assignment and proper
rotation alignment. Distance-only matches are candidates, not duplicate identity.
Symmetric ambiguous assignments remain unresolved rather than being called new.
Different global states are retained. Raw route equality is reported conservatively;
no records are automatically deleted, merged or accepted into training.

Force-bearing step counts are per document. They are not forces assigned to seeds.
Electronic-root, isotope, convergence and numerical-setting checks remain review
requirements. Existing raw file hashes are not in the old manifests: this first
pass detects incoming byte duplicates, and cross-warehouse geometry overlaps,
not verified byte identity against inaccessible historical originals.
No structural-family counts are inferred from file/record counts.

Local validation covers geometric invariance and an end-to-end synthetic three-
warehouse run. A live LONI run and compatibility with the actual collections remain
unverified until submission. No jobs have been submitted by this delivery.
