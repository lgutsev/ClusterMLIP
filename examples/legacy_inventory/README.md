# Package 26: ClusterMLIP legacy inventory and overlap audit

Owner: ClusterMLIP. Source: `examples/legacy_inventory/` in lgutsev/ClusterMLIP.
This package reads existing data only; no Gaussian calculations or training.

## Run on QB4

From the loni_smoke_tests checkout, after `git pull --ff-only`:

```bash
bash submit_smokes.sh --dry-run 26
bash submit_smokes.sh 26
```

The job uses `single`, 16 allocated cores for memory, a 24-hour cap, no `--mem`.
Parsing uses one CPU thread. Runtime defaults to
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

## Progress

New runs write `PROGRESS.json` (an atomically replaced snapshot) and
`progress.jsonl` (history) in the campaign's `run_<jobid>` directory. The Slurm
stdout log receives flushed `PROGRESS` lines. Each item is recorded before it is
processed; a heartbeat every 60 seconds reports the latest counters even while
a large file or overlap comparison is still in progress.

Counters include phase, completed/total items, phase percentage, current file or
record, extracted records, files with parsing errors, overlap pairs, and elapsed
seconds. Parsing counts source files, including archives and unsupported files;
overlap counts incoming records. These are separate phase percentages, not an
overall completion estimate. Reference loading happens before tracking starts.

```bash
cd /work/lgutsev/loni_smoke_tests/26_clustermlip_legacy_inventory
cat "$(cat outputs/result_path.txt)/PROGRESS.json"
```

The exit/TERM traps copy available progress files into `outputs/` for export.
An OOM kill or abrupt node loss may prevent those traps; the campaign files and
Slurm log still retain the latest writes. A `running` snapshot records the last
observation, so check the queue to determine whether the process is still alive.
An exception records `failed` and its message. This is monitoring, not a resume
checkpoint. Existing running jobs gain no progress instrumentation from this edit.

## Result checks

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
