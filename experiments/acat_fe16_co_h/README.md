# Fe16 + CO + 2H adsorbate-search pilot

Fixed catalyst, fixed composition: neutral Fe16 (geometry `fe16-200-49-gf`, the UBPW91 M = 49
optimum) with one CO and two H. Two search arms with the same budget: random placement and an
evolutionary search. Findings, limits and the recommendation are in
[`docs/acat_cluster_feasibility.md`](../../docs/acat_cluster_feasibility.md).

**No reference (DFT) energy exists for Fe16 + CO/H.** Everything the search prints is a model
energy from a foundation checkpoint that failed the reference gate. It is a software and
search-method benchmark and a source of structures to label, not a prediction.

## Files

| File | What it does |
|---|---|
| `catalyst/fe16_200_49_gf_m49.extxyz` | The catalyst: 16 Fe, atom IDs 0–15, charge 0, M = 49, provenance in the header |
| `common.py` | Loads the catalyst, builds the site list, makes the calculator |
| `run_search.py` | Runs the arms and seeds of one config |
| `configs/smoke_toy.json` | 20 relaxations per arm on the built-in toy potential (no model file, ~15 s) |
| `configs/pilot_mpa0_stageA.json` | 200 relaxations × 3 seeds × 2 arms on MACE-MPA-0, Fe fixed (Stage A) |
| `gate_small_clusters.py`, `gate_forces.py` | Reference gate on existing UBPW91 labels (Fe2/Fe4 with CO or H, bare Fe16) |
| `analyze_runs.py` | Arm comparison, pooled candidates, DFT shortlist, figures |
| `make_dft_inputs.py` | Draft Gaussian inputs for the shortlist (preparation only) |
| `check_acat_sites.py` | Tries ACAT's own cluster site finder; needs `asap3` (Linux) |
| `report/` | Small result files and figures from the runs described in the feasibility doc |

## Run

The environment is the laptop GPU env plus ACAT (`D:\MLIP_Work_Folder\envs\acat_fe16`, a venv on
top of `micromamba\envs\defects`). Run from this folder.

```bash
python run_search.py configs/smoke_toy.json --out OUT/smoke_toy
```

```bash
python run_search.py configs/pilot_mpa0_stageA.json --out OUT/pilot_mpa0_stageA
```

```bash
python analyze_runs.py OUT/pilot_mpa0_stageA --out OUT/analysis --second-model mace:~/.cache/mace/MACE-matpes-r2scan-omat-ft.model
```

```bash
python make_dft_inputs.py OUT/analysis/shortlist.extxyz --out OUT/dft_draft
```

Each `<arm>_s<seed>/` holds:

- `manifest.jsonl`: one line per proposal, including rejected ones. Fields: candidate ID, seed,
  operator, parent IDs, genes (species, site, mode, tilt, azimuth), the catalyst atoms of each
  site, status (`invalid` / `duplicate_start` / `relaxed`), convergence, steps, calculator calls
  (own and cumulative), wall time, model energy, fragments found after relaxation, flags,
  duplicate-of.
- `structures.extxyz`: start and relaxed geometry of every candidate with per-atom `role`
  (1 catalyst, 2 adsorbate), `mol_id` and `cluster`.
- `summary.json`: counts for the run and its config.

`provenance.json` next to them records the commit, catalyst hash, site counts and the model
file hash.

## Budget and matching

The budget counts relaxations started. Both arms use the same site list, builder, distance
filters, constraint (all Fe fixed), optimizer (BFGS, fmax 0.05 eV/Å, 200 steps), energy
definition and duplicate test. With the same seed the first 20 relaxations (the initial
population) are identical in both arms. Proposals rejected before relaxation cost no calculator
call and are counted separately.
