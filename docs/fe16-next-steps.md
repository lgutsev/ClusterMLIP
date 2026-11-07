# Fe16 MLIP: next steps (plan, 2026-10-07)

## Where things stand

- **The Fe16 MACE now fits its training data, but does not generalize.** The original stall was
  the learning rate (fix: lr 1e-3, forces_weight 10). Train force MAE is 2–5 meV/Å; on truly
  unseen frames it is ~51 meV/Å.
- **Local spin is not the cause.** Per-Fe Mulliken spin as an input did not help (A/B/C test,
  branch `test/fe16-local-spin-message-passing`). Fe16 Mulliken moments are partly a
  diffuse-basis artifact (central Fe ≈ −5.6), and the data has no same-geometry, same-M pairs
  with different spin arrangements to test it on.
- **The bottleneck is structural coverage.** The 51 Fe16 spin-ladder jobs cover 16 of the 62
  distinct UBPW91 minima (11 of the 29 within 1 eV of the global minimum), with optimization
  paths only and no rattles. 60 of the 90 held-out frames were near-copies of training frames,
  so earlier held-out errors were optimistic.
- `collect` on `main` now labels each frame by what it is (`frame_role`, #11). Frames treated as
  stationary points by the physical checks drop from 8,175 to 272 on `dataset_spin_v0`, which was
  collected with older code: every frame `labeled`, spins only in metadata.

## Steps, in order

### 1. Re-collect and fix the evaluation split (LONI, cheap)

- Collect the W2 spin campaign from `main` with `--frames all`, so frame roles and the job type
  recovered from the filename come through.
- Exclude the four 118–134 eV/Å force outliers (Fe2HO/Fe2NO, M=8).
- Split train/valid/test by **unique source geometry** (fingerprint), not by parent record, so
  near-copies of training frames cannot reach the test set.
- Always report independent geometries separately from frame counts.

### 2. Structural coverage campaign for Fe16 (through a grouping step)

- **Grouping/dedup step before launch (new, not in the repo yet).** About 47% of W2 jobs repeated
  another job's geometry, target M and search kind. Group planned jobs into
  motif × charge × spin-ladder study units, drop duplicates, set a budget per group, and carry
  `group_id` through `collect` and the splits.
- **The 17 missing minima within 1 eV**, run as spin ladders via `prepare-spins --record-id ...`.
  The record IDs are in the 2026-09-29 session notes; regenerate with the fingerprint match
  against `spin_plan.csv`.
- **Rattles on all covered minima** at σ = 0.05 and 0.10 Å, as single points, 2–4 per minimum.
- **Fe16 IRC points as fixed-geometry Force jobs, at the IRC's own M** (~460 jobs with Fe16N2;
  a subset along each path is enough). This depends on the IRC force-label review below.

### 3. Retrain, learning curve, then active learning

- Retrain the baseline (global charge/M) with the fixed settings. Draw a learning curve over
  25/50/100% of unique geometries.
- Once held-out forces are reasonable: 700 K MLIP MD with a committee, select frames by
  disagreement (`select-next-batch`), relabel, repeat.
- Free extra data to add: descent paths of the IRC-seeded W2 jobs (`frame_role`
  `optimization_step`), and legacy Fe16 optimization steps at the matching
  UBPW91/6-311++G* level (668 records).

### 4. Pending reviews and decisions (user)

- QB4 run of the IRC force-label review kit (`D:\MLIP_Work_Folder\Cluster_MLIP\w2_review_kit\`,
  `RUNS.md` lists the 13 inputs and 8 IRC logs). The IRC code exists only in the uncommitted
  `feat/irc-force-labels` checkout.
- Merge decisions: `feat/supported-delta-vasp`, `diag/fe16-baseline-training`,
  `test/fe16-local-spin-message-passing`, and the redundant `feat/collect-atomic-spins`
  (superseded by `local_spins.py`).

## Later (needs a working Fe16 model)

- **Supported clusters:** VASP cluster–support interaction labels (`vasp-prepare`/`vasp-collect`,
  NUPDOWN = M − 1, PW91/PBE calibrated against BPW91 on finite supports), the Δ-model, MD, and
  abTEM imaging (`examples/supported_fe16_tem/`). MACE-MP-0 alone is not usable: it gets
  graphene strain and the absence of Fe–MgO bonding wrong.
- **Reaction targets (G. Gutsev, 2026-10-07):** Fe16On (n = 17–22) bubble/shell formation; O
  removal from Fe16On via NOx/NOH channels (ΔG and rate constants); a reducible CeO2 support as an
  O sink (needs a reductant; DFT+U and Ce3+ polaron sampling on the support side).
- **Paper:** a spin-resolved MLIP for high-spin Fe clusters with buried sites, plus the dataset
  and the documented failure of the foundation models (POLAR-1 diverges once an Fe atom has 11 or
  more Fe neighbours).

## Constraints to keep

- UBPW91 is the reference level, by the group's choice. OMol25/hybrid models are a change of
  functional, not an upgrade.
- Laptop: float32 for relaxations and MD; float64 single points where energies near 5×10⁵ eV are
  compared.
