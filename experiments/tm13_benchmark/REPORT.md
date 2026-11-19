# Fe13 transfer benchmark: status report (2026-11-18)

Branch `exp/tm13-benchmark` (from `origin/main` d0f3ec0), code commit `8f2cfc2`. Units: eV, eV/Å, Å.

## Where it stands

| Step | Status |
|---|---|
| Fe13 held-out set (source + 4 displacements) | **done**, from the group's own UBPW91 Fe13 structure |
| Santos et al. coordinates | **blocked**: archive not found, ACS pages refuse automated access |
| Model evaluation on the 5 geometries, relaxations | **done** (7 models, CPU) |
| UBPW91 reference forces and spin survey | **pending**: package 144 (staged as 143) taken in by the desk, not yet submitted |
| Force scoring against the reference | **pending** on package 144; scripts ready |

No Fe13 force error against DFT exists yet, so the handoff's decision rule cannot be applied yet. What can
be said now comes from the legacy UBPW91 data and is listed under "Provisional findings".

## Inventory (handoff step 1.1)

- **Fe16 spin-aware model:** `fe16_spin_B_s1/s2` = `12_local_spin_mp/runs/long_B_s{1,2}` (MACE from scratch,
  charge + multiplicity input, 60 neutral Fe16 frames, M 47–53). On 21 unseen Fe16 frames its force MAE was
  0.051 eV/Å (CHECKPOINT.md of that experiment).
- **Other models scored:** `fe16coh_spin_fold1` (today's spin held-out experiment, Fe16 ladder + Fe16+CO+2H,
  M 39–49; read only), MACE-MPA-0, MACE-MATPES-r2SCAN, and their Fe16+CO+2H M = 49 fine-tunes. SHA-256 of
  every file is in `report/summary.json`.
- **Reference DFT:** UBPW91/6-311++G*, G09 D.01 on QB4. The route that completed cleanly is package 112/116's:
  per M one `Stable=Opt` link and one `Force` link, `Guess=Read`, `IOP(5/13=1,5/36=1,8/11=1)` in every
  link. The original "three Fe16 geometries × P0–P3" fragment pilot did not complete (packages 61/78/82
  failed to converge); the flip-down ladder (91, 99, 112, 116) is what works, and package 144 reuses it.
- **Local-spin output:** Mulliken puts −8.2 on the central Fe13 atom. This is the known diffuse-basis
  artifact, so the package prints Hirshfeld spins.

## Geometry source (step 1.2)

- **Santos et al.:** the Mendeley archive is not indexed (DataCite, Mendeley search, web), and the ACS article
  and SI pages return 403 (bot check) to both fetch and the browser pane. No coordinates were obtained and
  none were reconstructed. Whether the archive has forces is unknown.
- **Used instead:** set `fe13_legacy_ubpw91_m45`, the lowest neutral Fe13 state in the group's legacy archive
  (Oct9 `FE13.doc` block 0, state `2fdecba6912852bf`, UBPW91/6-311+G* optimisation, M = 45,
  −447086.38113 eV). Distorted icosahedron, centre CN 12, 42 bonds, mean nearest-neighbour distance 2.505 Å.
- **Agreement with the paper (cross-method only):** paper 44 μB (M = 45) and d_av 2.49 Å (ECN-weighted,
  PBE+TS) against legacy M = 45 and 2.505 Å.

Manifest with hashes, amplitudes and seed: `sets/fe13_legacy_ubpw91_m45/manifest.json`.

| Configuration | What | RMS / max displacement |
|---|---|---|
| c0_source | unchanged | 0 |
| c1_expand, c2_compress | radial scale 1 ± 0.01682 (shell atoms move 0.040 on average) | 0.038 / 0.040 |
| c3_surface | atom 2 moved 0.050 tangentially towards atom 6 | 0.014 / 0.050 |
| c4_random | seed 13, translation removed, rescaled | 0.040 / 0.076 |

## Provisional findings

These use the legacy data, not new DFT.

1. **Residual forces at the UBPW91 minimum.** If c0 is a stationary point of UBPW91 at M = 45 (it comes from
   a legacy `OPT` job; frame 0000 of package 144 checks this), the reference force there is about zero and a
   model's force is its error. Zero-force is exact there by construction, so this is not the displaced test.

   | Model (M = 45 where it takes spin) | Force RMS at c0 | Mean over 4 displaced |
   |---|---|---|
   | fe16_spin_B_s2 | 0.077 | 0.118 |
   | fe16_spin_B_s1 | 0.106 | 0.159 |
   | mace_mpa0 | 0.164 | 0.236 |
   | ft_mpa0_fe16coh_m49 | 0.202 | 0.206 |
   | fe16coh_spin_fold1 | 0.222 | 0.212 |
   | mace_matpes_r2scan | 0.327 | 0.428 |
   | ft_r2scan_fe16coh_m49 | 0.430 | 0.437 |

   The residuals at the minimum are as large as the models' whole response to a 0.04 Å displacement. Unless
   the UBPW91 forces on the displaced geometries are much larger than ~0.1 eV/Å, no model will beat zero-force.

2. **Relaxation from c0** (BFGS to 0.005 eV/Å; `report/relaxations.csv`). At M = 45, and for the spin-blind
   models, every model keeps the icosahedron. Aligned permutation-aware RMSD 0.08–0.16 Å; mean
   nearest-neighbour change from +0.02 (MPA-0) to −0.14 Å (fe16coh_spin_fold1 contracts the cluster).
   `fe16_spin_B_s1` at M = 47 leaves the icosahedron (38 bonds, RMSD 0.43 Å), while seed 2 keeps it. The
   legacy M = 47 state is a near-perfect icosahedron, so this endpoint is probably a model artifact; it has
   not been checked with DFT.

3. **Spin energetics do not transfer.** Vertical energy at c0 relative to M = 45:

   | Model | M = 43 | M = 47 |
   |---|---|---|
   | fe16_spin_B_s1 | +0.42 | +0.47 |
   | fe16_spin_B_s2 | +2.09 | +2.89 |
   | fe16coh_spin_fold1 | −1.28 | −1.00 |
   | legacy UBPW91 (adiabatic, each M at its own geometry) | +0.034 | +0.32 |

   Two seeds of the same model differ by 2 eV and the third model has the opposite sign. Forces also change
   with M by more than the displacement signal (s1: 0.106 at M = 45, 0.010 at M = 47).

4. **Spin ambiguity in the reference is real.** The legacy M = 43 state lies 34 meV above M = 45. Package 144
   measures the force difference between M = 45 and 43, and between ladder and fresh-guess solutions, on the
   same geometries before any geometry-only label is proposed.

## Pending: package 144 (staged as 143)

`D:\MLIP_Work_Folder\Cluster_MLIP\loni_package\143_clustermlip_fe13_heldout_reference`, hand-off note
`HANDOFF_143_TO_DISPATCH.md` approved and sent 2026-11-18; the desk took it in as **144_clustermlip_fe13_heldout_reference** (desk commit cc43c56), review pass, not yet submitted. 9 tasks, 38 links, at most 2,592 SU (expected a few hundred).

- Frames 0000–0004: all five geometries, ladder M = 47 → 45 → 43.
- Frames 0005–0008: fresh SCF at M = 45 and at M = 43 on c0 and c4.

When it returns:

```
python experiments/tm13_benchmark/collect_dft.py --package RETURNED_DIR --set experiments/tm13_benchmark/sets/fe13_legacy_ubpw91_m45 --out experiments/tm13_benchmark/report
python experiments/tm13_benchmark/score_models.py --set experiments/tm13_benchmark/sets/fe13_legacy_ubpw91_m45 --models experiments/tm13_benchmark/models.json --out experiments/tm13_benchmark/report --dft experiments/tm13_benchmark/report/dft_labels.extxyz
```

`collect_dft.py` screens every link (SCF convergence, stability, ⟨S²⟩, Force link kept the Stable link's
solution), keeps competing solutions, and reports their force spread. Its parser was checked on package 112's
log. `score_models.py` then fills the DFT columns of `report/results.csv` (now `status = dft_pending`).

## Not done

- No comparison on the Santos geometry.
- No DFT check of the `fe16_spin_B_s1` M = 47 endpoint (no input staged).
- Stage 3 (Co13, Ni13, Cu13, Fe13H and beyond): not started. 6-311+G* covers Co, Ni, Cu and H; the 4d/5d
  metals need a different basis or ECP and a relativistic treatment that the current route does not have.
- `score_models.py` with `--dft` has not run on real Fe13 labels.

## Recommendation

Package 144 is with the desk. Its result decides between the handoff's branches; on present evidence the likely outcome is
"model struggles", with spin transfer the clearest failure. Do not start the Co/Ni/Cu stage before that.
