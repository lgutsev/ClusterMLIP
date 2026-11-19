# Fe13 transfer benchmark: report (2026-11-19)

Branch `exp/tm13-benchmark` (from `origin/main` d0f3ec0), code commit `8f2cfc2`. Units: eV, eV/Å, Å.

## Result (2026-11-19, package 144 returned)

Reference: UBPW91/6-311++G*, G09 D.01, QB4 job 1191998. Tables: `report/dft_links.csv`, `report/dft_states.csv`,
`report/dft_labels.extxyz`, `report/results.csv`.

**Reference quality.** M = 45 and M = 47 are clean on all five geometries (converged SCF, stable, Force link
kept the solution). At M = 45 a fresh SCF on c0 and on c4 lands on the ladder state exactly (same energy to
1e-8 Ha, same forces), so there is one M = 45 solution. c0 is a UBPW91 stationary point: force RMS 0.0035 eV/Å.
**M = 43 did not converge anywhere** (129-cycle SCF in every Stable=Opt link, 7 of 7 states rejected; the
fresh M = 43 guess on c0, frame 0006, died in l508). No M = 43 label is used. The unconverged M = 43 energies
sit 0.21–0.30 eV above M = 45, not the 0.034 eV of the legacy adiabatic gap, so they are not to be trusted.

**Force error at M = 45** (RMSE, eV/Å; spin models evaluated at M = 45):

| Model | c0 | c1 expand | c2 compress | c3 surface | c4 random | mean of c1–c4 |
|---|---|---|---|---|---|---|
| DFT force RMS (= zero-force error) | 0.004 | 0.299 | 0.359 | 0.048 | 0.323 | 0.257 |
| mace_mpa0 | 0.165 | 0.198 | 0.108 | 0.168 | 0.220 | 0.174 |
| fe16_spin_B_s1 | 0.105 | 0.183 | 0.239 | 0.100 | 0.213 | 0.184 |
| fe16_spin_B_s2 | 0.077 | 0.236 | 0.244 | 0.075 | 0.213 | 0.192 |
| ft_mpa0_fe16coh_m49 | 0.202 | 0.088 | 0.306 | 0.210 | 0.329 | 0.233 |
| mace_matpes_r2scan | 0.327 | 0.336 | 0.298 | 0.321 | 0.312 | 0.317 |
| fe16coh_spin_fold1 | 0.223 | 0.133 | 0.581 | 0.233 | 0.440 | 0.347 |
| ft_r2scan_fe16coh_m49 | 0.430 | 0.396 | 0.349 | 0.427 | 0.427 | 0.400 |

- The Fe16 spin model beats zero-force by about 25–30 % on the displaced geometries, and is no better than
  untuned MACE-MPA-0. Its force MAE there is 0.09–0.19 eV/Å against 0.051 on unseen Fe16 frames.
- On c0 and c3, where the true forces are near zero, every model is worse than zero-force.
- Energy change against c0 at M = 45: the spin model is within 0.11 eV on every geometry (DFT changes are
  0.005–0.15 eV), always too low.

**Spin dependence is not captured.** DFT forces at M = 47 differ from M = 45 by 0.24–0.28 eV/Å RMS on the same
geometry, as large as the displacement signal. Scored at M = 47 against M = 47 forces, the spin models have
RMSE 0.39–0.44 eV/Å on the displaced geometries, no better than predicting zero. Vertical M = 47 − M = 45 at
c0 is +0.515 eV in DFT; the models give +0.47 (s1), +2.89 (s2) and −1.00 (fe16coh fold 1).

**Decision.** This is the "model struggles" branch: small gain over zero-force at the trained-like spin state,
none at the neighbouring one. Co13/Ni13/Cu13 and Fe13H should wait. The relaxation and provisional sections
below are from 2026-11-18 and stand, except that finding 1's assumption (c0 stationary) is now confirmed.

**Open.** M = 43 needs a different SCF strategy before it can be labelled (a new package, not a rerun of
frame 0006). The `fe16_spin_B_s1` M = 47 endpoint is still unchecked. Santos geometry still missing.

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
