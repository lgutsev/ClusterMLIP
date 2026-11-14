# Gaussian UBPW91 vs VASP: calibration set

Status: run as desk package **62** (staged 2026-10-16 as 56; results exported 2026-10-19).
**Outcome: delta only** (see [Results](#results-package-62-2026-10-19)).
Generator: `experiments/gaussian_vasp_calibration/make_calibration_package.py`.
Analysis: `experiments/gaussian_vasp_calibration/analyze_calibration.py`.

## Results (package 62, 2026-10-19)

The desk reported 144/147 jobs done. All 42 G09 tasks ended with 3 Normal terminations, and
102/105 VASP jobs finished. The 3 `probe_libxc_bpw91` jobs failed because `vasp6/6.6.1-cpu` has no
Libxc, so the code cannot run BPW91.

The full report is written to `D:\MLIP_Work_Folder\Cluster_MLIP\gv_calibration_62\`
(`report.md`, `report.json`, `states.csv`). To regenerate it:

```
python experiments/gaussian_vasp_calibration/analyze_calibration.py \
  --cluster-mlip-src .claude/worktrees/periodic-vasp-ingest/src \
  D:/MLIP_Work_Folder/hpc_smoke_tests/62_clustermlip_gaussian_vasp_calibration \
  -o D:/MLIP_Work_Folder/Cluster_MLIP/gv_calibration_62
```

**SCF screen.** Normal termination does not prove convergence. A Gaussian link counts as
unconverged if it printed "Convergence criterion not met", ended with Conv > 1e-6, or used
MaxCyc. These were unconverged:

- 14 of 42 UBPW91 links, including Fe16 M49/M51, three Fe16N2 states and the Fe4N2/Fe5N2 states;
- 4 UPW91PW91 links and 2 UPBEPBE links;
- 10 VASP runs that hit NELM = 300 (6 PW91 and 4 PBE, all Fe2–Fe5).

None of them enters a metric.

**G09 rerun control.** 7 of 9 archived dataset_spin_v0 states reproduce to 0.00 meV with identical
fmax. The other two are unconverged reruns:

- Fe16 M51: the default-guess UBPW91 SCF failed and ended 19.7 eV high.
- Fe16N2 off-equilibrium frame: 1.6 meV off.

**Pair c (VASP vs G09 UBPW91), matched states:**

| | PW91 | PBE |
|---|---|---|
| matched states / pattern match | 26 / 100 % | 28 / 100 % |
| offset-fit residual RMS / max (meV/atom) | 33.0 / 121.8 | 31.7 / 124.1 |
| relative-energy MAE (meV/atom) | 7.9 | 12.1 |
| spin-gap MAE / max (meV per cluster) | 46 / 83 | 70 / 166 |
| force RMSE Fe / N / O (meV/Å) | 208 / 272 / 169 | 227 / 337 / 164 |
| lowest-M disagreements in ladders | 0 / 4 | 0 / 4 |
| Fe moment, VASP − Hirshfeld (μB) | −0.25 ± 0.12 | −0.23 ± 0.15 |

Both functionals fail even the separate-head limits on the residual, the relative energies and
the forces, so the outcome is **delta only**.

The Fe ≥ 8 subset (the Fe16 regime) is reported for information:

- the residual drops to 8–11 meV/atom;
- the forces still differ by about 150 meV/Å;
- so that subset is also delta only.

**Attribution.**

- The code part (G09 PW91 vs VASP PW91, at the same functional) already fails: its force RMSE is
  163 / 270 / 160 meV/Å for Fe / N / O (118 for Fe at Fe ≥ 8). The VASP forces point the same way
  as the Gaussian ones but are 3–4× larger. The two codes therefore put the minima at different
  bond lengths (PAW vs 6-311++G*). More data cannot remove that difference, and the Libxc route
  that would remove the functional part is not available.
- The functional part on its own (G09 PW91 or PBE vs BPW91) is small at Fe ≥ 8: relative MAE
  0.7–2.1 meV/atom and force RMSE 16–63 meV/Å. It is size-dependent, though: Fe2 and Fe4 deviate
  by up to 77 meV/atom.
- Spin physics agrees. The lowest ladder M is the same in every ladder, and 97–100 % of the Fe
  sign patterns match.

**VASP convergence.**

- The box is converged: a 10 Å gap moves energies by 0.2 meV/atom relative to the 12 Å
  production gap, and a 14 Å gap by 0.05.
- PREC = Normal changes energies by 0.07 meV/atom.
- ENCUT 650 vs 520 moves Fe2N2O2 by 2.4 meV/atom and its forces by 58 meV/Å, while Fe8 and Fe16N2
  change by ≤ 0.3 meV/atom and ≤ 4 meV/Å.

The doc's 1 meV/atom redo trigger is therefore met for the small N/O-rich cluster. It cannot flip
the decision, though, because the Fe ≥ 8 force gap is about 30× the ENCUT effect. So nothing is
resubmitted unless the user asks.

**Consequence.** VASP stays out of absolute cluster energies and forces, as both the same head and
a separate head. It enters only through the supported-cluster dE_int (feat/supported-delta-vasp).
That term cancels the cluster-only gap only to the extent the gap is constant across the
configurations being compared; the Fe16 spread here is 0.8–2.5 meV/atom.

## Question

Can VASP labels join MACE training for Fe_n(N/O) clusters, and if so how?

- **Same head:** subtract per-element offsets, then train VASP frames alongside the UBPW91 frames.
- **Separate head:** multi-head MACE with a shared representation.
- **Delta only:** VASP enters only through the supported-cluster interaction term dE_int
  (`cluster_mlip.vasp`). It never supplies absolute cluster energies.

Answering this means separating the two parts of the gap:

- **functional:** BPW91 vs PW91 or PBE;
- **code:** all-electron 6-311++G* Gaussian vs PAW plane waves.

So every state is computed at all of these levels:

| code | BPW91 | PW91 | PBE |
|---|---|---|---|
| Gaussian 09 D.01, 6-311++G* | ✔ (the reference; reproduces the archived G09 label) | ✔ | ✔ |
| VASP, PAW (Fe_pv, N, O) | probe only (Libxc B88 + PW91c) | ✔ | ✔ |

Each comparison then isolates one part of the gap:

- Gaussian PW91 vs VASP PW91 (the same holds for PBE): the code/basis/PAW part at a fixed functional.
- Gaussian BPW91 vs Gaussian PW91 or PBE: the functional part at a fixed code.

UBPW91 stays the reference: the group's experience is that it beats hybrids for Fe clusters.

## Set (30 geometries, 42 states, all neutral)

The selection rules below are coded in the generator, which is deterministic. `states.csv` in
the package lists every state with its source record. All picks are q = 0, because `vasp-prepare`
refuses charged cells and a charged box needs its own correction.

| source | geometries | role |
|---|---|---|
| `FenOm_Warehouse2/dataset_spin_v0` (G09 UBPW91/6-311++G*, labels + Mulliken spins) | Fe2N, Fe2NO, Fe2N2O2, Fe2N2O4, Fe16, Fe16N2 | minimum = lowest-energy frame with fmax < 0.03 eV/Å. It prefers states with every Fe moment up, then frames with recorded spins |
| same | Fe2N2O2, Fe2N2O4, Fe16N2 (one frame each, at the minimum's M) | off-equilibrium mid-optimisation frame, force RMS closest to 0.5 eV/Å |
| Gutsev legacy warehouse (6-311 family bases only, M ≥ 2) | Fe2, Fe2O2, Fe3N2, Fe4, Fe4N2, Fe5N2, Fe6N, Fe7N, Fe8, Fe8N2, Fe9N, Fe10, Fe10N2, Fe12, Fe16NO2 | minimum = lowest legacy energy at the best-covered basis |
| same | Fe2N2O4 TS, Fe5N2 TS, Fe4N2 IRC, Fe16N2 IRC | saddle/IRC frames |
| seeded rattles (σ = 0.08 Å) | Fe8, Fe16N2 minima | off-equilibrium |

**Vertical spin ladders.** Six minima are also run at M − 2 and M + 2 at the same geometry:
Fe2N, Fe2N2O2, Fe4N2, Fe8, Fe16 and Fe16N2. These give spin gaps that both codes compute at
identical coordinates.

**Isomer pairs.** The set has several same-composition, same-M pairs: minimum vs off-equilibrium
vs rattle vs IRC/TS. These give the relative-energy test.

**Composition range.** The compositions span Fe, Fe–N, Fe–O and Fe–N–O, so the fit of
per-element offsets (Fe, N, O) is overdetermined.

The legacy geometries are only geometry sources: every label below is recomputed in G09 D.01.

Each dataset_spin_v0 state has an archived G09 label. The UBPW91 link reruns it in G09 D.01 on QB4,
which checks that the electronic root is reproduced.

## Gaussian (G09 D.01, QB4)

The user keeps these systems on G09 (decision of 2026-10-16), so the tasks load
`gaussian/g09-d01` and run `g09`. The generator's `--gaussian g16` profile exists only as an
alternative.

There is one task per state. It runs three links, with `Guess=Read` from the first link so the
functionals stay in the same SCF basin. Every link keeps `IOP(5/13=1,5/36=1,8/11=1)`:

```
#p UBPW91/6-311++G*    SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm Force IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine Pop=Hirshfeld
#p UPW91PW91/6-311++G* ... Force IOP(5/13=1,5/36=1,8/11=1) ... Pop=Hirshfeld Geom=Checkpoint Guess=Read
#p UPBEPBE/6-311++G*   ... Force IOP(5/13=1,5/36=1,8/11=1) ... Pop=Hirshfeld Geom=Checkpoint Guess=Read
```

Because IOP(5/13=1) lets an unconverged SCF continue, the analysis screens every link for
unconverged SCFs and drops those states. It does not change the route.

## VASP (vasp6/6.6.1-cpu, QB4, vasp_gam)

The inputs come from the existing `cluster_mlip.vasp.prepare_vasp_campaign`
(feat/periodic-vasp-ingest e9515bf). For a cluster alone in a box, its `A` role is the gas-phase
single point. The generator keeps only those job directories: `AB` is identical, and `B` would be
an empty support. The production INCAR is:

```
PREC = Accurate   ENCUT = 520   EDIFF = 1E-6   GGA = 91 | PE   ISMEAR = 0   SIGMA = 0.01
ISPIN = 2   NUPDOWN = M-1   MAGMOM = Gaussian Mulliken spins scaled to M-1 (all-up states) else (M-1)/n_Fe on Fe
ISYM = 0   LASPH = .TRUE.   LREAL = .FALSE.   IBRION = -1   NSW = 0   LORBIT = 11   NCORE = 8
AMIX = 0.2  BMIX = 0.0001  AMIX_MAG = 0.8  BMIX_MAG = 0.0001   (no IVDW, no dipole correction)
```

Other production settings:

- **Box:** orthorhombic, with a 12 Å gap between the cluster and its image. The +0.13 Å and
  +0.27 Å on b and c avoid cubic degeneracy.
- **Grid:** Γ only.
- **POTCARs:** Fe_pv, N and O from potpaw_PBE.

**Convergence variants (PBE).** These run on the Fe2N2O2, Fe8 and Fe16N2 minima:

- gaps of 8, 10 and 14 Å;
- ENCUT of 400 and 650 eV;
- PREC = Normal.

The production settings are deliberately conservative, so the convergence runs verify them after
the fact. If 520 eV or a 12 Å gap moves relative energies by more than 1 meV/atom, the production
runs are redone.

**`probe_libxc_bpw91`.** This variant sets `GGA = LIBXC`, `LIBXC1 = GGA_X_B88` and
`LIBXC2 = GGA_C_PW91` on the same three minima. If the module has Libxc, VASP can run BPW91
itself, which removes the functional part of the gap. If it fails, that is a result.

`vasp-ingest` (same branch) reads every output with
`--cluster-elements Fe --integer-tolerance 0.05`. It keeps `REF_energy` (the free energy, the one
the forces belong to), `energy_sigma0`, forces and `vasp_magmom`.

## Metrics

**Matched set.** All metrics use states whose two calculations are in the same electronic state:

- the converged integer total moment matches;
- the Fe moment sign pattern matches between Gaussian Hirshfeld and VASP LORBIT spins;
- the SCF converged in every link.

Mismatches are counted and reported, not averaged in.

For a code pair X/Y (for example VASP-PW91 vs G09-BPW91) and state i, the gap is
Δᵢ = E_Y,i − E_X,i.

1. **Per-element offset fit and spread.** Fit Δᵢ ≈ Σ_el n_el,i μ_el (Fe, N, O) by least squares.
   Report:
   - μ_el;
   - the RMS and maximum residual, in meV/atom;
   - the spread per composition (std of Δᵢ across the geometries and M of one formula).

   This is exactly what per-head E0s can absorb, and the residual is what they cannot.
2. **Relative energies.**
   - Vertical spin gaps: E(M ± 2) − E(M) at the same geometry.
   - Isomer and off-equilibrium differences: same composition and same M.
   - Report the MAE and maximum, in meV per cluster and in meV/atom.
   - Report whether the lowest M among the ladder members is the same in both codes.
3. **Forces.** Report the per-element RMSE and maximum |ΔF| in meV/Å. Report them separately for
   minima (where both should be near zero) and off-equilibrium frames, with the per-frame cosine
   similarity for the off-equilibrium frames.
4. **Magnetic moments.**
   - Per-Fe |m_VASP − m_Hirshfeld| in μB. The projections differ, so a systematic offset of
     roughly 0.1–0.3 μB is expected; report the mean and spread.
   - The fraction of states whose sign patterns match.
   - Total-moment agreement, which is exact by construction under NUPDOWN.
5. **Controls.**
   - G09 D.01 UBPW91 rerun vs the archived G09 label: |ΔE| ≤ 1 meV and force RMSE ≤ 5 meV/Å, or the state is
     flagged as a different root and excluded.
   - VASP convergence: the change in relative energies and forces for each convergence variant.

Each metric is reported for three pairs:

- (a) VASP-PW91 vs G09-PW91 and VASP-PBE vs G09-PBE (the code part);
- (b) G09-PW91 and G09-PBE vs G09-BPW91 (the functional part);
- (c) VASP-{PW91, PBE, Libxc-BPW91} vs G09-BPW91. This is the pair that matters for training.

## Decision rule

Apply the rule to pair (c) for the VASP functional that future VASP data would use:

| outcome | condition (all must hold) |
|---|---|
| **same head** (VASP frames enter the UBPW91 head after subtracting μ_el) | offset-fit residual RMS ≤ 2 meV/atom; relative-energy MAE ≤ 1 meV/atom and ≤ 30 meV per cluster on spin gaps; force RMSE ≤ 30 meV/Å per element; sign patterns match in ≥ 90 % of states; lowest ladder M identical in every ladder |
| **separate head** (multi-head MACE, shared trunk, own E0s) | not "same head", but residual RMS ≤ 10 meV/atom; relative-energy MAE ≤ 5 meV/atom; force RMSE ≤ 100 meV/Å; sign patterns match in ≥ 75 % of states; lowest ladder M differs in at most one ladder |
| **delta only** | anything worse. In particular, spin-gap ordering that differs in two or more ladders means VASP and UBPW91 disagree on which spin state is lowest, and a shared trunk would learn conflicting spin physics |

The thresholds are proposals. Before the decision is applied, compare them with the current
UBPW91 MACE validation errors: a code gap far below the model error favours sharing.

**Attribution.**

- If the code part (a) passes "same head" but the functional part (b) does not, the fix is the
  functional, not the code. Use Libxc BPW91 in VASP if the probe works, then re-apply the rule.
- If (a) itself fails, the difference comes from PAW and the basis. More data will not fix it, so
  the outcome is separate head or delta only.

**Delta-only consequence.** The supported-cluster dE_int = E_AB − E_A − E_B cancels most
cluster-only errors, so dE_int can still be used. Report the spread of the gas-cluster gap Δᵢ
alongside it: dE_int is trustworthy only where that gap is smooth across the configurations
being compared.

## Package and cost

The package is staged at
`D:\MLIP_Work_Folder\Cluster_MLIP\loni_package\56_clustermlip_gaussian_vasp_calibration`. Its
number is a proposal; the desk assigns the real one.

**Gaussian:** 42 G09 D.01 tasks on QB4 `single`, each with 12 cores, 24 GB and 12 h.

- The worst case is about 6 k SU.
- An Fe16 three-link set with one Force each should take a few hours, judging by the
  package-36 tandem at 1.8 h.
- The small clusters take minutes.

**VASP:** 105 single points on QB4 `workq`, each a 64-core node with a 12 h cap, at most 8 at
once.

- The Fe16 runs should take 1–3 h each; the small clusters take minutes.
- The expected total is about 5–8 k SU.

**Done** means:

- every Gaussian log has 3 `Normal termination` lines;
- every VASP output has `vasprun.xml` containing `</modeling>`.

A failed `probe_libxc_bpw91` job is a result.

## Not covered

- Charged clusters (the anions and cations in dataset_spin_v0).
- Ferrimagnetic Fe16 ground states. The Fe16 picks are the lowest all-Fe-up states; matching a
  flipped-site state across codes needs the fragment-guess tandem (`fix/bs-fragment-tandem`).
- Basis-set convergence on the Gaussian side.
