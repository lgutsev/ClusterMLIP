# Gaussian tandem fragment method for broken-symmetry (BS) states

Status, 2026-10-08: generator corrected and statically validated. **No new calculation has been
run.** The validation and pilot jobs below are prepared and need authorization to submit.

Code: `src/cluster_mlip/fragment_tandem.py` (fixed syntax, validation, inspection, log diagnostics);
`src/cluster_mlip/broken_symmetry.py` (Fe16 patterns); `experiments/fe16_bs_pilot/` (validation kit,
analysis). Tests: `tests/test_fragment_tandem.py`, `tests/test_broken_symmetry.py`.

## Background

The fragment guess builds the starting orbitals from fragments. Each fragment has an integer charge
(its electron count, N_e = ΣZ − q), a signed multiplicity (a negative value puts the fragment's
unpaired electrons in β orbitals) and an explicit atom list. Fragment charges are **initialization
parameters**. They are not oxidation states, Mulliken/Hirshfeld charges, or constraints on the
converged wavefunction, and the SCF decides which state results (F. R. Clemente, Gaussian Technical
Support, 2017-12-04). A normal termination therefore does not show that the intended BS state
survived.

## A. Historical-method audit (LG_Calcs.zip)

The archive contains 259 `.out` files and no inputs. Inputs were recovered from the echoed route,
Link 0 and molecule sections. All AFM fragment jobs ran on G09 D.01 (LONI, `%NProcShared=20`,
`%mem=50GB`). After removing byte-identical copies, **27 unique two-link fragment jobs** remain.

### The archived Link1 structure (Fe2O2N2_DimAd_AFM_1, reconstructed)

```
%NProcShared=20
%mem=50GB
%chk=Fe2O2N2_DimAd_AFM_1
# UBPW91/6-311++G* scf=(VSHIFT=5,NoIncFock,MAXCYC=200,Tight,NoVarAcc)
NOSYMM SP IOP(5/13=1,5/36=1,8/11=1) INT=UltraFine GUESS=(Fragment=3)

<title>

0 1 0 5 0 -5 0 1
Fe(Fragment=1)  -1.04914  -0.09571  -0.31538
Fe(Fragment=2)   1.08263   0.08187  -0.21416
O(Fragment=1)   -0.07622   1.35317  -0.69354
O(Fragment=2)    0.14998  -1.36616  -0.68541
N(Fragment=3)    0.52591   0.0409    1.58103
N(Fragment=3)   -0.66478  -0.05829   1.52448

--Link1--
%NProcShared=20
%mem=50GB
%chk=Fe2O2N2_DimAd_AFM_1
# UBPW91/6-311++G* scf=(VSHIFT=5,NoIncFock,MAXCYC=200,Tight,NoVarAcc)
NOSYMM OPT(TS,NoEigenTest,CalcFC) FREQ IOP(5/13=1,5/36=1,8/11=1) INT=UltraFine GEOM=CHECKPOINT GUESS=READ

<title>

0 1

```

(The archived jobs put the charge/multiplicity values on one line or several; Gaussian echoes them as
"Charge = 0 Multiplicity = 5 in fragment 1." and so on. Some routes were split over two `#` lines. Both
are cosmetic.)

### What the archived jobs actually did

1. **Link 0 never ran a supermolecule SCF.** In all 27 jobs, link 0 ran exactly one SCF per fragment
   ("Fragment guess: doing MCBS calculation for fragment k"), combined the fragment orbitals,
   wrote them to the checkpoint, and terminated. The archive line carries the last fragment's energy.
   In effect it was a guess-only stage. Its fragment orbitals, however, were **converged fragment
   SCFs**, not Harris guesses.
2. **The fragment guess survived Link1.** Link 1 printed "Initial guess from the checkpoint file".
   Its initial ⟨S²⟩ matched the ideal value of the fragment pattern, Sz(Sz+1) + n_β,unpaired, in 24 of
   27 jobs (3.97–4.03 against 4.0 for Fe(0,5)/Fe(0,−5); 5.005–5.009 against 5.0 for
   Fe⁺(1,6)/Fe⁺(1,−6)). The three outliers are genuinely defective (point 5).
3. **No archived job used `Stable`** in either link. Link 1 was always `Opt(TS,NoEigenTest,CalcFC) Freq`
   or `Opt Freq`, with `Geom=Checkpoint Guess=Read` and an explicit `Q M` line. None used `%oldchk`,
   `Geom=AllCheck` or `Guess=Only`.
4. **`IOP(5/13=1)` hid unconverged SCFs.** 10 of 27 jobs had at least one fragment SCF that stopped at
   129 cycles with "Convergence criterion not met"; the guess was built from it anyway. 6 jobs had
   unconverged SCFs inside the link-1 optimization. G09 printed "within 128 cycles" even though the
   route said `MAXCYC=200`. The smoke job must check which cap applies.
5. **Defective patterns that must be rejected:**
   - two jobs changed the multiplicity across Link1 (link 0 at M=1, link 1 at M=3 or M=5):
     `New/Fe2O2N2_DimAd_AFM_1`, `New/TS_AFM/Fe2O2H2_Dim3_TS_AFM_5`. Their guess ⟨S²⟩ was 5.64 and
     8.75, and neither finished;
   - `Fe2O2O2_Break_TS_AFM_1`: unconverged, contaminated fragment SCFs (⟨S²⟩ 6.41 and 6.94 against 6.0)
     gave a guess at ⟨S²⟩ 4.72 against 4.0.
6. **Fragment charges were initialization choices.** The same cluster families used neutral fragments
   (Fe(0,±5) or FeO(0,±5)) in some jobs and charge-separated ones (FeO⁺(1,±6) with O₂²⁻(−2,1)) in
   others. Every one satisfied charge conservation, electron-count parity, and
   Σ sign(m)(|m|−1) = M − 1.
7. **Different guesses can reach the same state.** In `AFM_Cr/`, each AFM job has a partner ("Cr")
   that starts from a triplet SCF and then `Guess=Read` at M=1, with no fragments. For Fe2O4N2_FeO_TS
   and Fe2O4O2_44_TS, both starts end at the same energy to 10⁻⁸ Ha and the same ⟨S²⟩ (−2938.08255071,
   1.6190; −2979.04463970, 1.4399). For Fe2O4_Fish and Fe2O4H2_FeFe_TS they do not (Δ 21 and 15 mHa).
   These are optimized TS geometries, so geometry differences contribute too.
8. **Outcomes:** 19 of 27 jobs completed (15 with no silently unconverged link-1 SCF); 8 ended in an
   error termination or are truncated.

### Reference table (27 unique archived fragment jobs)

Ideal guess ⟨S²⟩ = Sz(Sz+1) + n_β,unpaired. Fe spins are the last Mulliken values in link 1 (after the
geometry changed). A job is a **verified working pattern** only if it completed with every SCF
converged.

| Archived job | Fragments: composition(q m) | Link 1 job | Unconverged fragment SCFs | Link-1 guess <S�> / ideal | Link-1 first SCF: E (Ha), cycles | Link-1 last E (Ha) | Last Fe Mulliken spins | Outcome |
|---|---|---|---|---|---|---|---|---|
| `AFM_Cr/Fe2O2H2_Dim_TS_AFM_1` | FeO(0 5), FeO(0 -5), H2(0 1) | Opt(TS)+Freq | 0/3 | 3.9726 / 4.00 | -2679.349038, 32 | -2679.342313 | 2.95 / -2.89 | complete, NImag=1 |
| `AFM_Cr/Fe2O4_Fish_AFM_1` | FeO(1 6), FeO(1 -6), O2(-2 1) | Opt(TS)+Freq | 0/3 | 5.0052 / 5.00 | -2828.658786, 29 | -2828.663652 | 2.52 / -2.52 | complete, NImag=1 |
| `AFM_Cr/Fe2O4H2_FeFe_TS_AFM_1` | FeO(1 6), FeO(1 -6), O2(-2 1), H2(0 1) | Opt(TS)+Freq | 0/4 | 5.0082 / 5.00 | -2829.803130, 28 | -2829.805348 | 1.95 / -1.95 | complete, NImag=1 |
| `AFM_Cr/Fe2O4N2_FeO_TS_AFM_1` | FeO(1 6), FeO(1 -6), O2(-2 1), N2(0 1) | Opt(TS)+Freq | 1/4 | 5.0090 / 5.00 | -2938.082551, 37 | -2938.082551 | 1.09 / -1.24 | complete, NImag=1 |
| `AFM_Cr/Fe2O4O2_44_TS0_AFM_1` | FeO(1 6), FeO(1 -6), O2(-2 1), O2(0 1) | Opt(TS)+Freq | 1/4 | 5.0072 / 5.00 | -2979.039397, 41 | -2979.039397 | 1.61 / -2.53 | complete, NImag=1 |
| `AFM_Cr/Fe2O4O2_44_TS_AFM_1` | FeO(1 6), FeO(1 -6), O2(-2 1), O2(0 1) | Opt(TS)+Freq | 1/4 | 5.0061 / 5.00 | -2979.044640, 32 | -2979.044640 | 1.15 / -1.15 | complete, NImag=1 |
| `Fe2O2_Tet/Fe2O2_Tet_AFM_1` | Fe(1 6), Fe(1 -6), O2(-2 1) | Opt+Freq | 1/3 | 4.8812 / 5.00 | -2678.125710, 34 | -2678.154322 | 3.1 / -3.1 | complete, NImag=0 |
| `Fe2O2N2_DimAd/Fe2O2N2_DimAd_AFM_1` | FeO(0 5), FeO(0 -5), N2(0 1) | Opt(TS)+Freq | 1/3 | 3.9992 / 4.00 | -2787.667130, 49 | -2787.687998 | 3.19 / -3.16 | error termination |
| `Fe2O2N2_DimAd2/Fe2O2N2_DimAd2_AFM_1` | FeO(1 6), OFe(1 -6), N2(-2 1) | Opt+Freq | 0/3 | 4.9382 / 5.00 | -2787.638028, 50 | -2787.729731 | 3.06 / -2.95 | complete, NImag=0 |
| `Fe2O2O2_DimAd2/Fe2O2O2_DimAd2_AFM_1` | FeO(1 6), OFe(1 -6), O2(-2 1) | Opt+Freq | 0/3 | 4.9366 / 5.00 | -2828.503370, 47 | -2828.596561 | 2.69 / -3.11 | complete, NImag=0; 1 unconverged link-1 SCF(s) passed by IOP(5/13=1) |
| `New/AFM/!_Fe2O2H2_Dim3_TS_AFM_1` | FeO(0 5), FeO(0 -5), H2(0 1) | Opt(TS)+Freq | 0/3 | 4.0231 / 4.00 | -2679.253623, 31 | -2679.327255 | 3.1 / -3.1 | error termination |
| `New/AFM/Fe2O2H2_Dim2_TS_AFM_1` | FeO(0 5), FeO(0 -5), H2(0 1) | Opt(TS)+Freq | 0/3 | 3.9801 / 4.00 | -2679.311274, 80 | -2679.311274 | 3.18 / -3.31 | complete, NImag=1 |
| `New/AFM/Fe2O2H2_Fish_AFM_1` | FeOH(0 6), FeOH(0 -6) | Opt+Freq | 0/2 | 4.9430 / 5.00 | -2679.316896, 26 | -2679.323470 | 3.77 / -3.77 | complete, NImag=0 |
| `New/AFM/Fe2O2N2_Dim2_TS_AFM_1` | FeO(0 5), OFe(0 -5), N2(0 1) | Opt(TS)+Freq | 0/3 | 3.9755 / 4.00 | -2787.446925, 129 | -2787.557774 | 2.98 / -1.93 | complete, NImag=1; 5 unconverged link-1 SCF(s) passed by IOP(5/13=1) |
| `New/AFM/Fe2O2N2_Dim3_TS_AFM_1` | FeO(0 5), FeO(0 -5), N2(0 1) | Opt(TS)+Freq | 0/3 | 4.0200 / 4.00 | -2787.632612, 44 | -2787.575928 | 0.81 / -0.81 | complete, NImag=1; 2 unconverged link-1 SCF(s) passed by IOP(5/13=1) |
| `New/AFM/Fe2O2N2_Fish_AFM_1` | FeON(0 6), FeON(0 -6) | Opt+Freq | 0/2 | 4.9720 / 5.00 | -2787.629851, 37 | -2787.633818 | 0.95 / -0.95 | complete, NImag=0 |
| `New/AFM/Fe2O2O2_Break2_TS_AFM_1` | FeO2(0 5), FeO2(0 -5) | Opt(TS)+Freq | 0/2 | 4.0345 / 4.00 | -2828.611008, 59 | -2828.620260 | 2.17 / -2.53 | complete, NImag=1 |
| `New/AFM/Fe2O2O2_Break_AFM_1` | FeO2(0 5), FeO2(0 -5) | Opt(TS)+Freq | 0/2 | 3.9878 / 4.00 | -2828.626194, 35 | -2828.618835 | 1.7 / -2.1 | error termination |
| `New/AFM/Fe2O2O2_Break_TS_AFM_1` | FeO2(0 5), FeO2(0 -5) | Opt(TS)+Freq | 1/2 | 4.7206 / 4.00 | -2828.550073, 43 | -2828.550074 | 2.99 / -3.14 | complete, NImag=1 |
| `New/AFM/Fe2O2O2_Chain_AFM_1` | FeO2(0 5), FeO2(0 -5) | Opt+Freq | 0/2 | 3.9904 / 4.00 | -2828.620262, 34 | -2828.626194 | 1.7 / -2.1 | complete, NImag=0 |
| `New/AFM/Fe2O2O2_FeO_AFM_1` | FeO(0 5), FeO(0 -5), O2(0 1) | Opt+Freq | 2/3 | 4.0010 / 4.00 | -2828.324182, 49 | -2828.536556 | 3.12 / -2.72 | complete, NImag=0 |
| `New/AFM/Fe2O4_FishB_AFM_1` | FeO(1 6), FeO(1 -6), O2(-2 1) | Opt(TS)+Freq | 1/3 | 5.0067 / 5.00 | -2828.659025, 31 | -2828.618398 | 2.09 / -1.92 | error termination |
| `New/Fe2O2N2_DimAd_AFM_1` | FeO(0 5), FeO(0 -5), N2(0 1) | Opt+Freq at M=3 (link 0: M=1) | 1/3 | 5.6362 / 4.00 | -2787.664486, 47 | -2787.715677 | 3.17 / -1.42 | error termination; 2 unconverged link-1 SCF(s) passed by IOP(5/13=1) |
| `New/Fe2O2O2_DimAd_AFM_1` | FeO(1 6), OFe(1 -6), O2(-2 1) | Opt+Freq | 0/3 | 4.9366 / 5.00 | -2828.503370, 47 | -2828.596561 | 2.69 / -3.11 | complete, NImag=0; 1 unconverged link-1 SCF(s) passed by IOP(5/13=1) |
| `New/TS_AFM/Fe2O2H2_Dim3_TS_AFM_5` | FeO(0 5), FeO(0 -5), H2(0 1) | Opt(TS)+Freq at M=5 (link 0: M=1) | 0/3 | 8.7488 / 4.00 | -2679.236989, 71 | -2679.284464 | 3.48 / -0.07 | truncated; 1 unconverged link-1 SCF(s) passed by IOP(5/13=1) |
| `New/TS_AFM/Fe2O2N2_Dim_TS_AFM_1` | FeO(0 5), FeO(0 -5), N2(0 1) | Opt(TS)+Freq | 0/3 | 3.9705 / 4.00 | -2787.285625, 32 | -2787.698920 | 3.08 / -3.19 | error termination |
| `New/TS_AFM/Fe2O2O2_Dim2_TS_AFM_1` | FeO(0 5), FeO(0 -5), O2(0 1) | Opt(TS)+Freq | 1/3 | 3.9858 / 4.00 | -2828.091169, 59 | -2828.530897 | 2.57 / -3.19 | error termination |

### Validated input patterns (summary)

| Element | Archived, verified | Notes |
|---|---|---|
| Link count | 2 (`--Link1--` once) | fragment link, then the calculation |
| Checkpoint | one `%chk` in both links, no `%oldchk` | Gaussian appends `.chk` |
| Link 0 route | `<level> NoSymm SP IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine Guess=(Fragment=N)` | no `Only`, no `Always`, no `Stable` |
| Link 0 molecule | `Q M q1 m1 … qN mN`, then `El(Fragment=k) x y z` | signed m; one pair per fragment |
| Link 1 route | `<level> … <job> Geom=Checkpoint Guess=Read` | `Q M` line repeated, no coordinates |
| Link 1 job | `Opt(TS,NoEigenTest,CalcFC) Freq` or `Opt Freq` | geometry moved: not a fixed-geometry BS label |
| Fragments used | Fe(0,±5); FeO(0,±5); FeOH/FeON(0,±6); Fe⁺/FeO⁺(1,±6) with O₂²⁻ or N₂²⁻(−2,1); H₂/N₂/O₂(0,1) | all pass the new validator |

### Modern Gaussian documentation, compared

| | Archive (G09 D.01) | Guess keyword page (G16) | AFC example (gaussian.com/afc) |
|---|---|---|---|
| Fragment step | `SP Guess=(Fragment=N)`: converged SCF per fragment, combined, saved | `Guess=(Fragment=N,Only)` usually; without `Only` "a full SCF calculation is performed for each fragment" | `guess=(fragment=4,only)`: guess only, no SCF |
| Next step | same chk via `--Link1--`, `Geom=Checkpoint Guess=Read` + `Q M` | `Guess=Read Geom=AllCheck` via `--Link1--` | separate job, `%oldchk` copy, `guess=read geom=allcheck stable=opt` |
| Stability | none | n/a | `Stable=Opt` on the read fragment guess; confirmed by Mulliken spins (±3.93) and energy ordering |

`Guess=(Fragment=N,Only)` is **not** interchangeable with the archived link 0. `Only` skips the
per-fragment SCFs, so its fragment orbitals are cruder. The kit tests both forms on Fe₂ (V1 against
V2). The archive does not say whether G09/G16 *would* run a supermolecule SCF after the fragments
without `Only`; it never did on G09 D.01. The parser flags it if it happens (`s1_supermolecule_scf`).

## B. Audit of the existing ClusterMLIP generators

| Where | Discrepancy with the archived method | Consequence | Status |
|---|---|---|---|
| `main`: `spin.render_fragment_input` (`prepare-spins --strategy fragment`) | one link with `Guess=(Fragment=N,Always)` added to the campaign route (e.g. `Opt`) | not a tandem; `Always` rebuilds the fragment guess at **every optimization step** and discards the previous step's SCF; stability is never tested | not changed here (affects `prepare-spins` campaigns); follow-up below |
| `main`: `spin._validated_fragments` | `int()` coercion: charge 1.7 → 1, multiplicity 6.9 → 6, `True` → 1 | silent change of the guess's electron count | **fixed** (rejects non-integers; test added) |
| pilot (`feat/fe16-broken-symmetry-pilot`) stage 0 | `SP Stable=Opt Pop=Hirshfeld Guess=(Fragment=16,Always)` | `Stable=Opt` in the fragment link; a supermolecule SCF in link 0; `Always` not archived | **replaced** by the tandem below |
| pilot `IOP(5/13=1)` | dropped from every link | link 0 would abort whenever one of 16 atomic fragment SCFs hits the cycle cap (10/27 archived jobs had one) | kept in link 0 only (D3) |
| pilot P0 | default (Harris) guess | not a "high-spin-aligned" initialization | P0 is now every Fe α; the default guess is the control C0 |
| pilot P1 vs the Fe16 reference | central Fe started β | in all three reference geometries the central Fe already has Mulliken spin −5.0 to −5.9 (charge ≈ +12): P1 probably reproduces the reference state | P0 (aligned) is the decisive comparison; Hirshfeld settles whether the central moment really is reversed |
| pilot `analyze.py` | split logs on `--Link1--` | Gaussian logs never contain that string, so every check ran on the whole log | rewritten on `parse_tandem_log` |
| pilot kit `slurm_plan.json` | `scratch_root = C:/Program Files/Git/work/$USER/g09-scr` | Git Bash rewrote `/work/...`; the kit would have written scratch to a nonexistent path | new kit generated from PowerShell; plan shows `/work/$USER/g09-scr` |
| all | no check of the written input, no check that the guess reached link 1 | — | added (C) |

## C. Corrected generator

`TandemJob` + `render_tandem` write, with one `%chk` and no `%oldchk`:

```
link 0  #p UBPW91/6-311++G* SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm SP
           IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine Guess=(Fragment=N)
        Q M q1 m1 ... qN mN  /  El(Fragment=k) x y z
link 1  #p ... NoSymm Stable=Opt Pop=Hirshfeld IOP(5/36=1,8/11=1) Int=UltraFine Geom=Checkpoint Guess=Read
        Q M
link 2  #p ... NoSymm Force Pop=Hirshfeld IOP(5/36=1,8/11=1) Int=UltraFine Geom=Checkpoint Guess=Read
        Q M                                                     (optional; force_stage=False omits it)
```

The fixed syntax (link order, keywords, checkpoint continuity, charge/multiplicity layout) is in code.
The scientific settings (functional, basis, SCF options, grid, print IOps, resources) are a
`LevelOfTheory` and are identical in every link.

**Before writing** (`validate_fragment_plan`): integer charges and multiplicities with no coercion;
every atom in exactly one fragment, indices in range; N_e = ΣZ − q ≥ 0; |m| − 1 ≤ N_e; parity
(N_e + |m| − 1) even; m ≠ 0 and no "−1" (a singlet has no orientation); Σq = Q; Σ sign(m)(|m|−1) = M − 1;
total M possible for the total electron count.

**After writing** (`inspect_tandem_input`, run on every file; generation aborts on any problem). This
is an independent parser, not the renderer. It checks:
- the stage count;
- one identical `%chk` in every link and no `%oldchk`;
- link 0 is exactly `SP Guess=(Fragment=N)`, with N matching the charge/multiplicity line and every
  atom labelled;
- the fragment guess re-validated from the text, and equal to the request;
- coordinates identical to the source geometry;
- `Stable` only in link 1;
- no `Opt`/`Freq`/`IRC`;
- `Guess=Read` and `Geom=Checkpoint` in links 1–2, with no coordinates and the same `Q M`;
- `IOP(5/13=1)` absent from links 1–2;
- an identical level in every link;
- LF line endings;
- the exact template route.

**After running** (`parse_tandem_log`). It splits the log at termination lines and checks:
- link 0: fragment-SCF count, cycles, ⟨S²⟩, unconverged fragment SCFs, and any supermolecule SCF
  (flagged);
- link 1: the guess was read from the checkpoint and its ⟨S²⟩ matches the pattern (tolerance
  0.15 + 0.05 per open-shell fragment, which separates the 24 good archived jobs from the 3 bad ones);
  `Q M` is unchanged; every SCF energy and ⟨S²⟩ and every instability is recorded, which gives the
  state **before and after** `Stable=Opt` (`stable_opt_changed_state`, ΔE, Δ⟨S²⟩); the final verdict
  must be "stable"; no unconverged SCF;
- link 2: the SCF reproduces the link-1 energy within 10⁻⁵ Ha (the state survived Link1); the input
  orientation equals link 0's within 10⁻⁵ Å (geometry fixed); a force table is present.

Status is `ok` only if every check passes.

**State identity** (`compare_states`): two usable solutions at one geometry are the *same* state only
if all four agree: energy (1 meV), per-site Hirshfeld spin (0.2), ⟨S²⟩ (0.05) and force RMS
(5 meV/Å). They are *distinct* when spin and energy both differ beyond tolerance, or every criterion
differs. Otherwise they are *ambiguous*, which is reported and never counted. These are
reproducibility tolerances. A stable solution above the lowest one is a metastable BS state and is
kept.

### Deviations from the archived method (each needs your sign-off)

| # | Deviation | Reason |
|---|---|---|
| D1 | Link 1 is `Stable=Opt` at fixed geometry, not `Opt(TS)`/`Opt Freq` | the pilot needs fixed-geometry labels; `Stable=Opt` on the read fragment guess follows Gaussian's AFC example. The archive has no `Stable` at all, so this is new, not "historically validated" |
| D2 | Optional third link, `Force Geom=Checkpoint Guess=Read` | forces on the stable state; `Stable` is a job type and is not combined with `Force` in one route. Its SCF must reproduce link 1, or the job is flagged |
| D3 | `IOP(5/13=1)` kept in link 0, removed from links 1–2 | link 0 only seeds the guess (as archived); the labelled SCF must fail rather than continue unconverged |
| D4 | `Pop=Hirshfeld` added | Mulliken is unreliable for interior Fe with diffuse functions |
| D5 | `#p`, explicit `.chk` suffix, `%nprocshared=12`, `%mem=24GB` | cosmetic and resources; no change in method |

## D. Test inputs (prepared, not submitted)

Kit: `D:\MLIP_Work_Folder\Cluster_MLIP\fe16_bs_tandem_kit\`. It uses G09 D.01, the archive's version
(`mvapich2 gaussian/g09-d01`), on QB3 `workq` with account `loni_perovsk27`. Scratch is
`/work/$USER/g09-scr`.

| Sub-kit | Jobs | Purpose | Resources |
|---|---|---|---|
| `validation/` | V1 Fe₂ AFM, neutral fragments, archived link-0 syntax; V2 same with `,Only`; V3 Fe₂ charge-separated Fe(+1,6)/Fe(−1,−6); V4 Fe₂ default-guess control; V5 repeat of V1; H1/H2 replays of Fe2O2N2_DimAd_AFM_1 and Fe2O4H2_FeFe_TS_AFM_1 | does the tandem do what the archive did? H1/H2 must reproduce the archived fragment-SCF energies, guess ⟨S²⟩ (3.9992, 5.0082) and first link-1 SCF (−2787.66713025, −2829.80312953 Ha) | 7 jobs × 4 cores, 8 GB, one batch, 12 h limit; expected minutes to ~1 h each |
| `smoke/` | Fe16 geometry 1, P1 | wall time, cycle cap, `Stable=Opt` behaviour, Hirshfeld table, force parsing at Fe16 scale | 1 × 12 cores, 24 GB, 48 h |
| `pilot/` | 3 Fe16 geometries × P0–P3 (M=49, q=0) | distinct BS states at fixed geometry? | 12 jobs, 3 batches of 4 × 12 cores, 48 h |
| `charge_init/` | geometry 1: P0Q, P1Q (central Fe(−1), surface Fe(+1)), C0 | does fragment charge change convergence, iterations, final state, stability? | 3 × 12 cores, one batch, 48 h |

Fe16 run time is not measured. The previous two-link estimate was 1–4 h per job, and Stable=Opt
re-optimizations may double it; the smoke job measures it. Upper bound: 16 Fe16 jobs × 12 cores ×
≤ 8 h ≈ 1,500 core-hours.

**Gates:** validation (all `ok`; H1/H2 match the archive; V1 = V5) → smoke `ok` → pilot +
charge_init → `analyze.py`. Nothing enters the training dataset from this study until the analysis
shows consistent, reproducible states.

## E. Validation so far (static)

- `pytest`: 379 passed, 2 skipped. 70 of these are the new or rewritten tandem and BS tests.
- Regression: the generator rebuilds both archived link-0 routes token for token, along with the
  fragment charge/multiplicity sequence, the atom→fragment map and the single-checkpoint layout. The
  link-1 differences are exactly D1/D3/D4.
- The log parser, run on archived excerpts, reproduces: fragment-SCF counts, no supermolecule SCF,
  unconverged-fragment counts, guess read from the checkpoint, and guess ⟨S²⟩ within tolerance. It
  flags the archived multiplicity change across Link1 and the missing stability verdict.
- 15 deliberately broken inputs are each rejected with the specific reason: `Stable` in link 0,
  missing `Guess=Read`, `Geom=AllCheck`, M change, different chk, `%oldchk`, lenient IOP in link 1,
  `Opt`, repeated coordinates, level mismatch, unlabelled atom, wrong N, moved geometry, CRLF, missing
  link.
- **Not yet shown (needs the validation run):** that G09 D.01 builds the same guess for the replays;
  that `Stable=Opt` keeps or changes the BS state; that link 2 reproduces link 1; the actual SCF
  cycle cap; and any Fe16 result.

## F. Role in the Fe16 roadmap, and limitations

This is the targeted reference-data study of `ROADMAP_Backup_Plan.md` §8, run under separate
authorization. It addresses the roadmap gate "reliable same-(Q, M) states with distinct local spin
and forces". It tests whether unresolved electronic states could contribute to the poor force
generalization of the Fe16 MACE (≈ 51 meV/Å on unseen frames against 2–5 meV/Å on training). It does
not assume they do: the next-steps plan currently blames structural coverage.

Limitations:
- A fragment guess only biases the SCF. Different guesses may reach the same state (archived
  examples above), and a pattern that does not appear is not proof that no other state exists.
- `Stable=Opt` tests only the perturbations Gaussian considers; "stable" is necessary, not sufficient.
- ⟨S²⟩ of a BS determinant is not a spin eigenvalue and is supporting evidence only. Mulliken moments
  are unreliable for the interior Fe (the reference central Fe shows charge ≈ +12). Use Hirshfeld,
  and CM5 only as charges.
- One functional and basis (UBPW91/6-311++G*, the group's reference level), 16 per-atom fragments,
  M = 49 only, three geometries.
- Fragment charges in P0Q/P1Q are initialization perturbations, weakly motivated by interior atoms
  carrying more electron density. They are not oxidation states.

## Follow-ups (not done here)

- `prepare-spins --strategy fragment` on `main` still writes the single-link `Guess=(Fragment=N,Always)`
  form. Once the validation run confirms the tandem, port it to `fragment_tandem` (or retire it).
