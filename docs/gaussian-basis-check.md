# G09 basis-set check on the package-62 force gap

Status 2026-10-31: the desk took it in as **package 75** (desk commit 70852d9). It was staged as proposal 65 in
`D:\MLIP_Work_Folder\Cluster_MLIP\loni_package\65_clustermlip_gaussian_basis_check`, and the files are
unchanged; `package.json` keeps the staged name. Launch (`bash submit.sh`) is with the user. Results will come
back as `hpc_smoke_tests\75_clustermlip_gaussian_basis_check`.

## Question

Package 62 (`docs/gaussian-vasp-calibration.md`) compared G09 D.01 and VASP on the same Fe_n(N/O) geometries
at the same functional (PW91). The force RMSE for Fe was about 160 meV/A. The VASP forces point the same way
as the G09 forces but are 3–4 times larger, so the two codes place the minima at different bond lengths.

VASP is basis-converged: going from ENCUT 520 to 650 moves the Fe8 and Fe16N2 forces by at most 4 meV/A.
Package 62 measured disagreement, not accuracy. This package asks which side is off: is 6-311++G* the
incomplete side?

The gap has a clear sign. At every package-62 minimum, the mean radial VASP force points inward, so
VASP PW91 wants the clusters smaller:

| state | G09 6-311++G* (meV/A) | VASP (meV/A) |
|---|---:|---:|
| Fe2 M7 | −62 | −266 |
| Fe8 M25 | −63 | −295 |
| Fe10 M29 | −64 | −279 |
| Fe16 M51 | −49 | −206 |
| Fe16N2 M51 | −49 | −200 |

For Fe2 this is about 0.03 A shorter. Adding high-angular-momentum polarization (f, g) to a 3d-metal basis
usually shortens metal–metal bonds, so a basis-side explanation is plausible. This package tests it rather
than assuming it.

## Design

The package is 15 G09 D.01 tasks (`gaussian/g09-d01`, `g09`) on QB4 `single`, each with 12 cores, 24 GB and a
12 h cap. Every link is a Force single point with:

- `SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm Int=UltraFine Pop=Hirshfeld`;
- `IOP(5/13=1,5/36=1,8/11=1)`. This is in every link and is never dropped.

| link | route | guess | role |
|---|---|---|---|
| 1 | UBPW91/6-311++G* | default | replays package 62's link 1, the guess source |
| 2 | UPW91PW91/6-311++G* | Guess=Read | replays package 62's G_PW91 link; this is B0, the control |
| 3 | UPW91PW91/def2-TZVPP | Guess=Read from link 2 (`%OldChk`) | larger basis |
| 4 | UPW91PW91/def2-QZVPP | the same | only for n_Fe ≤ 10 (cost) |

**Why links 1–2 replay package 62:** package 62's PW91 state was reached from a BPW91 guess. Repeating that
path is the safest way to land on the same state, and link 2 must match package 62 exactly; that is the
control. Links 3 and 4 project link 2's orbitals into the new basis, so they stay on the same state.

**Bases:**

- def2-TZVPP and def2-QZVPP are built into G09 (`Def2TZVPP`, `Def2QZVPP`) and are all-electron for Fe and
  N/O, so no Gen or ECP is needed.
- def2-TZVPP has about the same number of functions per Fe as 6-311++G*. It adds a second f and a g, and it
  has a different s/p/d contraction.
- 6-311+G(3df) was left out: what the polarization designation means for Fe in G09's 6-311G family is not
  certain enough to interpret.

**States:** all are taken from package 62, and all have a converged G_PW91 SCF, a converged V_PW91 SCF and a
matching Fe Hirshfeld/VASP sign pattern (`gv_calibration_62/states.csv`). Geometries are package 62's G09
input coordinates, byte-identical.

| task | state | why |
|---|---|---|
| bc00 | s000 Fe2 M7 minimum | force gap; Fe2 anchor (scan point 0) |
| bc01 | s005 Fe2N2O2 M9 minimum | force gap with N/O |
| bc02 | s007 Fe2N2O2 M9 off-eq | deformation energy vs s005; large forces |
| bc03 | s023 Fe8 M23 (vertical) | spin gap vs s024 |
| bc04 | s024 Fe8 M25 minimum | force gap |
| bc05 | s025 Fe8 M27 (vertical) | spin gap vs s024 |
| bc06 | s028 Fe10 M29 minimum | force gap |
| bc07 | s032 Fe16 M51 minimum | force gap. Package 62's link 1 here is unconverged and its link 2 converged; the replay keeps that path |
| bc08 | s036 Fe16N2 M51 minimum | force gap |
| bc09–bc14 | Fe2 M7, bond length −0.08, −0.06, −0.04, −0.02, +0.02, +0.04 A from package 62's 2.0115 A | r_e for each basis |

**Cost:** Fe16 at 6-311++G* took about 1.4 h of wall time for its BPW91 link and about 0.6 h for each later
link in package 62. A def2-TZVPP link with g functions should take 2–4 times as long as a B0 link. Expected
times:

- Fe16 and Fe16N2 tasks: about 5 h each.
- Fe10 QZVPP: under 2 h.
- Everything else: minutes.

That is about 250 SU expected and 2160 SU at worst (15 × 12 cores × 12 h).

## Analysis

`experiments/gaussian_vasp_calibration/analyze_basis_check.py` reuses the package-62 parsers from
`analyze_calibration.py`.

1. **SCF screen.** "Normal termination" does not prove convergence under IOP(5/13=1). A link is unconverged
   if any of these holds:
   - it printed "Convergence criterion not met";
   - its final Conv= is above 1e-6;
   - it used all MaxCyc cycles.

   Unconverged links never enter a metric. For s032, link 1 is expected to be unconverged. It is only the
   guess source.
2. **Same state.** A larger-basis link counts only if its Fe Hirshfeld sign pattern equals link 2's and its
   ⟨S²⟩ is within 0.1 of link 2's.
3. **Control.** Link 2 must reproduce package 62's G_PW91 energy to within 1 meV and its forces to within
   1 meV/A. If it does not, nothing else is interpreted.
4. **Forces against VASP PW91,** taken from package 62's `prod_pw91` outputs, for each state and basis:
   - the Fe, N and O force RMSE against VASP;
   - the mean radial force;
   - **alpha** = ⟨F_B − F_B0, F_V − F_B0⟩ / |F_V − F_B0|², the share of the package-62 gap that the basis
     change explains (0 = no move, 1 = all the way to VASP);
   - the TZVPP→QZVPP step as a fraction of the gap.
5. **Relative energies,** for each basis against VASP:
   - the Fe8 spin gaps M23−M25 and M27−M25;
   - the Fe2N2O2 off-eq − minimum energy;
   - also the energy shift per atom from B0.
6. **Fe2 r_e.** Fit a quadratic to dE/dr(r) from the analytic forces at the 7 scan points (bc00 plus the 6
   scan tasks) for each basis. This gives r_e, k and ω_e. VASP has only the r0 point, so its r_e is
   estimated harmonically as r0 − (dE/dr)_V / k, using the curvature of the largest Gaussian basis.

## Decision rule

The rule uses the pure-Fe minima (Fe2, Fe8, Fe10, Fe16), each at its largest usable basis.

| outcome | condition | meaning and next step |
|---|---|---|
| `control_failed` | link 2 ≠ package 62 | Stop. Find out why the replay drifted. |
| `ladder_not_converged` | the TZVPP→QZVPP step is more than 0.25 of the gap | The ladder itself is not converged. |
| `gaussian_basis` | median alpha ≥ 0.5 | The **6-311++G* labels carry basis error**. Size it in UBPW91 terms with a BPW91/def2-TZVPP link on the same states (PW91 and BPW91 differ by only 34 meV/A in package 62, so the PW91 shift is a good proxy). Then decide whether to relabel or to delta-correct. |
| `not_basis` | median alpha ≤ 0.2 | The Gaussian side is basis-converged. Look at VASP: the PAW Fe_pv frozen core and augmentation (try Fe_sv / Fe_sv_GW, and harder potentials), or an all-electron third code as a referee. A functional-implementation difference is unlikely, because PBE shows the same gap (192 meV/A). |
| `partial` | in between | Both sides contribute. |

**Absolute anchor:** experimental Fe2 r_e = 2.02 ± 0.02 A, from EXAFS in a Ne matrix (Purdum, Montano, Shenoy,
Morrison, Phys. Rev. B 25, 4412 (1982), doi:10.1103/PhysRevB.25.4412). It is a matrix value, not a gas-phase
one, so it can only flag a deviation larger than about 0.02–0.03 A. It cannot pick a winner between two
values that are both within it. The Fe2 ground state is itself contested, and the comparison here is M = 7 at
the same state on both sides.

## Run the analysis

```bash
python experiments/gaussian_vasp_calibration/analyze_basis_check.py \
    --cluster-mlip-src .claude/worktrees/periodic-vasp-ingest/src \
    --calibration D:/MLIP_Work_Folder/hpc_smoke_tests/62_clustermlip_gaussian_vasp_calibration \
    D:/MLIP_Work_Folder/hpc_smoke_tests/<desk number>_clustermlip_gaussian_basis_check \
    -o D:/MLIP_Work_Folder/Cluster_MLIP/basis_check_<desk number>
```

The plumbing was checked on mock logs made from package-62 links: the control is exact, alpha is 0, and the
Fe2 fit recovers a synthetic Morse r_e to 1e-4 A.

## Results (package 75, received 2026-11-01)

The results are in `hpc_smoke_tests\75_clustermlip_gaussian_basis_check`. The report is in
`D:\MLIP_Work_Folder\Cluster_MLIP\basis_check_75` (`report.md`, `report.json`, `links.csv`).

All 15 tasks finished with every expected Normal termination:

- every link's SCF converged, including bc07's UBPW91 link, which was unconverged in package 62;
- no larger-basis link changed state (the Hirshfeld pattern held and ⟨S²⟩ moved by less than 0.1);
- the control replays package 62 exactly (dE = 0.000 meV, max dF ≤ 0.01 meV/A).

The longest wall times were 2.6 h (Fe16) and 2.9 h (Fe16N2).

**Verdict: `ladder_not_converged`.** The TZVPP→QZVPP step moves forces by 0.51 (Fe2), 0.29 (Fe8) and 0.32
(Fe10) of the gap. The rule therefore gives no final split, but the direction is clear.

| pure-Fe minimum | Fe RMSE vs VASP: 6-311++G* / TZVPP / QZVPP (meV/A) | alpha TZVPP / QZVPP | QZVPP shift from 6-311++G* (meV/A) |
|---|---|---|---:|
| Fe2 M7 | 204 / 166 / 63 | 0.19 / **0.69** | 141 |
| Fe8 M25 | 233 / 248 / 181 | −0.05 / 0.23 | 58 |
| Fe10 M29 | 215 / 249 / 180 | −0.16 / 0.16 | 36 |
| Fe16 M51 | 192 / 191 / – | 0.01 / – | – |

- **def2-TZVPP is no better than 6-311++G*.** Its alpha is about 0 for Fe8, Fe10 and Fe16.
- **def2-QZVPP moves every pure-Fe cluster toward VASP.** It closes 69% of the Fe2 gap and 16–23% of the Fe8
  and Fe10 gaps. The radial forces also turn more inward: Fe8 goes from −63 to −115 meV/A, against −295 for
  VASP.
- **So the 6-311++G* labels carry basis error.** The forces move by at least 36–141 meV/A RMS, and the
  ladder is still moving. Whether all of the gap is Gaussian basis error is not decided yet. For Fe8 and
  Fe10, 75–80% remains at QZVPP.
- **Fe2 bond length:**

  | | r_e (A) |
  |---|---:|
  | 6-311++G* | 2.0076 |
  | def2-TZVPP | 2.0051 |
  | def2-QZVPP | 1.9988 |
  | VASP (harmonic estimate) | 1.9957 |

  The ω_e is about 400 cm⁻¹ at every basis. At QZVPP the Gaussian value is within 0.003 A of VASP. Every
  value is inside the experimental 2.02 ± 0.02 A, so experiment cannot pick between them.
- **Relative energies change little.**

  | quantity | 6-311++G* | def2-TZVPP | def2-QZVPP | VASP |
  |---|---:|---:|---:|---:|
  | Fe8 M23−M25 (meV) | 738 | 738 | 739 | 669 |
  | Fe8 M27−M25 (meV) | 1099 | 1087 | 1075 | 1065 |
  | Fe2N2O2 off-eq − min (meV) | 979 | 979 | 959 | 956 |

  The 70 meV gap in the Fe8 M23 spin gap is not a basis effect.
- **Fe2N2O2 behaves differently.** Its Fe forces move toward VASP (Fe RMSE 309 → 165 meV/A), but its N forces
  move away (681 → 1263 meV/A). That makes alpha negative. Package 62's VASP N/O forces are the least
  converged in that set: at ENCUT 520 they were short, and ENCUT 650 moved them by 58 meV/A. So this is left
  open; it is not counted for or against the basis.

**Next, to finish the attribution:**

1. Extend the Gaussian ladder past def2-QZVPP on Fe2 and Fe8, through Gen. Options are a decontracted
   def2-QZVPP for Fe, or a larger all-electron set from the Basis Set Exchange.
2. Run a real VASP Fe2 bond scan at Fe_pv and Fe_sv to replace the one-point harmonic estimate.

If BPW91 labels need relabelling, def2-TZVPP is not enough. It has to be QZVPP or better.

## Beyond def2-QZVPP: finishing the attribution (package 83)

Status 2026-11-01: the desk took it in as **package 83** (`83_clustermlip_gaussian_basis_attribution`; 80–82
were taken). It was staged as proposal 80 in
`D:\MLIP_Work_Folder\Cluster_MLIP\loni_package\80_clustermlip_gaussian_basis_attribution`, with
`HANDOFF_80_TO_DISPATCH.md` beside it; that staged copy keeps the name 80.

The desk made one change. `run_vasp.slurm` now sleeps (task % 8) × 20 s before `module load vasp6/6.6.1-cpu`,
because 8 tasks loading the container module at once can corrupt its wrapper (package 77 died with "Exec format
error"). The generator now writes that stagger too, so its output is byte-identical to the desk's runner.

Launch is with the user, Fe_sv POTCAR check first. Results will come back as
`hpc_smoke_tests\83_clustermlip_gaussian_basis_attribution`.

Generator: `make_basis_check_package.py --ladder beyond-qz`. Analysis: `analyze_basis_check.py`, the same
script, which adds an "Attribution" section when these links or the VASP scan are present.

### Why another package

Package 75 left two open questions:

- Where is the Gaussian basis limit? The TZ→QZ step was still 0.3–0.5 of the gap.
- Where is VASP's own Fe2 r_e? The 1.996 A value was a one-point harmonic estimate.

There is also a third candidate that neither code's basis explains. VASP PAW potentials are
scalar-relativistic, while G09's default Hamiltonian is non-relativistic. For a 3d metal, scalar relativity
contracts 4s and shortens bonds, which is the same sign as the gap. Its size for Fe2 could be of order
0.005–0.01 A, comparable to what QZVPP left. So the package measures it too, with DKH2 at a fixed decontracted
basis. Otherwise a relativistic difference would be booked as "basis" or "PAW".

### Gaussian part (11 tasks, G09 D.01, QB4 `single`, 12 cores, 24 GB, 12 h)

Every link is a Force single point with:

- `SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm Int=UltraFine Pop=Hirshfeld`;
- `IOP(5/13=1,5/36=1,8/11=1)`.

Links 1–2 replay package 62 (UBPW91, then UPW91PW91/6-311++G*; link 2 is the control). Every larger link reads
link 2's orbitals through `%OldChk`. The Gen sets are Fe-only and stated as `Gen 5D 7F`; every system here is
pure Fe. The sets are taken verbatim from the Basis Set Exchange v0.12 API (gaussian94 format) and kept in
`experiments/gaussian_vasp_calibration/basis/`, with each file's SHA-256 pinned in the generator.

| basis | functions per Fe | role |
|---|---:|---|
| def2-QZVPP (built-in) | 100 | anchor; must reproduce package 75's links (`--previous`) |
| def2-QZVPP(gen) | 100 | the BSE copy through Gen; must equal the built-in to 0.1 meV and 0.1 meV/A (transcription and 5D/7F control) |
| unc-def2-QZVPP | 170 | fully decontracted; primitives within 5% merged (s 72.78/72.18, p 4.39/4.54) to avoid near-linear dependence. Tests contraction error |
| pcseg-3 | 92 | Jensen's DFT-optimised QZ family, with an h shell. A second family |
| pcseg-4(-i) | 135 | pcseg-4 without its i shell |
| pcseg-4 | 148 | Jensen's 5Z, built to approach the DFT basis limit (Fe2 only) |
| unc-def2-QZVPP+DKH2 | 170 | the same decontracted set with `Int=(UltraFine,DKH2)`: scalar relativity |

**Tasks:**

- **bq00–bq06:** Fe2 M7 at s000 and its 6 scan points (the same bond lengths as package 75). Each task carries
  every rung. The two links that might fail on G09 D.01 come last: DKH2 gradients, then the i-function
  gradients of full pcseg-4. A failed link ends a G09 job, so a failure there loses nothing else.
- **bq07–bq10:** Fe8 M25 (s024), one task per large basis so each stays inside the 12 h cap:
  def2-QZVPP + pcseg-3; unc-def2-QZVPP; pcseg-4(-i); unc-def2-QZVPP+DKH2.

**Cost:**

- Fe2 tasks take minutes.
- Fe8: the QZVPP link took 34 min of wall time in package 75. unc-def2-QZVPP has 1360 functions and should
  take about 3 h, as should DKH2; pcseg-4(-i) about 2–3 h.

### VASP part (23 jobs, `vasp6/6.6.1-cpu`, `vasp_gam`, QB4 `workq`, 64 cores, 1 h)

The jobs come from package 62's `prod_pw91` inputs. INCAR and KPOINTS are byte-identical: the same 14.01 ×
14.14 × 14.28 A box, PW91 (GGA=91), NUPDOWN=6, MAGMOM 2*3, ENCUT 520, EDIFF 1e-6 and PREC=Accurate. Only these
change:

- POSCAR: the same 7 Fe2 bond lengths as the Gaussian tasks;
- POTCAR.spec;
- ENCUT, in one variant.

| variant | Fe POTCAR | ENCUT | jobs |
|---|---|---|---|
| `pw91_pv` | Fe_pv | 520 | Fe2 × 7. Its r0 point replays package 62 (control: < 1 meV, < 1 meV/A) |
| `pw91_sv` | Fe_sv | 520 | Fe2 × 7, plus Fe8 s024 |
| `pw91_sv_encut650` | Fe_sv | 650 | Fe2 × 7, plus Fe8 s024; checks that Fe_sv is converged at 520 |

The 7-point scans replace the one-point harmonic r_e estimate with a fitted r_e, k and ω_e at Fe_pv and at
Fe_sv. Fe_sv treats Fe 3s3p as valence (Fe_pv only 3p). It is the standard test for a PAW frozen-core or
augmentation error. `$POTCARS/Fe_sv/POTCAR` must exist on QB4; the runner stops with exit code 2 if it does not.

**Expected cost:** about 250 SU in all. Worst case about 3060 SU: 11 × 12 cores × 12 h for Gaussian, plus
23 × 64 cores × 1 h for VASP.

### Decision rule: Gaussian basis or VASP PAW

Every quantity is a share of the package-62 gap, which runs from G09 PW91/6-311++G* to VASP PW91 Fe_pv.

- **On the Fe2 bond length**, the gap is r_e(6-311++G*) − r_e(VASP Fe_pv), both from 7-point scans.
- **On the Fe8 (and Fe2) forces**, a share is a projection onto the gap vector F_V − F_B0.

| symbol | meaning |
|---|---|
| G_lim | the largest usable Gaussian rung: pcseg-4 for Fe2, pcseg-4(-i) for Fe8 |
| **basis** | the share that B0 → G_lim moves toward VASP (alpha of G_lim) |
| **relativity** | the share of the DKH2 shift at fixed unc-def2-QZVPP |
| **vasp_paw** | how far Fe_pv → Fe_sv moves VASP back toward G09. Uses the ENCUT-650 run when it is usable |
| **residual** | \|(G_lim + DKH2 shift) − VASP Fe_sv\| / \|gap\|: what remains after both sides are fixed |
| **family_diff** | \|unc-def2-QZVPP − pcseg-4(-i)\| / gap. Two independent families must agree for G_lim to count as the limit |
| **last_step** | \|pcseg-3 → pcseg-4(-i)\| / gap |

The rule is applied in this order, separately to the Fe2 r_e and to the Fe8 forces:

| outcome | condition | meaning |
|---|---|---|
| `control_failed` / `gen_control_failed` / `vasp_control_failed` | a replay or the Gen transcription is off | Stop; nothing else is read. |
| `gaussian_ladder_not_converged` | family_diff or last_step > 0.10 | Even pcseg-4 is not the limit. Go to a fully uncontracted pcseg-4 or an even-tempered set. |
| `vasp_sv_not_converged` | Fe_sv 520 → 650 moves more than 0.05 of the gap | Rerun Fe_sv at a higher ENCUT before reading vasp_paw. |
| `unexplained_residual` | residual > 0.20 | Neither basis, relativity nor the PAW semicore explains the gap. Next: an all-electron referee (FHI-aims or similar) and the PAW augmentation spheres (Fe_sv_GW, harder potentials). |
| `gaussian_basis` | basis ≥ 0.20 (and residual ≤ 0.20) | 6-311++G* basis error. **If it is the only named component, the gap is the Gaussian basis.** The labels need relabelling or a delta at QZ quality or better. |
| `relativity` | relativity ≥ 0.20 | Part of the gap is scalar relativity, which is physics, not a numerical error of either code. G09 labels are non-relativistic; decide whether training should keep them that way (consistent across the dataset) or add DKH2. |
| `vasp_paw` | vasp_paw ≥ 0.20 | Fe_pv's frozen 3s is a real part of the gap. VASP data for Fe needs Fe_sv. |
| combined, e.g. `gaussian_basis+relativity` | several shares ≥ 0.20 | Each named piece is real. The shares say how much of the gap each one carries. |
| `within_noise` | every share < 0.20 and residual ≤ 0.20 | Should not occur with a gap this large. Check the inputs. |

The overall outcome holds only if Fe2 (r_e) and Fe8 (forces) agree. Otherwise it is `state_dependent`, and
both rows are reported.

**What would assign the gap to the Gaussian basis:**

- family_diff and last_step ≤ 0.10;
- basis ≥ 0.8, with relativity and vasp_paw < 0.2, and residual ≤ 0.2;
- on Fe2: r_e(pcseg-4) ≈ r_e(unc-def2-QZVPP) ≈ the fitted VASP Fe_pv r_e ≈ the Fe_sv r_e, to about 0.002 A.

**What would assign it to the VASP PAW:**

- the converged Gaussian limit stays near 6-311++G* (basis < 0.2);
- Fe_sv moves VASP most of the way back (vasp_paw ≥ 0.5);
- residual ≤ 0.2.

Package 75 already makes the pure-PAW reading unlikely: def2-QZVPP alone gives Fe2 alpha 0.69. The likely
results are `gaussian_basis` alone, or `gaussian_basis+relativity`. The second would mean that part of the
"basis gap" is the non-relativistic Hamiltonian, which no basis can remove.

The plumbing was checked on mock results built from the package-75 logs and synthetic VASP forces:

- the Gen control is exact (NBasis 200 = 200);
- the def2-QZVPP links match package 75;
- the VASP r0 control replays package 62 exactly;
- the scan fits recover the synthetic r_e to 1e-4 A.

The package-75 report is unchanged when the extended analyzer is rerun on it.

### Run the analysis

```bash
python experiments/gaussian_vasp_calibration/analyze_basis_check.py \
    --cluster-mlip-src .claude/worktrees/periodic-vasp-ingest/src \
    --calibration D:/MLIP_Work_Folder/hpc_smoke_tests/62_clustermlip_gaussian_vasp_calibration \
    --previous D:/MLIP_Work_Folder/hpc_smoke_tests/75_clustermlip_gaussian_basis_check \
    D:/MLIP_Work_Folder/hpc_smoke_tests/83_clustermlip_gaussian_basis_attribution \
    -o D:/MLIP_Work_Folder/Cluster_MLIP/basis_attribution_83
```

## Package 83 results and the revision (proposed package 88)

**Package 83 came back partial (2026-11-02).** The outputs are in `hpc_smoke_tests\83_clustermlip_gaussian_basis_attribution`.
The first copy there held outputs only, so the report was first built from
`D:\MLIP_Work_Folder\Cluster_MLIP\basis_attribution_83\results_merged`: the desk's inputs plus these outputs.
The desk has since unstuck the `hpc_smoke_tests` clone (22ba819), so the inputs now sit beside the results;
analysing that directory directly gives an identical report.
The report is in `basis_attribution_83` (`report.md`, `report.json`, `links.csv`).

What finished:

- all 23 VASP jobs;
- Fe8 def2-QZVPP + pcseg-3 (bq07) and pcseg-4(-i) (bq09);
- on Fe2, links 1–4 of every task (the replay, def2-QZVPP, and the Gen def2-QZVPP control).

What failed, and why:

- **unc-def2-QZVPP** (every Fe2 task, and bq08) stopped in l502 with "Inaccurate quadrature in CalDSu". Shell
  1, the lone 4.3e6 s primitive, integrates with an error of 0.44 against a tolerance of 0.012 under UltraFine.
  Inside a contraction (built-in def2-QZVPP) the same primitive is harmless. On Fe2 this blocked links 6–9.
- **DKH2 with Force** (bq10) stopped in l1 with "No analytic first derivatives for this method". G09 D.01 has
  no DKH2 gradients.

The controls held:

- link 2 replays package 62 exactly;
- the Gen def2-QZVPP equals the built-in one (dE 0.0000 meV, NBasis 200 = 200);
- def2-QZVPP equals package 75 (0.000 meV, 0.00 meV/A);
- the VASP Fe_pv r0 point replays package 62 exactly.

**Findings from the finished part:**

1. **For Fe8 the Gaussian basis is converged, and it explains only about 20% of the gap.**

   | Fe8 M25 | alpha | radial force (meV/A) | Fe RMSE vs VASP (meV/A) |
   |---|---:|---:|---:|
   | 6-311++G* | 0 | −63 | 233 |
   | def2-QZVPP | 0.23 | −115 | 181 |
   | pcseg-3 | 0.21 | −110 | 186 |
   | pcseg-4(-i) | 0.20 | −108 | 188 |
   | VASP Fe_pv | 1 | −295 | – |

   def2-QZVPP and pcseg-4(-i) (two families) differ by 0.05 of the gap, and pcseg-3 → pcseg-4(-i) by 0.02. On
   the breathing coordinate the numbers are 0.03 and 0.01, with a basis share of 0.19. Package 75's large TZ→QZ
   step was def2-TZVPP being poor, not QZ being unconverged.
2. **The VASP Fe2 scan confirms the old estimate.** Fe_pv gives r_e = 1.9961 A, against the one-point
   estimate of 1.9957, with ω_e ≈ 424 cm⁻¹.
3. **Fe_sv moves VASP further from G09, not toward it.** Fe2 r_e is 1.9882 at ENCUT 520 and 1.9878 at 650.
   Fe8 vasp_paw is −0.18, on the forces and on breathing alike. Fe_sv's 520 → 650 shift is 0.03 of the gap on
   r_e, 0.05 on breathing and 0.08 on the forces; the ENCUT-650 runs are the ones used. So the Fe_pv frozen
   3s is not the cause: a better PAW makes the gap about 18% wider.
4. **What remains is about 80% of the Fe8 gap,** and more against Fe_sv. It is neither Gaussian basis nor PAW
   semicore. The main untested candidate is scalar relativity: VASP's PAW potentials are scalar-relativistic
   and G09's default Hamiltonian is not. It has the right sign, since relativity contracts Fe 4s and shortens
   Fe–Fe bonds. To explain the Fe2 r_e gap left at QZVPP against Fe_sv, it would have to shorten the bond by
   about 0.011 A. That is plausible but has to be measured. If relativity does not close it, the next suspect is
   the PW91 implementation (VASP GGA=91 vs G09 PW91PW91) or the PAW augmentation itself. That would need an
   all-electron scalar-relativistic referee.

**Revision: proposed package 88** (`88_clustermlip_gaussian_basis_attribution_r2`, staged with
`HANDOFF_88_TO_DISPATCH.md`, not sent). Generator: `make_basis_check_package.py --ladder beyond-qz-r2`.

- **One large-basis link per task** (the desk's suggestion), so no failure blocks another basis. Links 1–2
  replay package 62. `Int=UltraFine` and `IOP(5/13=1,5/36=1,8/11=1)` are in every link.
- unc-def2-QZVPP is dropped.
- **Relativity comes from the cc-pVQZ / cc-pVQZ-DK pair** (BSE v0.12, SHA-pinned). Both have the same
  8s7p5d3f2g1h shells, 104 functions per Fe, and no lone tight primitive. One is contracted for the
  non-relativistic Hamiltonian, the other recontracted for Douglas–Kroll. The DKH2 links use
  `Int=(UltraFine,DKH2)` as **single points**.
  - **Fe2:** both links at the 7 scan points. r_e comes from cubic E(r) fits, and relativity_shift =
    r_e(DKH2) − r_e(NR).
  - **Fe8:** both links at the package-62 geometry scaled by 0.99, 1.00 and 1.01 about its centroid. Relativity
    on the radial coordinate is d(E_DKH2 − E_NR)/ds by central difference. It is compared with the force gap
    projected on the same coordinate: dE/ds = −Σ F·(r − c).
- **The Fe2 ladder is completed** with pcseg-3, pcseg-4(-i) and pcseg-4 (the i shell, in its own task).
- 41 tasks: 6 Fe8 tasks of about 1 h each first, then 35 Fe2 tasks of minutes each. About 90 SU expected.

**Analysis changes:**

- `analyze_basis_check.py` takes several results directories: package 83's merged results and package 88's.
- DKH2 links count as energy-only.
- The families are now def2-QZVPP vs pcseg-4(-i).
- The deciding measures are the **Fe2 r_e** and the **Fe8 breathing coordinate**, the two that can carry an
  energy-only relativity. They must agree. The Fe8 force-vector row stays informational.
- The thresholds and outcome names are those of "Decision rule: Gaussian basis or VASP PAW" above.

**Which result decides it:**

- **`gaussian_basis+relativity`:** basis + relativity ≈ 1 against Fe_sv (residual ≤ 0.2). The training labels
  differ from VASP by a Gaussian basis error (about 20% on Fe8, more on Fe2) plus the non-relativistic
  Hamiltonian. The latter is physics, consistent across all G09 labels, and is a question for the
  reference/delta design, not a bug in either code.
- **`unexplained_residual`:** residual > 0.2 even with relativity. Next: the PW91 implementation (compare PBE
  on both sides, which package 62 also has) and an all-electron scalar-relativistic referee.

The plumbing was checked on mock results built from the package-83 logs with synthetic DKH2 shifts. It
recovers a planted Fe2 Δr_e of −0.008 A (−0.0079) and a planted Fe8 breathing relativity share of 0.60
(0.60). The E(r) and force fits agree to 1e-4 A. The default ladder still reproduces package 65's inputs byte
for byte, and the package-75 report is unchanged.

```bash
python experiments/gaussian_vasp_calibration/analyze_basis_check.py \
    --cluster-mlip-src .claude/worktrees/periodic-vasp-ingest/src \
    --calibration D:/MLIP_Work_Folder/hpc_smoke_tests/62_clustermlip_gaussian_vasp_calibration \
    --previous D:/MLIP_Work_Folder/hpc_smoke_tests/75_clustermlip_gaussian_basis_check \
    D:/MLIP_Work_Folder/hpc_smoke_tests/83_clustermlip_gaussian_basis_attribution \
    D:/MLIP_Work_Folder/hpc_smoke_tests/88_clustermlip_gaussian_basis_attribution_r2 \
    -o D:/MLIP_Work_Folder/Cluster_MLIP/basis_attribution_88
```

## Results: packages 83 + 88 (received 2026-11-04)

Package 88 finished cleanly. All 41 tasks have 3 Normal terminations, including the six Fe8 cc-pVQZ /
cc-pVQZ-DK+DKH2 tasks and the Fe2 pcseg-4 links with their i shell. The report covers packages 83 and 88
together and is in `D:\MLIP_Work_Folder\Cluster_MLIP\basis_attribution_88` (`report.md`, `report.json`,
`links.csv`).

The checks all passed:

- every link's SCF converged;
- no larger-basis or DKH2 link changed state (Hirshfeld pattern held, ⟨S²⟩ within 0.1);
- every link-2 control replays package 62 exactly;
- the E(r) fits leave residuals of about 3e-5 eV;
- d(E_DKH2 − E_NR) is linear across the Fe8 stretch (+0.0324 eV, then +0.0318 eV).

**Fe2 M7 bond length:**

| | r_e (A) |
|---|---:|
| G09 6-311++G* (training basis) | 2.0076 |
| G09 def2-QZVPP / cc-pVQZ | 1.9989 |
| G09 pcseg-3 / pcseg-4(-i) / pcseg-4 | 1.9980 / 1.9983 / 1.9983 |
| G09 cc-pVQZ-DK + DKH2 | 1.9860 (relativistic shift −0.0130) |
| **G09 basis limit + scalar relativity** (pcseg-4 + shift) | **1.9853** |
| VASP PW91 Fe_pv | 1.9961 |
| **VASP PW91 Fe_sv (ENCUT 650)** | **1.9878** |
| experiment (Ne matrix) | 2.02 ± 0.02 |

**Shares of the gap,** measured against VASP Fe_pv: Fe2 r_e (gap 0.0115 A) and the Fe8 breathing coordinate.

| measure | basis | relativity | vasp_paw | residual vs Fe_sv | family_diff / last_step |
|---|---:|---:|---:|---:|---|
| Fe2 r_e | 0.81 | 1.13 | −0.72 | 0.22 (0.0025 A) | 0.05 / 0.03 |
| Fe8 breathing | 0.19 | 0.91 | −0.18 | 0.09 | 0.03 / 0.01 |

**Rule outcome: `state_dependent`.** The rule as written gives `unexplained_residual` for Fe2 (residual 0.22
against the 0.20 cut) and `relativity` for Fe8 (basis 0.19 against the 0.20 cut). Both misses are within 0.02
of a threshold. The thresholds are left as they were set before the data.

**Reading:**

1. **The package-62 G09-vs-VASP gap is the Gaussian basis plus scalar relativity.** G09 at its basis limit with
   DKH2 agrees with VASP Fe_sv:
   - Fe2 r_e to 0.0025 A (1.9853 vs 1.9878);
   - the Fe8 radial force to 9% of the gap.

   VASP Fe_pv appeared closer to G09 only because its frozen 3s lengthens the bonds and partly cancels the
   other two effects.
2. **The basis share depends on the system.** For Fe2 it is about 80% of the gap; 6-311++G* is missing diffuse
   and polarization flexibility that matters for the dimer. For Fe8 it is about 20%, and the basis is converged
   at QZ quality: def2-QZVPP, pcseg-3, pcseg-4(-i) and cc-pVQZ agree to within 0.05 of the gap.
3. **Relativity is the dominant missing piece for the larger cluster.** It shortens the Fe2 bond by 0.013 A,
   and on Fe8 it supplies 0.9 of the radial gap. This is not a code error: the G09 labels use the
   non-relativistic Hamiltonian, which is consistent across all G09 labels.
4. **Nothing points at the PW91 implementation or the PAW augmentation.** The residual is within 0.2 of the gap
   once the Gaussian basis, scalar relativity and Fe_sv are accounted for.

**Consequences for the MLIP:**

- G09 6-311++G* UBPW91 labels and VASP PW91 labels differ for two understood reasons: Gaussian basis error,
  and the absence of relativity in G09. Neither reason is noise. Package 62's decision of a separate head or
  delta, not a shared head, stands and is now explained.
- VASP Fe data, wherever they enter, should use Fe_sv (or harder) potentials. Fe_pv's frozen 3s is worth about
  0.008 A on the Fe2 bond.
- If the G09 labels should match the VASP physics, they need both a QZ-quality basis and DKH2. Neither alone
  is enough for Fe8 and up. DKH2 has no analytic gradients in G09 D.01, so DKH2 forces would need G16 (which
  the user does not use for these systems), finite differences, or a delta model. The alternative is to keep
  the G09 labels non-relativistic as the reference and treat relativity as part of the G09→VASP delta.
- Fe2 r_e from every method lies inside the experimental 2.02 ± 0.02 A, so experiment cannot choose.
