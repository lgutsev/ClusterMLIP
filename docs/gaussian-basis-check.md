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
