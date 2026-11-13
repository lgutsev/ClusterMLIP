# ACAT-assisted adsorbate search on Fe16: feasibility pilot

2026-11-09. Branch `feat/acat-adsorbate-search` (from `origin/main` `bfb9668`). Assignment:
`ClusterMLIP_ACAT_Worker_Assignment.md` (2026-11-09).

## Summary

- **Recommendation: adapt.** Keep the search adapter, the site/atom-role representation and the
  duplicate test. Do not use any available MLIP to rank Fe16 + CO + 2H structures. The next step
  is the reference panel in section 8 (20 draft Gaussian inputs, not submitted).
- **Pilot.** Neutral Fe16, geometry `fe16-200-49-gf` (UBPW91 M = 49 optimum), one CO and two H,
  all Fe fixed (Stage A). This is a software and model benchmark on a bare gas-phase cluster. It
  says nothing about supported Fe/K on θ-alumina or a working catalyst.
- **No Fe16 + CO/H reference label exists**, and no in-house checkpoint covers Fe with C, O and H.
  The only calculators that can run the composition are spin-blind foundation models.
- **The MLIP gate failed.** On existing UBPW91 labels for Fe2/Fe4 with CO or H, MACE-MPA-0 orders
  58–84% of same-multiplicity pairs correctly and MACE-MATPES-r2SCAN 42–66%; both have force errors
  larger than the reference forces themselves. The six lowest Fe16 + CO + 2H candidates span 0.016 eV on the model
  surface, far below the model's error, and the model puts desorbed CO only 0.02 eV above the best
  bound structure.
- **Search comparison (model surface only, 200 relaxations × 3 seeds per arm).** The evolutionary arm reached a lower model energy than the random arm in all three seeds, by 0.02–0.06 eV,
  using about 40% fewer calculator calls. That margin is smaller than the model's error, so it shows the
  search works, not that it found better structures.
- **ACAT itself.** Its cluster site finder cannot run here (it needs `asap3`, Linux only) and
  assumes fcc/icosahedral/decahedral shapes; Fe16 is a centred 15-atom shell. Sites come from a
  built-in convex-hull finder instead. ACAT's genetic operators depend on its own site objects, so
  the evolutionary arm uses equivalent operators written against our site list.

## 1. Project state found (audit)

| Item | State on 2026-11-09 |
|---|---|
| Repository | local checkout of `ClusterMLIP`, remote `github.com/lgutsev/ClusterMLIP` |
| Main checkout | branch `feat/irc-force-labels` at `73be9cf`, 21 modified and 13 untracked files (IRC force labels + local spins, uncommitted). Not touched. |
| `origin/main` | `bfb9668` (PR #22). This work is a separate worktree, `.claude/worktrees/acat-adsorbate-search`. |
| AGENTS.md | None in the repository or its worktrees. |
| Current priorities (`docs/fe16-next-steps.md`, 2026-10-07) | Fe16 structural coverage, re-collect and split by unique geometry, then active learning. Supported clusters and reactions are listed as later work. |
| Existing search / active learning | `active_learning.py`: committee force disagreement and `select-next-batch`. No structure search, no site finder, no adsorbate code. |
| Calculator interface | `delta.py`: `load_model`, `DeltaCalculator`, the per-atom `cluster` array (1 = cluster). Reused here; the new `role` array extends it. |
| Compute | Laptop GPU (8 GB) for inexpensive local runs. Gaussian jobs go through the dispatch desk and need the user's approval per package. Nothing was submitted for this pilot. |

### Fe16 history, as recorded

- **Three geometries, P0–P3.** The broken-symmetry pilot kit (`fe16_bs_tandem_kit/pilot`, 2026-10-08)
  has 12 jobs: geometries `fe16-200-49-gf`, `fe16-90-47-gf` and `fe16-M-49-gf`, each with four
  initial spin patterns at M = 49: P0 all α, P1 central Fe β, P2 and P3 one surface Fe β. Each job
  is three Gaussian links on one checkpoint: fragment-guess SP, `Stable=Opt` with Hirshfeld/CM5,
  then `Force`. Level UBPW91/6-311++G*.
- **What ran.** The 12-job pilot was never submitted. Only the smoke job ran (package 36,
  geometry 1, P1): the flip collapsed to the ferromagnetic state. The full-AFM start (packages 61,
  78, 82) never converged. The flip-down ladder from the converged M = 49 state worked (packages
  91 rev 2 and 99) and gave the first low-spin Fe16 labels: 53 force frames in
  `fe16_ladder_labels_v1`, all in the `fe16-200-49-gf` basin.
- **None of these jobs contains CO or H.** They are bare Fe16.

### Earlier model audits, as recorded

| Audit | Reference | Metric | Value |
|---|---|---|---|
| In-house Fe16 MACE (2026-10-06, `test/fe16-local-spin-message-passing`) | UBPW91/6-311++G*, gas phase | Force MAE, train / 21 unseen geometries | 2–5 / 51 meV/Å |
| MACE-MP-0 on Fe16 isomers (`examples/supported_fe16_tem`) | 62 UBPW91 minima | Distinct model basins after relaxation | 7 |
| Periodic MACE on Fe3/θ-Al2O3(010) ± N2, package 32 (2026-10-08) | PBE, VASP, periodic | Fe force RMSE on the test frames: fine-tune / two scratch seeds / reference force RMS | 0.27 / 0.036 and 0.059 / 0.010 eV/Å |

The package 32 numbers are force RMSE per element on the `A3_fe3n2_end` test frames, compared with
the RMS of the PBE forces on the same atoms ("zero model"). All three models are worse than zero
on Fe. Those checkpoints were trained on Fe/Al/O/N and are not usable for CO or H. Their status is
unchanged since October.

**Supported Fe/K on θ-Al2O3:** potassium does not appear in this
repository, its branches or the ClusterMLIP work folders. The only θ-alumina work recorded here is
stage A3 (Fe3/θ-Al2O3(010) ± N2, PBE). I could not verify any model coverage for that target or reference
protocol from here.

### Gap table for Fe / CO / H

| Need | What exists | Gap |
|---|---|---|
| Elements Fe + C + O + H in one reference set | Warehouse 2 seeds (UBPW91): Fe2/Fe4 with CO, with H/H2, with O and N | No structure with CO **and** H on the same cluster. No Fe16 with any adsorbate except N2 (8,097 Fe16N2 records). |
| Environment: Fe16 surface sites | 1,148 bare Fe16 records, 53 ladder frames | No adsorption geometry on Fe16 |
| Adsorption geometries | Fe4CO: 47 records (4 labelled minima). Fe4H2: 762. Fe2CO: 261. Fe2H2: 406 | Small clusters only; no bare Fe2/Fe4 in the seeds, so no adsorption energy can be formed from them |
| Relative energies | Energies for all of the above | Different M per record; few geometries have more than one M |
| Forces | `dataset_spin_v0`: Fe16 and Fe2H2 neutral frames | No CO-containing force frames in the collected set |
| Magnetic states | Fe16: M = 39–53 in one basin, 47–53 elsewhere | None for Fe16 + adsorbates |
| Charges / populations | Mulliken everywhere, Hirshfeld on packages 36, 91, 99 | Mulliken on the central Fe is a known artifact; Hirshfeld needed |
| Fragmentation (C + O, HCO, COH) | Fe2/Fe4 CO IRC and TS records | None on Fe16 |
| Consistent references | UBPW91/6-311++G* Gaussian, gas phase | Periodic PBE (A3 slab) is a different method and must not be mixed with it |
| MLIP covering Fe/C/O/H | Foundation models only (no charge or spin input) | No in-house checkpoint |

## 2. The pilot

| | |
|---|---|
| Catalyst | Fe16, `fe16-200-49-gf`, the UBPW91/6-311++G* optimum at M = 49 (package 91 frame 0, force link; max force 0.012 eV/Å) |
| File | `experiments/acat_fe16_co_h/catalyst/fe16_200_49_gf_m49.extxyz` |
| Why this geometry | The only Fe16 basin with a stability-checked ferromagnetic state and converged lower-spin labels (M = 47, 45, 41) |
| Charge, multiplicity of the bare cluster | 0, 49 (ferromagnetic, all Hirshfeld moments positive) |
| Periodicity | None. Isolated cluster, no cell, no vacuum setting |
| Catalyst atom IDs | 0–15, in the order of the file. Atom 3 is the interior atom (15 Fe neighbours within 3.0 Å); atoms 0–2 and 4–15 are the surface |
| Adsorbates | One intact CO and two H (4 atoms, appended after the catalyst) |
| Stage A constraint | All 16 Fe fixed at the UBPW91 positions; only C, O, H move |
| Stage B (not run) | Same search with Fe free in both arms, tracking Fe displacement. Only after a model passes the gate |

One CO + two H was kept as proposed. No label supports a better choice: the warehouse has no
coadsorbed CO + H structure on any cluster.

**Stage A is a constrained problem.** MACE-MPA-0 puts forces of up to 0.88 eV/Å on the bare Fe16
at the UBPW91 geometry, so the fixed cluster is not a minimum of the model. Rankings with Fe
fixed are not relaxed predictions on either surface.

## 3. Sites and atom roles

`find_cluster_sites` (in `src/cluster_mlip/adsorbate_search.py`) takes the convex hull of the
catalyst atoms. On this Fe16:

| | Count | Check |
|---|---|---|
| Surface atoms (top sites) | 15 | every atom except atom 3 |
| Bridge sites | 39 | every hull edge is shorter than 3.0 Å |
| 3-fold hollows | 26 | every hull face is a bonded triangle |
| 4-fold hollows | 0 | none on this cluster |
| Interior | atom 3 | 15 neighbours; never part of a site |

15 − 39 + 26 = 2, so the site list is a closed triangulated surface with nothing missing. Each
site stores its catalyst atom indices, position and outward normal (`sites.json` in each run).

Limits of the finder:

- It is for an isolated cluster and refuses a periodic cell. A slab or a supported cluster needs
  `sites_from_list` (manual list of catalyst atom indices) or a slab finder.
- A concave surface atom would be classed as interior. The finder reports any interior atom with
  fewer than 10 neighbours; none here.
- No subsurface sites.
- **ACAT cross-check not done.** `check_acat_sites.py` is ready for a Linux host with `asap3`.
  ACAT may still refuse this shape, since its classifier expects fcc-like facets.

Roles are stored per atom, never inferred from the element: `role` (0 support, 1 catalyst,
2 adsorbate), `mol_id` (−1, or the adsorbate number) and `cluster` (1 = catalyst, the existing
repository convention). A supported system keeps lattice O as role 0 and adsorbed O as role 2.

Candidates are built from one gene per adsorbate: species, site, binding mode, tilt and azimuth.
CO has three modes: C-bound, O-bound, and side-on (C on the site, axis tilted 55–85°). Before
relaxation a candidate is rejected if two adsorbates are too close (H–H < 1.30, H–C < 1.50,
H–O < 1.40 Å) or an adsorbate atom is closer to Fe than 0.75 × the covalent-radius sum. The
filter uses minimum-image distances in a periodic cell.

After relaxation every structure is read back: fragments (CO, H, H2, HCO, COH, separate C and O),
the Fe atoms each fragment is bonded to, desorption, atom overlap and Fe displacement. The
intended site is never assumed. Dissociated or reacted structures are kept and labelled, at the
same elemental composition; atom overlap is a failure.

Duplicates: for the isolated cluster, sorted interatomic distances per element pair (adsorbate–Fe
and adsorbate–adsorbate) within 0.15 Å and energies within 0.02 eV. This removes rigid motion
and the order of identical atoms and treats mirror images as the same. For a supported or
periodic system the comparison is in the fixed frame of the support, so a rigid rotation is a
different structure. The magnetic state is not part of the test: labels of different (charge, M)
must be kept apart by the caller.

## 4. Reference electronic state and energy convention

- **Level:** UBPW91/6-311++G*, Gaussian 09 D.01, the group's reference. `IOP(5/13=1)` stays in
  every link; frames whose SCF printed "Convergence criterion not met" are dropped afterwards.
- **Spin.** Fe16 + CO + 2H has an even electron count, so M is odd. The bare cluster's ground
  state in this basin is M = 49, with M = 47 at +0.45 eV and M = 45 at +1.11 eV (adiabatic). CO and
  H are expected to lower the moment, so the panel computes M = 49, 47 and 45 for every structure
  by the flip-down ladder (`Stable=Opt` then `Force` per M, `Guess=Read`), the procedure that
  converged for the bare cluster. Fresh low-spin guesses did not converge there (packages 61–82).
- **Stability and branch checks:** `Stable=Opt` at each M, ⟨S²⟩ before annihilation, Hirshfeld
  spins per atom. Two labels are the same state only if energy, ⟨S²⟩ and Hirshfeld spins agree.
- **What a geometry-only MLIP label means here.** The foundation models and any model without a
  spin input see one energy per geometry. The target for such a model is the **lowest-energy
  envelope over M = 49, 47, 45** at each geometry, evaluated the same way for every structure. A
  spin-aware model (the project's MACE with global charge/M) can be trained on each branch
  separately. If the lowest M changes between candidates within the low-energy window, a
  geometry-only model cannot represent the ranking without that envelope, and this must be stated
  with the result.
- **Energy comparisons.** Total energies are compared only at the same composition
  (Fe16 + CO + 2H), charge 0, and the same M protocol. For other coverages the convention is the
  adsorption energy E(Fe16·CO·2H) − E(Fe16) − E(CO) − E(H2), each term at its own lowest M with the
  same route; CO and H2 are in the warehouse at this level, bare Fe16 is package 91. A gas-phase
  thermodynamic extension needs reservoir chemical potentials μ(CO; T, p) and μ(H2; T, p). The
  electrode-potential formula of the Pt(100) study does not apply.
- Low-energy structures are not rates and not selectivity.

## 5. MLIP gate

### Checkpoints tested

| Name | File (sha256, first 12) | Training data | Charge / spin input |
|---|---|---|---|
| MACE-MPA-0 medium | `mace-mpa-0-medium.model` (`75428afe3a1d`) | MPtrj + sAlex, PBE(+U), periodic solids. Provenance of this local file was never checked (`MANIFEST.tsv`). | none |
| MACE-MATPES-r2SCAN-0 | `MACE-matpes-r2scan-omat-ft.model` (`8f147ecffa1d`) | MATPES r2SCAN, periodic solids | none |

Both through `mace.calculators.MACECalculator` (mace-torch 0.3.16, ASE 3.29), no dispersion
correction. Gate runs in float64; the search runs in float32 on the GPU.

### Acceptance thresholds

Set from the candidate separations found in section 6 and the convergence of the reference:

| Quantity | Threshold | Why |
|---|---|---|
| Relative energy within one composition and one spin branch | MAE ≤ 0.03 eV | the six lowest candidates span 0.016 eV on the model surface; motif minima are 0.01–0.09 eV apart |
| Pair ordering, pairs ≥ 0.05 eV apart | ≥ 90% | below this the top of a ranking is not informative |
| Forces on adsorbate atoms and on Fe, separately | RMSE ≤ 0.10 eV/Å and below half the reference force RMS | relaxations stop at 0.05 eV/Å |
| Adsorption geometry | Fe–C, Fe–H, C–O within 0.05 Å of reference minima; same site | |

The reference side is converged to about 0.01 eV for same-route differences (SCF `Tight`,
`Int=UltraFine`); the basis-set work of packages 75–88 concerns absolute Fe–Fe energetics and is
not part of this margin.

### Result on existing labels: fail

Energies (`gate_small_clusters.py`; neutral records, config types `minimum` and
`optimized_unverified`, one entry per distinct geometry). "Same M" compares only records of equal
multiplicity, the fair test for a model without spin input.

| Set | Distinct geometries | Model | Same-M pairs ordered | Same-M rel. energy MAE (eV) | All pairs ordered |
|---|---|---|---|---|---|
| Fe2(CO) | 85 | MPA-0 | 491 / 698 (70%) | 0.11 | 58% |
| | | r2SCAN | 459 / 698 (66%) | 0.31 | 44% |
| Fe4(CO) | 25 | MPA-0 | 31 / 52 (60%) | 0.83 | 29% |
| | | r2SCAN | 30 / 52 (58%) | 0.23 | 29% |
| Fe2H2 | 55 | MPA-0 | 335 / 400 (84%) | 0.13 | 66% |
| | | r2SCAN | 167 / 400 (42%) | 0.33 | 48% |
| Fe4H2 | 269 | MPA-0 | 1446 / 2365 (61%) | 0.15 | 56% |
| | | r2SCAN | 1489 / 2365 (63%) | 0.26 | 56% |
| Fe16 (bare) | 631 | MPA-0 | 26784 / 49400 (54%) | 0.62 | 52% |
| | | r2SCAN | 47425 / 49400 (96%) | 0.33 | 88% |

Forces (`gate_forces.py`; `dataset_spin_v0`, RMSE in eV/Å, next to the RMS of the reference
forces):

| Set | Frames | Element | MPA-0 | r2SCAN | Reference force RMS |
|---|---|---|---|---|---|
| Fe16 (bare) | 300 | Fe | 0.18 | 0.24 | 0.06 |
| Fe2H2 | 241 | Fe | 0.82 | 1.23 | 0.46 |
| | | H | 0.36 | 0.31 | 0.19 |

Reading these numbers:

- Every adsorbate-bearing set fails both thresholds for both models. r2SCAN orders bare Fe16
  well within one M (96%), but its forces there are four times the reference RMS.
- Most records are steps of geometry optimizations, so pairs from one trajectory are correlated
  and easy. The pass rates are upper bounds.
- The functional differs (PBE or r2SCAN against BPW91), and there is no spin input. Both are
  part of what is being tested, not excuses.
- Not measured, because no labels exist: anything on Fe16 with CO or H, coadsorption, CO
  orientation, short-contact failures, adsorption energies (no bare Fe2/Fe4 in the seeds), the
  CO-containing force errors, and same-geometry comparisons across M beyond the few in the table.
- There is no committee of independent Fe/C/O/H models. The difference between the two
  foundation models is recorded per candidate as a disagreement flag. It is not an uncertainty
  estimate and does not replace reference checks.

Because the gate failed, section 6 is a benchmark of the search methods on a model surface, and
its lowest structure is not a statement about Fe16.

## 6. Equal-budget search comparison (model surface)

Both arms: 200 relaxations per seed, seeds 0–2, MACE-MPA-0 (float32, GPU), all Fe fixed, BFGS to
0.05 eV/Å, at most 200 steps. Config `configs/pilot_mpa0_stageA.json`, commit `ca5f749`. Energies
are model energies relative to the lowest structure found by any run ("pooled minimum").
Reference DFT calls: 0.

| Arm | Seed | Proposals | Rejected before relaxing (distance / duplicate start) | Relaxations | Converged | Duplicates after relaxing | Unique | Motifs | Calculator calls | Best energy above pooled minimum (eV) |
|---|---|---|---|---|---|---|---|---|---|---|
| Random | 0 | 216 | 16 / 0 | 200 | 200 | 0 | 200 | 20 | 10,907 | 0.057 |
| Random | 1 | 211 | 11 / 0 | 200 | 200 | 1 | 199 | 20 | 11,538 | 0.036 |
| Random | 2 | 213 | 13 / 0 | 200 | 200 | 1 | 199 | 24 | 10,715 | 0.033 |
| Evolutionary | 0 | 244 | 5 / 39 | 200 | 199 | 41 | 159 | 18 | 6,898 | 0.000 |
| Evolutionary | 1 | 228 | 7 / 21 | 200 | 200 | 44 | 156 | 13 | 7,651 | 0.013 |
| Evolutionary | 2 | 226 | 4 / 22 | 200 | 200 | 43 | 157 | 15 | 6,080 | 0.008 |

No relaxation failed and no calculator error occurred. One evolutionary relaxation hit the step cap
and is counted as not converged.

Best energy so far above the pooled minimum (eV), seeds 0 / 1 / 2:

| Budget | Random | Evolutionary |
|---|---|---|
| 50 relaxations | 0.086 / 0.039 / 0.033 | 0.047 / 0.052 / 0.024 |
| 100 relaxations | 0.057 / 0.036 / 0.033 | 0.047 / 0.013 / 0.009 |
| 200 relaxations | 0.057 / 0.036 / 0.033 | 0.000 / 0.013 / 0.008 |
| 3,040 calls | 0.086 / 0.039 / 0.033 | 0.047 / 0.013 / 0.009 |
| 6,080 calls | 0.057 / 0.036 / 0.033 | 0.000 / 0.013 / 0.008 |

What the comparison supports:

- **The evolutionary arm is ahead in all three seeds** at 100 and 200 relaxations and at matched
  calculator calls. At 50 relaxations it is not (the first 20 are the shared random population).
- **It is cheaper per relaxation.** Children start near relaxed parents: 30–38 calls per
  relaxation against 54–58 for random starts.
- **It samples less widely.** 156–159 unique structures and 13–18 motifs per seed against
  199–200 and 20–24; about 21% of its relaxations end in a structure it already had. Within
  0.30 eV of the pooled minimum it still holds 6–9 motifs per seed, so it did not collapse onto one.
- **Coverage of the low-energy pool.** 451 distinct structures from all runs lie within 0.30 eV.
  Each evolutionary seed found 21–22% of them, each random seed 13–15%. Only one run
  (evolutionary, seed 0, at relaxation 125) found the pooled minimum. 200 relaxations do not
  exhaust this surface for either arm.
- **The margin is 0.02–0.06 eV**, on a surface whose six lowest structures span 0.016 eV, with a
  model whose error on comparable reference data is 0.1–0.8 eV. The result is about search
  efficiency on a given surface. It is not evidence that the structures are better.
- **Wall time is not reported as a comparison.** The runs shared the laptop GPU with each other
  and with an unrelated job; per-run wall times range from 399 to 2,860 s with no relation to the
  number of calls. Calculator calls are the cost measure here. The one run that had the GPU
  mostly to itself took 399 s for 200 relaxations.
- **Not measured:** recovery of DFT-confirmed motifs. No reference label exists, so "low-energy"
  above means low on the model surface, and coverage is counted against the pool of these six
  runs, not against all possible configurations.

Figures: `experiments/acat_fe16_co_h/report/progress.png` and `diversity.png`. Per-run table:
`report/runs.csv`. Comparison: `report/arms.json`.

## 7. Audit of search results

All 991 pooled structures were read back after relaxation (fragments, sites, contacts).

| Check | Result |
|---|---|
| Atom overlap, composition or role errors | 0 |
| CO dissociated, HCO/COH, H2 formed | 0 in 1,200 relaxations |
| Fe displacement | 0 (Fe fixed in Stage A) |
| CO with O nearer to Fe than C | 33 pooled structures, 33 of them with CO desorbed |
| **CO desorbed** (no C or O within 3.3 Å of Fe) | **46 pooled structures, 34 of them within 0.30 eV of the pooled minimum; the lowest is 0.022 eV above it** |

**The desorbed-CO result is a model artifact and the main finding of the audit.** On MACE-MPA-0 a
CO molecule 6–7 Å from the cluster, with both H still adsorbed, costs 0.02 eV relative to the best
bound structure. The model gives CO essentially no binding energy to this Fe16 with Fe fixed.
MACE-MATPES-r2SCAN puts the same structures 1.8–2.4 eV above the bound ones. The two cannot both
be right, and neither has been checked against UBPW91 for this system. The search did not treat
these as discoveries: they are flagged, excluded from the parent population, and two of them are
on the DFT shortlist because they measure the CO binding energy directly.

The two foundation models also disagree on the bound structures: over the 451 low-energy pool
entries they order 72% of pairs the same way, the root-mean-square difference in relative energy
over the whole pool is 0.43 eV, and the MPA-0 minimum is rank 28 for r2SCAN.

No CO activation appeared at all. With a model that barely binds CO, that absence carries no
information about Fe16.

## 8. Reference-validation matrix and DFT shortlist

**Prepared, not submitted. No compute authorization exists for it; it needs the user's approval
and a desk package.**

Files: `report/shortlist.csv`, `report/shortlist.extxyz`, `report/dft_draft_frames.csv`. Draft
inputs: `D:\MLIP_Work_Folder\Cluster_MLIP\acat_fe16_co_h\dft_draft\inputs\` (20 files, written by
`make_dft_inputs.py`; the folder is not a dispatch package).

| Group | Structures | Purpose |
|---|---|---|
| Lowest structure of each of the six lowest motifs | 6 (two have CO desorbed) | Motif ordering; CO binding energy |
| Near-degenerate competitors of the model minimum | 4 (within 0.016 eV) | Whether the reference separates what the model cannot |
| Largest disagreement between the two models, CO bound | 3 (H on top sites, 1.1–1.7 eV on MPA-0) | Extrapolative cases |
| Diverse controls from the random arm | 4 (0.10–0.51 eV) | Unbiased sample for error statistics |
| Constructed controls, not relaxed | 3: O-bound CO, side-on CO, separate C + O | High-energy controls and the dissociated state at the same composition |

Matrix per structure: charge 0; M = 49, 47, 45 by flip-down ladder; per M a `Stable=Opt` link and a
`Force` link with Hirshfeld populations; fixed geometry. 20 structures × 3 multiplicities = 60
labelled states, 120 links. Existing references used alongside: bare Fe16 at M = 49/47/45 (packages
91, 99), CO and H2 (warehouse).

Splits: the four diverse controls and the three constructed controls are the frozen audit set and
must never be used for fitting. The other 13 may enter training later; split by motif, not by
neighbouring frames.

Cost estimate (not measured for this composition): package 36 ran three links on bare Fe16 in
1.8 h on 12 cores. At a similar 0.6–1 h per link, one input (6 links) takes 4–6 h, so 20 inputs
come to roughly **1,000–1,500 core-hours** when the SCFs converge. The risk is the lower rungs:
for bare Fe16, M = 45 and below needed the 12 h link cap or did not converge. With a 12 h cap per
link and 48 h per job the ceiling is 11,520 core-hours. A sensible first submission is 4 inputs
(model minimum, one desorbed-CO structure, one near-degenerate competitor, the O-bound control)
to measure timing and SCF behaviour before the other 16.

Second tier, only after the first returns: constrained optimizations (Fe frozen) of the lowest
reference structures at their lowest M, then Stage B with Fe free.

## 9. Supported-catalyst extension (assessment)

What carries over from this pilot, and what does not.

**Carries over now (interfaces only):**

- `role` / `mol_id` arrays with a support role, `sites_from_list` for a manual site list on the
  supported Fe structure, minimum-image distance filters, fixed-frame duplicate test, the
  fragment analysis and the ledger. These are tested on a periodic slab toy case, not on alumina.
- Lattice O must be role 0 and adsorbed O/OH role 2. The fragment analysis only looks at role-2
  atoms, so an adsorbate O is never merged with the lattice. A CO that reacts with lattice O or a
  surface OH needs an added rule (role-0 atoms bonded to role-2 atoms) before such a search runs.

**Must be specified before any supported run:**

| Item | What to fix |
|---|---|
| Support | θ-Al2O3 termination and hydroxylation. Stage A3 used the (010) surface; its first slabs (A1/A2) were flawed and the A3 slab was later rejected (D-034). A validated slab is a precondition. |
| Fe structure | Size and shape on the support. A gas-phase Fe16 is not a supported particle: on graphene the cluster strain (1–2 eV in MACE-MP-0) exceeded the gas-phase isomer gaps. |
| K | Content and location (on the support, at the interface, on Fe). No K data exists in this project. |
| Catalyst state | Reduced, oxidized or carburized Fe are different materials with different references. The pilot represents bare metallic Fe only. |
| Adsorbates | CO, H, and what the question needs beyond them (OH, O, CHx). |
| Reference method | Periodic PBE (VASP) for the supported system against gas-phase UBPW91 (Gaussian) for clusters. Package 62 found only a difference (Δ) calibration between them and packages 83/88 trace the gap to the Gaussian basis and scalar relativity. The two label sets cannot be pooled. |
| Magnetism and charge transfer | Fe moments on an oxide support, K-induced charge transfer and the spin state per structure (NUPDOWN or a constrained moment) need their own convergence checks. |

**MLIP transfer** only after a support/interface gate like section 5, on supported labels. The
package 32 models were worse than zero on Fe forces, and the fine-tune had silently dropped its
spin embedding.

Adsorption thermodynamics from such a search is separate from barriers and from product
selectivity; neither follows from low-energy structures.

## 10. Handoff

**Branch and commits:** `feat/acat-adsorbate-search`, local only (not pushed), based on
`origin/main` `bfb9668`. Code and scripts: `ca5f749`; this document and the report files are in the
commit after it. The main checkout (`feat/irc-force-labels`, uncommitted work) was not touched.

**Files**

| Path | Content |
|---|---|
| `src/cluster_mlip/adsorbate_search.py` | Sites, genes, builder, filters, analysis, duplicate test, the two search arms, toy potential |
| `tests/test_adsorbate_search.py` | 24 checks |
| `experiments/acat_fe16_co_h/` | Catalyst file, configs, run/analysis/gate/draft scripts, README |
| `experiments/acat_fe16_co_h/report/` | Run table, arm comparison, shortlist, gate results, figures |
| `pyproject.toml`, `.github/workflows/tests.yml` | New optional extra `search` (numpy, scipy, ASE); CI installs it so the new tests run |
| `D:\MLIP_Work_Folder\Cluster_MLIP\acat_fe16_co_h\` | Full run output: six manifests and structure files, pooled candidates, pictures, draft Gaussian inputs |
| `D:\MLIP_Work_Folder\envs\acat_fe16` | Environment: venv on top of `defects`, plus `acat 2.0.3` and `ase-ga` |

**Checks run**

- `python -m unittest discover -s tests`: 351 tests pass, 2 skipped (system Python 3.14, ASE 3.29).
- `mypy src`: clean.
- New tests cover: site counts on an icosahedron and a cube; refusal of periodic cells by the
  cluster finder; composition and role preservation; placement distances and CO modes; rejection
  of crowded starts; minimum-image distance filter against the same structure without a cell;
  dissociated CO, H2, formyl, desorption and overlap recognition; duplicates under swapping
  identical atoms; rigid rotation as a duplicate for an isolated cluster and not for a supported
  one; ledger accounting, shared initial population, parent records and same-seed repeatability.
- Toy-potential smoke run repeated: identical final structures.

**Limits**

- No reference energy for Fe16 with CO or H exists. Nothing here is a physical result for Fe16.
- Both available models failed the gate; the search ran on MACE-MPA-0, which gives CO almost no
  binding to the cluster.
- Stage A only (Fe fixed at the UBPW91 geometry, which is not a minimum of the model). Stage B not run.
- One catalyst geometry, one composition, three seeds per arm.
- ACAT's own site finder and operators were not exercised (`asap3` unavailable on Windows).
- The duplicate test uses distance lists: mirror images count as one structure, and two different
  structures with identical distance lists would be merged (not observed, not excluded).
- Wall-time comparison is unusable (shared GPU).
- The draft Gaussian inputs have not been run through the desk's package generator or its checks.

**Next action (as written on 2026-11-09):** submit the 4-input first batch of the reference panel
through the desk. Its result decides everything else: if UBPW91 binds CO by more than about 1 eV
and orders the motifs differently from both models, the path is to label the remaining 16, add
Fe16 + CO/H frames to the project's spin-aware MACE training set, and rerun this same comparison
on that model. That batch has run; see section 11.

## 11. First reference results (desk package 112, 2026-11-11)

Four of the 20 shortlisted structures at UBPW91/6-311++G*, G09 D.01, fixed geometry, charge 0, ladder
M = 49, 47, 45 (`Stable=Opt` then `Force`). Review script: `experiments/acat_fe16_co_h/review_reference_batch.py`;
numbers in `experiments/acat_fe16_co_h/report/reference_batch1_review.json`.

**Run quality.** All 4 tasks: 6 of 6 Normal terminations, no watchdog stop, 4.9 to 5.2 h on 12 cores.
Every `Stable=Opt` reports a stable wavefunction; ⟨S²⟩ after annihilation is 600.01 / 552.01 / 506.02 to
506.03 against 600 / 552 / 506. Two links are flagged "Convergence criterion not met" (M = 49 `Stable=Opt`
of frames 1 and 3, 129 cycles); in both, the following `Force` link converged in 3 to 5 cycles to the
same energy to 1e-7 Ha. Energies and forces are taken from the `Force` links only. All 12 states are usable.

**Energies at M = 49, relative to the model minimum (eV).** Fixed model geometries, not UBPW91 minima.

| Frame | Structure | UBPW91 | MACE-MPA-0 | MATPES-r2SCAN |
|---|---|---|---|---|
| 0 | model minimum: CO on top through C, 2 H in hollows | 0 | 0 | 0 |
| 1 | CO desorbed (about 9 Å away), 2 H on the cluster | +1.54 | +0.02 | +1.78 |
| 2 | near-degenerate competitor | +0.16 | +0.01 | −0.12 |
| 3 | CO bound through O (control) | +1.95 | +2.32 | +2.30 |

- **UBPW91 binds CO by about 1.5 eV here.** The near-degeneracy of desorbed and bound CO on MPA-0 is a
  model artifact, as suspected in section 7. This is not a clean adsorption energy: the two H sit on
  different sites in frames 0 and 1, the geometries are MPA-0's, and CO is at 9 Å, not at infinity.
- **Neither model orders the two bound structures as UBPW91 does.** MPA-0 has them degenerate; r2SCAN
  has the wrong sign.
- Both models get the O-bound control roughly right.

**Forces against UBPW91 (RMSE, eV/Å; zero-force baseline in brackets), 4 frames.**

| M | MPA-0 Fe | MPA-0 adsorbate | r2SCAN Fe | r2SCAN adsorbate |
|---|---|---|---|---|
| 49 | 0.22 (0.21) | 0.49 (0.44) | 0.27 (0.21) | 0.77 (0.44) |
| 47 | 0.25 (0.13) | 0.45 (0.37) | 0.30 (0.13) | 0.81 (0.37) |
| 45 | 0.29 (0.18) | 0.46 (0.35) | 0.33 (0.18) | 0.85 (0.35) |

Both models are again no better than predicting zero force. The gate verdict of section 5 stands, now on
the target composition itself.

**What the UBPW91 forces say about the search.** At MPA-0's minimum (frame 0, M = 49) the UBPW91 forces are
large: up to 1.6 eV/Å on an adsorbate atom and 0.25 eV/Å RMS on Fe, on a cluster whose bare optimum had
0.012 eV/Å. So the model's adsorbate positions are far from UBPW91 stationary points, and freezing Fe
(Stage A) is not a small approximation once CO and H are on the surface.

**Spin.** M = 49 is lowest in all four frames. The vertical 49 → 47 gap is 0.13 to 0.15 eV for the two
bound structures, 0.21 eV for the O-bound control and 0.35 eV with CO desorbed, against 0.52 eV for bare
Fe16 at the same Fe positions (package 91). Adsorption narrows the spin gap by a factor of about 3.5.
For bare Fe16, relaxing at M = 47 gained 0.07 eV (package 99), so M = 47 could come within 0.1 eV of
M = 49 for bound structures once geometries relax. A spin-blind model cannot represent that.

**Consequences.**

- The condition set on 2026-11-09 is met: CO binds by more than 1 eV and the models do not reproduce the
  UBPW91 ordering. The remaining 16 inputs are worth sending. Measured cost: about 60 core-hours per
  structure, so about 1,000 core-hours for 16 (ceiling 9,216 with the 48 h cap).
- These labels are off-equilibrium by construction, which is what a correction model or a training set
  needs; they are not UBPW91 minima. A constrained optimization of the two lowest structures at M = 49
  and 47 is the cheapest way to get the first real minima and to test the narrowing spin gap.
- Of the three routes in the roadmap, the spin result argues for carrying M: a Δ-model or fine-tune that
  ignores M would have to be fitted per multiplicity.

Limits: four structures, one composition, fixed model geometries. No statement about the ground state of
Fe16 + CO + 2H follows from this.

## 12. The full 20-structure reference panel (desk packages 112 and 116, 2026-11-12)

The other 16 shortlisted structures ran as package 116, built exactly as 112. Scripts:
`review_reference_batch.py` per package, `combine_reference_panel.py` for the panel. Numbers:
`experiments/acat_fe16_co_h/report/reference_panel.csv` and `.json`.

**Run quality.** 16 of 16 tasks: 6 of 6 Normal terminations, no watchdog stop, 4.7 to 6.8 h each. Every
`Stable=Opt` reports a stable wavefunction. The whole panel cost 1,264 core-hours.

**One state is excluded.** Frame 0 of package 116 (`fe16coh2_001`) at M = 49: both the `Stable=Opt` link and
the following `Force` link stopped at 129 cycles with "Convergence criterion not met" (RMSDP 1e-5, MaxDP
2e-3, gap 0.001 Ha). `IOP(5/13=1)` let the job continue, so the log holds an energy and forces from an
unconverged density. They are not used. The desk's summary counted this link as converged; it is not. The
other three flagged links (112 frames 1 and 3, 116 frames 6 and 15) were followed by a converged `Force`
link with the same energy and are used. Usable states: 19 at M = 49, 20 at M = 47, 20 at M = 45.

**UBPW91 energies at M = 49, relative to the lowest structure (eV), with both models on the same geometries.**

| Structure (draft id) | Motif | UBPW91 | MPA-0 | r2SCAN | 49 → 47 gap |
|---|---|---|---|---|---|
| 000 | CO top, H hollow + hollow | 0.00 | 0.00 | 0.13 | 0.15 |
| 007 | CO top, H bridge + hollow | 0.02 | 0.01 | 0.06 | 0.24 |
| 008 | CO top, H hollow + hollow | 0.07 | 0.02 | 0.15 | 0.13 |
| 006 | CO top, H bridge + hollow | 0.16 | 0.01 | 0.00 | 0.13 |
| 004 | CO bridge, H bridge + hollow | 0.17 | 0.07 | 0.12 | 0.25 |
| 003 | CO bridge, H hollow + hollow | 0.18 | 0.06 | 0.19 | 0.12 |
| 014 | CO top, H bridge + hollow | 0.22 | 0.10 | 0.17 | 0.19 |
| 009 | CO top, H hollow + hollow | 0.26 | 0.02 | 0.19 | 0.04 |
| 013 | CO top, H hollow + hollow | 0.44 | 0.14 | 0.25 | 0.12 |
| 015 | CO top, H bridge + hollow | 0.45 | 0.13 | 0.23 | 0.08 |
| 016 | CO bridge, H bridge + hollow | 0.48 | 0.51 | 0.69 | 0.03 |
| 012 | CO bridge, H bridge + top | 0.92 | 1.15 | 0.74 | 0.08 |
| 011 | CO hollow, H top + top | 1.17 | 1.55 | 1.12 | 0.25 |
| 010 | CO bridge, H top + top | 1.51 | 1.67 | 1.14 | 0.12 |
| 005 | CO desorbed, H hollow + hollow | 1.52 | 0.09 | 2.20 | 0.50 |
| 002 | CO desorbed, H bridge + hollow | 1.54 | 0.02 | 1.91 | 0.35 |
| 018 | CO side-on (control) | 1.89 | 4.40 | 1.82 | 0.09 |
| 017 | CO through O (control) | 1.95 | 2.32 | 2.43 | 0.21 |
| 019 | C + O dissociated (control) | 3.27 | 6.67 | 2.46 | −0.14 |
| 001 | CO top, H bridge + hollow | excluded | 0.01 | 0.00 | n/a |

Model columns are relative to each model's own lowest structure. For 001 the unconverged M = 49 energy
would put it near +0.19 eV; its converged M = 47 state is 0.29 eV above the lowest structure.

**What the panel shows.**

- **Both desorbed-CO structures sit 1.5 eV up in UBPW91.** MPA-0 has them at 0.02 and 0.09 eV. The artifact
  is systematic, not a one-off.
- **MPA-0's pick of the lowest structure is confirmed** (000), and r2SCAN's is not (it picks 006, +0.16 eV).
- **Among the 11 bound structures within 0.5 eV, both models rank fairly well:** MPA-0 orders 41 of 45
  resolved pairs correctly (91 %), r2SCAN 39 of 45 (87 %); correlation with UBPW91 about 0.7. But MPA-0
  compresses the scale: structures 0.16 to 0.45 eV up in UBPW91 appear within 0.14 eV.
- **Over all usable structures:** MPA-0 mean absolute error 0.60 eV (worst 3.4 eV, the dissociated control);
  r2SCAN 0.27 eV (worst 0.93 eV). MPA-0 is badly wrong wherever CO is not bound through C on Fe: desorbed,
  side-on, dissociated.
- **CO dissociation is 3.3 eV uphill at these fixed geometries.** Nothing was relaxed, so this is an upper
  bound for that particular C and O placement, not a reaction energy.

**Forces against UBPW91 on the 16 structures of package 116 (RMSE, eV/Å; zero baseline in brackets).**

| M | MPA-0 Fe | MPA-0 adsorbate | r2SCAN Fe | r2SCAN adsorbate |
|---|---|---|---|---|
| 49 (15 structures) | 0.24 (0.28) | 1.22 (1.31) | 0.28 (0.28) | 0.64 (1.31) |
| 47 | 0.25 (0.23) | 1.08 (1.22) | 0.28 (0.23) | 0.77 (1.22) |
| 45 | 0.29 (0.21) | 1.02 (1.18) | 0.31 (0.21) | 0.86 (1.18) |

This refines section 11. On these structures, which carry larger forces, r2SCAN halves the adsorbate force
error against the zero baseline; MPA-0 barely improves on it. Neither model beats the zero baseline on Fe
forces at any multiplicity. So the gate still fails, but the two models fail differently: MPA-0 ranks bound
structures well and is wrong on forces and on unbound or broken CO; r2SCAN is better on forces and energies
overall and worse at picking the minimum.

**Spin.**

- M = 49 is the lowest converged state in 18 of 20 structures. The dissociated control (019) is lowest at
  M = 47, by 0.14 eV. For 001 only M = 47 and 45 are usable, so its lowest state is undetermined.
- The 49 → 47 gap over 19 structures: median 0.13 eV, range −0.14 to 0.50 eV; six are below 0.10 eV, the
  smallest among bound structures 0.03 eV (016). Bare Fe16: 0.52 eV.
- The gap is largest with CO desorbed (0.35 and 0.50 eV, close to bare Fe16) and smallest for some bound
  structures. Binding CO is what narrows it. A model without M as input cannot describe this.

**Consequences.**

- The search adapter's output was useful: the model's lowest structure is UBPW91's lowest of the 19, and
  its near-degenerate set is the UBPW91 low-energy set. What the model cannot do is tell bound from
  unbound CO, give forces, or handle spin.
- The panel now holds 59 usable UBPW91 states (energies, forces, Hirshfeld spins) for Fe16 + CO + 2H. That
  is enough to score any corrected model against, and a start for training; it is not enough to train on
  alone, and every geometry is a model geometry.
- Next reference step, if wanted: optimize the three lowest structures (000, 007, 008) at M = 49 and 47
  with UBPW91, Fe frozen first. That gives the first real minima, tests whether M = 47 becomes competitive
  after relaxation, and yields on-path frames with small forces, which the panel lacks. The excluded state
  can be retried in the same package with a different SCF start.

Limits: one composition, fixed model geometries, 20 structures chosen by the models' own ranking. The
lowest structure here is the lowest of this panel, not a claim about the ground state.

## 13. UBPW91 optimizations with Fe frozen (desk package 118, 2026-11-13)

The three lowest panel structures (000, 007, 008) at M = 49 and M = 47: the 112 / 116 ladder to the target
M, then `Opt=ModRedundant` with `X n F` on the 16 Fe atoms. A seventh task retried the excluded state.
Review script: `experiments/acat_fe16_co_h/review_opt_package.py`; numbers in
`report/opt_frozen_fe_review.json`, final geometries in `report/opt_frozen_fe_final.extxyz`.

**Run quality.**

- The `X n F` freeze form was accepted. Fe atoms moved at most 0.003 Å in every optimization.
- Three optimizations reached "Stationary point found" (007 at M = 49; 008 at M = 49 and 47). Three were
  stopped by the 12 h link cap while still running: 000 at M = 49 (23 steps), 000 at M = 47 (22 steps),
  007 at M = 47 (13 steps). No Error termination. Tasks took 10.4 to 15.6 h.
- One optimization step (007 at M = 47, not the last) had an unconverged SCF. No other step did.

**Two of the three starting structures are the same minimum.** 000 and 008 relax to adsorbate positions
0.008 Å apart at M = 49 (0.06 Å at M = 47) with energies within 1.4 meV. So the capped 000 runs add
nothing that 008's converged runs do not already give, and need no restart. Only 007 at M = 47 is
genuinely unfinished (energy still falling by 1 meV per step).

**Relaxed energies (Fe frozen).**

| Minimum | M = 49 (Ha) | M = 47 (Ha) | 49 → 47 gap | Relative at M = 49 | Gain from relaxing |
|---|---|---|---|---|---|
| A (from 000 and 008): CO on Fe3, both H in hollows | −20336.37722 (stationary) | −20336.37172 (stationary) | 0.150 eV | 0 | 0.21 to 0.28 eV |
| B (from 007): CO on Fe7, H in bridge + hollow | −20336.37488 (stationary) | above −20336.36498 (unfinished) | at most 0.27 eV | +0.064 eV | 0.17 eV |

- **Relaxing the adsorbates does not close the spin gap.** For minimum A the gap is 0.150 eV relaxed against
  0.13 to 0.15 eV vertical. The suggestion in section 11 that M = 47 might come within 0.1 eV is not
  supported with Fe frozen.
- **A stays below B**, by 0.064 eV (0.02 eV before relaxation).

**What moved.** CO stays on its top site, linear (Fe–C–O 178°), Fe–C 1.80 Å, C–O 1.174 Å, but leans towards
a second Fe: that Fe–C distance shortens from 3.1 to 3.3 Å in the model geometry to 2.5 Å. C moves about
0.9 Å and O about 1.5 Å; the H atoms barely move (under 0.1 Å). The search model places CO too upright.

**The frozen cluster is under large forces.** At the relaxed minimum A, the UBPW91 forces on the frozen Fe
atoms are 0.24 eV/Å RMS and up to 1.1 eV/Å at M = 49, and 0.15 eV/Å RMS, up to 0.44 eV/Å at M = 47. The
bare cluster's optimum had 0.012 eV/Å. Relaxing the adsorbates did not reduce them. So these are minima of
the adsorbates on a strained cluster, not minima of Fe16 + CO + 2H, and the Fe forces differ between the two
multiplicities, so a full relaxation can change the gap in either direction. Stage A (Fe frozen) is good
for ranking sites; it is not good enough for adsorption energies or spin gaps.

**The retry failed.** For 001 at M = 49 all three links (`Stable=Opt`, `Force`, `Force`) again stopped at 129
cycles with "Convergence criterion not met" (MaxDP about 1e-3). The energy is steady to 6e-6 Ha at
−20336.36241 Ha, about 0.19 eV above the lowest structure, but the density is not converged and the state
stays excluded. Repeating the same SCF does not help; it needs a different SCF strategy, which is a choice
for the group's Gaussian practice and is low priority at this energy.

**Consequences.**

- No restart is needed for the capped runs: 000 is covered by 008. Restarting 007 at M = 47 would only
  refine a state at least 0.2 eV up.
- The next reference step that would change conclusions is a full relaxation (Fe free) of minimum A at
  M = 49 and 47, restarted from the checkpoints kept on LONI. Package 99's bare-Fe16 optimizations needed 14
  to 25 steps and hit the 12 h link cap, so this needs a longer link limit.
- For training, this package adds about 120 optimization frames with energies and forces, the first for
  this composition that lie along a relaxation path and end at small adsorbate forces.
