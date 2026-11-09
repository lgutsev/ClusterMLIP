# ACAT-assisted adsorbate search on Fe16: feasibility pilot

2026-11-09. Branch `feat/acat-adsorbate-search` (from `origin/main` `bfb9668`). Assignment:
`ClusterMLIP_ACAT_Worker_Assignment.md` (2026-11-09).

## Summary

- **Recommendation: adapt.** Keep the search adapter, the site/atom-role representation and the
  duplicate test. Do not use any available MLIP to rank Fe16 + CO + 2H structures. The next step
  is the reference panel in section 8 (21 draft Gaussian inputs, not submitted).
- **Pilot.** Neutral Fe16, geometry `fe16-200-49-gf` (UBPW91 M = 49 optimum), one CO and two H,
  all Fe fixed (Stage A). This is a software and model benchmark on a bare gas-phase cluster. It
  says nothing about supported Fe/K on θ-alumina or a working catalyst.
- **No Fe16 + CO/H reference label exists**, and no in-house checkpoint covers Fe with C, O and H.
  The only calculators that can run the composition are spin-blind foundation models.
- **The MLIP gate failed.** On existing UBPW91 labels for Fe2/Fe4 with CO or H, MACE-MPA-0 orders
  58–84% of same-multiplicity pairs correctly and MACE-MATPES-r2SCAN 42–66%; both have force errors
  larger than the reference forces themselves. Low-lying Fe16 + CO + 2H candidates sit about
  @@GAP@@ apart on the model surface, far below the model's error.
- **Search comparison (model surface only, 200 relaxations × 3 seeds per arm).** @@HEADLINE@@
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
| Relative energy within one composition and one spin branch | MAE ≤ 0.03 eV | low-lying candidates are about @@GAP@@ apart on the model surface |
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

@@BENCH@@

## 7. Audit of search results

@@AUDIT@@

## 8. Reference-validation matrix and DFT shortlist

**Prepared, not submitted. No compute authorization exists for it; it needs the user's approval
and a desk package.**

@@SHORTLIST@@

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

@@HANDOFF@@
