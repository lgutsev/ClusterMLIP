# G09 workflow comparison — literature recheck, code audit, and an assessment plan

Prepared 2026-09-05. This is the reviewed audit and plan; it is kept as the
record of what was wrong and why. Implementation landed piecemeal:

- **Phase 1.1** (SCF-convergence gate in `collect`, `--no-scf-continue`) and
  **Phase 5.1** (same-state label-conflict report) are on `main`.
- **Phase 1.2** (multi-step Link1 completion detection), **Phase 2.1**
  (`collect` reads `spin_jobs.csv` with per-multiplicity lineage) and the
  `validate-spins` stability hygiene of **Phase 4.2** arrived independently
  via the ladder-hardening / checkpoint-restart work already on `main`.
- Phases 2.2, 3, and the rest of 4 (opt-trajectory frames, orientation
  reconciliation, basis fallback, calc-profile names) are **not on `main`**
  and remain open items from this plan.

Scope: re-check the literature behind the earlier comparison, verify against the
code which of the six findings are actually fixed, and lay out a phased plan to
assess before launching all 382 batches. No Gaussian was run; this is a code +
documentation audit.

---

## Part A — Literature recheck

The earlier reading holds up. Nothing found argues for replacing the ClusterMLIP
approach; the precedents converge on "one consistent electronic surface + curate
the DFT, train a conventional structural potential, keep local spin as
diagnostics." New/confirmed points:

| Work | What it is | What it says that matters here |
|---|---|---|
| Romano et al., *Structure and Dynamics of the Magnetite(001)/Water Interface…*, JCTC 2025 ([arXiv:2408.11538](https://arxiv.org/abs/2408.11538), [JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.4c01507)) | Behler–Parrinello NNP for Fe₃O₄(001)/water; reactive MD (adsorption, diffusion, dissociation, recombination) | Closest data-curation precedent. VASP optB88-vdW+U (U_eff = 3.61 eV), SCF to 1e-5 eV. They **removed structures with total magnetization error > 0.01 μB**, monitored **force consistency**, and used a committee "**agree to disagree**" signal as a flag for corrupted DFT — explicitly because "multiple local minima can be reached in the DFT SCF loop." A conventional structural NNP was enough for the reactive chemistry. |
| *Machine-Learned Force Field for MD of Nonequilibrium Ammonia Synthesis on Iron Catalysts*, J. Phys. Chem. C 2025 ([doi:10.1021/acs.jpcc.4c07596](https://pubs.acs.org/doi/10.1021/acs.jpcc.4c07596)) | ML force field for N₂/H₂/NHx on Fe, reactive MD | Spin-polarised DFT references; emphasis is on **reaction coverage** (approach, adsorbed intermediates, bond breaking/forming) rather than an exhaustive magnetic-configuration search. Direct support for prioritising chemical coverage. |
| Eckhoff & Behler, *Insights into lithium manganese oxide–water interfaces using ML potentials*, J. Chem. Phys. 155, 244703 (2021) ([arXiv:2109.14068](https://arxiv.org/abs/2109.14068)) | Water dissociation / proton transfer at a magnetic oxide/water interface (Li_xMn₂O₄) | Energy potential **restricted to the magnetic ground state**; a **separate** network (HDNNS) predicts local spins / Mn oxidation states for *analysis only* — local spins are not dynamical inputs to the energy model. |
| *Development of an atomic cluster expansion potential for iron and its oxides*, npj Comput. Mater. 2025 ([article](https://www.nature.com/articles/s41524-025-01574-w), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11946894/)) | ACE for metallic Fe + oxide phases spanning magnetic configs | The "if spin matters, here is the approximate fix": Fe↑ / Fe↓ / Fe_NM as **distinct atomic species** (0.1 μB threshold), ~40k configs, each magnetic structure included **twice** with inverted spins for spin-inversion symmetry. Authors concede this misses non-collinear and finite-T magnetic excitations. |
| *Accuracy and Limitations of MLIPs for Magnetic Systems: A Case Study on Fe-Cr-C*, Phys. Rev. E 112, 055302 (2025) ([arXiv:2507.18935](https://arxiv.org/abs/2507.18935)) | DeePMD, one nonmagnetic vs one spin-polarised model | Nonmagnetic potential reproduces **dynamic/collective** properties (viscosity, T_melt); spin-polarised needed for **static/local** properties (equilibrium density), because local moments **self-average over space and time** at high T. Caveat for us: local/near-active-site energetics are exactly the "static/local" regime. |
| cDFT magnetic MLIP for Fe–Al ([Sci. Rep. 2023](https://www.nature.com/articles/s41598-023-46951-x)) | Constrained-DFT training data with non-equilibrium moments | The heavyweight option if we ever need explicit moment control. Almost certainly overkill for this project. |
| *Modeling Fe(II) Complexes Using Neural Networks*, JCTC 2024 ([doi:10.1021/acs.jctc.4c00063](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00063)) | 23k conformers, low-spin and high-spin | Precedent for the "same geometry, two roots, conflicting labels" case: they make spin state an **explicit label / separate heads** and predict the splitting (MAE ≈ 0.03 eV). This is the concrete form of "a defined root-selection policy, or a model that receives magnetic information." |
| cclib Gaussian parser ([data notes](https://cclib.github.io/data_notes.html), [issue](https://lightrun.com/answers/cclib-cclib-the-orientation-of-gaussian-grads-is-differ-from-that-of-gaussian-atomcoords)) | Reference parser behaviour | Confirms: Gaussian prints the **gradient in input orientation** while coordinates in the summary are **standard orientation**; cclib rotates forces into standard orientation to pair them. **With `Symmetry=None` everything is input orientation and they already match.** |
| *Challenges of Pretrained MLIPs in Heterogeneous Catalysis* ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12976938/)); CLAM catalysis dataset ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12728600/)) | Foundation-model behaviour on catalysis | MACE-MP-0 puts TS geometries roughly right but energies "quite wrong" for catalytic systems → keep the from-scratch model as the control; a foundation fine-tune is not a shortcut past your own consistent labels. |

**Bearing on the plan.** The literature strengthens exactly the three items you
already prioritised — consistent/converged DFT labels, complete frame
collection, reliable completion detection — and it demotes exhaustive magnetic
root-matching to a *measured* question (Part D). It also gives two ready
fallbacks if that measurement says spin matters: Fe↑/Fe↓/Fe_NM species (ACE
Fe-oxide paper) or explicit spin-state heads (Fe(II) NN paper).

---

## Part B — Code audit: status of the six findings

Verified against `HEAD` (`6bf13c1`). "Open" means the earlier finding still
stands in the current tree.

### 1. `Guess=(Read,Always)` on checkpoint continuation — **FIXED** (`6bf13c1`)

- Ladder continuation stages now emit `Geom=Checkpoint Guess=Read`
  ([spin.py:412](../src/cluster_mlip/spin.py)); the fragment path correctly keeps
  `Guess=(Fragment=N,Always)` ([spin.py:599](../src/cluster_mlip/spin.py)); the
  main-campaign force stage uses `Guess=Read Geom=Checkpoint`
  ([jobs.py:32](../src/cluster_mlip/jobs.py)).
- **Assess:** one real G09 smoke test of a 2-stage ladder input; confirm stage 1
  starts from the stage-0 checkpoint (`Initial guess read from the checkpoint
  file` in the log) and that l1.exe does not abort.

### 2. `IOP(5/13=1)` + no SCF-convergence gate on exported frames — **OPEN**

- `5/13=1` ("continue despite SCF non-convergence") is still in
  `_LEGACY_IOP` ([jobs.py:16](../src/cluster_mlip/jobs.py)) and
  `DEFAULT_SPIN_ROUTE` ([spin.py:25](../src/cluster_mlip/spin.py)).
- `parse_final_force_frame` takes the last `SCF Done` energy and the last
  `Forces` block with **no convergence check**
  ([gaussian.py:417](../src/cluster_mlip/gaussian.py)); `command_collect` does
  not gate either ([cli.py:334](../src/cluster_mlip/cli.py)).
- The README claims "A non-converged SCF root … shows up here [label_report]
  before it ever reaches training" — only true when it also produces a
  force-RMS outlier. A non-converged root with plausible-looking forces passes
  silently.
- Romano et al. is the precedent for not trusting these frames.

### 3. Batch "resume" / completion detection mistakes a partial Link1 job for a finished one — **OPEN**

- Every generated shell check is `grep -q 'Normal termination of Gaussian'`
  — a single match ([slurm.py:324,419,448,555,593](../src/cluster_mlip/slurm.py)).
- Every input is multi-step: the main campaign `.gjf` has 2 steps
  (`Opt[/Freq]` then `Force`), a spin ladder has 2–7. Gaussian prints "Normal
  termination" **once per step**, so a job whose step 1 finished and step 2
  crashed (no force label; or a ladder that died at m25) is treated as
  complete and **skipped on resume**.
- `campaign-status` (`progress.py`) is already better — it splits Link1
  sections and checks per stage, and flags `terminated_no_force`
  ([progress.py:77-111](../src/cluster_mlip/progress.py)) — but the sbatch
  resume logic does not use it.
- "Resume" also means "skip finished inputs, re-run the rest **from scratch**",
  never "restart the interrupted optimisation from its checkpoint".

### 4. Collector does not match the spin-campaign design — **OPEN**

- `command_collect` reads `jobs.csv` only
  ([cli.py:287](../src/cluster_mlip/cli.py)); `spin_jobs.csv` is never consulted.
- `parse_final_force_frame` keeps **only the last `Forces` block**
  (`force_headers[-1]`, [gaussian.py:418](../src/cluster_mlip/gaussian.py)) — so
  for a ladder only the final multiplicity's final optimisation step survives;
  intermediate multiplicities and all optimisation-trajectory frames are dropped.
- Spin-checkpoint provenance (`spin_jobs.csv` lineage columns) is not carried
  into the dataset. Per-atom Fe moments from `spin_inventory.csv` are not carried
  into `seeds.extxyz`/`all.extxyz` at all — `write_labeled_extxyz` writes graph
  charge/spin only ([dataset.py:17-34](../src/cluster_mlip/dataset.py)).
- Harvesting archived IRC geometries **with energy/force labels** is a separate
  missing capability (`extract_records` splits IRC points as seeds; `collect`
  never labels them).

### 5. Force/coordinate orientation not reconciled — **OPEN (archive path only)**

- `parse_final_force_frame` pairs the **last** orientation table (which may be
  `Standard orientation`) with the `Forces` block (always **input** orientation)
  and applies **no rotation** ([gaussian.py:441-444](../src/cluster_mlip/gaussian.py)).
- New runs use `NoSymm`, so standard ≡ input and this is harmless there
  (cclib confirms). **Archive parsing** — the spin inventory and any legacy
  `.log` with a gradient — is unprotected, and `extract_records` explicitly
  accepts `Standard orientation` tables ([gaussian.py:44](../src/cluster_mlip/gaussian.py)).

### 6. "Same spin root" comparison sorts away site identity — **OPEN (deliberately demoted to a diagnostic)**

- `spin_density_distance` sorts `(symbol, spin)` on both sides before comparing
  ([spin.py:1105-1106](../src/cluster_mlip/spin.py)); `root_signature` sorts by
  `(symbol, spin)` ([spin.py:89-96](../src/cluster_mlip/spin.py)). ↑↓↑↓ and
  ↑↑↓↓ get identical signatures and zero distance.
- Consequence inside the tool: `validate_spin_campaign` can label a genuine
  alternative AFM root as `matched_same_root`
  ([spin.py:1487-1495](../src/cluster_mlip/spin.py)).
- Per your steer, this stays a **diagnostic** (Part D), not a launch blocker.

### 7. Defaults / validation drift — **OPEN**

- `DEFAULT_SPIN_ROUTE` = `UBPW91/6-311++G*`
  ([spin.py:24](../src/cluster_mlip/spin.py)); `DEFAULT_ROUTE` =
  `UBPW91/Gen` (6-31G* light + def2-TZVP-no-f Fe,
  [jobs.py:26](../src/cluster_mlip/jobs.py)). Deliberate now and documented, but
  it means **spin-campaign and main-campaign energies/forces are on different
  bases** and cannot share one MACE target head. `scan_dataset`'s
  method-consistency guard reads the route from `link1_route`/`first_route`
  metadata ([training.py:213](../src/cluster_mlip/training.py)), which `collect`
  only writes for `jobs.csv` campaigns — so if spin outputs were ever collected,
  the guard would not fire.
- `validate-spins --strict` counts `planned_states_without_stability`
  (`stability != "stable"`) as a failure
  ([cli.py:477](../src/cluster_mlip/cli.py),
  [spin.py:1704](../src/cluster_mlip/spin.py)). Since `Stable=Opt` was
  deliberately removed and no separate stability stage is generated, **every**
  observed stage reports `not_tested` and `--strict` always returns non-zero on
  a spin campaign.

### 8. Modeling consequence — confirmed, no code change yet

- Training conditions on `--total_charge_key=charge --total_spin_key=spin`
  (multiplicity) only ([training.py:309-310](../src/cluster_mlip/training.py)).
  Different AFM roots at the same geometry/charge/multiplicity are
  indistinguishable inputs; if their labels differ materially the model must
  average them. More data cannot fix this — it needs either extra state info or
  a root-selection policy (Part D).

### Minor, found in passing

- The main-campaign Link1 stage is labelled "MLIP diffuse-basis force label"
  ([jobs.py:176](../src/cluster_mlip/jobs.py)) but uses the **same** `/Gen`
  basis as stage 1 — a stale label from an abandoned diffuse-basis plan, worth
  removing or implementing.
- Rattled seeds run `SP` then a separate `Force` Link1 on the same basis
  ([jobs.py:29-31](../src/cluster_mlip/jobs.py)); the first `SP` is redundant
  work (the `Force` step redoes the SCF).

---

## Part C — Phased plan to assess

Each item lists the change, the acceptance check, and the test to add. Ordering
is by "what blocks a trustworthy 382-batch launch."

### Phase 1 — label integrity (blocks launch)

**1.1 Gate every exported frame on a converged, clean SCF.**
- Add an SCF-convergence + normal-per-step check to `parse_final_force_frame`
  and a hard filter in `command_collect`: reject a frame unless its owning step
  shows `SCF Done` **without** a preceding unconverged marker
  (`>>>>>>>>>> Convergence criterion not met`, `Convergence failure`,
  `SCF has not converged`) in that step's text, and the step ends in `Normal
  termination`.
- Write rejected frames + reason to `failed_outputs.tsv` (already the sink for
  parse failures).
- Keep `IOP(5/13=1)` in the route **or** drop it — decide explicitly. Dropping
  it makes Gaussian stop on non-convergence (cleaner training signal, some jobs
  fail outright); keeping it plus the collector gate gets the same training set
  with more salvageable partial runs. Recommendation: **keep the IOP, add the
  gate**, and surface the non-converged count in `label_report.md`.
- *Assess:* run `collect` on a deliberately non-converged fixture log and
  confirm it lands in `failed_outputs.tsv`, not `all.extxyz`.
- *Test:* new fixture `tests/fixtures/scf_not_converged.log`; assert 0 frames,
  1 rejection.

**1.2 Correct completion detection in the launcher.**
- Replace `grep -q 'Normal termination'` with an expected-count check:
  `expected = 1 + (count of '^--Link1--' lines in the .gjf)`; a job is complete
  iff `grep -c 'Normal termination of Gaussian' == expected` **and** no
  `Error termination` / `Erroneous write` appears. Apply in `_worker_script`,
  `_batch_script`, `_submit_all_script`, `_status_script`.
- Emit a tiny `check_complete.sh <input> <log>` helper from `prepare-slurm` so
  the four scripts and `campaign-status` share one definition.
- *Assess:* a hand-built log with 2 "Normal termination" lines but an
  `Error termination` in step 2 must report incomplete and be re-submitted by
  `RUN_POLICY=resume`.
- *Test:* extend `tests/test_slurm.py` — generate scripts, run the completion
  predicate over synthetic logs (full / step-1-only / error-in-step-2).

**1.3 Make `resume` able to continue an interrupted optimisation.**
- Smaller lift: for an incomplete job that has a usable `.chk`, regenerate a
  one-shot restart input (`%oldchk`, `Geom=Checkpoint Guess=Read`, same route
  minus `Opt`'s already-done work → `Opt=Restart`) rather than restarting the
  whole multi-step input. Gate behind `RUN_POLICY=continue` so current
  behaviour is unchanged by default.
- *Assess:* kill a job mid-`Opt`, run with `RUN_POLICY=continue`, confirm the
  restart reads the checkpoint and the final step count matches expected.
- Lower priority than 1.1/1.2 — acceptable to launch with "resume = rerun from
  scratch" as long as 1.2 correctly identifies what to rerun.

### Phase 2 — spin-campaign collection (blocks using spin data, not the main launch)

**2.1 Teach `collect` about `spin_jobs.csv`.**
- When `outputs/spin_jobs.csv` exists, iterate its rows, and for each ladder
  stage parse **that stage's** section (reuse `spin.parse_spin_diagnostics`
  section splitting) rather than the whole log.
- Emit one labeled frame per (stage, converged optimisation step you choose to
  keep). Start with: final converged geometry of **each** multiplicity (not
  just the last), tagged with `intended_multiplicity`, `chain_id`,
  `stage_index`, `checkpoint_lineage`, `high_spin_inference`.
- Carry per-atom Fe Mulliken moments into the frame as an extxyz per-atom
  column (`REF_local_spin`) or `atoms.info` metadata — diagnostic only, not a
  model input yet.
- *Assess:* `collect` on a 3-rung ladder fixture yields ≥3 frames with distinct
  `intended_multiplicity` and populated lineage columns.

**2.2 Optionally harvest optimisation-trajectory frames.**
- Add `--include-opt-trajectory` (default off) to keep intermediate `Forces`
  blocks from each `Opt` as additional off-equilibrium training frames, tagged
  `config_type=opt_step`. This is free force data that the current
  last-frame-only path throws away.
- *Assess:* frame count with the flag ≈ number of `Forces` blocks in the log;
  without it, unchanged.

**2.3 Method-consistency guard that actually covers spin data.**
- Have `collect` write the resolved route into every frame's metadata for both
  campaign types, so `scan_dataset` can enforce "one method per head" even for
  spin outputs. Then either keep spin (6-311++G*) and main (`/Gen`) as
  **separate datasets / separate heads**, or re-run one campaign on the other's
  basis. Recommendation: separate datasets until a deliberate decision.

### Phase 3 — archive-parse robustness (blocks trusting archive-derived labels/inventory)

**3.1 Orientation reconciliation in `extract_records` / `parse_final_force_frame`.**
- When the geometry table paired with a `Forces` block is `Standard
  orientation` and the route lacks `NoSymm`, compute the rotation between the
  standard-orientation coords and the nearest `Input orientation` table and
  apply it to the forces (cclib's approach), or refuse to emit a labeled frame
  and log it. Prefer: pair forces with the **input-orientation** table when one
  exists within the same step.
- *Assess:* a fixture log from a symmetric molecule run **without** `NoSymm`:
  collected forces must satisfy Σ F ≈ 0 and match a cclib reference to ~1e-5.
- *Test:* add such a fixture; compare against a small hand-checked reference.

**3.2 Link1 / section-boundary coverage in the parser.**
- Add fixtures with multi-section logs (Link1 opt+force, Link1 ladder, restarted
  jobs) and assert one record per real state, correct energy↔geometry↔force
  pairing, no cross-section bleed. This is the "cclib regression-log coverage"
  gap.

### Phase 4 — defaults & strict-mode hygiene (cleanup, do before launch)

**4.1 One explicit calculation profile per campaign type**, named and recorded
in the campaign manifest (`profile: "spin_g09_6311ppgs"` /
`profile: "main_gen_bpw91"`), so "which method produced this frame" is a lookup,
not an inference.

**4.2 `validate-spins --strict`:** move `planned_states_without_stability` out of
the hard-failure sum (it is expected when `Stable=Opt` is intentionally absent);
report it, and add a separate `--require-stability` opt-in for campaigns that
did run a stability stage.

**4.3** Remove or implement the "diffuse-basis force label"; drop the redundant
`SP` before `Force` for rattled seeds (or document why it stays).

### Phase 5 — validation additions (parallel; informs go/no-go on spin)

**5.1** `evaluate` / `physical_checks`: add an explicit **inconsistent
broken-symmetry SCF** check over the training set (frames sharing
geometry+charge+multiplicity whose energies or near-active-site forces disagree
beyond a threshold) — this is the "explicit checks for inconsistent
broken-symmetry SCF roots" item still marked *not implemented* in the README.

**5.2** Committee "agree to disagree" curation pass (Romano et al.): once ≥2
seeds are trained, flag training frames where all committee members are
confidently wrong in the same place as **candidates for DFT recheck**, not as
automatic removals.

---

## Part D — the spin-sensitivity diagnostic (replaces eager root-matching)

Goal: measure whether alternative spin solutions change the chemistry you care
about, before deciding to build root-matching or a spin-aware model.

**Design.**
1. Pick ~8–12 representative geometries: bare cluster, gas molecule approaching,
   adsorbed intermediate, barrier-region, product — across 2–3 cluster sizes.
2. At **fixed charge and multiplicity**, initialise 2–3 alternative spin
   arrangements per geometry (different `Guess`, fragment guesses, or a
   swapped-site broken-symmetry start).
3. Converge each; record total energy, per-atom Fe moments, `<S²>`, and forces.
4. Metrics that matter:
   - Δ(adsorption energy) and Δ(barrier height) across roots — compare to the
     model accuracy target (e.g. 50 meV, 0.1 eV/Å).
   - Force RMS **restricted to the active-site atoms** (adsorbate + first Fe
     shell), not whole-cluster per-atom — a per-atom cluster average hides local
     disagreement (Fe-Cr-C paper's static/local caveat).
5. Decision:
   - Differences ≪ target → keep the current graph-level charge/spin model;
     local moments stay diagnostic. Root-matching adds little.
   - Differences ≳ target → adopt one of: **(a)** a documented root-selection
     policy (always take the lowest-energy converged root per
     geometry/charge/multiplicity, recorded in provenance), or **(b)** Fe↑/Fe↓/
     Fe_NM species input (ACE Fe-oxide paper), or **(c)** explicit spin-state
     heads (Fe(II) NN paper).

This is cheap (a few dozen extra single points) and it is the piece of evidence
that actually settles findings #6 and #8.

---

## Part E — recommended sequencing before the 382-batch launch

1. **Phase 1 in full** (1.1 label gate, 1.2 completion detection) — these
   determine whether the batch outputs can be trusted at all. 1.3 is optional.
2. **Phase 3.1 + 3.2** if any batch inputs are seeded from archive-parsed
   geometries/forces; skip if every seed is a fresh `extract` structure.
3. **Phase 4** (defaults, strict mode) — small, removes false signals during the
   run.
4. Launch a **pilot range** (`submit_gaussian_batches.sh --start 1 --end 8`),
   run `campaign-status`, and hand-check ~10 logs against the new gates before
   releasing the rest.
5. **Phase 2** once the first spin outputs exist; **Phase D** in parallel with
   the pilot.
6. **Phase 5** after the first committee is trained.

The earlier passing unit tests covered the generators, not the
output-parsing / completion / collection paths that a production launch depends
on. Phases 1–3 are mostly new fixtures + predicates, not architectural change.

---

## Open questions for you

- Drop `IOP(5/13=1)` or keep-and-gate? (recommendation: keep-and-gate)
- Are any 382-batch seeds archive-derived, or all from fresh `extract`? (decides
  whether Phase 3 blocks launch)
- Spin campaign (6-311++G*) and main campaign (`/Gen`): separate heads
  permanently, or converge on one basis?
- Diagnostic (Part D) cluster sizes / reactions — which specific systems are
  representative of the real target?
