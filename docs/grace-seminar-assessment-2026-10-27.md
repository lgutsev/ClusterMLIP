# GRACE seminar vs. ClusterMLIP: assessment (2026-10-27)

Source: R. Drautz, *Machine Learned Potentials: Foundational and Fine-Tuned, at Scale with GRACE*,
Materials Design UGM webinar, 2026-10-27. The input was a set of notes on the visible slides plus
automatic captions of the Q&A, not a transcript. Gaps in the notes stay gaps here. Every external
claim that drives a choice below was checked against a primary source (tables at the end); the
rest is labelled *seminar only*.

Status words used throughout:

| Word | Meaning |
|---|---|
| **code** | code present, unit-tested |
| **smoke** | run end to end on real data or checkpoints at least once |
| **trained** | a model was trained with it and its numbers are on record |
| **validated** | passed a held-out scientific test on the target chemistry |
| **proposed** | written down only |

Controls that this assessment does **not** change:

- The compact, from-scratch MACE with graph-level charge/multiplicity embeddings stays the control.
  Foundation fine-tuning stays deferred and unverified (README "Fine-tuning a foundation model").
- Total-charge/multiplicity conditioning is not local-spin dynamics, a spin-crossing probability,
  spin–orbit coupling, an oxidation state, charge transfer or nonadiabatic dynamics. Native
  charge/spin inputs in another model do not supply any of these either.
- Mulliken local spins are diagnostics (`docs/fe16-next-steps.md`; the Fe16 central-atom moment is
  partly a basis artifact). Reproducing every archived broken-symmetry root is not a prerequisite.
- **Gaussian–VASP: delta only** (package 62, 2026-10-19; `docs/gaussian-vasp-calibration.md` on
  `feat/gaussian-vasp-calibration`, commit 210018a). VASP never trains absolute cluster energies or
  forces, including through a shared-trunk second head. It enters only through the supported-cluster
  interaction correction. The seminar's mention of multi-head training and its criticism of
  EAM-baseline delta learning (Q7, Q18) are general arguments. They are not matched evidence and do
  not reopen this decision. Our delta is a correction to a DFT-quality gas-phase term, fitted to
  same-geometry VASP labels, not a correction to a weak empirical baseline.
- UBPW91 is the group's reference functional, not a stopgap. A model pretrained on a different
  functional is a change of method, not an upgrade.

## 1. Seminar idea → repository

| Seminar idea | What ClusterMLIP already has (status, evidence) | Genuine gap | Useful action | Verification needed |
|---|---|---|---|---|
| **Foundation fine-tuning** beats training from scratch with limited data (B–W table: specific GRACE 1.96 meV/atom vs foundation 16.3; Q2: hundreds to thousands of structures) | `cluster-mlip train --finetune` for `polar-1-{s,m,l}`, `mace-omol`, `small/medium/large`: **code**. POLAR-1 Fe16 fine-tune: **smoke, failed** (diverges; pretrained POLAR breaks once an Fe has ≥ 11 Fe neighbours; `fe16_v1/08_polar`). MACE-MP-0 fine-tune on periodic A3: **trained**, but the saved model had no charge/spin embedding, because mace-torch 0.3.16 drops `--embedding_specs` on fine-tune; Fe and N forces were worse than zero force (package 32). The guard is on unmerged `fix/train-embedding-model-check` (36215c2). Scratch control: **trained** on Fe16 (fits training data; unseen-frame force MAE ≈ 51 meV/Å vs 119 zero-force RMSE; `experiments/fe16_local_spin` on that branch). | No transfer experiment has run under matched budgets and splits with conditioning verified in the saved model. No GRACE path exists. Released GRACE checkpoints have no total-charge or multiplicity input (verified, §4). | One bounded transfer pilot ([spec](../experiments/foundation_transfer_pilot/SPEC_2026-10-27.md)). The arms are the scratch control, a MACE-MP-0 fine-tune that keeps the embeddings, and a single-state GRACE arm only where state conditioning isn't needed. | Saved-model embedding check. Matched-budget held-out comparison on independent DFT frames. Falsification rule in the spec. |
| **Native charge/spin candidates** | `polar-1-*` and `mace-omol` are wired in as native charge/spin (**code**). Both were trained at ωB97M-V/def2-TZVPD on OMol25 (verified). | They change method (hybrid vs UBPW91) and failed or diverged on Fe16. | Keep them as recorded negatives on Fe16. Re-test only on small Fe_n(N,H) (n ≤ 6), where the coordination failure may not occur. | A zero-shot check on the pilot's test set before any fine-tune. |
| **GRACE as backend** | None, by design (2026-10-27 decision to stay on MACE for spin training). Local install: tensorpotential 0.6.2 in `D:\MLIP_Work_Folder\envs\grace_tp062v` (also 0.6.0). | GRACE has no graph-level charge/multiplicity input. Per-atom magnetic-moment input appears in the calculator, but the module it imports (`tensorpotential.experimental.mag`) is **not shipped** in 0.6.2 (checked in the installed package). Weights are ASL (academic). | GRACE is an *assessed candidate* for single-state subsets and periodic support-only work. It does not replace the backend. | Zero-shot on pilot test frames, matched to the MACE-MP-0 zero-shot in package 27. |
| **Distillation** into a light model from synthetic labels (B–W light GRACE 5.84 meV/atom; Q3, Q9) | Nothing. `select-next-batch` and `examples/supported_fe16_tem` generate candidates but never label them with a model. | No teacher-label provenance field, so a synthetic label could be mistaken for DFT. No teacher exists that is validated on our chemistry. | Specify only (pilot spec §5). Gate it on a validated teacher. Carry teacher provenance as a separate label level that `train` refuses to mix with DFT unless asked. | Student vs teacher and teacher vs DFT reported separately on independent DFT frames. |
| **CALM extrapolation grade** (latent Mahalanobis, percent-level overhead) vs our **committee** | `select-next-batch`: worst-atom committee force disagreement, **code** (and now **smoke**, below). New `calibrate-committee` checks disagreement against held-out DFT error: **code + smoke** on Fe16 (this branch). | CALM is GRACE-only (verified). MACE has no built-in equivalent; published latent-GMM work on MACE/NequIP exists (arXiv:2402.03753, 2211.09866). We had no test of whether our committee ranks real errors. | Keep the committee as baseline and calibrate it (done once; §3). A MACE latent-distance grade is a later experiment, not a dependency. | Calibrate on independent DFT frames, deliberately held-out environments, and other-multiplicity frames (§3). |
| **Task-specific evaluation** (Q13, Q16: "very small errors on a narrow dataset can mislead"; sample the configurations that decide the property) | `evaluate`: errors by q/M, PES region, coordination, compactness, provenance (**code**, **smoke**). Spin-ordering physical check (**code**). Vertical spin-gap errors on verified ladder pairs: **code** in the uncommitted IRC/local-spin checkout only. Per-element forces were missing. | No per-element vs zero-force view (the package-32 lesson). No adsorption/reaction-energy metric. No barrier metric (no TS↔IRC pairing). No split by unique geometry on `main` (it exists in the local-spin checkout). | **Added here:** per-element force table with zero-force baseline (`by_element.csv`, report section). Adsorption-energy and barrier metrics are specified in the pilot, not built. | Run on any new model. Adsorption metrics need same-level labels for X, Fe_n and Fe_nX. |
| **Label quality limits accuracy** (Q4, Q8, Q11: noise floor; converged DFT; PBE without U) | `label-levels` classifies frames by functional, basis, grid and extras (**code**, **smoke** on Fe16 legacy). `collect` drops unconverged-SCF frames (IOP(5/13=1)), with outlier report and route checks (**code**, **smoke**). Package 62: 14/42 UBPW91 links unconverged despite Normal termination; VASP forces 3–4× Gaussian. Basis check (package 65, proposed). | No measured label-noise floor for UBPW91 forces (grid or basis sensitivity on identical geometries). Acceptance thresholds therefore cannot yet be tied to measured reference uncertainty. | Use package 65 and package 62 residuals as the provisional noise estimate. State thresholds as proposed (pilot spec §6). | A repeat-label check on ~20 frames (grid/basis) before claiming any error below the noise floor. |
| **Fe–O–H reactive examples** (H uptake, Fe oxidation by water, hematite/wüstite reduction; specific ACE) | Fe_nO_m UBPW91 warehouses and the W2 spin campaign (data). Fe–O basin hopping deferred (ROADMAP). | These are bulk/surface Fe–O–H at 1000 K, with periodic PBE-type references and specific (not foundation) ACE. Our target is small gas-phase or supported Fe_n with N2/H2 in declared (q, M). The primary paper for the Fe–O–H reduction work was not found; the Fe–O magnetic ACE paper is (Bienvenu et al., npj Comput. Mater. 11, 81, 2025). | Motivation only: a specific, task-trained potential can run reactive Fe chemistry when trained on targeted data. It is not evidence for our N2/H2 chemistry. | None for ClusterMLIP now. |

## 2. Ranked actions

1. **Calibrate the committee before trusting selection** (done once, on Fe16; repeat on every new
   committee). Cost: minutes of CPU. Reason: §3 shows the overall rank correlation is inflated by
   memorized twins, and the committee is over-confident on frames whose geometry it saw at another
   multiplicity.
2. **Report per-element forces against the zero-force baseline** in every evaluation (done here).
3. **Merge the saved-model embedding check** (`fix/train-embedding-model-check`) before any
   fine-tune. Without it the pilot's fine-tune arm can silently train spin-blind (package 32). User
   decision; not merged here.
4. **Build the pilot dataset** (§4 of the spec): Fe_n, Fe_nN2 and Fe_nH_k (n = 3–6) at the UBPW91
   label level, split by unique geometry with ladder relatives kept together, plus an independent
   DFT test set. Collection runs on LONI (qbd). Needs authorization.
5. **Run the transfer pilot** (scratch vs MACE-MP-0 fine-tune with embeddings, matched budgets).
   GPU-hours estimated in the spec. Needs authorization.
6. **Measure the UBPW91 label-noise floor** on ~20 repeated geometries (grid/basis), so acceptance
   thresholds rest on measured reference uncertainty. Fold into package 65 if it runs.
7. **Later, gated:** distillation (needs a validated teacher), a MACE latent-distance grade beside
   the committee, and a GRACE single-state arm if the MACE fine-tune shows transfer at all.

Not recommended: switching backend to GRACE; using the B–W or 170× numbers as predictions; and
reopening the delta-only decision on the strength of the seminar.

## 3. Inexpensive check run today: is the committee's disagreement calibrated?

The data were the 90 Fe16 held-out frames and the two-seed model-B committee (global q/M, 800 epochs)
from the local-spin experiment. These are existing checkpoints; no training was run.

Command (laptop CPU, 46 s):

```bash
cluster-mlip calibrate-committee fe16_v1/12_local_spin_mp/data/heldout.extxyz \
  --models runs/long_B_s1/long_B_s1.model runs/long_B_s2/long_B_s2.model \
  -o fe16_v1/13_seminar_checks/calib_longB_heldout
```

Outputs are in `D:\MLIP_Work_Folder\Cluster_MLIP\fe16_v1\13_seminar_checks\`. As a consistency check,
the single-model `evaluate` on the same frames gives a force MAE of 19.3 meV/Å. That agrees with the
experiment's own per-class numbers (≈ 20 meV/Å weighted), so charge and multiplicity reach the model.

The experiment's README classifies each held-out frame by its nearest training frame. Results
broken down by that class:

| Held-out class | Frames | Spearman ρ (disagreement vs committee error) | Median disagreement (eV/Å) | Median error (eV/Å) | Median error / disagreement |
|---|---:|---:|---:|---:|---:|
| all | 90 | 0.955 | — | — | — |
| same-state twin (memorized) | 60 | 0.94 | 0.003 | 0.008 | 1.9 |
| same geometry, trained only at another M | 9 | 0.26 | 0.068 | 0.319 | 4.3 |
| unseen geometry | 21 | 0.77 | 0.083 | 0.248 | 2.2 |

- Disagreement separates novel frames from memorized ones well (AUC 0.955). Most of the overall ρ
  comes from that separation, not from ranking errors within the frames that matter.
- **Top-9 error recall is 0.33** (chance 0.10). Of the 9 worst-error frames, 5 are other-M twins.
  **None of the 5 is in the top-9 by disagreement**; all 9 of those are unseen geometries. A
  committee asked to pick the worst frames would miss the ones where the geometry is familiar but
  the multiplicity is new.
- The median-based "confident failure" count is 0. The median over all 90 frames is dominated by
  memorized twins, so that summary hides this case. Read the per-class table, not only the headline.
- Limits: two seeds trained on the same 60 frames, one composition, one charge, three
  multiplicities. This is a diagnostic of the selector, **not** evidence about accuracy on the
  target chemistry. A CALM-like latent grade was not tested. Whether it would flag other-M frames
  depends on whether the multiplicity embedding reaches the latent features; that is a hypothesis.

**Consequence for selection:** build every calibration set with deliberately held-out multiplicities
for seen geometries, and keep selecting some frames by state coverage, not only by disagreement.

## 4. External claims checked

| Claim (as used above) | Status | Primary source |
|---|---|---|
| GRACE architecture | verified | Bochkarev, Lysogorskiy, Drautz, *Phys. Rev. X* 14, 021036 (2024) |
| GRACE foundation models, fine-tuning and distillation | verified (abstract and docs) | Lysogorskiy, Bochkarev, Drautz, *npj Comput. Mater.* 12, 114 (2026), doi:10.1038/s41524-026-01979-1 (arXiv:2508.17936). gracemaker docs, "foundation" and tutorials |
| CALM = calibrated Mahalanobis grade on a projection of GRACE's invariant basis, per element/cluster, percent-level cost, correlates with force errors on OMat24/SMAX | verified (abstract and docs). No correlation numbers retrieved | Lysogorskiy, Bochkarev, Drautz, arXiv:2609.40060 (2026). gracemaker `uq` docs. `tensorpotential/uq/calm_numpy.py` in the installed 0.6.2 |
| Released checkpoints: 1L/2L/3L OMAT, -ft-AM, SMAX, SMAX-OMAT | verified | `tensorpotential/calculator/foundation_models.py` (0.6.2). HuggingFace AMS-ICAMS-RUB/grace-foundation-models (ASL) |
| No total charge or multiplicity input. Magnetic-moment module not shipped | verified locally; "not found" in docs | Installed package source (above) |
| SMAX dataset, "Mendeleev materials" | verified (abstract) | Bochkarev et al., arXiv:2602.23489 (2026) |
| Non-collinear magnetic ACE for Fe, positions + moments, spin-DFT | verified (abstract) | Rinaldi et al., *npj Comput. Mater.* 10, 12 (2024), doi:10.1038/s41524-024-01196-8 |
| Fe–O magnetic ACE | verified (abstract) | Bienvenu et al., *npj Comput. Mater.* 11, 81 (2025), doi:10.1038/s41524-025-01574-w |
| Fe–O–H reduction by H2 (Adv. Sci. e76709, 2026) | **not found** | conference abstracts only |
| "GRACE warp", ≈ 170× time at ≈ 1.9× error | **not found**; seminar only | — |
| MACE-OMOL-0: OMol25, ωB97M-V/def2-TZVPD, charge/spin embeddings, ASL | verified | github.com/ACEsuit/mace-foundations |
| MACE-POLAR-1: total charge/spin control, OMol25 | verified (abstract); weight license not confirmed | arXiv:2602.19411 |
| MACE multi-head replay fine-tuning | verified | mace-docs "multihead_finetuning" |
| OMat24 / Alexandria journal versions as cited on the slide | not found; preprints arXiv:2410.12771, 2512.09169 | — |

Full PDFs were not read, so no page-level quotes are available. The verification covered abstracts,
official documentation and the installed source.
