# Does local Fe spin make the Fe16 force surface learnable?

**Verdict, in two parts:**
- **NOT SUPPORTED as the cause of the observed Fe16 force-learning failure.** That failure was the optimizer
  (lr 5e-3, forces_weight 100). The input the model needed was the global multiplicity M, which ClusterMLIP
  already supplies.
- **INCONCLUSIVE for the general hypothesis.** The hypothesis is that same-geometry, same-M broken-symmetry
  states make force labels multivalued. The current labels cannot test it: they contain no such alternative
  states, and their Mulliken "antiparallel" moments are almost certainly a basis-set partitioning artifact
  (section 3).

**Is local atomic spin justified as a first-class ClusterMLIP feature? No, not on this evidence.**

Branch `test/fe16-local-spin-message-passing` (experimental, not for merge). Outputs too large for git are under
`D:/MLIP_Work_Folder/Cluster_MLIP/fe16_v1/12_local_spin_mp/` (`data/`, `runs/`). This version includes the
corrections from an independent review (2026-10-06): the held-out leak, the Mulliken artifact, convergence, and
several figures.

## Hypothesis

The model sees E(R, Z, q, M). For Fe clusters, geometrically similar structures at the same M can be different
broken-symmetry states (↑↑↑ vs ↑↓↑). If so, the labels are multivalued (same geometry + same global state →
different forces), and the network averages incompatible force surfaces. That could explain the earlier
60-frame overfit that stalled at poor force accuracy.

## Implementation (minimal)

MACE 0.3.16 accepts a per-atom continuous input through `--embedding_specs`
(`{"type": "continuous", "per": "atom", "key": "local_moment", "in_dim": 1, "emb_dim": 32}`). Its
`GenericJointEmbedding` output is **added to `node_feats` before the first interaction block**
(`mace/modules/models.py`, `ScaleShiftMACE.forward`). So s_i is part of h_i^(0) and enters every message. No MACE
code was changed. The training logs confirm that model C loaded the column (`local_moment: 60`, train and valid).

- s_i is the signed Mulliken spin population (α−β) per Fe, as printed by G09. It is continuous and unclipped.
- The runs call `mace_run_train` directly with explicit argv (`s5_run_abc.py`, `runs/*/argv.json`). Between
  models these differ only in `--embedding_specs` / `--use_embedding_readout`.
- The equivalent ClusterMLIP path, `cluster-mlip train --local-moment-key` (cherry-picked c1c8c25, which refuses
  frames missing the column), is on this branch but **was not exercised by these runs**.

Scripts, in order:
1. `s1_inventory.py`
2. `s2_pairs.py`
3. `s3_hidden_switches.py`
4. `s4_build_subset.py`
5. `s5_run_abc.py write`, then `runs/run_all.sh` and `runs/run_long_{0,1}.sh`
6. `s6_evaluate.py`
7. `s7_report.py`
8. `s8_examples.py`
9. `s9_unseen.py`

Python: the `defects` env (mace-torch 0.3.16, float64, CUDA).

## 1. Frame inventory (`results/inventory.json`, `results/frames.csv`, `results/heldout_unseen.json`)

| | |
|---|---|
| collected Fe16 force frames (02d, UBPW91/6-311++G(d)) | 1,251 |
| usable: complete per-Fe Mulliken table | **150** |
| rejected: no table associated with the frame (G09 prints populations only at the first and final opt step) | 1,101 |
| unique geometries / unique (geometry, M) states among the 150, 0.01 Å | **34 / 56** |
| source geometry families / jobs | 10 / 91 |
| multiplicity | M47: 5, M49: 30, M51: 55, M53: 60 |
| charge | 0 only |
| atom order, elements, column = table, geometry + forces identical to collect | 150/150 each |
| Σ s_i = M − 1 | 150/150, max error 2e-6 (exact by construction for Mulliken; not a physical check) |

The 150 tables were verified against the raw logs earlier (`fe16_v1/07_crosswalk`). No value was filled. The
effective sample is far smaller than 150: 74 same-state pairs are exact duplicates (RMSD < 0.001 Å).

## 2. Direct data test (`results/pairs_summary.json`, `results/pairs_diagnostic.png`, `results/examples.md`)

All 11,175 pairs were permutation- and rotation-matched (Hungarian assignment + Kabsch from 24 principal-axis
starts). D_s and D_F are computed on the same correspondence. The reviewer found no pair whose identity-permutation
Kabsch beats the match, and no hidden matches: no pair with near-identical sorted distances but RMSD > 0.1 Å, and
mirror images included.

![pairs](results/pairs_diagnostic.png)

| same q, M, RMSD bin | pairs, same pattern (D_s < 0.25) | median D_F | pairs, different pattern (D_s ≥ 1) |
|---|---:|---:|---:|
| < 0.01 Å | 284 (many exact duplicates) | 3.1 meV/Å | **0** |
| 0.01–0.03 Å | 12 | 4.4 | **0** |
| 0.03–0.1 Å | 267 | 130.5 | **0** |
| 0.1–0.3 Å | 91 | 245.6 | **0** |
| 0.3–1.0 Å | 786 | 131.0 | 1,064 |

- **No near-identical geometry at the same (q, M) has a different local-spin pattern.** The closest such pair is
  0.79 Å apart (0.825 Å with mirror images). Below 0.1 Å, D_s ≤ 0.144 μB; at 0.03–0.3 Å, D_s ≤ 0.194 μB.
- The large D_F at 0.03–0.3 Å comes from geometry: first (unrelaxed) frames paired with relaxed ones, with the
  same pattern.
- **Total multiplicity strongly changes forces at fixed geometry** (RMSD < 0.01 Å, 333 pairs). Model B already
  receives M.

  | ΔM | pairs | D_F range (meV/Å) | median |
  |---|---:|---:|---:|
  | 2 | 305 | 108–178 | 131 |
  | 4 | 26 | 204–282 | 255 |
  | 6 | 2 | 337–373 | 355 |

## 3. Spin diagnostics (`results/hidden_switches.json`)

**The "antiparallel" Fe is a Mulliken artifact, not a broken-symmetry moment.**
- 105 frames have one Fe at s_i < −1 (−5 to −6 μB). In **all 105** it is the most central, highest-coordination
  Fe.
- That same atom carries a Mulliken **charge** of +11.8 to +12.5 e; the other Fe average −0.8 e.
- The before-annihilation ⟨S²⟩ excess is only 0.39–0.72 in every frame. A genuine antiparallel site of several μB
  would add several units of contamination.
- This is the known failure of Mulliken partitioning for an interior atom with diffuse functions
  (6-311++G(d)). s_i is therefore a deterministic function of geometry here, not an independent electronic state
  variable.
- So the earlier description of these frames as "ferrimagnetic-like states" is withdrawn. Nothing was discarded,
  but the pattern should not be read physically.

**No hidden state switches inside optimizations.** This covers 1,158 adjacent steps across all 1,251 frames.
- The before-annihilation ⟨S²⟩ excess jumps by at most 0.063 (0.004 after annihilation).
- The energy–force work residual |ΔE + ½(F_a+F_b)·Δx| has median 3e-6 eV, p99 1.9e-3 eV and max 0.028 eV.
- The residual follows step size (Spearman 0.85), and its median is 0.5% of the second-order work term
  |ΔF·Δx|, as expected for trapezoid error on a smooth surface.
- The four largest residuals (0.009–0.028 eV) come from one stage (`e54a588d…-ladder-m53-m49-s00`, frames
  40–46).
  - These are large (0.16–0.21 Å), back-and-forth optimizer steps with alternating residual sign.
  - Through that stretch ⟨S²⟩ moves by at most 0.009.
  - Integration error is the likely explanation. The stage is not proven clean.
- **Caveat:** ⟨S²⟩ cannot detect a switch between two solutions with equal contamination (e.g. a moved minority
  site). Only the energy residual constrains that.

**Stability of Mulliken values.**
- First vs final frame of a stage (57 stages, median RMSD 0.06 Å): the largest per-site change has a median of
  0.19 μB and a maximum of 0.68 μB.
- 12 stages show one sign change, always on a weak site (|s| < 1) crossing zero.
- M → M−2 hand-offs at the same geometry (305 pairs): Σs drops by exactly 2, and 4 hand-offs gain a negative
  site.

## 4–6. Controlled A/B/C comparison (`results/comparison.md`, `results/training_curves.png`)

The setup is identical for all models:
- the same 60 frames as train and valid (`data/train.extxyz`), plus 90 held-out labelled frames;
- the same seeds, architecture (2 × RealAgnosticResidualNonLinear, 128x0e+128x1o+128x2e, L=3, r_max 6 Å), EMA,
  batch 8, float64, and ReduceLROnPlateau on valid (= train).

Only the inputs differ:
- **A:** R + Z, with no embedding block.
- **B:** + total charge and multiplicity, using ClusterMLIP's current graph-level embedding with readout.
- **C:** B + per-Fe s_i.

Model C was not tuned. The subset is seeded with the 16 frames of the closest different-pattern same-M pairs and
12 same-geometry cross-M frames, then filled stratified over (family, M).

Regimes:
- **ref:** the earlier failing settings: lr 5e-3, forces_weight 100, 60 epochs.
- **fixed:** lr 1e-3, forces_weight 10, 400 epochs, 2 seeds.
- **long:** B and C only, as fixed but 800 epochs.

The long runs exist because the reviewer noted that fixed-regime forces were still falling steeply at epoch 400.
Training is deterministic: the first 400 epochs of the long runs reproduce the fixed runs exactly.

![curves](results/training_curves.png)

### Memorization (60 training frames; frame-level, final EMA model, mean over seeds)

| regime | model | E MAE (meV/atom) | F MAE (meV/Å) | F RMSE (meV/Å) |
|---|---|---:|---:|---:|
| ref (60 ep) | A R+Z | 41.2 | 56.4 | 106.8 |
| ref (60 ep) | B +q, M | 42.8 | 54.6 | 90.5 |
| ref (60 ep) | C +q, M, s_i | 19.0 | 54.3 | 91.4 |
| fixed (400 ep) | A R+Z | 20.0 | 55.0 | 89.7 |
| fixed (400 ep) | B +q, M | 4.9 | 5.0 | 7.6 |
| fixed (400 ep) | C +q, M, s_i | 4.3 | 4.9 | 7.6 |
| **long (800 ep)** | **B +q, M** | 2.8 | **2.0** | 3.3 |
| **long (800 ep)** | **C +q, M, s_i** | 2.4 | **2.1** | 3.4 |
| long | C, s_i shifted 5 sites (control) | 4.6 | 30.5 | 55.3 |

Zero-force RMSE on these frames: 111 meV/Å. Per seed at epoch 795 (MACE log, EMA): B 1.88 / 2.10, C 2.17 / 1.98
meV/Å. The B–C difference is smaller than the seed spread.

### Held-out frames, split by what they actually test (`s9_unseen.py`)

`s4` split by frame, so held-out frames were classed by their nearest training frame (0.01 Å):

| held-out class | frames |
|---|---:|
| same-state twin (memorization, not a test) | 60 |
| same geometry in training only at another M | 9 |
| **unseen** (no training frame within 0.01 Å at any M) | **21** |

| class (frames) | model | fixed F MAE | long F MAE | long F RMSE | zero-force F RMSE | long E MAE |
|---|---|---:|---:|---:|---:|---:|
| **unseen (21)** | A | 60.9 | – | 114.0 (fixed) | 119.4 | 22.5 (fixed) |
| **unseen (21)** | B | 51.1 | **51.2** | 78.8 | 119.4 | 7.4 |
| **unseen (21)** | C | 58.5 | **58.8** | 89.1 | 119.4 | 6.6 |
| other-M twin (9) | B | 62.4 | 63.3 | 89.5 | 148.8 | 5.9 |
| other-M twin (9) | C | 69.9 | 70.5 | 96.1 | 148.8 | 2.2 |
| same-state twin (60) | B | 5.0 | 2.7 | 5.2 | 77.5 | 3.7 |
| same-state twin (60) | C | 5.5 | 3.0 | 6.3 | 77.5 | 3.6 |

F in meV/Å, E in meV/atom, mean over 2 seeds. Longer training improves only the memorized twins. On unseen frames
it changes nothing, and C stays slightly worse than B.

### Findings

- **The reference failure is an optimizer failure.** At lr 5e-3 all three models, C included, stay on the
  zero-force plateau.
- **B memorizes the forces. C is not better.** After 800 epochs, both reach a training force MAE of about
  2 meV/Å (B 2.0, C 2.1; seed spread 1.9–2.2). The curves overlap throughout. The 400-epoch figures were still
  falling, and the extra 400 epochs did not open a gap.
- **Global M is what matters.** A cannot separate the same geometry at different M and stays at about 55 meV/Å.
- **C's energy advantage is transient.** It is large early (ref: 19 vs 43 meV/atom; fixed epoch 50: 10.9 vs
  27.3) and shrinks at convergence. It is consistent with the geometry-independent embedding readout using s_i,
  which here encodes the central atom's position, as a per-frame energy offset.
- **On truly unseen frames neither model generalizes forces well, and C is not better.** Errors are a large
  fraction of the zero-force reference. The earlier "held-out 21.5 vs 24.3 meV/Å" mostly measured memorized
  twins and is superseded.
- **Step 6, closest different-pattern same-M pairs.** Both frames are in training, the median RMSD is 0.81 Å, and
  they come down to a few unique geometries because of duplicates. On the force difference between the frames, errors on the
  force difference are B 6.8/8.6 vs C 7.0/7.4 meV/Å at 400 epochs, and B 3.7/3.4 vs C 3.8/3.8 at 800 epochs (per
  seed): identical within seed spread (`results/comparison.md`).
- **Wrong-site control.** Each frame's moments were shifted by 5 atoms. This raises C's training force MAE from
  about 2 to about 31 meV/Å (800 epochs), which shows C reads s_i. Shifting moves the −5.6 value onto a surface atom, an
  input never seen in training, so it does not show that s_i carries extra information.
- **Outliers are not enriched in spin patterns.** B and C errors are similar for frames with 0 or 1 negative
  site, and the worst frames are the unrelaxed first frames.

## 7. Acceptance criteria

1. *Near-identical geometries at the same M show force differences correlated with different local-spin
   patterns:* **not met. It also cannot be met with these labels.** No such pairs exist, the high-spin-start
   checkpoint protocol never produces alternative solutions at fixed geometry and M, and the one recurring
   "pattern" is a Mulliken artifact that is a function of geometry.
2. *C overfits Fe16 forces dramatically better than B:* **not met.** With no conflicting labels, criterion 2 has
   little discriminating power: any flexible model with M can memorize the 42 unique (geometry, M) training
   states, and s_i duplicates geometry.

**Conclusion.**
- **The earlier failure: NOT SUPPORTED.** The local-spin hypothesis does not explain it. The causes were the
  learning rate (fixed: lr 1e-3, forces_weight 10) and, for unseen structures, limited structural coverage
  (34 unique geometries, mostly near minima).
- **The general hypothesis: INCONCLUSIVE (untestable on current labels).**
- **First-class local spin in ClusterMLIP: not justified.** M already resolves the spin dependence present in the
  data. C needs DFT moments at inference, and the only available moments (Mulliken, diffuse basis) are
  unreliable for interior atoms.

## What a real test would need (not done: requires new Gaussian jobs)

1. Alternative broken-symmetry SCF solutions at fixed geometry and M. For example, `guess=fragment` or
   `guess=alter` with the minority site moved, then `stable=opt`, keeping only solutions with a real ⟨S²⟩
   increase and distinct energies.
2. A partition-robust local moment: Hirshfeld/CM5 or NBO/NAO spin densities, or Mulliken in a basis without
   diffuse functions, on a few frames, to confirm the central-atom artifact.
3. Force labels for those solutions. Then repeat the B vs C memorization and pair test on genuinely multivalued
   data.

## Limitations

- Only first and final optimization frames carry Mulliken tables (150/1,251), so spin inside trajectories is
  probed indirectly (⟨S²⟩ and energy–force consistency).
- Fe16, q = 0, M 47–53 only.
- The memorization subset is small: 60 frames, 42 unique (geometry, M) states.
