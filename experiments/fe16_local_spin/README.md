# Does local Fe spin make the Fe16 force surface learnable?

**Verdict: NOT SUPPORTED.** Local atomic spin is not justified as a first-class ClusterMLIP feature on the
current Fe16 data.

Branch `test/fe16-local-spin-message-passing` (experimental, not for merge). Outputs too large for git are under
`D:/MLIP_Work_Folder/Cluster_MLIP/fe16_v1/12_local_spin_mp/` (`data/`, `runs/`).

## Hypothesis

The model sees E(R, Z, q, M). For Fe clusters, geometrically similar structures at the same M can be different
broken-symmetry states (↑↑↑ vs ↑↓↑). If so, the training labels are multivalued (same geometry + same global state
→ different forces), and the network averages incompatible force surfaces. That could explain the earlier
60-frame overfit that stalled at poor force accuracy.

## Implementation (minimal)

MACE 0.3.16 already accepts a per-atom continuous input through `--embedding_specs`
(`{"type": "continuous", "per": "atom", "key": "local_moment", "in_dim": 1, "emb_dim": 32}`). Its
`GenericJointEmbedding` output is **added to `node_feats` before the first interaction block**
(`mace/modules/models.py`, `ScaleShiftMACE.forward`). So s_i is part of h_i^(0) and enters every message. It is
not passive metadata. No MACE code was changed.

- `cluster-mlip train --local-moment-key local_moment` (cherry-picked c1c8c25) adds the spec. It refuses any
  frame that lacks the column, so missing values are never zero-filled.
- s_i is the signed Mulliken spin population (α−β) on each Fe, as printed by G09. It is continuous and unclipped.
- The wrong-site control proves the input is used: the same frames with the moments cyclically shifted by 5 atoms
  (same values, same sum, wrong sites) raise C's training force MAE from 4.9 to 31.2 meV/Å.

Scripts: `s1_inventory.py` → `s2_pairs.py` → `s3_hidden_switches.py` → `s4_build_subset.py` → `s5_run_abc.py write`
+ `runs/run_all.sh` → `s6_evaluate.py` → `s7_report.py` → `s8_examples.py`. Python: the `defects` env
(mace-torch 0.3.16, float64, CUDA).

## 1. Frame inventory (`results/inventory.json`, `results/frames.csv`)

| | |
|---|---|
| collected Fe16 force frames (02d, UBPW91/6-311++G(d)) | 1,251 |
| usable: complete per-Fe Mulliken table | **150** |
| rejected: no table associated with the frame (G09 prints populations only at the first and final opt step) | 1,101 |
| independent source geometries / jobs | 10 / 91 |
| multiplicity | M47: 5, M49: 30, M51: 55, M53: 60 |
| charge | 0 only |
| atom order, elements, column = table, geometry + forces identical to collect | 150/150 each |
| Σ s_i = M − 1 | 150/150, max error 2e-6 (holds exactly by construction for Mulliken) |
| antiparallel Fe sites per frame | 0: 25 frames, 1: 125 frames |

The 150 tables were verified against the raw logs earlier (`fe16_v1/07_crosswalk`). No value was filled.

## 2. Direct data test (`results/pairs_summary.json`, `results/pairs_diagnostic.png`)

The analysis covers all 11,175 frame pairs. Each pair is permutation- and rotation-matched (Hungarian assignment
+ Kabsch from 24 principal-axis starts). D_s and D_F are computed on the same atom correspondence.

![pairs](results/pairs_diagnostic.png)

| same q, M, RMSD bin | pairs, same spin pattern (D_s < 0.25) | median D_F | pairs, different pattern (D_s ≥ 1) |
|---|---:|---:|---:|
| < 0.01 Å | 284 | 3.1 meV/Å | **0** |
| 0.01–0.03 Å | 12 | 4.4 | **0** |
| 0.03–0.1 Å | 267 | 130.5 | **0** |
| 0.1–0.3 Å | 91 | 245.6 | **0** |
| 0.3–1.0 Å | 786 | 131.0 | 1,064 |

- **No near-identical geometry at the same (q, M) has a different local-spin pattern.** The closest such pair is
  0.79 Å apart. Below 0.1 Å, D_s never exceeds 0.15 μB.
- The large D_F at 0.03–0.3 Å comes from geometry: first (unrelaxed) frames paired with relaxed frames. The spin
  pattern is the same, and D_s is 0.03–0.14 μB, drifting smoothly with geometry.
- **Total multiplicity strongly changes forces.** For the same geometry (RMSD < 0.01 Å), D_F is 200–370 meV/Å
  between M and M−2/M−4 (333 pairs). Model B already receives M.

Concrete examples are in `results/examples.md`.

## 3. Spin diagnostics (`results/hidden_switches.json`)

- **No hidden state switches inside optimizations.** Over 1,158 adjacent opt steps (all 1,251 frames), the
  energy–force work residual |ΔE + ½(F_a+F_b)·Δx| has a median of 3e-6 eV and a maximum of 0.028 eV. ⟨S²⟩ excess
  jumps by at most 0.004.
- **Stability between the first and final frame of a stage** (57 stages, median RMSD 0.06 Å):
  - The largest per-site change has a median of 0.19 μB and a maximum of 0.68 μB.
  - 12 stages show one sign change. All of them are on weak sites (|s| < 1) crossing zero, never a flip of a
    ~4 μB moment.
- **M → M−2 hand-offs at the same geometry** (305 pairs):
  - Σs drops by exactly 2.
  - 301 keep the antiparallel pattern; 4 create a new antiparallel site. None lose one.
- **The ferrimagnetic-like states are kept.** In 125/150 frames one Fe (often the central, high-coordination
  Fe12) is antiparallel at −5 to −6 μB, including at M = 53. These frames were all kept.

## 4–6. Controlled A/B/C comparison (`results/comparison.md`, `results/training_curves.png`)

The setup is identical for all three models:
- the same 60 frames, used as both train and valid (`data/train.extxyz`), with 90 held-out labelled frames;
- the same seed(s), architecture (2 × RealAgnosticResidualNonLinear, 128x0e+128x1o+128x2e, L=3, r_max 6 Å),
  EMA, batch 8 and float64.

Only the inputs differ:
- **A:** R + Z, with no embedding block.
- **B:** + total charge and multiplicity, using ClusterMLIP's current graph-level embedding with readout.
- **C:** B + per-Fe s_i.

Model C was not tuned. The subset is seeded with the 16 frames of the closest different-pattern same-M pairs and
12 same-geometry cross-M frames, then filled stratified over (source geometry, M).

Two regimes were run:
- **ref:** the earlier failing settings: lr 5e-3, forces_weight 100, 60 epochs.
- **fixed:** the repaired defaults: lr 1e-3, forces_weight 10, 400 epochs, 2 seeds.

![curves](results/training_curves.png)

| regime | model | train E MAE (meV/atom) | train F MAE (meV/Å) | held-out E MAE | held-out F MAE |
|---|---|---:|---:|---:|---:|
| ref | A R+Z | 41.2 | 56.4 | 60.0 | 53.5 |
| ref | B +q, M | 42.8 | 54.6 | 64.3 | 54.6 |
| ref | C +q, M, s_i | 19.0 | 54.3 | 26.1 | 54.6 |
| fixed | A R+Z | 20.0 | 55.0 | 26.2 | 56.7 |
| fixed | **B +q, M** | **4.9** | **5.0** | 6.4 | **21.5** |
| fixed | C +q, M, s_i | 4.3 | 4.9 | 5.5 | 24.3 |
| fixed | C, s_i shifted 5 sites (control) | 6.0 | 31.2 | 7.1 | 59.9 |

These are frame-level evaluations of the final EMA models, averaged over 2 seeds in the fixed regime. The
zero-force RMSE is 111 meV/Å on train and 98 meV/Å on held-out.

- **The reference failure is an optimizer failure, not missing spin information.** At lr 5e-3 all three models
  stay on the zero-force plateau, C included.
- **With the repaired optimizer, B memorizes forces to 5.0 meV/Å (≈ 4% of the force scale), and C does the
  same (4.9).** Their force curves overlap within seed spread over all 400 epochs. There is no qualitative change.
- **Global M is what matters.** A cannot separate the same geometry at different M and stays at 55 meV/Å.
- **C's energy advantage is transient.** It is large early (ref: 19 vs 43 meV/atom; fixed epoch 50: 11 vs 29) and
  small at convergence (4.3 vs 4.9). Most likely the geometry-independent embedding readout lets C use the spin
  vector as a per-frame energy offset.
- **Step 6, closest different-pattern same-M pairs** (66 pairs, both frames in training, median RMSD 0.81 Å,
  D_s 1.27): the error on the force *difference* is B 7.5 vs C 7.1 meV/Å. B already assigns distinct forces to
  these structures from geometry alone. No near-identical pairs with different spin exist to test the stronger
  version of this question (section 2).
- **Outliers are not enriched in unusual spin states.** B and C force RMSE is 7–9 meV/Å for 0- and 1-antiparallel
  frames and for frames with weak sites alike. The worst frames are first (unrelaxed, high-force) frames.

## 7. Acceptance criteria

1. *Near-identical geometries at the same M show force differences correlated with different local-spin
   patterns:* **not met.** No such pairs exist in the data, and adjacent-step energy–force consistency rules out
   hidden switches.
2. *C overfits Fe16 forces dramatically better than B:* **not met** (4.9 vs 5.0 meV/Å, overlapping curves).

**Conclusion: NOT SUPPORTED.** The poor Fe16 force learning had two causes:
- the learning rate (fixed: lr 1e-3, forces_weight 10);
- limited structural coverage. Held-out forces here (21.5 meV/Å) and in the leave-one-family-out runs are limited
  by having about 10 source geometries, mostly near minima.

Local magnetic state is not the missing input.

**Does this justify first-class local atomic spin in ClusterMLIP?** No, not on this data. Total multiplicity
already resolves the observed spin dependence, and C needs DFT moments at inference anyway, so it cannot be used
as a predictive potential. The question would only reopen with labels that contain genuine same-geometry,
same-(q, M) alternative broken-symmetry solutions, for example a moved antiparallel site via alternative SCF
guesses at fixed geometry. Those would need new Gaussian jobs, which were out of scope here.

## Limitations

- Only first and final optimization frames carry Mulliken tables (150/1,251), so spin inside trajectories is
  probed only indirectly, through ⟨S²⟩ and energy–force consistency. Both are clean.
- Only Mulliken populations were used; other partitionings (NAO, Hirshfeld) were not tested.
- The test covers Fe16, q = 0, M 47–53 only.
