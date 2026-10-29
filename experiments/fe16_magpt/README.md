# Fe16 MagPT diagnostic (bounded pilot)

**Question:** at fixed Fe16 geometry, does the public MagPT spin–lattice model give different energies to the
different Gaussian (UBPW91/6-311++G*) spin solutions, and does it order them the way Gaussian does?

This is a diagnostic only. It changes nothing in the Gaussian broken-symmetry campaign (packages 36/41), and
it writes nothing into any ClusterMLIP training set. A failure here is a recorded limitation, not a reason to
change the Fe16 workflow.

**Status (2026-10-28): run complete. Verdict: close the pilot; a larger pilot is NOT justified.** See Results.

## Model and environment

| item | value |
|---|---|
| Model repository | https://github.com/MagTheoryLab/MagPT @ `dd27a5d389227f1f60caf3e65e6bc5f904d7ac6d` (2026-10-20) |
| Checkpoint | `model/MagPT_399168.pth`, 24,171,352 B, sha256 `3eeb45355c741ac1a5609dfd06fecd04d6c6ce5c80edbb6a45eda9a1ab23956e` |
| What it is | retrained on the **released 399,168-frame subset** only, not the paper's full 912,492-frame model (not public) |
| Paper | Shi et al., arXiv:2610.24407 |
| Inference code | https://github.com/MagTheoryLab/deepmd-kit, branch `spin_virial_master_port` @ `4e72376de5072ad9bd54f3076003f54c6687c0d4` (DeePMD-kit v3 PyTorch backend + spin changes) |
| Architecture | DPA3 descriptor, DeepSpin energy E(R, m) |
| Training labels | ABACUS, PBE, non-collinear + SOC, **DeltaSpin constrained moments**; 48 metals; bulk/periodic cells ≤ 60 atoms/primitive cell |
| Units (README) | positions Å, spin μB, energy eV, force eV/Å, magnetic force eV/μB |
| Inputs | `DeepPot.eval(coords, cells, atom_types, spin=(nframes, N, 3))`; spin = moment vector per atom in μB |
| Isolated clusters | not mentioned anywhere in the paper or repo; training data are all periodic crystals. The PyTorch spin path accepts `cells=None`, so we run both nopbc (as a cross-check) and cubic boxes |
| Local env | `D:\MLIP_Work_Folder\envs\magpt_dp` — Python 3.12.11 (conda-forge), torch 2.8.0+cpu, ase 3.25.0, numpy 2.2.6; CPU only, 1 thread |
| Caches (not in Git) | `D:\MLIP_Work_Folder\external\magpt\MagPT` (repo + checkpoint), `D:\MLIP_Work_Folder\external\magpt\deepmd-kit-magpt` |

Install, as actually run on Windows (MSVC 2022, CPU). The checkpoint is TorchScript and references the compiled
`deepmd::border_op`, so the PyTorch custom-op library must be built (`DP_ENABLE_PYTORCH=1`). That op is only
executed for multi-rank LAMMPS halo exchange, never by Python `DeepPot.eval`. Two Windows fixes are passed as CMake
flags; the branch source is unmodified:

```bash
P=D:/MLIP_Work_Folder/envs/magpt_dp/python.exe
$P -m pip install scipy pyyaml "dargs>=0.4.7" typing_extensions h5py wcmatch packaging ml_dtypes mendeleev   array-api-compat lmdb msgpack "scikit-build-core>=0.11,<0.13" dependency_groups cmake setuptools_scm   "hatch-fancy-pypi-readme>=23.2" pytest
cd D:/MLIP_Work_Folder/external/magpt
MSYS_NO_PATHCONV=1 CMAKE_ARGS="-DCMAKE_CXX_FLAGS=-D_USE_MATH_DEFINES -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON"   DP_ENABLE_TENSORFLOW=0 DP_ENABLE_PYTORCH=1 DP_VARIANT=cpu SETUPTOOLS_SCM_PRETEND_VERSION=3.1.0.dev0+magpt.4e72376   $P -m pip install --no-deps --no-build-isolation ./deepmd-kit-magpt
```

Installed: deepmd-kit 3.1.0.dev0+magpt.4e72376. Model type map has 64 entries including Fe; rcut 6.0 Å. Build logs
are in `D:\MLIP_Work_Folder\external\magpt\deepmd_build9.log`.

**Determinism.** The model computes in float32. The first call after loading gives forces 1 ulp (4.8e-7 eV/Å) away
from all later calls, because TorchScript's profiling executor re-optimizes the graph. `DeepSpinEvaluator` therefore
does two warm-up calls, after which repeated inference is bitwise identical (tested). The runner uses 1 CPU thread.

## Run

```bash
# state selection only (no MagPT; any env with numpy + ase)
python experiments/fe16_magpt/run_magpt_diagnostic.py --select-only \
  --dataset D:/MLIP_Work_Folder/Cluster_MLIP/fe16_v1/02_fe16_dataset/all.extxyz --out experiments/fe16_magpt/report

# full diagnostic (MagPT env)
D:/MLIP_Work_Folder/envs/magpt_dp/python.exe experiments/fe16_magpt/run_magpt_diagnostic.py \
  --dataset D:/MLIP_Work_Folder/Cluster_MLIP/fe16_v1/02_fe16_dataset/all.extxyz \
  --checkpoint D:/MLIP_Work_Folder/external/magpt/MagPT/model/MagPT_399168.pth \
  --deepmd-src D:/MLIP_Work_Folder/external/magpt/deepmd-kit-magpt \
  --out experiments/fe16_magpt/report

# tests (the 3 real-model tests run only when MAGPT_CHECKPOINT is set)
MAGPT_CHECKPOINT=D:/MLIP_Work_Folder/external/magpt/MagPT/model/MagPT_399168.pth \
  D:/MLIP_Work_Folder/envs/magpt_dp/python.exe -m pytest tests/test_magpt_diagnostic.py
```

Outputs in `report/`: `selected_states.csv`, `selected_pairs.csv`, `results.csv` (one row per state × condition,
with model/provenance columns), `summary.json`/`summary.md` (per-pair metrics + the automatic gate), `provenance.json`,
and `plots/<geometry_id>.png` (Gaussian vs MagPT relative energies, with the moment-scale 0.8–1.2 range as a bar).

## Data: what exists and what was selected

Source: `fe16_v1/02_fe16_dataset/all.extxyz` (`cluster-mlip collect`, UBPW91/6-311++G(d), q = 0; energies in eV and
forces in eV/Å, already converted from Hartree and Hartree/Bohr by collect). 150 of 1251 frames carry their own
per-atom spin table.

- **Only Mulliken populations exist.** No Hirshfeld table is in any collected Fe16 log. The `Pop=Hirshfeld` jobs
  are the broken-symmetry smoke and pilot (packages 36/41), and their results have not come back. The loader
  (`load_states_from_extxyz(spin_array=...)`) can take a Hirshfeld array once those logs are collected.
- **Same-geometry states come only from verified Link1 M → M−2 hand-offs.** The final frame of the M stage and
  the first frame of the M−2 stage have identical coordinates (≤ 2e-5 Å), and each is a converged SCF with forces.
  No geometry has more than two states, and there are no same-M alternative solutions (the 2026-10-05 pair audit
  found none closer than 0.79 Å).
- Selection: group spin-labelled frames by identical coordinates (1e-3 Å), drop repeated SCF solutions, then keep
  one representative per structural family (sorted-distance RMS < 0.05 Å and state energies within 10 meV). Result:
  **11 geometries, 22 states, 11 pairs** (6 × M53/M51, 5 × M51/M49). Gaussian ΔE(M−2 − M) runs from −338 to
  +591 meV, with both signs, so ordering is testable. Two families are ferro-like (all sites +); the others have one
  antiparallel central Fe.

**Limitation for Q1, which matters for interpretation:** within each pair, *no strong site changes sign*. The M−2
state redistributes 2 μB over all sites (RMS local change 0.13–0.18 μB, max 0.19–0.42 μB). These are genuinely
different SCF states (different S, ΔE up to 0.6 eV, force differences of 0.11–0.18 eV/Å RMS), but they are **not
different sign patterns**. Distinct sign patterns at one geometry only appear with the BS pilot. Until then, this
diagnostic tests longitudinal (magnitude) sensitivity, not flip-pattern discrimination.

## Conversion: Gaussian populations → MagPT moments

`m_i = (0, 0, s_i)` μB, with `s_i = n_α − n_β` the Mulliken spin population (spin-only, g = 2, no orbital moment).
Signs and per-site magnitudes are kept. There is **no** renormalization, so Σ m_i = M − 1 exactly for every state.
The z axis is arbitrary. With SOC in MagPT, an `axis_x` condition measures the anisotropy this choice introduces.

Known mismatches, which the diagnostic does not try to correct:
- **Moment definition:** Mulliken (diffuse 6-311++G* basis) vs DeltaSpin's projection onto ABACUS atomic orbitals.
  The central Fe's ≈ −5 μB Mulliken moment comes with a Mulliken charge of ≈ +12 e. This is a basis-set artifact
  (see the 2026-10-06 local-spin review), and it lies outside the physical Fe range that MagPT was trained on.
- **Constrained vs unconstrained:** MagPT learned E(R, m) on moment-constrained states. The Gaussian states are
  unconstrained self-consistent solutions. If the moment definitions matched, an unconstrained solution would sit at
  ∂E/∂m ≈ 0, so the MagPT magnetic force at the Gaussian moments shows how far MagPT puts that pattern from its own
  stationary moments. It is reported, but it is **not** compared quantitatively with any Gaussian quantity.
- **Level:** PBE + SOC, plane waves vs UBPW91 with Gaussian basis functions. Only relative energies within one
  geometry are compared. Absolute energies are never put on a common scale.

## Conditions evaluated per state

Cubic boxes of 20, 25 and 30 Å (the Fe16 extent is about 7 Å, so the vacuum gap is ≥ 13 Å), nopbc if supported,
moment scale 0.8/0.9/1.1/1.2 (same signs), x axis, a repeat run (determinism), and two controls that are not
Gaussian states: time-reversed moments (−m; should be degenerate) and zero moments (the size of the magnetic
energy).

## Decision gate (computed in `summary.json`)

- Q1 distinguishes: |ΔE_MagPT| > max(1 meV, 10 × box/repeat noise) for every pair.
- Q2 ordering: the ΔE sign agrees with Gaussian in ≥ 2 whole geometries, and the one-sided binomial p < 0.05 over all pairs.
- Q3 stable: the ΔE sign is unchanged over moment scales 0.9–1.1 for every pair, and the box spread is below the resolution.
- Q4 larger pilot: Q1 ∧ Q2 ∧ Q3. Otherwise, record the limitation and close.

## Results (2026-10-28)

11 geometries, 22 states, 11 same-geometry pairs. All numbers below are ΔE = E(M−2) − E(M) at fixed geometry. MagPT
was evaluated in a 30 Å box; full tables are in `report/summary.md`, `report/results.csv` and
`report/bulk_fe_moment_scan.csv`, and plots are in `report/plots/` (`pooled_dE.png` plus one per geometry).

| check | result |
|---|---|
| Box size 20 / 25 / 30 Å and nopbc | ΔE spread ≤ 0.06 meV; nopbc vs 30 Å ≤ 0.04 meV. Isolated clusters work without periodic-image effects |
| Different energies for different states | yes, all 11 pairs: \|ΔE_MagPT\| = 0.99–1.85 eV (resolution 1 meV) |
| ΔE_MagPT sign | **negative in 11/11 pairs**: MagPT always prefers the lower moment |
| ΔE_Gaussian | −338 to +591 meV; 6 negative, 5 positive |
| Sign agreement | 5/11 (one-sided binomial p = 0.73). The agreements are exactly the M53→M51 pairs where Gaussian also goes down. MagPT never predicts an increase |
| Rank correlation, pooled | Spearman 0.56, Pearson 0.36. This reflects the two ladder steps forming separate clusters, not tracking within either group |
| Magnitude | MagPT ΔE is 3–60× the Gaussian ΔE; Gaussian \|ΔE\| ≤ 0.59 eV |
| Moment scale 0.9–1.1 (signs fixed) | sign kept in 10/11 pairs, but ΔE changes by 0.5–4 eV. At scale 0.8 every pair flips sign (+0.14 to +0.39 eV) |
| Moment axis z → x | ΔE changes by up to 0.8 eV (SOC anisotropy should be meV) |
| Time reversal m → −m | E changes by up to **0.61 eV** per state. Bulk bcc Fe ferro is invariant to 1e-5 eV, so this is an out-of-domain failure, not an input bug |
| E(m) − E(m = 0) | +4 to +6 eV for ferrimagnetic states: MagPT says these moments are strongly *destabilizing* |
| Atomic forces vs Gaussian | at Gaussian minima (Gaussian RMS ≈ 0.003 eV/Å), MagPT RMS is 0.3–0.9 eV/Å and cosine ≈ 0 (−0.28 to 0.32) |
| Spin-induced force change ΔF | Gaussian 0.11–0.18 eV/Å RMS, MagPT 0.11–0.80; cosine −0.22 to 0.43 |
| Magnetic forces (−∂E/∂m) | longitudinal RMS 0.43–1.56 eV/μB, far from stationary (MagPT's training RMSE is 31 meV/μB). Not compared with Gaussian: no equivalent quantity exists |

**Mechanism.** In bulk bcc Fe, MagPT's E(|m|) has its minimum at about 2.5 μB, rises steeply above 3 μB (+0.48
eV/atom at 4 μB) and turns over above 4 μB (an extrapolation artifact). The Fe16 Mulliken moments are 3.6–4.2 μB,
plus −5 μB on the central atom, so every state sits on or beyond that wall. Taking 2 μB out of the cluster therefore
lowers MagPT's energy by about 1–1.8 eV whatever the geometry or electronic state. This also explains the extreme
sensitivity to moment scale, the large magnetic forces and the time-reversal and axis violations. Out-of-domain causes
are compounded: a moment definition (diffuse-basis Mulliken, with the central-atom artifact) that differs from
DeltaSpin's projection, a 16-atom open cluster vs bulk training data, and unconstrained vs constrained states.

### Answers

1. **Does MagPT distinguish the supplied Fe16 spin patterns?** Formally yes: every pair gets a different energy, far
   above numerical noise. But the difference is a near-constant ~−1.5 eV moment-magnitude term, not a state-specific
   signal. Note too that the supplied pairs do not differ in sign pattern (no strong site flips; the pilot data
   contain no flip-pattern alternatives), so flip-pattern discrimination is untested.
2. **Does it reproduce the qualitative ordering?** No. It predicts "lower M is lower" for all 11 pairs. Gaussian has
   5 pairs that go the other way (up to +0.59 eV), and agreement is at chance level (5/11, p = 0.73).
3. **Stable to box size and moment perturbations?** Box size: yes, fully (≤ 0.06 meV; nopbc agrees). Moment
   magnitude: no. ±10% changes ΔE by eV, −20% flips every sign, and axis choice and global moment reversal each change
   energies by up to 0.6–0.8 eV.
4. **Larger pilot justified?** **No.** The automatic gate gives Q1 ✓, Q2 ✗, Q3 ✗, so Q4 ✗. Record the limitation
   and close the pilot without changing the Fe16 Gaussian workflow.

What could change this conclusion (not done here; it would be a new pilot): the BS-pilot Hirshfeld moments
(packages 36/41), which are smaller and closer to an orbital-projected definition, at same-geometry flip patterns.
Even then, the time-reversal and axis violations on Fe16 mean the released 399k checkpoint is outside its domain for
open Fe clusters.
