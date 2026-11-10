# NVIDIA ALCHEMI batch backend for the Fe16 CO/H pilot: feasibility

Date: 2026-11-09, timings 2026-11-10. Branch `feat/alchemi-batch-backend` (stacked on `feat/acat-adsorbate-search`).
Scope: an optional batch backend for the fixed-catalyst candidate screening of
[acat_cluster_feasibility.md](acat_cluster_feasibility.md). No new campaign, no multi-GPU run, no distillation.

## Recommendation

**Adopt as an optional backend, for one job only: relaxing independent candidates with a model that takes
no charge or spin input.** Keep the native workflow as the default and the fallback.

**Defer** for the project's spin-conditioned MACE models and for anything beyond Stage A screening.

Reasons, all measured here:

- Same physics. Energies and forces match native MACE to rounding (1e-14 in float64; 1.5e-5 eV in float32,
  equal to native's own repeatability), in any batch size, order or company.
- Reliable. 128 of 128 candidates reach a terminal status in every toolkit run; frozen atoms never move;
  a deliberately broken input fails alone; converged forces are confirmed by a native recompute.
- Faster. On the 128-candidate pool, three repeats on an idle GPU: native 260 s, fixed batches of 16
  41 s (6.4×), inflight batching with 64 slots 22 s (12×). The threshold was 1.5×.
- Small to maintain: about 250 lines of adapter in `experiments/`, and one contract in the package.

Limits that keep this from being more than optional:

- The toolkit's MACE wrapper does not pass charge or spin. On the project spin model it stops with
  `KeyError: 'total_spin'` (it does not silently drop the input). A 20-line subclass fixes it and
  reproduces native exactly, but that subclass is ours to maintain against a library at version 0.3.
- Only independent proposals can be batched without changing the search. That is all of the random arm but
  only the first 50 of 200 evaluations of the evolutionary arm.
- The MLIP gate of the ACAT pilot still fails. Faster relaxations on MACE-MPA-0 produce candidates for DFT
  faster; they do not produce physical conclusions.
- This Windows laptop runs the toolkit without its compiled kernels path (`torch.compile` needs Triton) and
  without cuEquivariance. A Linux check is needed before any number here is quoted for LONI.

## Stack and inputs

| Item | Value |
|---|---|
| GPU | RTX 5070 Laptop, 8 GB, driver 591.44; single GPU |
| Environment | `D:\MLIP_Work_Folder\envs\alchemi_030` (venv over the `defects` micromamba env; `constraints.txt` in the experiment folder) |
| Toolkit | nvalchemi-toolkit 0.3.0, nvalchemi-toolkit-ops 0.5.0, warp-lang 1.19.0, tensordict 0.15.5 |
| Model stack | torch 2.11.0+cu128, mace-torch 0.3.16, e3nn 0.4.4, ASE 3.29.0 |
| Deviation | the toolkit's `mace` extra pins mace-torch 0.3.15; 0.3.16 (already installed) was kept so native and batch use the same MACE |
| Not available | cuEquivariance (neither backend uses it); Triton, so everything ran with `TORCHDYNAMO_DISABLE=1` |
| Screening model | MACE-MPA-0 medium, sha256 `75428afe…`, float32, same as the ACAT pilot |
| Spin model (input test only) | `fe16_v1/12_local_spin_mp/runs/long_B_s1/long_B_s1.model`, Fe only |
| Pool | first 128 valid starts of the pilot's `random_s0` run, `pool128.extxyz`, sha256 `80cb2d23…`; Fe16 + CO + 2H, 20 atoms |
| Fixture | 12 structures: clean Fe16, 6 relaxed pilot minima, 2 unrelaxed starts, 3 distorted |
| Relaxation rule | 16 Fe frozen, 4 adsorbate atoms free, largest force on a free atom below 0.05 eV/Å, cap 200 steps, no force clamping |

Two Windows defects worked around in the adapter, not in the toolkit: `MACEWrapper.from_checkpoint` treats
`D:/…` as a URL (the model is loaded with `torch.load` and wrapped), and `NeighborListHook` must be given
its stage explicitly.

## Electronic-state inputs

Defined per structure, never per batch: total charge `q` and multiplicity `M`, carried as per-graph fields.

- MACE-MPA-0 has no such input. The pilot's Stage A energies are spin-blind by construction; that is
  recorded in the ACAT document and unchanged here.
- The project spin model declares `total_spin` and `total_charge` (categorical, per graph). Native energies
  of one Fe16 geometry at M = 49, 47, 51, 53 span 0.93 eV, so the input matters.
- Built-in `MACEWrapper` on that model: **error**, `KeyError: 'total_spin'`. Loud, not silent.
- `SpinMACEWrapper` (`electronic_inputs.py`): passes every name in the model's `embedding_specs` from the
  batch and raises if one is missing. Mixed-M batch against native: energy difference 0.0 eV, force
  difference 6.5e-16 eV/Å; reversed order the same; missing input raises
  `model needs per-graph input 'total_spin'; the batch does not carry it`.

Not tested: relaxation with the spin model, because it has no C, O or H.

## Single-point parity

`parity_single_point.py`, 12-structure fixture. Tolerance is the backend's own repeatability.

| Comparison | float64 max ΔE (eV) / max ΔF (eV/Å) | float32 max ΔE / max ΔF |
|---|---|---|
| native vs native (yardstick) | 2.8e-14 / 1.9e-14 | 1.5e-5 / 7.4e-6 |
| native vs batch of 1 | 2.8e-14 / 1.1e-14 | 1.5e-5 / 1.5e-5 |
| native vs batch of 12 | 2.8e-14 / 3.0e-14 | 1.5e-5 / 9.1e-6 |
| batch of 12 vs permuted order | 2.8e-14 / 2.8e-14 | 1.5e-5 / 1.1e-5 |
| batch of 12 vs different companions | 2.8e-14 / 1.6e-14 | 1.5e-5 / 1.1e-5 |

Atom permutation and rigid rotation plus translation agree to the same level (float32 rigid motion 3.9e-5
eV/Å). Neighbour lists equal ASE's, with no edge between structures. Energy ranking identical.

## Relaxation and reliability

`relax_bench.py` on the 128-candidate pool, `reliability_checks.py` on 8 of them.

| Check | Result |
|---|---|
| Terminal status per input | native 127 converged + 1 step cap; every toolkit run 128 converged; each input exactly once |
| Converged means converged | 0 structures above 0.05 eV/Å when forces are recomputed natively; reported vs recomputed energy ≤ 3.1e-5 eV |
| Frozen atoms | moved 0.0 Å at each of 40 checked steps; ≤ 1.1e-7 Å at the end (float32 round trip); stored force on them 0 |
| Step cap | cap of 5 steps: 8 of 8 reported `step_cap`, none as converged |
| Failure isolation | one input with two atoms on one point: `numerical_failure` at step 1; its 8 companions all converge as before; 9 outputs for 9 inputs |
| Mixed sizes | Fe16, Fe16+CO, Fe16+2H, Fe16+CO+2H (16/18/18/20 atoms) in one batch: same structures as alone |
| Chemistry audit | all runs: 128 × CO+H+H, no dissociation, no H2, no overlap; 9 desorbed-CO in each backend; O-bound CO 7 (native) vs 6 |
| Composition, roles, molecule ids | kept in every output |

**Relaxed structures against native** (127 candidates converged in both):

| Toolkit run | Same structure | Same motif | Median \|ΔE\| | Ranking pairs agree / disagree | Best energy vs native best |
|---|---|---|---|---|---|
| batch 1 | 85 | 98 | 1.0 meV | 3,496 / 0 | −1.5 meV |
| batch 16 | 84 | 100 | 1.1 meV | 3,383 / 0 | −1.3 meV |
| batch 64 | 84 | 99 | 1.0 meV | 3,413 / 0 | −1.3 meV |
| inflight, 16 slots | 85 | – | 1.0 meV | – | +4.1 meV |

About a third of the candidates end in a geometry that fails the 0.15 Å distance test against native, with
the toolkit lower in energy as often as higher (60 vs 59). The cause is the optimizer, not the batching:
the toolkit uses L-BFGS, native uses ASE BFGS, and the adsorbates sit on a soft surface. Batch 1 against
batch 16 with the same optimizer agrees on 116 of 128.

**Neither backend is repeatable run to run in float32.** The same 8 candidates relaxed twice:

| | Same structure | Max \|ΔE\| | Max position difference |
|---|---|---|---|
| native vs native | 7 of 8 | 48 meV | 0.45 Å |
| toolkit batch 8 vs itself | 7 of 8 | 13 meV | 0.32 Å |
| toolkit batch 1 vs batch 8 | 7 of 8 | 12 meV | 0.27 Å |
| toolkit batch 8 vs reversed order | 8 of 8 | 2 meV | 0.13 Å |

So batch-size and order effects are inside the noise native shows against itself. Consequence for the
search: a ledger is reproducible in its proposals, not bit-for-bit in its minima, on either backend.

One trap, handled in the adapter: in a fixed batch the toolkit keeps stepping structures that have already
converged. The adapter records each structure at the step it first converges.

## Throughput

128 candidates, float32, same model, same criterion, three repeats each on an idle GPU (0 MiB in use and
0 to 1 % utilisation logged before every configuration; 2026-11-10). Every run converged 128 of 128.
Times are the relaxation loop; the first repeat of each process adds 3 to 22 s of model load and warm-up.

| Backend | Run (s), 3 repeats | Median (s) | Candidates / hour | Speed-up vs native | Model calls | Peak VRAM (MB) |
|---|---|---|---|---|---|---|
| native (ASE BFGS) | 238 / 260 / 276 | 260 | 1,770 | 1.0 | 7,160 | 98 |
| toolkit, batch 1 | 317 / 305 / 177 | 305 | 1,510 | 0.85 | 5,127 | 98 |
| toolkit, batch 4 | 69 / 62 / 60 | 62 | 7,380 | 4.2 | 1,887 | 209 |
| toolkit, batch 16 | 41 / 40 / 45 | 41 | 11,350 | 6.4 | 634 | 665 |
| toolkit, batch 64 | 42 / 44 / 40 | 42 | 10,920 | 6.2 | 180 | 2,480 |
| toolkit, batch 128 | 36 / 51 / 48 | 48 | 9,580 | 5.4 | 110 | 4,979 |
| toolkit, inflight, 16 slots | 29 / 30 / 24 | 29 | 15,840 | 8.9 | 357 | 664 |
| toolkit, inflight, 64 slots | 21 / 22 / 24 | 22 | 21,240 | 12.0 | 124 | 2,480 |

How to read it:

- The adoption threshold of 1.5× end to end is met from batch 4 upward. Including a cold start, inflight
  with 64 slots takes 23 to 27 s against 241 to 276 s for native: about 10×.
- Batch 1 is no faster than native. The gain comes from batching, not from the toolkit as such.
- Fixed batches stop paying beyond 16: they keep stepping structures that have already finished (slot use
  48 % at batch 16, 43 % at 64, 37 % at 128). Inflight batching replaces a finished structure at once, so
  it keeps improving with slots.
- L-BFGS needs fewer steps than ASE BFGS here (about 40 against 56 per candidate). That part of the gain,
  1.4×, is the optimizer, not batching.
- Run-to-run spread is about ±10 %, except batch 1 (177 to 317 s).
- An earlier single run, taken while another job held 7.3 of 8 GB of the card, is kept in
  `report/bench128_log.txt`. Its ratios were inflated (native suffers most from sharing) and it is superseded.

Sensible setting on this card: inflight, 64 slots (2.5 GB).

## Integration with the search

- `cluster_mlip.adsorbate_search` gains `BatchEvaluator`, `native_evaluator` (the fallback, same ASE BFGS
  as before) and `AdsorbateSearch.run_batched`. `step` and `run` behave as before.
- Only proposals that do not depend on earlier results are batched: the random arm, and the initial
  population of the evolutionary arm. Children are still evaluated one at a time, in order, so selection,
  the population and the ledger order are unchanged. Batching children would be a different algorithm
  (generational instead of steady-state) and is not done.
- Test: with the native evaluator, `run_batched` reproduces the sequential ledger exactly for both arms.
- With the toolkit evaluator: 24-evaluation random search in batches of 12, 25 proposals, ledger in
  proposal order, 24 converged, composition, roles and frozen catalyst kept.
- Expected gain per arm at 200 evaluations: random arm, all 200 batched; evolutionary arm, 50 of 200.

## Gaps before wider use

| Gap | Effect | Size of fix |
|---|---|---|
| No charge/spin in the toolkit's MACE wrapper | spin models need our subclass | done here; must be re-checked on each toolkit release |
| No per-structure step cap in inflight mode | a structure that never converges holds a slot until the global cap | small: evict by step count in the adapter |
| Windows: no Triton, no cuEquivariance, checkpoint path bug | toolkit runs in its slow path here | check on Linux (QB4) before quoting numbers |
| mace-torch pin 0.3.15 vs 0.3.16 | none seen | keep watching |
| Float32 relaxations not repeatable | minima differ run to run on both backends | float64 for final ranking, or accept and deduplicate |
| Free-catalyst relaxation (Stage B) | not exercised | small |

## Conditional notes

- **NEB**: not attempted. Worth a look only after a model passes the gate; batched images are a natural fit,
  but the toolkit's NEB support was not examined.
- **Distillation**: not started, as instructed. There is no teacher that passes the gate.
- **Supported Fe / K promotion**: mixed sizes in one batch work, and frozen-atom masks are per atom, so a
  frozen support costs nothing extra in the adapter. Periodic cells were not tested. The blocker remains
  the reference data, not the backend.

## Files

- `src/cluster_mlip/adsorbate_search.py`: batch contract and native fallback. `tests/test_adsorbate_search.py`: 26 tests.
- `experiments/alchemi_fe16_co_h/`: `alchemi_backend.py`, `alchemi_evaluator.py`, `electronic_inputs.py`,
  `build_pool.py`, `parity_single_point.py`, `relax_bench.py`, `inflight_bench.py`, `compare_backends.py`,
  `inflight_compare.py`, `reliability_checks.py`, `native_repeatability.py`, `constraints.txt`, `README.md`.
- `experiments/alchemi_fe16_co_h/report/`: parity, reliability, benchmark and inflight JSON, the benchmark
  log, and per-candidate manifests for native and batch 16.
- Full outputs: `D:\MLIP_Work_Folder\Cluster_MLIP\alchemi_fe16_co_h\`.

## Next action

Repeat the parity check and one benchmark row on a Linux node. Nothing else is needed to use the backend
for random-arm screening today.
