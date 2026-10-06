# Fe16 broken-symmetry pilot kit

Generated 2026-10-06 by `cluster-mlip prepare-broken-symmetry` (branch `feat/fe16-broken-symmetry-pilot`).
**Nothing has been submitted.** Submitting is your decision.

## Priority (from ROADMAP_Backup_Plan.md §8)
This is a small, separately scoped study. It ranks **after** the legacy warehouse inventory (package 26) and the
rattle/coverage campaign, and it should not take queue time from them.

Before submitting the pilot batch, check package 26's `overlap.csv`. If it shows same-(q, M) near-duplicate
geometries with different energies in the legacy warehouses, those are free candidates for alternative solutions
and may change which geometries this pilot should use.

## The question
At a fixed Fe16 geometry and fixed q = 0, M = 49, does UBPW91/6-311++G* have more than one stable SCF solution,
with different site-resolved spins and different nuclear forces? If it does, the roadmap gate "reliable same-(Q,M)
states have distinct local spin and forces" opens.

## Jobs
- **Geometries:** 3 converged M=49 minima from 3 families: `78e5b1fe…frame000042`, `4134201881…frame000041`,
  `a09438f8…frame000032` (from `fe16_v1/02d_fe16_tol05`).
- **Patterns:**
  - **P0:** default guess.
  - **P1:** central (highest-coordination) Fe started β.
  - **P2, P3:** two different surface Fe started β.
- **Fragments:** one per Fe atom, neutral, odd multiplicity only (m = 3 or 5; a neutral Fe fragment has 26
  electrons). Each fragment is validated, and the signed total is 48 = M − 1.
- **Stage 0:** `SP Stable=Opt Pop=Hirshfeld`.
- **Stage 1** (`--Link1--`): `Force Guess=Read Geom=Checkpoint Pop=Hirshfeld`.
- **Level:** the dataset level, except that `IOP(5/13=1)` is dropped, so an unconverged SCF fails instead of
  continuing.
- **Resources:** `%nprocshared=12`, `%mem=24GB`; QB3 `workq`, 48 h, 4 jobs per node, account `loni_perovsk27`.
- **Expected run time:** about 1–4 h per job (not yet measured; the smoke job measures it).

## Run (generate on the laptop, submit from qbc / QB3)
1. Copy this folder to `/work/$USER/fe16_bs_pilot_kit` on QB3.
2. **Smoke first:**

   ```bash
   cd smoke && bash submit_gaussian_batches.sh
   ```

   This runs P1 on geometry 1, a single job. When it finishes:

   ```bash
   bash gaussian_batch_status.sh
   ```

   Then check the `.log` for:
   - two `Normal termination` lines;
   - the last `Stable=Opt` message ("The wavefunction is stable under the perturbations considered");
   - a Hirshfeld table in the force stage;
   - the wall time.
3. **Only if the smoke job looks right:**

   ```bash
   cd ../pilot && bash submit_gaussian_batches.sh
   ```

   This submits 3 batches of 4 jobs. The pilot reruns the smoke input; delete that one row from
   `pilot/spin_jobs.csv` if you want to save about 2 h.
4. **Pack results:**

   ```bash
   bash pack_results.sh
   ```

   Then copy `fe16_bs_pilot_results.tar.gz` back to the laptop.

## Analysis (laptop)

```bash
PYTHONPATH=src python experiments/fe16_bs_pilot/analyze.py <kit>/pilot
```

This writes:
- `bs_states.csv`: per job, the energy, the stability verdict, ⟨S²⟩, Mulliken and Hirshfeld spins, and CM5
  charges;
- `bs_pairs.csv`: ΔE and D_F between distinct states;
- `bs_summary.md`, with the verdict.

**Rules:**
- Two stable, converged solutions are the same state when |ΔE| ≤ 1 meV and every per-site Hirshfeld spin agrees
  within 0.2. Otherwise they are distinct.
- ⟨S²⟩ is supporting evidence only. CM5 values are charges, not spins.
- **Success:** at least 2 geometries with at least 2 distinct stable states.
- **If everything collapses to P0:** this pilot found no alternative stable state for these geometries, this M and
  these guesses. That is the whole conclusion; it does not say local spin is generally irrelevant.

## Known unverified points (the smoke job checks them)
- G09's `Guess=(Fragment=16)` with per-atom fragments, combined with `Stable=Opt`, has not been run in this
  campaign before.
- The Hirshfeld parser was tested on a synthetic table only. Check it against the real one.
- `parse_force_frames` on a `Geom=Checkpoint` Force stage that follows an SP stage.
