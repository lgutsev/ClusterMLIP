# Local-spin A/B: model A (global charge/multiplicity) vs model B (+ per-atom moments)

Model B adds the signed per-atom spin population supplied by the DFT (Mulliken, alpha minus
beta, unclipped) as a per-atom continuous MACE embedding. It is a **diagnostic**: it consumes
DFT moments at inference and does not predict them; its forces are derivatives at fixed
supplied moments, not moment-relaxed forces.

    python build_ab_dataset.py SRC_COLLECT_DATASET DEST          # population-labelled frames only
    cluster-mlip train DEST -o runs/A ...                          # model A ignores the column
    cluster-mlip train DEST -o runs/B --local-moment-key local_moment ...
    python smoke_local_moment.py runs/B/seed_1/<name>.model DEST/train.extxyz

`build_ab_dataset.py` keeps a frame only when its metadata table covers every atom in order
with matching elements, writes the values as supplied, and appends the column last in
`Properties`. Model A and B therefore see identical frames and inherited group splits.

## Using model B from ASE

    MACECalculator(model_paths=..., arrays_keys={"local_moment": "local_moment"})

- A missing input raises `KeyError` (verified both without `arrays_keys` and with the array
  absent); nothing is zero-filled.
- **ASE caches results** and compares positions, numbers, cell, pbc and magmoms, but not custom
  arrays: at a fixed geometry, changing `local_moment` returns the stale energy. Call
  `calc.reset()` before each evaluation, or mirror the moments into
  `atoms.set_initial_magnetic_moments(...)`, which ASE does compare (both verified).

## Smoke results (provisional A/B data, frame with atom 12 = −5.61, M = 53, sum 52)

| check | untrained B | B after 20 epochs |
|---|---|---|
| move the antiparallel moment to another atom (same geometry, M, sum) | 1.4–2.8 meV | 36–66 meV |
| missing input | KeyError (both cases) | KeyError (both cases) |
| rotation: ΔE / max ΔF | 0 / 4e-16 eV/Å | 0 / 2e-15 eV/Å |
| permutation (atoms + moments): ΔE / max ΔF | 0 / 8e-17 eV/Å | 0 / 1e-16 eV/Å |
| reload | identical | identical |
| global reversal (all moments negated) | +1.06 eV | **−13.98 eV** |

**Spin reversal is not a symmetry of model B.** A negated moment vector (sum −2S) never occurs
in Gaussian output (alpha excess is positive), so the model extrapolates. Averaging
E(m) and E(−m) would contaminate every in-distribution prediction with that extrapolation; the
exact alternative is sign canonicalization (multiply the moments by the sign of their sum before
the model). All Fe16 training frames already satisfy it (M ≥ 47), so it changes no label.
Augmentation with reversed copies would only approximate the symmetry and is not used.
