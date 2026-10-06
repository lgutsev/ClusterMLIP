"""Per-atom local-moment inputs for models trained with ``--local-moment-key``.

Two rules every caller of such a model must follow:

* Sign canonicalization. A generic per-atom embedding is not symmetric under global spin
  reversal (all moments negated), and a negated vector -- moment sum -2S -- never occurs in
  Gaussian output, where alpha excess is positive. Multiplying the moments by the sign of
  their sum makes reversal an exact symmetry and leaves every training frame unchanged.
  A vector whose sum is exactly zero is ambiguous; it is refused rather than guessed.
* ASE result caching. ASE reuses the previous result when positions, numbers, cell, pbc and
  magmoms are unchanged and does not compare custom arrays, so a fixed-geometry scan over
  moments would return stale energies. Mirroring the moments into the initial magnetic
  moments makes ASE see the change.
"""
from __future__ import annotations

import numpy as np

DEFAULT_KEY = "local_moment"


def canonicalize(moments) -> np.ndarray:
    """Return the moments multiplied by the sign of their sum (values otherwise unchanged)."""
    m = np.asarray(moments, dtype=float)
    total = float(m.sum())
    if total == 0.0:
        raise ValueError("moment sum is exactly zero: the reversal sign is ambiguous")
    return m if total > 0 else -m


def set_local_moments(atoms, moments, key: str = DEFAULT_KEY):
    """Attach canonical moments to ``atoms`` for a model B calculator; returns ``atoms``."""
    m = canonicalize(moments)
    if m.shape != (len(atoms),):
        raise ValueError(f"expected {len(atoms)} moments, got shape {m.shape}")
    atoms.arrays[key] = m.copy()
    atoms.set_initial_magnetic_moments(m)  # lets ASE's cache notice a changed input
    return atoms
