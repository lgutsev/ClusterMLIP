"""Bounded MagPT diagnostic: can a spin-lattice foundation model tell Fe16 spin states apart?

The Gaussian side supplies unconstrained UBPW91 broken-symmetry solutions at fixed geometries (energies in
eV, forces in eV/A, per-atom Mulliken or Hirshfeld spin populations). The MagPT side (DeePMD-kit spin model,
public checkpoint ``MagPT_399168.pth``) is evaluated on the same coordinates with explicit collinear moment
vectors ``m_i = (0, 0, m_i)`` in Bohr magnetons. Nothing here relaxes atoms or spins, and nothing here writes
into a training dataset: the outputs are diagnostic tables only.

Everything except :class:`DeepSpinEvaluator` is plain numpy so it can be unit-tested without DeePMD-kit.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .gaussian import BOHR_TO_ANG, HARTREE_TO_EV

# A spin population (n_alpha - n_beta electrons on a site) is a spin-only moment in Bohr magnetons with
# g = 2: m = g * S_z = 2 * (n_alpha - n_beta) / 2.  No orbital moment is available from the Gaussian side.
SPIN_POPULATION_TO_MU_B = 1.0

# Geometry and state matching tolerances.  Link1 checkpoint hand-offs reuse coordinates exactly
# (max displacement <= 2e-5 A in the Fe16 pair audit), so 1e-3 A only absorbs print rounding.
SAME_GEOMETRY_TOL_ANG = 1e-3
# Two SCF solutions at one geometry are the same state only if both energy and every local moment agree.
DUPLICATE_STATE_ENERGY_TOL_EV = 1e-4
DUPLICATE_STATE_SPIN_TOL_MU_B = 0.05

AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}


def hartree_to_ev(energy_hartree: float) -> float:
    return energy_hartree * HARTREE_TO_EV


def hartree_per_bohr_to_ev_per_ang(force_au: np.ndarray) -> np.ndarray:
    return np.asarray(force_au, dtype=float) * (HARTREE_TO_EV / BOHR_TO_ANG)


@dataclass
class SpinState:
    """One converged Gaussian SCF solution at one geometry."""

    frame_id: str
    symbols: list[str]
    positions: np.ndarray  # (N, 3) Angstrom
    multiplicity: int
    charge: int
    energy_ev: float
    local_spins: np.ndarray  # (N,) spin population, n_alpha - n_beta
    spin_method: str
    forces_ev_ang: np.ndarray | None = None  # (N, 3)
    source: str = ""
    geometry_id: str = ""
    state_id: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def total_moment(self) -> float:
        return float(self.local_spins.sum())

    @property
    def abs_moment(self) -> float:
        return float(np.abs(self.local_spins).sum())


def moment_vectors(
    local_spins: Sequence[float], scale: float = 1.0, axis: str = "z", to_mu_b: float = SPIN_POPULATION_TO_MU_B
) -> np.ndarray:
    """Collinear moment vectors m_i = scale * s_i * axis, in Bohr magnetons.

    Signs and per-site magnitudes are kept as given: there is deliberately no renormalization to a common
    moment, so a weak site stays weak and an antiparallel site stays antiparallel.
    """
    spins = np.asarray(local_spins, dtype=float)
    if spins.ndim != 1:
        raise ValueError(f"local spins must be one value per atom, got shape {spins.shape}")
    if not np.all(np.isfinite(spins)):
        raise ValueError("local spins contain non-finite values")
    if axis not in AXES:
        raise ValueError(f"axis must be one of {sorted(AXES)}, got {axis!r}")
    return (scale * to_mu_b * spins)[:, None] * AXES[axis][None, :]


def sorted_distance_fingerprint(positions: np.ndarray) -> np.ndarray:
    """Permutation- and rotation-invariant fingerprint: sorted interatomic distances."""
    pos = np.asarray(positions, dtype=float)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    return np.sort(dist[np.triu_indices(len(pos), k=1)])


def same_geometry(a: np.ndarray, b: np.ndarray, tol: float = SAME_GEOMETRY_TOL_ANG) -> bool:
    """Identical geometry in the same atom order and frame (what a Link1 hand-off produces)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and float(np.abs(a - b).max()) <= tol


def geometry_id(positions: np.ndarray) -> str:
    """Stable short id from the permutation-invariant fingerprint, rounded to 1e-3 A."""
    fp = np.round(sorted_distance_fingerprint(positions), 3)
    return "g" + hashlib.sha256(fp.tobytes()).hexdigest()[:10]


def group_states_by_geometry(states: Iterable[SpinState], tol: float = SAME_GEOMETRY_TOL_ANG) -> list[list[SpinState]]:
    groups: list[list[SpinState]] = []
    for state in states:
        for group in groups:
            ref = group[0]
            if ref.symbols == state.symbols and same_geometry(ref.positions, state.positions, tol):
                group.append(state)
                break
        else:
            groups.append([state])
    return groups


def is_duplicate_state(a: SpinState, b: SpinState) -> bool:
    """Same multiplicity, same energy and same local moments: a repeated SCF solution, not a new state."""
    return (
        a.multiplicity == b.multiplicity
        and abs(a.energy_ev - b.energy_ev) <= DUPLICATE_STATE_ENERGY_TOL_EV
        and float(np.abs(a.local_spins - b.local_spins).max()) <= DUPLICATE_STATE_SPIN_TOL_MU_B
    )


def distinct_states(group: Sequence[SpinState]) -> list[SpinState]:
    kept: list[SpinState] = []
    for state in group:
        if not any(is_duplicate_state(state, other) for other in kept):
            kept.append(state)
    return kept


def drop_twin_geometries(
    groups: Sequence[list[SpinState]], fingerprint_tol: float = SAME_GEOMETRY_TOL_ANG, energy_tol: float = 1e-3
) -> list[list[SpinState]]:
    """Drop groups that repeat an earlier group's structure (relabelled/permuted reruns of the same job).

    Two groups are twins when their geometries match up to permutation and rotation (sorted-distance
    fingerprint) and they carry the same multiplicities with matching energies.  Looser tolerances
    (e.g. 0.05 A / 0.01 eV) collapse near-identical reruns of one structural family to one representative.
    """
    kept: list[list[SpinState]] = []
    for group in groups:
        fp = sorted_distance_fingerprint(group[0].positions)
        sig = sorted((s.multiplicity, s.energy_ev) for s in group)
        twin = False
        for other in kept:
            if len(other[0].positions) != len(group[0].positions):
                continue
            ofp = sorted_distance_fingerprint(other[0].positions)
            if float(np.sqrt(np.mean((fp - ofp) ** 2))) > fingerprint_tol:
                continue
            osig = sorted((s.multiplicity, s.energy_ev) for s in other)
            if len(osig) == len(sig) and all(
                m == om and abs(e - oe) <= energy_tol for (m, e), (om, oe) in zip(sig, osig)
            ):
                twin = True
                break
        if not twin:
            kept.append(group)
    return kept


def assign_ids(groups: Sequence[list[SpinState]]) -> None:
    for group in groups:
        gid = geometry_id(group[0].positions)
        for state in sorted(group, key=lambda s: (-s.multiplicity, s.energy_ev)):
            state.geometry_id = gid
            n_same_m = sum(1 for s in group if s.multiplicity == state.multiplicity and s.state_id)
            state.state_id = f"{gid}:M{state.multiplicity}" + (f"#{n_same_m + 1}" if n_same_m else "")


def spin_pattern_distance(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    strong = (np.abs(a) >= 0.5) & (np.abs(b) >= 0.5)
    return {
        "rms_local_moment_diff": float(np.sqrt(np.mean((a - b) ** 2))),
        "max_local_moment_diff": float(np.abs(a - b).max()),
        "n_sign_changes_strong_sites": int(np.sum(strong & (np.sign(a) != np.sign(b)))),
    }


def relative_energies(energies: Sequence[float], reference_index: int) -> np.ndarray:
    """E_i - E_ref for a fixed reference state (the same state is the reference on both sides)."""
    e = np.asarray(energies, dtype=float)
    return e - e[reference_index]


def pairwise_ordering(reference: Sequence[float], predicted: Sequence[float], tie_ev: float = 0.0) -> dict:
    """Pairwise order agreement and rank correlations between two energy lists over the same states."""
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    concordant = discordant = ties = 0
    for i, j in combinations(range(len(ref)), 2):
        dr = ref[j] - ref[i]
        dp = pred[j] - pred[i]
        if abs(dr) <= tie_ev or abs(dp) <= tie_ev:
            ties += 1
        elif np.sign(dr) == np.sign(dp):
            concordant += 1
        else:
            discordant += 1
    n_pairs = concordant + discordant + ties
    kendall = (concordant - discordant) / n_pairs if n_pairs else math.nan
    return {
        "n_states": int(len(ref)),
        "n_pairs": n_pairs,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "tied_pairs": ties,
        "kendall_tau_a": kendall,
        "spearman_rho": spearman(ref, pred),
    }


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ranks for exact ties
    for value in np.unique(x):
        idx = np.where(x == value)[0]
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3:
        return math.nan
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def force_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    diff = pred - ref
    rn = np.linalg.norm(ref)
    pn = np.linalg.norm(pred)
    return {
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff**2).mean())),
        "ref_rms": float(np.sqrt((ref**2).mean())),
        "pred_rms": float(np.sqrt((pred**2).mean())),
        "cosine": float((ref * pred).sum() / (rn * pn)) if rn > 0 and pn > 0 else math.nan,
    }


def place_in_cubic_box(positions: np.ndarray, box_length: float) -> tuple[np.ndarray, np.ndarray]:
    """Centre a cluster in a cubic periodic box; returns (positions, cell)."""
    pos = np.asarray(positions, dtype=float)
    centred = pos - pos.mean(axis=0) + box_length / 2.0
    return centred, np.eye(3) * box_length


def minimum_vacuum_gap(positions: np.ndarray, box_length: float) -> float:
    """Smallest distance between the cluster and its nearest periodic image along a box axis."""
    pos = np.asarray(positions, dtype=float)
    extent = pos.max(axis=0) - pos.min(axis=0)
    return float(box_length - extent.max())


def load_states_from_extxyz(path: str | Path, spin_array: str = "mulliken_spin") -> list[SpinState]:
    """Frames of a ``cluster-mlip collect`` extxyz that carry their own per-atom spin table.

    ``REF_energy`` is in eV and ``REF_forces`` in eV/A there (collect converts from Hartree and
    Hartree/Bohr); frames without the spin array are skipped rather than given invented moments.
    """
    from ase.io import read  # lazy: only the diagnostic needs ASE

    states: list[SpinState] = []
    for atoms in read(str(path), ":"):
        if spin_array not in atoms.arrays:
            continue
        info = atoms.info
        forces = atoms.arrays.get("REF_forces")
        states.append(
            SpinState(
                frame_id=str(info.get("record_id", "")),
                symbols=atoms.get_chemical_symbols(),
                positions=np.array(atoms.positions, dtype=float),
                multiplicity=int(info["multiplicity"]),
                charge=int(info.get("charge", 0)),
                energy_ev=float(info["REF_energy"]),
                local_spins=np.array(atoms.arrays[spin_array], dtype=float),
                spin_method=spin_array.split("_")[0],
                forces_ev_ang=None if forces is None else np.array(forces, dtype=float),
                source=str(info.get("source", "")),
                extra={"parent_record_id": str(info.get("parent_record_id", ""))},
            )
        )
    return states


# evaluator(positions[N,3], cell[3,3] | None, moments[N,3], symbols) -> (energy_ev, forces[N,3], mag_forces[N,3])
Evaluator = Callable[[np.ndarray, "np.ndarray | None", np.ndarray, Sequence[str]], tuple[float, np.ndarray, np.ndarray]]


class DeepSpinEvaluator:
    """Thin wrapper over DeePMD-kit's ``DeepPot`` for a spin model (MagPT branch of DeePMD-kit)."""

    def __init__(self, model_path: str | Path):
        from deepmd.infer import DeepPot  # imported lazily: only the inference environment has it

        self.model_path = str(model_path)
        self.model = DeepPot(self.model_path)
        self.type_map = list(self.model.get_type_map())
        self.rcut = float(self.model.get_rcut())
        # TorchScript's profiling executor re-optimizes the graph after the first call, which moves float32
        # forces by ~1 ulp; warm up so every recorded evaluation comes from the same (bitwise-stable) graph.
        if "Fe" in self.type_map:
            dimer = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2]])
            for _ in range(2):
                self(dimer, np.eye(3) * 20.0, moment_vectors([2.0, 2.0]), ["Fe", "Fe"])

    def __call__(self, positions, cell, moments, symbols):
        missing = sorted(set(symbols) - set(self.type_map))
        if missing:
            raise ValueError(f"elements {missing} are not covered by the model type map")
        atype = np.array([self.type_map.index(s) for s in symbols])
        e, f, _v, fm, _mask = self.model.eval(
            coords=np.asarray(positions, dtype=float)[None],
            cells=None if cell is None else np.asarray(cell, dtype=float)[None],
            spin=np.asarray(moments, dtype=float)[None],
            atom_types=atype,
        )
        n = len(symbols)
        return float(np.asarray(e).ravel()[0]), np.asarray(f).reshape(n, 3), np.asarray(fm).reshape(n, 3)


def evaluate_state(
    evaluator: Evaluator,
    state: SpinState,
    box_length: float | None,
    scale: float = 1.0,
    axis: str = "z",
    spins: np.ndarray | None = None,
) -> dict:
    """One fixed-geometry, fixed-moment MagPT evaluation. ``box_length=None`` means no periodic cell."""
    moments = moment_vectors(state.local_spins if spins is None else spins, scale=scale, axis=axis)
    if box_length is None:
        positions, cell = np.asarray(state.positions, dtype=float), None
    else:
        positions, cell = place_in_cubic_box(state.positions, box_length)
    energy, forces, mag_forces = evaluator(positions, cell, moments, state.symbols)
    axis_vec = AXES[axis]
    longitudinal = mag_forces @ axis_vec
    transverse = mag_forces - longitudinal[:, None] * axis_vec[None, :]
    return {
        "energy_ev": energy,
        "forces": forces,
        "mag_forces": mag_forces,
        "mag_force_longitudinal_rms": float(np.sqrt(np.mean(longitudinal**2))),
        "mag_force_longitudinal_max": float(np.abs(longitudinal).max()),
        "mag_force_transverse_rms": float(np.sqrt(np.mean((transverse**2).sum(-1)))),
        "net_force_norm": float(np.linalg.norm(forces.sum(axis=0))),
    }
