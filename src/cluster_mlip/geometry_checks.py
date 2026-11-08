"""Numerical geometry evidence for isolated clusters and supported/adsorbed systems.

Every structural statement this module makes (contact, coordination, site,
orientation, displacement, bond change) comes from computed distances, never
from reading a coordinate listing. Interpretations need numbers behind them;
language models in particular are poor at extracting geometry from text-encoded
coordinates (Alampara, Miret & Jablonka, Nat. Comput. Sci. 2026,
doi:10.1038/s43588-026-01045-2).

Scope is geometry only. Bond lengths do not establish oxidation states, local
spins, broken-symmetry character or barriers; every report carries that caveat
and those claims must come from electronic-structure outputs and their
convergence checks.

Conventions (all reported back in the output so the numbers can be re-derived):
- Bonded: d <= bond_tolerance * (r_i + r_j), Cordero covalent radii, the same
  table and default tolerance (1.2) as stratify.bonding_graph.
- Close contact: d < close_contact_scale * (r_i + r_j), default 0.6 (Fe-Fe 1.82 A
  with the high-spin Fe radius; 0.75 flagged ordinary 2.15-2.27 A cluster Fe-Fe).
  Per-element-pair minima are always reported so the evidence is visible.
- Periodic directions use the minimum-image convention via ASE. A group
  (cluster/adsorbate) is unwrapped by bond-graph traversal, so it may straddle
  a cell boundary; groups wider than half a periodic cell length are refused.
- Atom correspondence for isolated clusters: element-preserving Hungarian
  assignment alternated with proper (det=+1) Kabsch rotation, from several
  starts; the lowest-RMSD mapping wins. Reflections are never used. This is a
  heuristic search: it has no error bound and does not prove the global
  optimum. Its settings (starts, seed, iterations) are returned with every
  result so a run can be reproduced. Exact-identity claims belong to the
  exhaustive spin_audit.alignments.
- "Bond" always means the cutoff-defined connectivity graph above, an
  operational geometric criterion, not a chemical-bonding assignment.
- Adsorption-site labels (top/bridge/hollow-n) are given only when the
  contact pattern is close to ideal; anything else is reported as
  irregular-n with the reason, never forced into an ideal label.
- Supported systems are compared in the substrate frame: no rotation, the
  rigid lateral drift of the substrate is removed, displacements use the
  minimum image.

Needs numpy, scipy and ase (`pip install -e .[geometry]`); imported only when
this module is used.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Sequence

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from .stratify import COVALENT_RADII_ANGSTROM, DEFAULT_BONDING_TOLERANCE

DEFAULT_CLOSE_CONTACT_SCALE = 0.6
ELECTRONIC_CAVEAT = (
    "Geometry only: distances and coordination do not establish oxidation states, "
    "local spins, broken-symmetry solutions or reaction barriers."
)


# ------------------------------------------------------------------ basics
def _radii(atoms: Atoms) -> np.ndarray:
    symbols = atoms.get_chemical_symbols()
    missing = sorted({s for s in symbols if s not in COVALENT_RADII_ANGSTROM})
    if missing:
        raise ValueError(f"no tabulated covalent radius for {missing}; refusing to guess")
    return np.array([COVALENT_RADII_ANGSTROM[s] for s in symbols])


def _periodic(atoms: Atoms) -> bool:
    return bool(np.any(atoms.pbc))


def distance_matrix(atoms: Atoms) -> np.ndarray:
    """All pair distances; minimum image along periodic directions."""
    return atoms.get_all_distances(mic=_periodic(atoms))


def parse_indices(spec: str | None, natoms: int) -> list[int]:
    """'0:192,200,205:208' -> indices (Python half-open slices, 0-based)."""
    if not spec:
        return []
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            lo, hi = part.split(":")
            out.extend(range(int(lo or 0), int(hi) if hi else natoms))
        elif part:
            out.append(int(part))
    if any(i < 0 or i >= natoms for i in out) or len(set(out)) != len(out):
        raise ValueError(f"index spec {spec!r} is out of range or repeats atoms (natoms={natoms})")
    return out


@dataclass
class Partition:
    """Which atoms are cluster, adsorbate and substrate. Unlisted atoms are cluster."""

    cluster: list[int]
    adsorbate: list[int] = field(default_factory=list)
    substrate: list[int] = field(default_factory=list)

    @classmethod
    def from_specs(cls, natoms: int, substrate: str | None = None,
                   adsorbate: str | None = None) -> "Partition":
        sub, ads = parse_indices(substrate, natoms), parse_indices(adsorbate, natoms)
        if set(sub) & set(ads):
            raise ValueError("substrate and adsorbate index sets overlap")
        taken = set(sub) | set(ads)
        return cls([i for i in range(natoms) if i not in taken], ads, sub)

    def label(self, i: int) -> str:
        if i in self.substrate:
            return "substrate"
        if i in self.adsorbate:
            return "adsorbate"
        return "cluster"

    def as_dict(self) -> dict[str, Any]:
        def ranges(ix: list[int]) -> str:
            ix, parts, start = sorted(ix), [], None
            for k, i in enumerate(ix):
                start = i if start is None else start
                if k + 1 == len(ix) or ix[k + 1] != i + 1:
                    parts.append(f"{start}:{i + 1}" if i > start else str(i))
                    start = None
            return ",".join(parts)
        return {"cluster": ranges(self.cluster), "adsorbate": ranges(self.adsorbate),
                "substrate": ranges(self.substrate)}


# ------------------------------------------------------- contacts, bonding
def bond_matrix(atoms: Atoms, tolerance: float = DEFAULT_BONDING_TOLERANCE,
                distances: np.ndarray | None = None) -> np.ndarray:
    d = distance_matrix(atoms) if distances is None else distances
    r = _radii(atoms)
    bonded = d <= tolerance * (r[:, None] + r[None, :])
    np.fill_diagonal(bonded, False)
    return bonded


def contact_report(atoms: Atoms, partition: Partition | None = None,
                   close_contact_scale: float = DEFAULT_CLOSE_CONTACT_SCALE) -> dict[str, Any]:
    """Minimum distances (overall and per group pair) and flagged close contacts."""
    partition = partition or Partition(list(range(len(atoms))))
    d = distance_matrix(atoms)
    r = _radii(atoms)
    symbols = atoms.get_chemical_symbols()
    iu = np.triu_indices(len(atoms), 1)
    report: dict[str, Any] = {}
    if len(atoms) > 1:
        k = int(np.argmin(d[iu]))
        i, j = int(iu[0][k]), int(iu[1][k])
        report["min_distance"] = {"d": round(float(d[i, j]), 4), "i": i, "j": j,
                                  "pair": f"{symbols[i]}-{symbols[j]}"}
    groups = {"cluster": partition.cluster, "adsorbate": partition.adsorbate,
              "substrate": partition.substrate}
    per_pair: dict[str, Any] = {}
    names = [n for n, ix in groups.items() if ix]
    for a, b in [(n, n) for n in names] + list(combinations(names, 2)):
        ia, ib = np.array(groups[a]), np.array(groups[b])
        block = d[np.ix_(ia, ib)].copy()
        if a == b:
            if len(ia) < 2:
                continue
            block[np.tril_indices(len(ia))] = np.inf
        ka, kb = np.unravel_index(int(np.argmin(block)), block.shape)
        i, j = int(ia[ka]), int(ib[kb])
        per_pair[f"{a}-{b}"] = {"d": round(float(d[i, j]), 4), "i": i, "j": j,
                                "pair": f"{symbols[i]}-{symbols[j]}"}
    report["min_distance_by_group"] = per_pair
    by_elements: dict[str, float] = {}
    for p, q in zip(*iu):
        key = "-".join(sorted((symbols[p], symbols[q])))
        by_elements[key] = min(by_elements.get(key, np.inf), float(d[p, q]))
    report["min_distance_by_element_pair"] = {k: round(v, 4) for k, v in sorted(by_elements.items())}
    threshold = close_contact_scale * (r[:, None] + r[None, :])
    flagged = [(int(i), int(j)) for i, j in zip(*iu) if d[i, j] < threshold[i, j]]
    report["close_contacts"] = [
        {"i": i, "j": j, "pair": f"{symbols[i]}-{symbols[j]}", "d": round(float(d[i, j]), 4),
         "threshold": round(float(threshold[i, j]), 4),
         "groups": f"{partition.label(i)}-{partition.label(j)}"} for i, j in flagged]
    return report


def coordination_report(atoms: Atoms, indices: Sequence[int], partition: Partition | None = None,
                        tolerance: float = DEFAULT_BONDING_TOLERANCE,
                        metal_pairs: str | None = None) -> dict[str, Any]:
    """Per-atom coordination split by neighbor group and element.

    `metal_pairs`: element (e.g. "Fe") whose like-pair distance statistics
    (bonded pairs only, within `indices`) are reported.
    """
    partition = partition or Partition(list(range(len(atoms))))
    d = distance_matrix(atoms)
    bonded = bond_matrix(atoms, tolerance, d)
    symbols = atoms.get_chemical_symbols()
    per_atom = []
    for i in indices:
        nbrs = np.flatnonzero(bonded[i])
        by_group = Counter(partition.label(int(j)) for j in nbrs)
        by_element = Counter(symbols[j] for j in nbrs)
        per_atom.append({"i": int(i), "el": symbols[i], "cn": int(len(nbrs)),
                         "by_group": dict(by_group), "by_element": dict(by_element),
                         "nearest": round(float(min(d[i, j] for j in range(len(atoms)) if j != i)), 4)})
    cns = [a["cn"] for a in per_atom]
    out: dict[str, Any] = {
        "per_atom": per_atom,
        "cn_histogram": dict(sorted(Counter(cns).items())),
        "cn_mean": round(float(np.mean(cns)), 3) if cns else None,
    }
    if metal_pairs:
        ix = [i for i in indices if symbols[i] == metal_pairs]
        pairs = sorted(float(d[i, j]) for i, j in combinations(ix, 2) if bonded[i, j])
        if pairs:
            out[f"{metal_pairs}-{metal_pairs}_bonded"] = {
                "n": len(pairs), "min": round(pairs[0], 4), "max": round(pairs[-1], 4),
                "mean": round(float(np.mean(pairs)), 4)}
    return out


# ----------------------------------------------------------- unwrapping
def unwrapped_positions(atoms: Atoms, indices: Sequence[int],
                        tolerance: float = DEFAULT_BONDING_TOLERANCE) -> np.ndarray:
    """Contiguous Cartesian positions for a group that may cross cell boundaries.

    Traverses the group's bond graph and places each atom at the minimum image
    of its already-placed neighbor; disconnected pieces are placed at the
    minimum image of the group's first atom. Refuses groups wider than half of
    any periodic cell length, where the minimum image is ambiguous.
    """
    ix = list(indices)
    pos = atoms.positions
    if not _periodic(atoms) or not ix:
        return pos[ix].copy()
    sub = atoms[ix]
    bonded = bond_matrix(sub, tolerance)
    placed = np.full((len(ix), 3), np.nan)
    for root in range(len(ix)):
        if not np.isnan(placed[root, 0]):
            continue
        offset = find_mic(pos[ix[root]] - pos[ix[0]], atoms.cell, atoms.pbc)[0] if root else 0.0
        placed[root] = pos[ix[0]] + offset
        stack = [root]
        while stack:
            k = stack.pop()
            for m in np.flatnonzero(bonded[k]):
                if np.isnan(placed[m, 0]):
                    vec = find_mic(pos[ix[m]] - pos[ix[k]], atoms.cell, atoms.pbc)[0]
                    placed[m] = placed[k] + vec
                    stack.append(int(m))
    frac = placed @ np.linalg.inv(atoms.cell.array)
    if any(np.ptp(frac[:, axis]) > 0.5 for axis in np.flatnonzero(atoms.pbc)):
        raise ValueError("group spans more than half a periodic cell; unwrapping is ambiguous")
    return placed


# ------------------------------------------------ isolated-cluster mapping
def _kabsch(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Proper rotation R minimizing |p R - q| for centered p, q."""
    u, _, vt = np.linalg.svd(p.T @ q)
    c = np.eye(3)
    c[-1, -1] = np.sign(np.linalg.det(u @ vt)) or 1.0
    return u @ c @ vt


def _assign(x: np.ndarray, y: np.ndarray, sx: list[str], sy: list[str],
            cost: np.ndarray | None = None) -> np.ndarray:
    """perm with y[perm[i]] matched to x[i], element-preserving."""
    if cost is None:
        cost = np.linalg.norm(x[:, None] - y[None, :], axis=2) ** 2
    cost = cost.copy()
    cost[np.array(sx)[:, None] != np.array(sy)[None, :]] = 1e12
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(len(x), dtype=int)
    perm[rows] = cols
    return perm


MATCH_DEFAULTS = {"max_iter": 50, "random_starts": 64, "seed": 0}


def match_atoms(reference: Atoms, moving: Atoms, max_iter: int = MATCH_DEFAULTS["max_iter"],
                random_starts: int = MATCH_DEFAULTS["random_starts"],
                seed: int = MATCH_DEFAULTS["seed"]) -> dict[str, Any]:
    """Atom correspondence + optimal proper superposition of two isolated clusters.

    Starts: distance-fingerprint assignment, identity order, the four proper
    principal-axis frames and `random_starts` rotations drawn with `seed`.
    Each start alternates Kabsch and Hungarian assignment until the mapping
    repeats (at most `max_iter` rounds). Returns perm (moving index for each
    reference atom), rotation, RMSD, max and per-atom displacement, and the
    search settings.

    Heuristic: the result is the best mapping among these starts, not a
    proven global optimum, and there is no error bound. Agreement between
    forward and reverse searches (observed within ~1e-3 A on near-spherical
    Fe16) is a consistency check only. For exact-identity claims use the
    exhaustive spin_audit.alignments.
    """
    sx, sy = reference.get_chemical_symbols(), moving.get_chemical_symbols()
    if sorted(sx) != sorted(sy):
        raise ValueError("compositions differ; no atom correspondence exists")
    x = reference.positions - reference.positions.mean(0)
    y = moving.positions - moving.positions.mean(0)
    elements = sorted(set(sx))

    def fingerprints(z: np.ndarray, s: list[str]) -> np.ndarray:
        dd = np.linalg.norm(z[:, None] - z[None, :], axis=2)
        mask = {e: np.array(s) == e for e in elements}
        return np.array([np.concatenate([np.sort(dd[i, mask[e]]) for e in elements])
                         for i in range(len(z))])

    starts: list[np.ndarray] = []
    fx, fy = fingerprints(x, sx), fingerprints(y, sy)
    starts.append(_assign(x, y, sx, sy, np.linalg.norm(fx[:, None] - fy[None, :], axis=2)))
    if sx == sy:
        starts.append(np.arange(len(x)))

    def axes(z: np.ndarray) -> np.ndarray:
        _, vecs = np.linalg.eigh(z.T @ z)
        return vecs

    ax, ay = axes(x), axes(y)
    for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        rot = ay @ np.diag(signs) @ ax.T
        if np.linalg.det(rot) < 0:
            rot = ay @ np.diag(signs) @ np.diag([1, 1, -1]) @ ax.T
        starts.append(_assign(x, y @ rot, sx, sy))
    # Near-spherical clusters (Fe16) have nearly degenerate principal moments,
    # so the axis frames above are unstable; fixed-seed random rotations make
    # the search reproducible and keep it out of local optima.
    for rot in Rotation.random(random_starts, random_state=seed).as_matrix() if random_starts else []:
        starts.append(_assign(x, y @ rot, sx, sy))

    best: dict[str, Any] | None = None
    for perm in starts:
        seen = {perm.tobytes()}
        for _ in range(max_iter):
            new = _assign(x, y @ _kabsch(y[perm], x), sx, sy)
            if new.tobytes() in seen:  # converged, or cycling between mappings
                break
            seen.add(new.tobytes())
            perm = new
        rot = _kabsch(y[perm], x)
        disp = np.linalg.norm(y[perm] @ rot - x, axis=1)
        rmsd = float(np.sqrt(np.mean(disp ** 2)))
        if best is None or rmsd < best["rmsd"] - 1e-12:
            best = {"perm": perm, "rotation": rot, "rmsd": rmsd,
                    "max_displacement": float(disp.max()), "displacement": disp}
    assert best is not None
    # Independent, mapping-free cross-check: sorted same-element-pair distances.
    best["fingerprint_rms"] = _fingerprint_rms(reference, moving)
    best["search"] = {"method": "multi-start Hungarian+Kabsch (heuristic)", "n_starts": len(starts),
                      "random_starts": random_starts, "seed": seed, "max_iter": max_iter}
    return best


def _fingerprint_rms(a: Atoms, b: Atoms) -> float:
    def fp(z: Atoms) -> np.ndarray:
        dd, s = distance_matrix(z), z.get_chemical_symbols()
        groups: dict[tuple[str, str], list[float]] = {}
        for i, j in combinations(range(len(z)), 2):
            groups.setdefault(tuple(sorted((s[i], s[j]))), []).append(float(dd[i, j]))
        return np.concatenate([np.sort(groups[k]) for k in sorted(groups)])
    return float(np.sqrt(np.mean((fp(a) - fp(b)) ** 2)))


def bond_changes(reference: Atoms, moving: Atoms, perm: Sequence[int],
                 tolerance: float = DEFAULT_BONDING_TOLERANCE,
                 indices: Sequence[int] | None = None) -> dict[str, Any]:
    """Bonds formed/broken between mapped structures (reference numbering)."""
    m = moving[list(perm)]
    m.set_chemical_symbols(reference.get_chemical_symbols())
    d0, d1 = distance_matrix(reference), distance_matrix(m)
    b0, b1 = bond_matrix(reference, tolerance, d0), bond_matrix(m, tolerance, d1)
    ix = range(len(reference)) if indices is None else indices
    s = reference.get_chemical_symbols()
    keep = set(ix)
    changes: dict[str, list] = {"formed": [], "broken": []}
    for i, j in combinations(sorted(keep), 2):
        if b0[i, j] != b1[i, j]:
            changes["formed" if b1[i, j] else "broken"].append(
                {"i": i, "j": j, "pair": f"{s[i]}-{s[j]}",
                 "d_ref": round(float(d0[i, j]), 4), "d_new": round(float(d1[i, j]), 4)})
    return changes


# ------------------------------------------------------- adsorption sites
def _lateral_vectors(atoms: Atoms, origin: np.ndarray, targets: Sequence[int],
                     normal_axis: int) -> np.ndarray:
    vec = atoms.positions[list(targets)] - origin
    if _periodic(atoms):
        vec = find_mic(vec, atoms.cell, atoms.pbc)[0]
    vec[:, normal_axis] = 0.0
    return vec


def surface_layer(atoms: Atoms, substrate: Sequence[int], column_radius: float = 1.0,
                  normal_axis: int = 2) -> list[int]:
    """Substrate atoms with no other substrate atom above them within `column_radius` Å laterally.

    Column-based rather than a global z-window, so a bent or rumpled support
    (freestanding graphene under a cluster) keeps its whole top sheet.
    """
    sub = list(substrate)
    z = atoms.positions[sub, normal_axis]
    top = []
    for k, i in enumerate(sub):
        lateral = np.linalg.norm(_lateral_vectors(atoms, atoms.positions[i], sub, normal_axis), axis=1)
        if not np.any((lateral < column_radius) & (z > z[k] + 1e-6)):
            top.append(int(i))
    return top


def site_report(atoms: Atoms, group: Sequence[int], substrate: Sequence[int],
                tolerance: float = DEFAULT_BONDING_TOLERANCE, column_radius: float = 1.0,
                local_radius: float = 3.0, normal_axis: int = 2,
                site_elements: Sequence[str] | None = None,
                lateral_tolerance: float = 0.5, height_spread_tolerance: float = 0.3) -> dict[str, Any]:
    """Adsorption-site assignment for each atom of `group` that touches the substrate.

    Site atoms are the surface-layer atoms within the bond cutoff (of the
    `site_elements`, if given). An ideal label needs an ideal pattern:
    top (1 atom), bridge (2) or hollow-n (n >= 3) only if the adsorbing atom
    sits within `lateral_tolerance` Å (laterally) of the site atom / pair
    midpoint / polygon centroid, no site atom of a bridge/hollow lies within
    `lateral_tolerance` Å directly below it, and the site atoms' heights spread
    by at most `height_spread_tolerance` Å. Otherwise the site is "irregular-n"
    with the reasons (off-centre, atom-beneath, buckled), as on reconstructed or strongly deformed
    supports. Labels carry the site composition ("top:O", "hollow-6:C6").

    `site_elements` is required whenever cations sit inside the bond cutoff
    of the adsorbing atom, i.e. on ionic supports: on MgO pass ["O"], because
    Fe-Mg is 3.5 A at tolerance 1.2 and an Fe above O would count its four Mg
    neighbours. All bonded surface atoms are still listed.

    Height is local: above the mean normal coordinate of surface atoms within
    `local_radius` Å laterally (minimum image). On a buckled support this
    differs from the height above the highest support atom by up to the
    buckling amplitude; both should be quoted with their definitions.
    """
    d = distance_matrix(atoms)
    r = _radii(atoms)
    s = atoms.get_chemical_symbols()
    top = surface_layer(atoms, substrate, column_radius, normal_axis)
    z_top = atoms.positions[top, normal_axis]
    lateral_axes = [a for a in range(3) if a != normal_axis]
    contacts, heights = [], []
    for i in group:
        bonded = [j for j in top if d[i, j] <= tolerance * (r[i] + r[j])]
        defining = [j for j in bonded if site_elements is None or s[j] in site_elements]
        nearest = min(top, key=lambda j: d[i, j])
        lateral = np.linalg.norm(_lateral_vectors(atoms, atoms.positions[i], top, normal_axis), axis=1)
        local = z_top[lateral <= local_radius]
        z_local = float(local.mean()) if len(local) else float(atoms.positions[nearest, normal_axis])
        height = float(atoms.positions[i, normal_axis] - z_local)
        heights.append(height)
        entry: dict[str, Any] = {
            "i": int(i), "el": s[i], "height": round(height, 4),
            "nearest_surface": {"j": int(nearest), "el": s[nearest], "d": round(float(d[i, nearest]), 4)},
        }
        if bonded:
            vecs = find_mic(atoms.positions[bonded] - atoms.positions[i], atoms.cell, atoms.pbc)[0]
            if defining:
                site_vecs = vecs[[bonded.index(j) for j in defining]]
                label, geometry = _site_label(site_vecs, lateral_axes, normal_axis,
                                              lateral_tolerance, height_spread_tolerance)
                composition = "".join(f"{e}{n if n > 1 else ''}"
                                      for e, n in sorted(Counter(s[j] for j in defining).items()))
                entry.update(site=f"{label}:{composition}", site_geometry=geometry)
            else:
                entry["site"] = "no-site-element-contact"
            entry.update(
                site_atoms=[{"j": int(j), "el": s[j], "d": round(float(d[i, j]), 4),
                             "lateral": round(float(np.linalg.norm(v[lateral_axes])), 4)}
                            for j, v in sorted(zip(bonded, vecs), key=lambda t: d[i, t[0]])])
            contacts.append(entry)
    lowest = int(np.argmin(heights)) if heights else None
    near = min(((int(i), int(j)) for i in group for j in top), key=lambda p: d[p], default=None)
    return {
        "surface_layer": {"n": len(top), "elements": dict(Counter(s[j] for j in top)),
                          "z_range": [round(float(z_top.min()), 4), round(float(z_top.max()), 4)],
                          "column_radius": column_radius, "local_radius": local_radius,
                          "site_elements": list(site_elements) if site_elements else "all",
                          "lateral_tolerance": lateral_tolerance,
                          "height_spread_tolerance": height_spread_tolerance},
        "n_in_contact": len(contacts),
        "min_height": None if lowest is None else {"i": int(group[lowest]), "h": round(heights[lowest], 4)},
        "closest_contact": None if near is None else
        {"i": near[0], "j": near[1], "pair": f"{s[near[0]]}-{s[near[1]]}", "d": round(float(d[near]), 4),
         "bond_cutoff": round(float(tolerance * (r[near[0]] + r[near[1]])), 4)},
        "contacts": contacts,
        "site_counts": dict(Counter(c["site"] for c in contacts)),
    }


def _site_label(vecs: np.ndarray, lateral_axes: list[int], normal_axis: int,
                lateral_tolerance: float, height_spread_tolerance: float) -> tuple[str, dict[str, Any]]:
    """Ideal label only for an ideal pattern; `vecs` are adsorbate->site-atom vectors."""
    n = len(vecs)
    offset = float(np.linalg.norm(vecs.mean(0)[lateral_axes]))  # to atom / midpoint / centroid
    spread = float(np.ptp(vecs[:, normal_axis]))
    geometry: dict[str, Any] = {"lateral_offset": round(offset, 4), "site_height_spread": round(spread, 4)}
    reasons = []
    if offset > lateral_tolerance:
        reasons.append("off-centre")
    if n > 1 and spread > height_spread_tolerance:
        reasons.append("buckled")
    if n > 1 and float(np.linalg.norm(vecs[:, lateral_axes], axis=1).min()) < lateral_tolerance:
        reasons.append("atom-beneath")  # a bridge/hollow has no site atom directly below
    if reasons:
        geometry["irregular"] = reasons
        return f"irregular-{n}", geometry
    return {1: "top", 2: "bridge"}.get(n, f"hollow-{n}"), geometry


def adsorbate_report(atoms: Atoms, adsorbate: Sequence[int], host: Sequence[int],
                     tolerance: float = DEFAULT_BONDING_TOLERANCE,
                     surface_normal: Sequence[float] | None = (0.0, 0.0, 1.0)) -> dict[str, Any]:
    """Adsorbate internal bonds, host-adsorbate contacts and orientation.

    Orientation: angle between the adsorbate's principal axis (first bonded
    pair for diatomics) and the reference direction, which is the surface
    normal for supported systems, else the vector from the host centroid to
    the adsorbate centroid. The axis is oriented from the host-bound end
    outward, so 0 deg = upright (bound end down), 90 deg = flat.
    """
    d = distance_matrix(atoms)
    bonded = bond_matrix(atoms, tolerance, d)
    s = atoms.get_chemical_symbols()
    ads = list(adsorbate)
    internal = [{"i": i, "j": j, "pair": f"{s[i]}-{s[j]}", "d": round(float(d[i, j]), 4)}
                for i, j in combinations(ads, 2) if bonded[i, j]]
    contacts = [{"i": i, "j": int(j), "pair": f"{s[i]}-{s[j]}", "d": round(float(d[i, j]), 4)}
                for i in ads for j in host if bonded[i, j]]
    out: dict[str, Any] = {"internal_bonds": internal, "host_contacts": contacts,
                           "host_bound_atoms": sorted({c["i"] for c in contacts})}
    pos = unwrapped_positions(atoms, ads, tolerance)
    if len(ads) >= 2:
        centered = pos - pos.mean(0)
        axis = np.linalg.eigh(centered.T @ centered)[1][:, -1]
        bound = [k for k, i in enumerate(ads) if i in out["host_bound_atoms"]]
        if bound:
            # point the axis from the bound end toward the free end
            if (pos.mean(0) - pos[bound].mean(0)) @ axis < 0:
                axis = -axis
        if surface_normal is not None:
            ref = np.asarray(surface_normal, dtype=float)
            ref_name = "surface_normal"
        else:
            host_pos = atoms.positions[list(host)]
            ref = find_mic(pos.mean(0) - host_pos.mean(0), atoms.cell, atoms.pbc)[0] \
                if _periodic(atoms) else pos.mean(0) - host_pos.mean(0)
            ref_name = "host_centroid_to_adsorbate"
        cosang = float(axis @ ref / np.linalg.norm(ref))
        out["orientation"] = {
            "reference": ref_name,
            "tilt_deg": round(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), 2),
            "axis_oriented": bool(bound),
        }
    return out


# --------------------------------------------------------- comparisons
def cluster_internal_rmsds(reference: Atoms, moving: Atoms) -> dict[str, Any]:
    """Two different questions, two numbers (both after proper superposition):

    - trajectory_rmsd: same atom order; how far each atom moved during the
      relaxation/MD that produced `moving` from `reference`.
    - shape_rmsd: optimal element-preserving mapping; how different the two
      shapes are. Lower than trajectory_rmsd when atoms exchanged
      symmetry-equivalent positions.
    cluster_atoms_exchanged counts non-identity entries of the best mapping;
    in a symmetric cluster it can be nonzero with no physical exchange, so
    read it together with the gap between the two RMSDs.
    """
    a = reference.positions - reference.positions.mean(0)
    b = moving.positions - moving.positions.mean(0)
    trajectory = float(np.sqrt(np.mean(np.sum((b @ _kabsch(b, a) - a) ** 2, 1))))
    m = match_atoms(reference, moving)
    return {"cluster_trajectory_rmsd": round(trajectory, 4),
            "cluster_shape_rmsd": round(m["rmsd"], 4),
            "cluster_shape_max_displacement": round(m["max_displacement"], 4),
            "cluster_atoms_exchanged": int(np.sum(m["perm"] != np.arange(len(a)))),
            "cluster_matching": m["search"]}


def bond_change_sensitivity(reference: Atoms, moving: Atoms, perm: Sequence[int],
                            tolerances: Sequence[float] = (1.10, 1.15, 1.20, 1.25, 1.30),
                            indices: Sequence[int] | None = None) -> list[dict[str, Any]]:
    """Formed/broken counts in the cutoff-defined graph across bond tolerances.

    A count that moves with the tolerance comes from distances near the
    cutoff and should be quoted with that sensitivity, not as a fixed number.
    """
    out = []
    for tol in tolerances:
        ch = bond_changes(reference, moving, perm, tol, indices)
        pair_types = sorted({p["pair"] for k in ch for p in ch[k]})
        out.append({"tolerance": tol, "formed": len(ch["formed"]), "broken": len(ch["broken"]),
                    "pair_types": pair_types})
    return out


def compare_isolated(reference: Atoms, moving: Atoms,
                     tolerance: float = DEFAULT_BONDING_TOLERANCE,
                     top: int = 5) -> dict[str, Any]:
    m = match_atoms(reference, moving)
    order = np.argsort(m["displacement"])[::-1][:top]
    s = reference.get_chemical_symbols()
    return {
        "mode": "isolated",
        "rmsd": round(m["rmsd"], 4),
        "max_displacement": round(m["max_displacement"], 4),
        "fingerprint_rms": round(m["fingerprint_rms"], 4),
        "mapping_is_identity": bool(np.array_equal(m["perm"], np.arange(len(reference)))),
        "perm": [int(p) for p in m["perm"]],
        "largest_displacements": [{"i": int(i), "el": s[i], "moving_index": int(m["perm"][i]),
                                   "d": round(float(m["displacement"][i]), 4)} for i in order],
        "bond_changes": bond_changes(reference, moving, m["perm"], tolerance),
        "bond_change_sensitivity": bond_change_sensitivity(reference, moving, m["perm"]),
        "matching": m["search"],
    }


def compare_supported(reference: Atoms, moving: Atoms, partition: Partition,
                      tolerance: float = DEFAULT_BONDING_TOLERANCE, top: int = 5,
                      site_elements: Sequence[str] | None = None) -> dict[str, Any]:
    """Same atom ordering assumed (relaxation of one system); checked by element.

    Displacements are minimum-image, after removing the substrate's mean
    lateral+normal drift. The cluster's internal change is also reported after
    unwrapping and proper superposition (cluster-only RMSD).
    """
    if reference.get_chemical_symbols() != moving.get_chemical_symbols():
        raise ValueError("supported comparison needs the same atom order; map atoms first")
    if not np.allclose(reference.cell.array, moving.cell.array, atol=1e-6):
        raise ValueError("cells differ; displacements across different cells are not defined")
    raw = moving.positions - reference.positions
    vec = find_mic(raw, reference.cell, reference.pbc)[0] if _periodic(reference) else raw
    drift = vec[partition.substrate].mean(0) if partition.substrate else np.zeros(3)
    vec = vec - drift
    disp = np.linalg.norm(vec, axis=1)
    s = reference.get_chemical_symbols()
    out: dict[str, Any] = {"mode": "supported", "substrate_drift": [round(float(v), 4) for v in drift]}
    for name, ix in (("cluster", partition.cluster), ("adsorbate", partition.adsorbate),
                     ("substrate", partition.substrate)):
        if ix:
            dd = disp[ix]
            out[f"{name}_displacement"] = {"rms": round(float(np.sqrt(np.mean(dd ** 2))), 4),
                                           "max": round(float(dd.max()), 4),
                                           "mean_vector": [round(float(v), 4) for v in vec[ix].mean(0)]}
    if len(partition.cluster) >= 3:
        symbols = [s[i] for i in partition.cluster]
        a = Atoms(symbols, unwrapped_positions(reference, partition.cluster, tolerance))
        b = Atoms(symbols, unwrapped_positions(moving, partition.cluster, tolerance))
        out.update(cluster_internal_rmsds(a, b))
    order = np.argsort(disp)[::-1][:top]
    out["largest_displacements"] = [{"i": int(i), "el": s[i], "group": partition.label(int(i)),
                                     "d": round(float(disp[i]), 4)} for i in order]
    out["bond_changes"] = bond_changes(reference, moving, list(range(len(reference))), tolerance,
                                       sorted(partition.cluster + partition.adsorbate))
    out["bond_changes_cluster_substrate"] = _interface_bond_changes(reference, moving, partition, tolerance)
    if partition.substrate:
        group = partition.cluster + partition.adsorbate
        sr, sm = (site_report(x, group, partition.substrate, tolerance, site_elements=site_elements)
                  for x in (reference, moving))
        out["sites"] = {"reference": sr["site_counts"], "moving": sm["site_counts"],
                        "n_in_contact": [sr["n_in_contact"], sm["n_in_contact"]]}
    return out


def _interface_bond_changes(a: Atoms, b: Atoms, partition: Partition, tolerance: float) -> dict[str, int]:
    if not partition.substrate:
        return {}
    ba, bb = bond_matrix(a, tolerance), bond_matrix(b, tolerance)
    up, sub = partition.cluster + partition.adsorbate, partition.substrate
    block_a, block_b = ba[np.ix_(up, sub)], bb[np.ix_(up, sub)]
    return {"reference": int(block_a.sum()), "moving": int(block_b.sum()),
            "formed": int((block_b & ~block_a).sum()), "broken": int((block_a & ~block_b).sum())}


# ------------------------------------------------------------- top level
def structure_report(atoms: Atoms, partition: Partition | None = None,
                     tolerance: float = DEFAULT_BONDING_TOLERANCE,
                     close_contact_scale: float = DEFAULT_CLOSE_CONTACT_SCALE,
                     metal: str | None = None,
                     site_elements: Sequence[str] | None = None) -> dict[str, Any]:
    partition = partition or Partition(list(range(len(atoms))))
    report: dict[str, Any] = {
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "periodic": [bool(p) for p in atoms.pbc],
        "partition": partition.as_dict(),
        "conventions": {"radii": "Cordero 2008 (stratify.COVALENT_RADII_ANGSTROM)",
                        "bonded": f"d <= {tolerance} * (r_i + r_j)",
                        "close_contact": f"d < {close_contact_scale} * (r_i + r_j)",
                        "distances": "minimum image" if _periodic(atoms) else "Cartesian"},
        "contacts": contact_report(atoms, partition, close_contact_scale),
        "cluster_coordination": coordination_report(atoms, partition.cluster, partition, tolerance, metal),
        "caveat": ELECTRONIC_CAVEAT,
    }
    if len(partition.cluster) >= 2:
        pos = unwrapped_positions(atoms, partition.cluster, tolerance)
        centered = pos - pos.mean(0)
        evals = np.sort(np.linalg.eigvalsh(centered.T @ centered / len(pos)))[::-1]
        rg = float(np.sqrt(evals.sum()))
        report["cluster_shape"] = {
            "radius_of_gyration": round(rg, 4),
            "principal_moments": [round(float(v), 4) for v in evals],
            "asphericity": round(float((evals[0] - 0.5 * (evals[1] + evals[2])) / evals.sum()), 4),
            "connected": _connected(bond_matrix(atoms[partition.cluster], tolerance)),
        }
    if partition.substrate:
        report["sites"] = site_report(atoms, partition.cluster + partition.adsorbate,
                                      partition.substrate, tolerance, site_elements=site_elements)
    if partition.adsorbate:
        host = partition.cluster + partition.substrate
        report["adsorbate"] = adsorbate_report(
            atoms, partition.adsorbate, host, tolerance,
            (0.0, 0.0, 1.0) if partition.substrate else None)
    return report


def _connected(bonded: np.ndarray) -> bool:
    n = len(bonded)
    seen, stack = {0}, [0]
    while stack:
        for j in np.flatnonzero(bonded[stack.pop()]):
            if int(j) not in seen:
                seen.add(int(j))
                stack.append(int(j))
    return len(seen) == n


def _read_frame(spec: str) -> Atoms:
    """'path' or 'path@index' (ASE index syntax, default last frame)."""
    from ase.io import read
    path, _, index = spec.partition("@")
    frame = read(path, index=int(index) if index else -1)
    assert isinstance(frame, Atoms)
    return frame


def run_cli(args: Any) -> int:
    import json
    from pathlib import Path

    reference = _read_frame(args.structure)
    partition = Partition.from_specs(len(reference), args.substrate, args.adsorbate)
    sites = args.site_elements.split(",") if args.site_elements else None
    report: dict[str, Any] = {"reference": structure_report(
        reference, partition, args.bond_tolerance, args.close_contact_scale, args.metal, sites)}
    if args.compare:
        moving = _read_frame(args.compare)
        report["moving"] = structure_report(
            moving, partition, args.bond_tolerance, args.close_contact_scale, args.metal, sites)
        if partition.substrate:
            report["comparison"] = compare_supported(reference, moving, partition, args.bond_tolerance,
                                                     site_elements=sites)
        else:
            report["comparison"] = compare_isolated(reference, moving, args.bond_tolerance)
    text = json.dumps(report, indent=1)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Geometry report: {args.output}")
    else:
        print(text)
    flagged = len(report["reference"]["contacts"]["close_contacts"]) + \
        len(report.get("moving", {}).get("contacts", {}).get("close_contacts", []))
    if flagged:
        print(f"WARNING: {flagged} close contact(s) below {args.close_contact_scale} x (r_i + r_j)")
    return 1 if flagged and args.fail_on_close_contact else 0
