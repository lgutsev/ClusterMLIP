"""Fixed-composition adsorbate search on a fixed catalyst (random vs evolutionary).

The pieces, in the order a candidate passes through them:

* ``find_cluster_sites`` / ``sites_from_list``: adsorption sites on the actual
  catalyst geometry, with explicit catalyst atom indices. The built-in finder
  works on the convex hull of an isolated cluster; it does not assume a
  crystalline nanoparticle template (ACAT's ``ClusterAdsorptionSites`` does, and
  needs ``asap3``).
* ``Gene`` / ``build_candidate``: one gene per adsorbate (species, site,
  binding mode, tilt, azimuth). Building writes per-atom ``role`` and ``mol_id``
  arrays so catalyst, support and adsorbate atoms are never told apart by
  element.
* ``validate``: composition, roles and distance exclusions (minimum image when
  the cell is periodic).
* ``analyze``: after relaxation, what the structure became: fragments (intact
  CO, HCO, COH, H2, C + O ...), observed site of each fragment, desorption,
  overlaps, catalyst displacement.
* ``same_structure``: permutation-aware duplicate test. An isolated cluster is
  compared by sorted interatomic distances (rotation/translation removed); a
  supported or periodic system is compared in the fixed frame of the support.
* ``AdsorbateSearch``: the two arms. Both draw the same random initial
  proposals and use the same builder, filters, relaxation and duplicate test;
  the evolutionary arm then proposes children with move / rotate / recombine.

Atom roles (``atoms.arrays["role"]``): 0 support, 1 catalyst, 2 adsorbate.
``atoms.arrays["cluster"]`` (1 = catalyst) is kept as well, the convention of
`delta.py` and `periodic.py`.

This module needs numpy, scipy and ASE; the rest of the package does not import it.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import find_mic, get_distances

ROLE_SUPPORT, ROLE_CATALYST, ROLE_ADSORBATE = 0, 1, 2

# Start bond lengths (Angstrom) between the binding atom and the catalyst atoms
# of a site. Starting guesses only: the relaxation decides the final geometry.
START_BOND = {
    "H": {"top": 1.55, "bridge": 1.68, "hollow3": 1.78, "hollow4": 1.85},
    "C": {"top": 1.78, "bridge": 1.95, "hollow3": 2.05, "hollow4": 2.10},
    "O": {"top": 1.95, "bridge": 2.05, "hollow3": 2.15, "hollow4": 2.20},
}
CO_BOND = 1.15
CO_MODES = ("C", "O", "side")  # C-bound, O-bound, side-on (C at the site, axis tilted)

# Distance exclusions for a *starting* structure (Angstrom).
MIN_ADS_ADS = {("H", "H"): 1.30, ("C", "H"): 1.50, ("H", "O"): 1.40}
MIN_ADS_ADS_DEFAULT = 1.60
MIN_ADS_CAT_SCALE = 0.75  # times the sum of covalent radii

# Post-relaxation thresholds.
BOND_SCALE = 1.25  # bonded if d < BOND_SCALE * (r_i + r_j)
OVERLAP_SCALE = 0.65  # failure if d < OVERLAP_SCALE * (r_i + r_j)
CO_INTACT_MAX = 1.45
DESORB_SCALE = 1.60  # fragment desorbed if no atom within this * (r_i + r_j) of the catalyst


def _radius(symbol: str) -> float:
    return float(covalent_radii[atomic_numbers[symbol]])


# --------------------------------------------------------------------------- sites


@dataclass(frozen=True)
class Site:
    id: int
    kind: str  # top | bridge | hollow3 | hollow4
    indices: tuple[int, ...]  # catalyst atom indices in the full Atoms object
    position: tuple[float, ...]
    normal: tuple[float, ...]

    @property
    def p(self) -> np.ndarray:
        return np.array(self.position)

    @property
    def n(self) -> np.ndarray:
        return np.array(self.normal)


@dataclass
class SiteReport:
    sites: list[Site]
    surface_ids: list[int]
    interior_ids: list[int]
    notes: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.sites:
            out[s.kind] = out.get(s.kind, 0) + 1
        return out


def catalyst_indices(atoms: Atoms) -> np.ndarray:
    if "role" in atoms.arrays:
        return np.where(atoms.arrays["role"] == ROLE_CATALYST)[0]
    return np.arange(len(atoms))


def find_cluster_sites(
    atoms: Atoms, catalyst_ids: Sequence[int] | None = None, *, bond_cutoff: float = 3.0
) -> SiteReport:
    """Top/bridge/hollow sites on the convex hull of an isolated cluster.

    A hull vertex is a surface atom and a top site. A hull edge shorter than
    ``bond_cutoff`` is a bridge. A hull triangle with three bonded edges is a
    3-fold hollow; two triangles sharing an unbonded edge form a 4-fold hollow
    when their four outer edges are bonded. Catalyst atoms off the hull are
    reported as interior; a concave surface atom would land there too, so the
    report flags any interior atom with fewer than 10 catalyst neighbours.
    """
    from scipy.spatial import ConvexHull

    if any(atoms.pbc):
        raise ValueError("find_cluster_sites is for an isolated cluster (pbc all False); "
                         "use sites_from_list or a slab site finder for periodic systems")
    ids = np.array(catalyst_indices(atoms) if catalyst_ids is None else catalyst_ids, dtype=int)
    pos = atoms.positions[ids]
    if len(ids) < 4:
        raise ValueError("need at least 4 catalyst atoms for a convex hull")
    hull = ConvexHull(pos)
    centroid = pos.mean(axis=0)
    surface_local = sorted(int(v) for v in hull.vertices)
    interior_local = sorted(set(range(len(ids))) - set(surface_local))
    notes: list[str] = []
    for loc in interior_local:
        cn = int((np.linalg.norm(pos - pos[loc], axis=1) < bond_cutoff).sum()) - 1
        if cn < 10:
            notes.append(f"interior atom {int(ids[loc])} has only {cn} catalyst neighbours within "
                         f"{bond_cutoff} A: possibly a concave surface atom missed by the hull")

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(pos[a] - pos[b]))

    faces = [tuple(sorted(int(v) for v in simplex)) for simplex in hull.simplices]
    normals = [np.array(eq[:3]) for eq in hull.equations]  # outward unit normals
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[0], face[2]), (face[1], face[2])):
            edge_faces.setdefault((a, b), []).append(fi)

    raw: list[tuple[str, tuple[int, ...], np.ndarray, np.ndarray]] = []
    for v in surface_local:
        nsum = sum((normals[fi] for fi, f in enumerate(faces) if v in f), np.zeros(3))
        if np.linalg.norm(nsum) < 1e-6:
            nsum = pos[v] - centroid
        raw.append(("top", (v,), pos[v].copy(), nsum))
    for (a, b), fis in sorted(edge_faces.items()):
        if dist(a, b) < bond_cutoff:
            raw.append(("bridge", (a, b), (pos[a] + pos[b]) / 2, sum((normals[fi] for fi in fis), np.zeros(3))))
    for fi, face in enumerate(faces):
        a, b, c = face
        if max(dist(a, b), dist(a, c), dist(b, c)) < bond_cutoff:
            raw.append(("hollow3", face, pos[list(face)].mean(axis=0), normals[fi]))
    for (a, b), fis in sorted(edge_faces.items()):
        if dist(a, b) >= bond_cutoff and len(fis) == 2:
            others = sorted(set(faces[fis[0]]) ^ set(faces[fis[1]]))
            outer = [(min(x, y), max(x, y)) for x in (a, b) for y in others]
            if len(others) == 2 and all(dist(x, y) < bond_cutoff for x, y in outer):
                quad = tuple(sorted((a, b, *others)))
                raw.append(("hollow4", quad, pos[list(quad)].mean(axis=0), normals[fis[0]] + normals[fis[1]]))
            else:
                notes.append(f"hull edge {int(ids[a])}-{int(ids[b])} is unbonded ({dist(a, b):.2f} A) "
                             "and does not close a 4-fold site: no site assigned there")

    sites: list[Site] = []
    for kind, local, p, n in raw:
        n = n / np.linalg.norm(n)
        sites.append(Site(len(sites), kind, tuple(int(ids[i]) for i in local),
                          tuple(float(x) for x in p), tuple(float(x) for x in n)))
    return SiteReport(sites, [int(ids[i]) for i in surface_local], [int(ids[i]) for i in interior_local], notes)


def sites_from_list(atoms: Atoms, entries: Iterable[dict[str, Any]]) -> SiteReport:
    """Manual site list: each entry has ``indices`` and optionally ``kind``/``normal``.

    The site position is the mean of the listed catalyst atoms (minimum image
    around the first one). Without ``normal`` a periodic system uses +z and an
    isolated one the direction away from the catalyst centroid.
    """
    cat = catalyst_indices(atoms)
    centroid = atoms.positions[cat].mean(axis=0)
    kinds = {1: "top", 2: "bridge", 3: "hollow3", 4: "hollow4"}
    sites: list[Site] = []
    for entry in entries:
        idx = tuple(int(i) for i in entry["indices"])
        if not set(idx) <= set(int(i) for i in cat):
            raise ValueError(f"site {idx} names atoms that are not catalyst atoms")
        rel = [np.zeros(3)] + [find_mic(atoms.positions[i] - atoms.positions[idx[0]], atoms.cell, atoms.pbc)[0]
                               for i in idx[1:]]
        p = atoms.positions[idx[0]] + np.mean(rel, axis=0)
        n = np.array(entry["normal"], dtype=float) if "normal" in entry else (
            np.array([0.0, 0.0, 1.0]) if any(atoms.pbc) else p - centroid)
        n = n / np.linalg.norm(n)
        sites.append(Site(len(sites), entry.get("kind", kinds[len(idx)]), tuple(sorted(idx)),
                          tuple(float(x) for x in p), tuple(float(x) for x in n)))
    return SiteReport(sites, sorted({i for s in sites for i in s.indices}), [], ["manual site list"])


# --------------------------------------------------------------------------- genes and building


@dataclass(frozen=True)
class Gene:
    species: str  # "H" or "CO"
    site: int
    mode: str = "atom"  # H: "atom"; CO: "C" | "O" | "side"
    tilt: float = 0.0  # degrees from the site normal
    azimuth: float = 0.0  # degrees around the site normal

    def key(self) -> tuple:
        return (self.species, self.site, self.mode, round(self.tilt, 1), round(self.azimuth, 1))


def _tangent_basis(site: Site, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    n = site.n
    ref = atoms.positions[site.indices[0]] - site.p
    if np.linalg.norm(ref - (ref @ n) * n) < 1e-6:
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = ref - (ref @ n) * n
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1)


def _height(site: Site, atoms: Atoms, bond: float) -> float:
    r = float(np.mean([np.linalg.norm(find_mic(atoms.positions[i] - site.p, atoms.cell, atoms.pbc)[0])
                       for i in site.indices]))
    return float(np.sqrt(max(bond * bond - r * r, 0.3 ** 2)))


def gene_atoms(gene: Gene, site: Site, catalyst: Atoms) -> list[tuple[str, np.ndarray]]:
    if gene.species == "H":
        return [("H", site.p + _height(site, catalyst, START_BOND["H"][site.kind]) * site.n)]
    if gene.species != "CO":
        raise ValueError(f"unsupported adsorbate {gene.species!r}")
    if gene.mode not in CO_MODES:
        raise ValueError(f"unknown CO binding mode {gene.mode!r}")
    t1, t2 = _tangent_basis(site, catalyst)
    th, ph = np.radians(gene.tilt), np.radians(gene.azimuth)
    axis = np.cos(th) * site.n + np.sin(th) * (np.cos(ph) * t1 + np.sin(ph) * t2)
    bound, far = ("O", "C") if gene.mode == "O" else ("C", "O")
    p0 = site.p + _height(site, catalyst, START_BOND[bound][site.kind]) * site.n
    return [(bound, p0), (far, p0 + CO_BOND * axis)]


def build_candidate(catalyst: Atoms, sites: Sequence[Site], genes: Sequence[Gene]) -> Atoms:
    """Catalyst (+ support) followed by the adsorbates, with role and mol_id arrays."""
    atoms = catalyst.copy()
    atoms.calc = None
    n0 = len(atoms)
    role = np.array(catalyst.arrays["role"]) if "role" in catalyst.arrays else np.full(n0, ROLE_CATALYST)
    mol = np.full(n0, -1)
    for k, gene in enumerate(genes):
        for symbol, p in gene_atoms(gene, sites[gene.site], catalyst):
            atoms.append(symbol)
            atoms.positions[-1] = p
            role = np.append(role, ROLE_ADSORBATE)
            mol = np.append(mol, k)
    atoms.set_array("role", role.astype(int))
    atoms.set_array("mol_id", mol.astype(int))
    atoms.set_array("cluster", (role == ROLE_CATALYST).astype(int))
    return atoms


def composition_of(genes: Sequence[Gene]) -> dict[str, int]:
    out: dict[str, int] = {}
    for g in genes:
        out[g.species] = out.get(g.species, 0) + 1
    return out


# --------------------------------------------------------------------------- validation


def _dmat(atoms: Atoms, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return get_distances(atoms.positions[a], atoms.positions[b], cell=atoms.cell, pbc=atoms.pbc)[1]


def validate(atoms: Atoms, reference: Atoms, composition: dict[str, int]) -> list[str]:
    """Reasons a starting structure is rejected (empty list = valid)."""
    problems: list[str] = []
    role, mol = atoms.arrays["role"], atoms.arrays["mol_id"]
    n0 = len(reference)
    sym = np.array(atoms.get_chemical_symbols())
    if list(sym[:n0]) != reference.get_chemical_symbols() or not np.allclose(atoms.positions[:n0], reference.positions):
        problems.append("catalyst_changed")
    if (role[:n0] == ROLE_ADSORBATE).any() or (role[n0:] != ROLE_ADSORBATE).any() or (mol[:n0] != -1).any():
        problems.append("role_map_broken")
    expected: dict[str, int] = {}
    for species, count in composition.items():
        for el in ("C", "O") if species == "CO" else (species,):
            expected[el] = expected.get(el, 0) + count
    found = {el: int((sym[n0:] == el).sum()) for el in set(sym[n0:])}
    if found != expected:
        problems.append("composition_mismatch")
    ads = np.where(role == ROLE_ADSORBATE)[0]
    cat = np.where(role != ROLE_ADSORBATE)[0]
    if len(ads) == 0:
        return problems
    d_ac = _dmat(atoms, ads, cat)
    limit = MIN_ADS_CAT_SCALE * np.add.outer([_radius(s) for s in sym[ads]], [_radius(s) for s in sym[cat]])
    if (d_ac < limit).any():
        problems.append("adsorbate_catalyst_too_close")
    d_aa = _dmat(atoms, ads, ads)
    for i in range(len(ads)):
        for j in range(i + 1, len(ads)):
            if mol[ads[i]] == mol[ads[j]]:
                continue
            pair = tuple(sorted((sym[ads[i]], sym[ads[j]])))
            if d_aa[i, j] < MIN_ADS_ADS.get(pair, MIN_ADS_ADS_DEFAULT):
                problems.append("adsorbates_too_close")
                break
        else:
            continue
        break
    return problems


# --------------------------------------------------------------------------- analysis after relaxation


@dataclass
class Analysis:
    topology: str  # e.g. "CO+H+H", "HCO+H", "C+O+H+H", "CO+H2"
    motif: str  # topology with the observed site kind of each fragment
    fragments: list[dict[str, Any]]
    flags: list[str]
    failure: bool  # overlap / composition error: not a physical candidate
    catalyst_max_disp: float
    min_ads_cat: float


def _fragments(atoms: Atoms, ads: np.ndarray) -> list[list[int]]:
    sym = atoms.get_chemical_symbols()
    d = _dmat(atoms, ads, ads)
    parent = list(range(len(ads)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(ads)):
        for j in range(i + 1, len(ads)):
            if d[i, j] < BOND_SCALE * (_radius(sym[ads[i]]) + _radius(sym[ads[j]])):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(ads)):
        groups.setdefault(find(i), []).append(int(ads[i]))
    return list(groups.values())


def _fragment_name(symbols: list[str], atoms: Atoms, members: list[int]) -> str:
    counts = {s: symbols.count(s) for s in sorted(set(symbols))}
    formula = "".join(f"{s}{n if n > 1 else ''}" for s, n in counts.items())
    if counts == {"C": 1, "H": 1, "O": 1}:
        # formyl (H on C) vs hydroxycarbyne (H on O)
        h = next(m for m in members if atoms.symbols[m] == "H")
        c = next(m for m in members if atoms.symbols[m] == "C")
        o = next(m for m in members if atoms.symbols[m] == "O")
        return "HCO" if atoms.get_distance(h, c, mic=True) <= atoms.get_distance(h, o, mic=True) else "COH"
    return {"CO": "CO", "H2": "H2", "HO": "OH", "CH": "CH"}.get(formula, formula)


def observed_site(atoms: Atoms, atom_index: int, sites: Sequence[Site]) -> tuple[str, tuple[int, ...], int | None]:
    """Catalyst atoms bonded to one adsorbate atom, and the listed site they form (if any)."""
    cat = np.where(atoms.arrays["role"] == ROLE_CATALYST)[0]
    d = _dmat(atoms, np.array([atom_index]), cat)[0]
    sym = atoms.get_chemical_symbols()
    bonded = tuple(sorted(int(c) for c, dist in zip(cat, d)
                          if dist < BOND_SCALE * (_radius(sym[atom_index]) + _radius(sym[c]))))
    if not bonded:
        return "none", bonded, None
    for s in sites:
        if tuple(sorted(s.indices)) == bonded:
            return s.kind, bonded, s.id
    return {1: "top", 2: "bridge"}.get(len(bonded), f"cn{len(bonded)}"), bonded, None


def analyze(atoms: Atoms, reference: Atoms, sites: Sequence[Site]) -> Analysis:
    role = atoms.arrays["role"]
    sym = atoms.get_chemical_symbols()
    ads = np.where(role == ROLE_ADSORBATE)[0]
    cat = np.where(role != ROLE_ADSORBATE)[0]
    flags: list[str] = []
    failure = False

    d_all = atoms.get_all_distances(mic=True)
    rsum = np.add.outer([_radius(s) for s in sym], [_radius(s) for s in sym])
    close = (d_all < OVERLAP_SCALE * rsum) & ~np.eye(len(atoms), dtype=bool)
    if close.any():
        flags.append("atom_overlap")
        failure = True

    fragments = []
    for members in _fragments(atoms, ads):
        symbols = [sym[m] for m in members]
        name = _fragment_name(symbols, atoms, members)
        d_fc = _dmat(atoms, np.array(members), cat)
        limit = DESORB_SCALE * np.add.outer([_radius(s) for s in symbols], [_radius(sym[c]) for c in cat])
        desorbed = not (d_fc < limit).any()
        # binding atom = the fragment atom closest to the catalyst
        binder = members[int(np.argmin(d_fc.min(axis=1)))]
        kind, bonded, site_id = observed_site(atoms, binder, sites)
        fragments.append({"name": name, "atoms": members, "mol_ids": sorted({int(atoms.arrays["mol_id"][m]) for m in members}),
                          "binding_atom": int(binder), "binding_element": sym[binder], "site_kind": kind,
                          "site_atoms": list(bonded), "site_id": site_id, "desorbed": bool(desorbed)})
        if desorbed:
            flags.append(f"desorbed_{name}")
    names = sorted(f["name"] for f in fragments)
    topology = "+".join(names)
    if "CO" not in names and "C" in sym and "O" in sym:
        c = [i for i in ads if sym[i] == "C"]
        o = [i for i in ads if sym[i] == "O"]
        dco = _dmat(atoms, np.array(c), np.array(o)).min()
        flags.append("co_dissociated" if dco > CO_INTACT_MAX and not any("C" in n and "O" in n for n in names)
                     else "co_reacted")
    for f in fragments:
        if f["name"] == "CO":
            c = next(m for m in f["atoms"] if sym[m] == "C")
            o = next(m for m in f["atoms"] if sym[m] == "O")
            f["d_CO"] = float(atoms.get_distance(c, o, mic=True))
            if f["d_CO"] > 1.30:
                flags.append("co_stretched")
            if f["binding_element"] == "O":
                flags.append("co_o_bound")
    if "H2" in names:
        flags.append("h2_formed")
    h_sites = [tuple(f["site_atoms"]) for f in fragments if f["name"] == "H" and f["site_atoms"]]
    if len(h_sites) != len(set(h_sites)):
        flags.append("h_pooling")

    n0 = len(reference)
    disp = np.linalg.norm(atoms.positions[:n0] - reference.positions, axis=1)
    cat_ref = np.where(role[:n0] == ROLE_CATALYST)[0]
    max_disp = float(disp[cat_ref].max()) if len(cat_ref) else 0.0
    if max_disp > 0.5:
        flags.append("catalyst_rearranged")
    d_ac = _dmat(atoms, ads, cat)
    motif = "|".join(sorted(f"{f['name']}@{'gas' if f['desorbed'] else f['binding_element'] + '-' + f['site_kind']}"
                            for f in fragments))
    return Analysis(topology, motif, fragments, sorted(set(flags)), failure, max_disp, float(d_ac.min()))


def observed_genes(atoms: Atoms, analysis: Analysis, intended: Sequence[Gene], sites: Sequence[Site]) -> list[Gene] | None:
    """Genes re-read from a relaxed structure, or None when it no longer maps onto the gene model.

    Only intact, adsorbed CO + separate H on listed sites map back; anything else
    (dissociated, reacted, desorbed, off-list site) is kept as a result but is
    not used as a parent.
    """
    by_mol = {}
    for f in analysis.fragments:
        if f["desorbed"] or f["site_id"] is None or len(f["mol_ids"]) != 1 or f["name"] not in ("CO", "H"):
            return None
        by_mol[f["mol_ids"][0]] = f
    if len(by_mol) != len(intended):
        return None
    genes = []
    for k, g in enumerate(intended):
        f = by_mol[k]
        site = sites[f["site_id"]]
        if g.species == "H":
            genes.append(Gene("H", site.id))
            continue
        c = next(m for m in f["atoms"] if atoms.symbols[m] == "C")
        o = next(m for m in f["atoms"] if atoms.symbols[m] == "O")
        bound, far = (o, c) if f["binding_element"] == "O" else (c, o)
        axis = find_mic(atoms.positions[far] - atoms.positions[bound], atoms.cell, atoms.pbc)[0]
        axis /= np.linalg.norm(axis)
        t1, t2 = _tangent_basis(site, atoms)
        tilt = float(np.degrees(np.arccos(np.clip(axis @ site.n, -1, 1))))
        azimuth = float(np.degrees(np.arctan2(axis @ t2, axis @ t1))) % 360.0
        mode = "O" if f["binding_element"] == "O" else ("side" if tilt > 50 else "C")
        genes.append(Gene("CO", site.id, mode, tilt, azimuth))
    if len({g.site for g in genes}) != len(genes):
        return None
    return genes


# --------------------------------------------------------------------------- duplicates


def fingerprint(atoms: Atoms, *, include_catalyst: bool = False) -> dict[str, np.ndarray]:
    """Sorted distance lists per pair class: invariant to rotation, translation and
    to permuting identical atoms. For an isolated cluster only."""
    role = atoms.arrays["role"]
    sym = np.array(atoms.get_chemical_symbols())
    ads = np.where(role == ROLE_ADSORBATE)[0]
    cat = np.where(role != ROLE_ADSORBATE)[0]
    fp: dict[str, np.ndarray] = {}
    for el in sorted(set(sym[ads])):
        sel = ads[sym[ads] == el]
        fp[f"{el}-cat"] = np.sort(_dmat(atoms, sel, cat).ravel())
    els = sorted(set(sym[ads]))
    for i, a in enumerate(els):
        for b in els[i:]:
            sa, sb = ads[sym[ads] == a], ads[sym[ads] == b]
            d = _dmat(atoms, sa, sb)
            vals = d[np.triu_indices(len(sa), k=1)] if a == b else d.ravel()
            if len(vals):
                fp[f"{a}-{b}"] = np.sort(vals)
    if include_catalyst:
        d = _dmat(atoms, cat, cat)
        fp["cat-cat"] = np.sort(d[np.triu_indices(len(cat), k=1)])
    return fp


def fingerprint_distance(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    if a.keys() != b.keys() or any(len(a[k]) != len(b[k]) for k in a):
        return float("inf")
    return max((float(np.abs(a[k] - b[k]).max()) for k in a), default=0.0)


def fixed_frame_distance(a: Atoms, b: Atoms) -> float:
    """Largest atom displacement between two structures in the same frame, after the
    best assignment of identical adsorbate atoms (no rotation or translation)."""
    from scipy.optimize import linear_sum_assignment

    if len(a) != len(b) or a.get_chemical_symbols() != b.get_chemical_symbols():
        return float("inf")
    role = a.arrays["role"]
    fixed = np.where(role != ROLE_ADSORBATE)[0]
    worst = 0.0
    if len(fixed):
        worst = float(np.linalg.norm(find_mic(a.positions[fixed] - b.positions[fixed], a.cell, a.pbc)[0], axis=1).max())
    sym = np.array(a.get_chemical_symbols())
    ads = np.where(role == ROLE_ADSORBATE)[0]
    for el in set(sym[ads]):
        sel = ads[sym[ads] == el]
        cost = get_distances(a.positions[sel], b.positions[sel], cell=a.cell, pbc=a.pbc)[1]
        rows, cols = linear_sum_assignment(cost)
        worst = max(worst, float(cost[rows, cols].max()))
    return worst


def same_structure(a: Atoms, b: Atoms, *, tol: float = 0.15, frame: str = "auto", include_catalyst: bool = False) -> bool:
    """``frame="free"``: isolated cluster, rigid motions removed. ``frame="fixed"``:
    supported/periodic system, where a rigid rotation is a different structure.
    Magnetic branch is not part of this test: compare only within one (charge, M) label."""
    if frame == "auto":
        frame = "fixed" if any(a.pbc) or (a.arrays["role"] == ROLE_SUPPORT).any() else "free"
    if frame == "fixed":
        return fixed_frame_distance(a, b) < tol
    return fingerprint_distance(fingerprint(a, include_catalyst=include_catalyst),
                                fingerprint(b, include_catalyst=include_catalyst)) < tol


# --------------------------------------------------------------------------- relaxation


class CallCounter(Calculator):
    """Wraps a calculator and counts energy/force evaluations."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, inner: Any):
        super().__init__()
        self.inner = inner
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        bare = atoms.copy()
        bare.set_constraint()
        bare.calc = self.inner
        self.results = {"energy": float(bare.get_potential_energy()), "forces": np.array(bare.get_forces())}


@dataclass
class RelaxResult:
    converged: bool
    steps: int
    calls: int
    energy: float
    fmax: float
    wall_s: float
    error: str | None = None


def relax(atoms: Atoms, counter: CallCounter, *, fmax: float, max_steps: int, free_catalyst: bool = False,
          maxstep: float = 0.1) -> RelaxResult:
    """BFGS relaxation. Stage A: every non-adsorbate atom fixed. Stage B
    (``free_catalyst``): catalyst atoms move, support atoms stay fixed."""
    from ase.optimize import BFGS

    role = atoms.arrays["role"]
    fixed = np.where(role == ROLE_SUPPORT)[0] if free_catalyst else np.where(role != ROLE_ADSORBATE)[0]
    atoms.set_constraint(FixAtoms(indices=fixed) if len(fixed) else None)
    atoms.calc = counter
    calls0, t0 = counter.calls, time.perf_counter()
    try:
        opt = BFGS(atoms, logfile=None, maxstep=maxstep)
        converged = bool(opt.run(fmax=fmax, steps=max_steps))
        forces = atoms.get_forces()
        result = RelaxResult(converged, int(opt.nsteps), counter.calls - calls0, float(atoms.get_potential_energy()),
                             float(np.linalg.norm(forces, axis=1).max()), time.perf_counter() - t0)
    except Exception as exc:  # calculator blow-up: keep the attempt in the ledger
        result = RelaxResult(False, 0, counter.calls - calls0, float("nan"), float("nan"),
                             time.perf_counter() - t0, f"{type(exc).__name__}: {exc}")
    atoms.calc = None
    return result


# --------------------------------------------------------------------------- search


@dataclass
class SearchConfig:
    arm: str = "random"  # "random" | "evolutionary"
    seed: int = 0
    budget: int = 200  # relaxations started
    composition: dict[str, int] = field(default_factory=lambda: {"CO": 1, "H": 2})
    population: int = 20  # evolutionary arm: random initial candidates and population cap
    niche_cap: int = 4  # at most this many population members per motif
    tournament: int = 3
    operators: dict[str, float] = field(default_factory=lambda: {
        "move_H": 0.30, "move_CO": 0.20, "rotate_CO": 0.15, "recombine": 0.25, "immigrant": 0.10})
    co_mode_weights: dict[str, float] = field(default_factory=lambda: {"C": 0.80, "side": 0.15, "O": 0.05})
    fmax: float = 0.05
    max_steps: int = 200
    free_catalyst: bool = False  # Stage B
    dedup_tol: float = 0.15
    dedup_energy_tol: float = 0.02  # eV
    frame: str = "auto"
    max_proposals_factor: int = 50  # stop after budget * factor proposals (guards against a stuck arm)


# A batch evaluator relaxes a list of start structures (role/mol_id arrays set) and returns,
# in the same order, the relaxed structure and its RelaxResult. It must apply the same
# constraints and convergence rule as :func:`relax`.
BatchEvaluator = Callable[[list[Atoms]], list[tuple[Atoms, RelaxResult]]]


def native_evaluator(calculator: Any, *, fmax: float, max_steps: int, free_catalyst: bool = False) -> BatchEvaluator:
    """The native fallback behind the batch contract: ASE BFGS, one structure at a time."""
    counter = CallCounter(calculator)

    def evaluate(structures: list[Atoms]) -> list[tuple[Atoms, RelaxResult]]:
        out = []
        for atoms in structures:
            final = atoms.copy()
            out.append((final, relax(final, counter, fmax=fmax, max_steps=max_steps, free_catalyst=free_catalyst)))
        return out

    return evaluate


def geometry_sha(atoms: Atoms) -> str:
    text = ";".join(f"{s}:{x:.5f},{y:.5f},{z:.5f}" for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class AdsorbateSearch:
    """One arm, one seed. ``calculator`` is any ASE calculator; every call is counted.

    Budget is counted in relaxations started. Proposals rejected before
    relaxation (distance filter, duplicate start) cost no calculator call and
    are kept in the ledger with their reason.
    """

    def __init__(self, catalyst: Atoms, sites: Sequence[Site], calculator: Any, config: SearchConfig,
                 *, clock: Callable[[], float] = time.perf_counter):
        if config.arm not in ("random", "evolutionary"):
            raise ValueError(f"unknown arm {config.arm!r}")
        self.catalyst = catalyst.copy()
        if "role" not in self.catalyst.arrays:
            self.catalyst.set_array("role", np.full(len(catalyst), ROLE_CATALYST))
        self.sites = list(sites)
        self.cfg = config
        self.counter = CallCounter(calculator)
        self.rng = np.random.default_rng(config.seed)
        self.records: list[dict[str, Any]] = []
        self.structures: dict[str, tuple[Atoms, Atoms | None]] = {}
        self._starts: list[Atoms] = []
        self._unique: list[tuple[str, Atoms, float]] = []
        self._population: list[str] = []
        self._neighbors = {s.id: [t.id for t in self.sites if t.id != s.id and set(t.indices) & set(s.indices)]
                           for s in self.sites}
        self._clock = clock
        self._t0 = clock()
        self._pending = 0  # proposals accepted for a batch but not yet recorded
        self._external_calls = 0  # calculator calls made by a batch evaluator

    # ---- proposals

    def _random_co(self, site: int) -> Gene:
        modes = list(self.cfg.co_mode_weights)
        w = np.array([self.cfg.co_mode_weights[m] for m in modes], dtype=float)
        mode = str(self.rng.choice(modes, p=w / w.sum()))
        tilt = float(self.rng.uniform(55, 85)) if mode == "side" else float(self.rng.uniform(0, 30))
        return Gene("CO", site, mode, tilt, float(self.rng.uniform(0, 360)))

    def random_genes(self) -> list[Gene]:
        n = sum(self.cfg.composition.values())
        chosen = [int(i) for i in self.rng.choice(len(self.sites), size=n, replace=False)]
        genes: list[Gene] = []
        for species in sorted(self.cfg.composition):
            for _ in range(self.cfg.composition[species]):
                site = chosen.pop()
                genes.append(self._random_co(site) if species == "CO" else Gene(species, site))
        return genes

    def _free_site(self, genes: Sequence[Gene], moving: int) -> int:
        taken = {g.site for i, g in enumerate(genes) if i != moving}
        local = [s for s in self._neighbors[genes[moving].site] if s not in taken]
        pool = local if local and self.rng.random() < 0.5 else [s.id for s in self.sites
                                                                 if s.id not in taken and s.id != genes[moving].site]
        return int(self.rng.choice(pool))

    def _select(self) -> dict[str, Any]:
        by_id = {r["candidate_id"]: r for r in self.records}
        picks = self.rng.choice(len(self._population), size=min(self.cfg.tournament, len(self._population)), replace=False)
        return min((by_id[self._population[int(i)]] for i in picks), key=lambda r: r["energy"])

    def evolutionary_genes(self) -> tuple[list[Gene], str, list[str]]:
        ops = list(self.cfg.operators)
        w = np.array([self.cfg.operators[o] for o in ops], dtype=float)
        op = str(self.rng.choice(ops, p=w / w.sum()))
        if op == "immigrant" or not self._population:
            return self.random_genes(), "immigrant", []
        pa = self._select()
        genes = [Gene(**g) for g in pa["parent_genes"]]
        if op in ("move_H", "move_CO"):
            species = op.split("_")[1]
            idx = [i for i, g in enumerate(genes) if g.species == species]
            i = int(self.rng.choice(idx))
            site = self._free_site(genes, i)
            g = genes[i]
            genes[i] = Gene(g.species, site, g.mode, g.tilt, g.azimuth)
            return genes, op, [pa["candidate_id"]]
        if op == "rotate_CO":
            i = next(i for i, g in enumerate(genes) if g.species == "CO")
            genes[i] = self._random_co(genes[i].site)
            return genes, op, [pa["candidate_id"]]
        # recombine: CO from parent A, the other adsorbates from parent B
        pb = self._select()
        other = [Gene(**g) for g in pb["parent_genes"]]
        child = [g for g in genes if g.species == "CO"]
        taken = {g.site for g in child}
        for g in other:
            if g.species == "CO":
                continue
            site = g.site
            if site in taken:
                free = [s.id for s in self.sites if s.id not in taken]
                site = int(self.rng.choice(free))
            taken.add(site)
            child.append(Gene(g.species, site, g.mode, g.tilt, g.azimuth))
        child.sort(key=lambda g: g.species)
        return child, "recombine", [pa["candidate_id"], pb["candidate_id"]]

    # ---- one proposal

    def _update_population(self) -> None:
        by_id = {r["candidate_id"]: r for r in self.records}
        ranked = sorted((cid for cid, _, _ in self._unique if by_id[cid]["parent_genes"] is not None
                         and not by_id[cid]["failure"] and by_id[cid]["converged"]),
                        key=lambda cid: by_id[cid]["energy"])
        kept: list[str] = []
        per_motif: dict[str, int] = {}
        for cid in ranked:
            motif = by_id[cid]["motif"]
            if per_motif.get(motif, 0) < self.cfg.niche_cap:
                kept.append(cid)
                per_motif[motif] = per_motif.get(motif, 0) + 1
            if len(kept) == self.cfg.population:
                break
        self._population = kept

    def _propose(self) -> tuple[dict[str, Any], Atoms, list[Gene], bool]:
        """Next proposal, validated and checked against earlier starts. The bool says
        whether it should be relaxed; a rejected proposal is already in the ledger."""
        cfg = self.cfg
        n_relaxed = sum(1 for r in self.records if r["status"] == "relaxed") + self._pending
        if cfg.arm == "random" or n_relaxed < cfg.population:
            genes, op = self.random_genes(), "random"
            parents: list[str] = []
        else:
            genes, op, parents = self.evolutionary_genes()
        proposal = len(self.records) + self._pending
        cid = f"{cfg.arm}-s{cfg.seed}-{proposal:05d}"
        rec: dict[str, Any] = {
            "candidate_id": cid, "arm": cfg.arm, "seed": cfg.seed, "proposal_index": proposal, "operator": op,
            "parent_ids": parents, "genes": [asdict(g) for g in genes],
            "site_atoms": [list(self.sites[g.site].indices) for g in genes],
            "site_kinds": [self.sites[g.site].kind for g in genes],
            "status": "", "reject_reasons": [], "duplicate_of": None, "unique": False, "converged": False,
            "steps": 0, "calc_calls": 0, "energy": None, "fmax": None, "topology": None, "motif": None,
            "flags": [], "failure": False, "parent_genes": None, "catalyst_max_disp": None, "novelty": None,
            "start_sha": None, "final_sha": None,
        }
        start = build_candidate(self.catalyst, self.sites, genes)
        rec["start_sha"] = geometry_sha(start)
        problems = validate(start, self.catalyst, cfg.composition)
        if problems:
            rec["status"], rec["reject_reasons"] = "invalid", problems
        elif any(same_structure(start, s, tol=0.05, frame=cfg.frame) for s in self._starts):
            rec["status"], rec["reject_reasons"] = "duplicate_start", ["duplicate_start"]
        else:
            self._starts.append(start)
            return rec, start, genes, True
        self._close(rec, start, None)
        return rec, start, genes, False

    def _close(self, rec: dict[str, Any], start: Atoms, final: Atoms | None) -> None:
        rec["cum_relaxations"] = sum(1 for r in self.records if r["status"] == "relaxed") + (rec["status"] == "relaxed")
        rec["cum_calc_calls"] = self.counter.calls + self._external_calls
        rec["cum_wall_s"] = self._clock() - self._t0
        self.records.append(rec)
        self.structures[rec["candidate_id"]] = (start, final)
        if rec["unique"]:
            self._update_population()

    def _finish(self, rec: dict[str, Any], start: Atoms, genes: list[Gene], final: Atoms, res: RelaxResult) -> None:
        """Record one relaxation result: analysis, duplicate test, population update."""
        cfg = self.cfg
        cid = rec["candidate_id"]
        rec.update(status="relaxed", converged=res.converged, steps=res.steps, calc_calls=res.calls,
                   energy=None if np.isnan(res.energy) else res.energy,
                   fmax=None if np.isnan(res.fmax) else res.fmax)
        if res.error:
            rec["flags"], rec["failure"] = ["calculator_error"], True
            rec["reject_reasons"] = [res.error]
        else:
            an = analyze(final, self.catalyst, self.sites)
            rec.update(topology=an.topology, motif=an.motif, flags=an.flags, failure=an.failure,
                       catalyst_max_disp=an.catalyst_max_disp, fragments=an.fragments,
                       final_sha=geometry_sha(final))
            if not res.converged:
                rec["flags"] = sorted(set(rec["flags"]) | {"not_converged"})
            og = observed_genes(final, an, genes, self.sites)
            rec["parent_genes"] = None if og is None else [asdict(g) for g in og]
            include_cat = cfg.free_catalyst
            fp = fingerprint(final, include_catalyst=include_cat) if cfg.frame != "fixed" and not any(final.pbc) else None
            dists = []
            for other_id, other, e_other in self._unique:
                if fp is not None:
                    dist = fingerprint_distance(fp, fingerprint(other, include_catalyst=include_cat))
                else:
                    dist = fixed_frame_distance(final, other)
                dists.append(dist)
                if dist < cfg.dedup_tol and abs(e_other - res.energy) < cfg.dedup_energy_tol:
                    rec["duplicate_of"] = other_id
                    break
            finite = [d for d in dists if np.isfinite(d)]
            rec["novelty"] = min(finite) if finite else None
            if rec["duplicate_of"] is None:
                rec["unique"] = True
                self._unique.append((cid, final, res.energy))
        self._close(rec, start, final)

    def step(self) -> dict[str, Any]:
        rec, start, genes, needs_relax = self._propose()
        if needs_relax:
            final = start.copy()
            res = relax(final, self.counter, fmax=self.cfg.fmax, max_steps=self.cfg.max_steps,
                        free_catalyst=self.cfg.free_catalyst)
            self._finish(rec, start, genes, final, res)
        return rec

    def _budget_left(self) -> int:
        return self.cfg.budget - sum(1 for r in self.records if r["status"] == "relaxed")

    def run(self) -> list[dict[str, Any]]:
        limit = self.cfg.budget * self.cfg.max_proposals_factor
        while self._budget_left() > 0 and len(self.records) < limit:
            self.step()
        return self.records

    def run_batched(self, evaluator: "BatchEvaluator", batch_size: int) -> list[dict[str, Any]]:
        """Same search, with relaxations handed to ``evaluator`` in batches.

        Only proposals that do not depend on earlier results are batched: all of the
        random arm, and the random initial population of the evolutionary arm. They
        are proposed, filtered and recorded in the same order as in :meth:`run`, so
        with an evaluator that returns the same result per structure the ledger is
        the same. Evolutionary children are still evaluated one at a time (through
        the same evaluator), because each child is selected from the population the
        previous child left; batching them would be a different algorithm.
        """
        cfg = self.cfg
        limit = cfg.budget * cfg.max_proposals_factor
        while self._budget_left() > 0 and len(self.records) < limit:
            relaxed = cfg.budget - self._budget_left()
            independent = cfg.arm == "random" or relaxed < cfg.population
            room = self._budget_left() if cfg.arm == "random" else min(self._budget_left(), cfg.population - relaxed)
            want = min(batch_size, room) if independent else 1
            held: list[tuple[dict[str, Any], Atoms, list[Gene]]] = []
            attempts = 0
            while len(held) < want and attempts < 50 * want:
                attempts += 1
                rec, start, genes, needs_relax = self._propose()
                if needs_relax:
                    held.append((rec, start, genes))
                    self._pending += 1
            self._pending = 0
            if not held:
                break
            # rejected proposals made while filling the batch were recorded first; keep ledger order by index
            results = evaluator([start.copy() for _, start, _ in held])
            if len(results) != len(held):
                raise RuntimeError(f"evaluator returned {len(results)} results for {len(held)} structures")
            for (rec, start, genes), (final, res) in zip(held, results):
                self._external_calls += res.calls
                self._finish(rec, start, genes, final, res)
            self.records.sort(key=lambda r: r["proposal_index"])
        return self.records

    # ---- output

    def summary(self) -> dict[str, Any]:
        relaxed = [r for r in self.records if r["status"] == "relaxed"]
        good = [r for r in relaxed if r["unique"] and not r["failure"] and r["converged"]]
        return {
            "arm": self.cfg.arm, "seed": self.cfg.seed, "proposals": len(self.records),
            "invalid": sum(r["status"] == "invalid" for r in self.records),
            "duplicate_start": sum(r["status"] == "duplicate_start" for r in self.records),
            "relaxations": len(relaxed), "converged": sum(r["converged"] for r in relaxed),
            "duplicates_after_relax": sum(r["duplicate_of"] is not None for r in relaxed),
            "unique": sum(r["unique"] for r in relaxed), "failures": sum(r["failure"] for r in relaxed),
            "unique_converged_valid": len(good), "motifs": len({r["motif"] for r in good}),
            "topologies": sorted({r["topology"] for r in good}),
            "calc_calls": self.counter.calls, "wall_s": self._clock() - self._t0,
            "best_energy": min((r["energy"] for r in good), default=None),
        }

    def write(self, out_dir: Path) -> None:
        from ase.io import write as ase_write

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for r in self.records:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        frames = []
        for r in self.records:
            start, final = self.structures[r["candidate_id"]]
            for stage, atoms in (("start", start), ("relaxed", final)):
                if atoms is None:
                    continue
                a = atoms.copy()
                a.set_constraint()
                a.info = {"candidate_id": r["candidate_id"], "stage": stage, "status": r["status"],
                          "operator": r["operator"], "seed": r["seed"]}
                if stage == "relaxed" and r["energy"] is not None:
                    a.info["model_energy"] = r["energy"]
                    a.info["motif"] = r["motif"]
                frames.append(a)
        ase_write(out_dir / "structures.extxyz", frames)
        (out_dir / "summary.json").write_text(json.dumps({"config": asdict(self.cfg), **self.summary()}, indent=2,
                                                         sort_keys=True) + "\n", encoding="utf-8")


def site_table(report: SiteReport) -> list[dict[str, Any]]:
    return [{"id": s.id, "kind": s.kind, "indices": list(s.indices), "position": [round(x, 5) for x in s.position],
             "normal": [round(x, 5) for x in s.normal]} for s in report.sites]


# --------------------------------------------------------------------------- toy calculator


class ToyPairCalculator(Calculator):
    """Deterministic Morse pair potential for software checks. No chemistry is claimed:
    it only gives the search something smooth to relax on without a model file."""

    implemented_properties = ["energy", "forces"]
    # (D eV, r0 A, a 1/A) per unordered element pair
    PAIRS = {
        ("Fe", "Fe"): (0.6, 2.45, 1.4), ("Fe", "H"): (0.9, 1.70, 1.8), ("C", "Fe"): (1.2, 1.90, 1.8),
        ("Fe", "O"): (0.5, 2.10, 1.6), ("C", "O"): (11.0, 1.15, 2.3),
    }
    DEFAULT = (0.05, 2.6, 1.2)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        sym = atoms.get_chemical_symbols()
        vec, dist = get_distances(atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
        energy, forces = 0.0, np.zeros((len(atoms), 3))
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                d0, r0, a = self.PAIRS.get(tuple(sorted((sym[i], sym[j]))), self.DEFAULT)
                x = np.exp(-a * (dist[i, j] - r0))
                energy += d0 * (x * x - 2 * x)
                pull = d0 * (-2 * a * x * x + 2 * a * x) * vec[i, j] / dist[i, j]  # vec[i, j] = r_j - r_i
                forces[i] += pull
                forces[j] -= pull
        self.results = {"energy": float(energy), "forces": forces}
