from __future__ import annotations

import math
from typing import TypedDict, cast

from .models import Atom, Record

# Cordero et al., "Covalent radii revisited", Dalton Trans., 2008, 2832-2838 --
# the same single-bond radii ase.data.covalent_radii is built from. Mn/Fe/Co
# have distinct low-/high-spin values in the source; this warehouse is
# specifically about high-spin Fe clusters (README: ladders up to
# multiplicity 29), so the high-spin radius is used for Fe/Co/Mn to avoid
# systematically under-connecting exactly the atoms this project cares about
# most. An element missing from this table never gets a guessed radius --
# bonding_graph() returns None instead, same "don't guess silently" rule as
# state_inference.
COVALENT_RADII_ANGSTROM: dict[str, float] = {
    "H": 0.31, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.61, "Fe": 1.52, "Co": 1.50, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Rb": 2.20,
    "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47,
    "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42,
    "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Cs": 2.44, "Ba": 2.15,
    "W": 1.62, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Pb": 1.46, "Bi": 1.48,
}

DEFAULT_BONDING_TOLERANCE = 1.2
DEFAULT_HIGH_SPIN_THRESHOLD = 5


def covalent_radius(symbol: str) -> float | None:
    return COVALENT_RADII_ANGSTROM.get(symbol)


def bonding_graph(
    atoms: list[Atom], tolerance: float = DEFAULT_BONDING_TOLERANCE
) -> dict[int, set[int]] | None:
    """Bonded-neighbor adjacency (indices into `atoms`): i-j bonded iff
    distance(i, j) <= tolerance * (r_cov[i] + r_cov[j]).

    Returns None if any atom's element has no tabulated covalent radius --
    callers must treat that as "cannot classify", never guess a fallback
    radius.
    """
    radii = [covalent_radius(atom.symbol) for atom in atoms]
    if any(r is None for r in radii):
        return None
    graph: dict[int, set[int]] = {i: set() for i in range(len(atoms))}
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dx = atoms[i].x - atoms[j].x
            dy = atoms[i].y - atoms[j].y
            dz = atoms[i].z - atoms[j].z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            cutoff = tolerance * (cast(float, radii[i]) + cast(float, radii[j]))
            if distance <= cutoff:
                graph[i].add(j)
                graph[j].add(i)
    return graph


def connected_components(graph: dict[int, set[int]]) -> list[set[int]]:
    seen: set[int] = set()
    components: list[set[int]] = []
    for start in graph:
        if start in seen:
            continue
        stack = [start]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(graph[node] - component)
        seen |= component
        components.append(component)
    return components


def coordination_class(atoms: list[Atom], tolerance: float = DEFAULT_BONDING_TOLERANCE) -> str:
    """Frame-level label from the *worst* (lowest-degree) atom, since that's
    the atom whose force an MLIP is most likely to fit poorly -- matching the
    original "low-coordination" concern from periodic surface/interface work.
    """
    graph = bonding_graph(atoms, tolerance)
    if not graph:
        return "unclassified"
    min_degree = min(len(neighbors) for neighbors in graph.values())
    if min_degree <= 1:
        return "low_coordination"
    if min_degree <= 3:
        return "typical"
    return "well_coordinated"


def compactness_class(atoms: list[Atom], tolerance: float = DEFAULT_BONDING_TOLERANCE) -> str:
    graph = bonding_graph(atoms, tolerance)
    if not graph:
        return "unclassified"
    return "compact" if len(connected_components(graph)) == 1 else "fragmenting"


_SADDLE_TYPES = {"transition_state", "first_order_saddle", "higher_order_saddle"}
_IRC_TYPES = {"irc_forward", "irc_reverse", "irc_checkpoint", "irc_input_seed"}


def pes_region(config_type: str) -> str:
    base = config_type[: -len("_rattled")] if config_type.endswith("_rattled") else config_type
    if base == "minimum":
        return "minimum"
    if base in _SADDLE_TYPES:
        return "saddle"
    if base in _IRC_TYPES:
        return "irc"
    return "other"


def displacement_class(config_type: str) -> str:
    return "rattled" if config_type.endswith("_rattled") else "relaxed"


def charge_spin_class(
    charge: int, multiplicity: int, high_spin_threshold: int = DEFAULT_HIGH_SPIN_THRESHOLD
) -> str:
    if charge != 0:
        return "charged"
    if multiplicity <= 1:
        return "neutral_closed_shell"
    if multiplicity < high_spin_threshold:
        return "neutral_open_shell_low"
    return "neutral_open_shell_high"


def provenance_tier(record: Record) -> str:
    tag = record.metadata.get("state_inference", "")
    if tag == "":
        return "validated"
    if tag == "filename":
        return "filename_derived"
    return "fallback_guess"


class Strata(TypedDict):
    pes_region: str
    displacement_class: str
    coordination_class: str
    compactness_class: str
    charge_spin_class: str
    provenance_tier: str


STRATA_FIELDS: tuple[str, ...] = (
    "pes_region", "displacement_class", "coordination_class",
    "compactness_class", "charge_spin_class", "provenance_tier",
)


def classify_record(
    record: Record,
    *,
    bonding_tolerance: float = DEFAULT_BONDING_TOLERANCE,
    high_spin_threshold: int = DEFAULT_HIGH_SPIN_THRESHOLD,
) -> Strata:
    """The single source of truth for "what kind of frame is this" --
    reused by grouped_split (stratified splitting), label_report and
    evaluate (per-stratum reporting), and physical_checks (which axis a
    check applies to). Same "one function, reused everywhere" pattern as
    models.geometry_signature().
    """
    return {
        "pes_region": pes_region(record.config_type),
        "displacement_class": displacement_class(record.config_type),
        "coordination_class": coordination_class(record.atoms, bonding_tolerance),
        "compactness_class": compactness_class(record.atoms, bonding_tolerance),
        "charge_spin_class": charge_spin_class(record.charge, record.multiplicity, high_spin_threshold),
        "provenance_tier": provenance_tier(record),
    }


def strata_value(strata: Strata, field: str) -> str:
    """Dynamic-key read of a Strata TypedDict. Field names routinely come
    from a runtime --stratify-by list rather than a static literal, so plain
    strata[field] indexing doesn't type-check against a TypedDict; this
    contains the one cast the whole pipeline needs for that.
    """
    return cast(dict[str, str], strata)[field]


def stratum_key(strata: Strata, fields: tuple[str, ...]) -> str:
    return "|".join(strata_value(strata, field) for field in fields)
