"""Per-state spin labels from the legacy Gaussian summary documents (the Oct9 Fe_n series).

Each Word document is a hand-curated record for one cluster size: a title naming the level
(``Fe_13  BPW91/6-311+G*``), then one block per state with ``Charge = q Multiplicity = M``, the geometry
table, the Mulliken spin table, optional frequencies, and the archive line ``HF=...\\S2=...``. Some blocks carry
their own route line (other functionals and basis sets are mixed in); some are vertical-ion summaries
(``IP = ... eV`` plus a bare ``HF=``) with no geometry or spins; some are counterpoise runs with ghost atoms.

``parse_document_states`` turns the text into one :class:`LegacyState` per complete block, counts in
``report`` why the other blocks were skipped, and records (without dropping anything) why a state may not be
usable:

* ``level`` / ``level_source``: the block's own route if it has one (``ambiguous`` if its ``$$$`` title names a
  different functional or basis); else a functional/basis named in the block title (``header``); else the
  document title, but only if no differently-leveled route appeared earlier in the file (otherwise ``ambiguous``).
* ``artifact_sites``: sites with |m| > 5 mu_B, beyond the physical range of an Fe atom. In the diffuse
  6-311+G* basis these are Mulliken artifacts (the central atom of compact clusters, cf. Fe16).
* ``scf_flags``: explicit convergence-failure text, a spin sum that differs from M - 1, a large <S^2> excess,
  and ``IOP(5/13=1)`` in the route (which lets an unconverged SCF continue silently).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

HARTREE_TO_EV = 27.211386245988
ARTIFACT_MOMENT_MU_B = 5.0
SPIN_SUM_TOL = 0.05
S2_EXCESS_FLAG = 1.0

_PRINTABLE = re.compile(rb"[\x20-\x7e\t\r\n]{4,}")
_LEVEL_TOKEN = re.compile(r"^U?[A-Z][A-Z0-9-]*/[0-9A-Za-z+*(),-]+$")
_ROUTE = re.compile(r"^\s*#[pPnNtT]?\s*(?P<route>[A-Za-z][A-Za-z0-9-]*/[^\s]+[^\n]*)$", re.M)
_CHARGE = re.compile(r"Charge\s*=\s*(?P<q>-?\d+)\s+Multiplicity\s*=\s*(?P<m>\d+)(?P<rest>[^\n]*)")
# Geometry tables with or without the "... orientation:" title; 6 columns (G03+) or 5 (G98, no "Atomic Type").
_FLOAT = r"-?\d*\.\d+"
_ORIENT = re.compile(
    r"Center\s+Atomic[^\n]*\n[^\n]*Number[^\n]*\n\s*-+\s*\n"
    rf"(?P<body>(?:\s*\d+\s+\d+(?:\s+-?\d+)?\s+{_FLOAT}\s+{_FLOAT}\s+{_FLOAT}\s*\n)+)"
)
_SPIN = re.compile(r"(?:atomic spin densities|charges and spin densities)[^\n]*\n(?P<body>(?:[^\n]*\n){1,120}?)"
                   r"\s*Sum of Mulliken", re.I)
_SPIN_ROW_1 = re.compile(rf"^\s*\d+\s+(?P<sym>[A-Z][a-z]?)\s+(?P<v>{_FLOAT})\s*$")
_SPIN_ROW_2 = re.compile(rf"^\s*\d+\s+(?P<sym>[A-Z][a-z]?)\s+{_FLOAT}\s+(?P<v>{_FLOAT})\s*$")
# Word wraps long archive lines, so "HF= -15166.23493 58\S2=..." means -15166.2349358. Digit groups are only
# re-joined when the archive's \S2 field follows directly, so unrelated text is never glued onto an energy.
_HF_WRAPPED = re.compile(r"HF=\s*(?P<e>-?\d+\.\d+(?:[ \n]\d+)+)\\S2=(?P<s2>\d+\.\d+)")
_HF_PLAIN = re.compile(r"HF=\s*(?P<e>-?\d+\.\d+)(?:\\S2=(?P<s2>\d+\.\d+))?")


def _find_hf(block: str) -> re.Match | None:
    hits = [m for m in (_HF_WRAPPED.search(block), _HF_PLAIN.search(block)) if m]
    return min(hits, key=lambda m: (m.start(), -len(m.group("e")))) if hits else None
# Dimers are often recorded only as an optimized Z-matrix bond length: "! r  2.2617 -DE/DX" or "! R1 R(2,1) 1.83".
_DIMER_R = re.compile(r"!\s+(?:r|R1\s+R\(2,1\))\s+(?P<r>\d+\.\d+)\s+-DE/DX")
# Block titles such as "$$$$  Fe_4-   ANION   S = 16  C1  BLYP" name the functional (and sometimes the basis).
_BLOCK_HEADER = re.compile(r"^\s*\${3,}[^\n]*$", re.M)
_HEADER_METHODS = ("B3LYP", "BLYP", "PBEPBE", "BPBE", "BP86", "LSDA", "SVWN", "PW91PW91", "TPSSTPSS",
                   "REVTPSS", "M06L", "MP2", "B3PW91", "BPW91")
_HEADER_BASES = ("6-311+G*", "6-311G*", "6-311G", "GEN")
_FAIL = re.compile(r"Convergence failure|SCF has not converged|Convergence criterion not met", re.I)
_SYMBOL = {26: "Fe", 1: "H", 6: "C", 7: "N", 8: "O"}


def document_text(raw: bytes) -> str:
    """Printable text of a legacy .doc/.log; Word paragraph marks (\\r) become newlines."""
    return "\n".join(r.decode("ascii") for r in _PRINTABLE.findall(raw)).replace("\r", "\n")


def normalize_level(route_or_level: str) -> str:
    """'UBPW91/6-311+G** scf=(...)' -> 'BPW91/6-311+G*' (G** equals G* for Fe-only systems)."""
    token = route_or_level.strip().split()[0].upper()
    method, _, basis = token.partition("/")
    if method.startswith(("RO", )) and len(method) > 3:
        method = method[2:]
    elif method[:1] in ("U", "R") and len(method) > 2 and method[1] not in "0123456789-":
        method = method[1:]
    basis = basis.replace("G**", "G*").replace("(D,P)", "*").replace("(D)", "*")
    return f"{method}/{basis}"


def document_level(text: str) -> str:
    """The level named in the document's title lines, e.g. 'Fe_4 --- Fe_4- BPW91/6-311+G*'."""
    for line in text.splitlines()[:40]:
        if "#" in line or "=" in line:
            continue
        tokens = [t for t in line.split() if _LEVEL_TOKEN.match(t)]
        if tokens and re.search(r"fe", line, re.I):
            return normalize_level(tokens[-1])
    return ""


def header_tags(header: str) -> tuple[str | None, str | None]:
    """(functional, basis) named in a block title such as '$$$ Fe_4- ANION S = 16 C1 BLYP'; None where silent."""
    tokens = header.upper().split()
    return (next((m for m in _HEADER_METHODS if m in tokens), None),
            next((b for b in _HEADER_BASES if b in tokens), None))


def _conflicts(level: str, tags: tuple[str | None, str | None]) -> bool:
    method, _, basis = level.partition("/")
    return (tags[0] is not None and tags[0] != method) or (tags[1] is not None and tags[1] != basis)


def _spin_values(body: str) -> tuple[list[str], list[float]]:
    syms, vals = [], []
    for line in body.splitlines():
        m = _SPIN_ROW_1.match(line) or _SPIN_ROW_2.match(line)
        if m:
            syms.append(m.group("sym"))
            vals.append(float(m.group("v")))
    return syms, vals


@dataclass
class LegacyState:
    source: str
    block_index: int
    charge: int
    multiplicity: int
    symbols: list[str]
    positions: np.ndarray
    spins: np.ndarray
    energy_ev: float
    s2: float | None
    level: str
    level_source: str  # "route" | "header" | "document" | "ambiguous"
    route: str = ""
    n_imaginary: int | None = None
    scf_flags: list[str] = field(default_factory=list)
    artifact_sites: list[int] = field(default_factory=list)

    @property
    def formula(self) -> str:
        counts: dict[str, int] = {}
        for s in self.symbols:
            counts[s] = counts.get(s, 0) + 1
        return "".join(f"{s}{c if c > 1 else ''}" for s, c in counts.items())

    @property
    def s2_excess(self) -> float | None:
        if self.s2 is None:
            return None
        s = (self.multiplicity - 1) / 2
        return self.s2 - s * (s + 1)

    @property
    def state_id(self) -> str:
        key = np.round(self.positions, 4).tobytes() + f"{self.charge}|{self.multiplicity}|{self.energy_ev:.6f}".encode()
        return hashlib.sha256(key).hexdigest()[:16]

    @property
    def genuine_antiparallel(self) -> bool:
        """An antiparallel site (< -0.5 mu_B) that is not itself an artifact site."""
        return any(v < -0.5 and i not in self.artifact_sites for i, v in enumerate(self.spins))


def parse_document_states(text: str, source: str = "", report: dict | None = None) -> list[LegacyState]:
    """``report``, if given, counts parsed blocks and why the other Charge/Multiplicity blocks were skipped."""
    report = {} if report is None else report

    def skip(why: str) -> None:
        report[why] = report.get(why, 0) + 1

    doc_level = document_level(text)
    charges = list(_CHARGE.finditer(text))
    routes = [(m.start(), normalize_level(m.group("route")), m.group("route").strip()) for m in _ROUTE.finditer(text)]
    states: list[LegacyState] = []
    for k, cm in enumerate(charges):
        end = charges[k + 1].start() if k + 1 < len(charges) else len(text)
        block = text[cm.start():end]
        if "IP =" in cm.group("rest") or "EA =" in cm.group("rest"):
            skip("vertical_ion_summary")
            continue
        orients = list(_ORIENT.finditer(block))
        spin = _SPIN.search(block)
        hf = _find_hf(block)
        syms, vals = _spin_values(spin.group("body")) if spin else ([], [])
        dimer = _DIMER_R.findall(block) if not orients and len(vals) == 2 else []
        if (not orients and not dimer) or not spin or not hf:
            missing = [n for n, x in (("geometry", orients or dimer), ("spins", spin), ("energy", hf)) if not x]
            skip("missing_" + "_".join(missing))
            continue
        if orients:
            before = [o for o in orients if o.start() < spin.start()]
            geo = before[-1] if before else orients[0]
            rows = [line.split() for line in geo.group("body").strip().splitlines() if line.strip()]
        else:  # last optimized bond length of a homonuclear dimer, from the spin table's element
            z = {v: k for k, v in _SYMBOL.items()}.get(syms[0], 0)
            rows = [["1", str(z), "0.0", "0.0", "0.0"], ["2", str(z), "0.0", "0.0", dimer[-1]]]
        numbers = [int(r[1]) for r in rows]
        if any(n not in _SYMBOL for n in numbers) or "Bq" in syms:
            skip("ghost_or_unknown_atoms")
            continue
        if len(vals) != len(numbers) or syms != [_SYMBOL[n] for n in numbers]:
            skip("spin_table_does_not_match_geometry")
            continue
        positions = np.array([[float(x) for x in r[-3:]] for r in rows])
        spins = np.array(vals)
        prev_start = charges[k - 1].start() if k else 0
        own = [r for r in routes if prev_start <= r[0] < cm.start()]
        earlier = [r for r in routes if r[0] < cm.start()]
        headers = [h for h in _BLOCK_HEADER.finditer(text, prev_start, cm.start())]
        tags = header_tags(headers[-1].group()) if headers else (None, None)
        d_method, _, d_basis = doc_level.partition("/")
        if own:
            level, level_source, route = own[-1][1], "route", own[-1][2]
            if _conflicts(level, tags):
                level_source = "ambiguous"  # the block's title and its own route disagree
        elif tags != (None, None):
            method, basis = tags[0] or d_method, tags[1] or d_basis
            level, route = f"{method}/{basis}", earlier[-1][2] if earlier else ""
            level_source = "header" if method and basis else "ambiguous"
        elif earlier and earlier[-1][1] != doc_level:
            level, level_source, route = earlier[-1][1], "ambiguous", earlier[-1][2]
        else:
            level, level_source, route = doc_level, "document", earlier[-1][2] if earlier else ""
        m = int(cm.group("m"))
        st = LegacyState(source, k, int(cm.group("q")), m, [_SYMBOL[n] for n in numbers], positions, spins,
                         float(re.sub(r"\s", "", hf.group("e"))) * HARTREE_TO_EV,
                         float(hf.group("s2")) if hf.group("s2") else None,
                         level, level_source, route)
        if _FAIL.search(block):
            st.scf_flags.append("convergence_failure_text")
        if abs(spins.sum() - (m - 1)) > SPIN_SUM_TOL:
            st.scf_flags.append("spin_sum_mismatch")
        if st.s2_excess is not None and st.s2_excess > S2_EXCESS_FLAG:
            st.scf_flags.append("s2_excess")
        if "5/13=1" in route.replace(" ", ""):
            st.scf_flags.append("iop_5_13")
        freqs = [float(x) for line in re.findall(r"Frequencies --([^\n]*)", block)
                 for x in line.split() if re.fullmatch(_FLOAT, x)]
        if freqs:
            st.n_imaginary = sum(1 for f in freqs if f < 0)
        st.artifact_sites = [i for i, v in enumerate(spins) if abs(v) > ARTIFACT_MOMENT_MU_B]
        states.append(st)
        skip("parsed")
    return states


def deduplicate(states: list[LegacyState], energy_tol_ev: float = 1e-4, fingerprint_tol: float = 1e-2) -> list[LegacyState]:
    """Drop repeated blocks (the same state pasted twice, e.g. 'similar to the above')."""
    def fp(s: LegacyState) -> np.ndarray:
        d = np.linalg.norm(s.positions[:, None] - s.positions[None], axis=-1)
        return np.sort(d[np.triu_indices(len(d), 1)])

    kept: list[LegacyState] = []
    for s in states:
        if not any(o.formula == s.formula and o.charge == s.charge and o.multiplicity == s.multiplicity
                   and abs(o.energy_ev - s.energy_ev) <= energy_tol_ev
                   and float(np.abs(fp(o) - fp(s)).max()) <= fingerprint_tol for o in kept):
            kept.append(s)
    return kept
