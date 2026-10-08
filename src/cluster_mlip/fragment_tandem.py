"""Gaussian tandem fragment-guess inputs for fixed-geometry broken-symmetry (BS) work.

The Link1 structure is copied from the archived 2019 AFM calculations (LG_Calcs,
G09 D.01, 27 unique fragment jobs; see docs/fragment-tandem-method.md):

  link 0   %chk=NAME
           # <level> NoSymm SP IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine Guess=(Fragment=N)
           Q M q1 m1 ... qN mN        (signed m: negative = unpaired electrons start beta)
           El(Fragment=k)  x  y  z
  link 1   %chk=NAME                  (same file, no %oldchk)
           # <level> NoSymm <job> Geom=Checkpoint Guess=Read
           Q M                        (no coordinates)

In every archived log, link 0 ran one SCF per fragment, combined the fragment
orbitals into the checkpoint and ran no supermolecule SCF. Link 1 read that guess:
its initial <S**2> matched the ideal value of the fragment pattern (3.999 for
Fe(S=2) up / Fe(S=2) down).

What this module changes, and why (each is a documented deviation):
  * link 1 is ``Stable=Opt`` at fixed geometry instead of the archived ``Opt``.
    The archive has no Stable keyword anywhere; this follows Gaussian's AFC example
    (gaussian.com/afc), which applies Stable=Opt to the read fragment guess.
  * an optional link 2, ``Force Geom=Checkpoint Guess=Read``, labels forces on the
    stability-optimized state; its SCF must reproduce the link-1 energy.
  * IOP(5/13=1) stays in every link, as archived. Without it these Fe SCFs very
    often abort (user, from the group's experience, 2026-10-08), even though
    Gaussian describes it as optional. The cost is that an unconverged SCF continues
    silently, so parse_tandem_log flags every "Convergence criterion not met" in the
    labelled links; such a job is never "ok" and never becomes a label.

Fixed syntax (stage keywords, checkpoint continuity, charge/multiplicity layout)
lives in this module. The scientific settings (functional, basis, SCF options,
grid, resources) are passed in as a LevelOfTheory.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .gaussian import ATOMIC_SYMBOLS, parse_force_frames, parse_hirshfeld_table

SYMBOL_TO_Z = {symbol: z for z, symbol in ATOMIC_SYMBOLS.items()}

# --------------------------------------------------------------------------- level


@dataclass(frozen=True)
class LevelOfTheory:
    """Scientific settings shared by every link. They are not Gaussian control syntax."""

    method: str = "UBPW91"
    basis: str = "6-311++G*"
    scf: str = "SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc)"
    grid: str = "Int=UltraFine"
    print_iops: str = "5/36=1,8/11=1"  # archived print options; not SCF control

    def common(self) -> str:
        return f"{self.method}/{self.basis} {self.scf} NoSymm"


ARCHIVED_LEVEL = LevelOfTheory()

# Fixed per-stage syntax. Changing any of these changes execution semantics.
STAGE_FRAGMENT = "fragment_init"
STAGE_STABLE = "stable_opt"
STAGE_FORCE = "force"
STAGE_CONTROL_STABLE = "control_stable_opt"
# Continue past an SCF that misses convergence instead of aborting (archived, every link).
CONTINUE_SCF_IOP = "5/13=1"
# prepare-spins --fragment-layout tandem: link 1 runs the campaign job (Opt, Opt(TS), ...)
# on the read fragment guess, as every archived job did, instead of Stable=Opt.
STAGE_CAMPAIGN = "campaign_job"


def _iop(level: LevelOfTheory) -> str:
    parts = [CONTINUE_SCF_IOP] + [p for p in level.print_iops.split(",") if p]
    return f"IOP({','.join(parts)})"


def fragment_route(level: LevelOfTheory, n_fragments: int, guess_only: bool = False) -> str:
    guess = f"Guess=(Fragment={n_fragments},Only)" if guess_only else f"Guess=(Fragment={n_fragments})"
    return f"#p {level.common()} SP {_iop(level)} {level.grid} {guess}"


def stable_route(level: LevelOfTheory, from_checkpoint: bool = True) -> str:
    tail = " Geom=Checkpoint Guess=Read" if from_checkpoint else ""
    return f"#p {level.common()} Stable=Opt Pop=Hirshfeld {_iop(level)} {level.grid}{tail}"


def force_route(level: LevelOfTheory) -> str:
    return f"#p {level.common()} Force Pop=Hirshfeld {_iop(level)} {level.grid} Geom=Checkpoint Guess=Read"


# ------------------------------------------------------------------ fragment plan


@dataclass(frozen=True)
class Fragment:
    atoms: tuple[int, ...]  # 1-based atom indices
    charge: int
    multiplicity: int  # signed; negative = unpaired electrons start in beta orbitals

    @property
    def unpaired(self) -> int:
        return abs(self.multiplicity) - 1

    @property
    def signed_unpaired(self) -> int:
        return self.unpaired if self.multiplicity > 0 else -self.unpaired


@dataclass(frozen=True)
class FragmentPlan:
    symbols: tuple[str, ...]
    charge: int
    multiplicity: int
    fragments: tuple[Fragment, ...]
    electrons: tuple[int, ...]  # per fragment: sum(Z) - q

    @property
    def atom_fragment(self) -> dict[int, int]:
        return {atom: k for k, frag in enumerate(self.fragments, start=1) for atom in frag.atoms}

    @property
    def n_alpha_unpaired(self) -> int:
        return sum(f.unpaired for f in self.fragments if f.multiplicity > 0)

    @property
    def n_beta_unpaired(self) -> int:
        return sum(f.unpaired for f in self.fragments if f.multiplicity < 0)

    @property
    def open_shell_fragments(self) -> int:
        return sum(1 for f in self.fragments if f.unpaired)

    @property
    def ideal_guess_s2(self) -> float:
        """<S**2> of the combined guess if fragment magnetic orbitals do not overlap.

        One determinant with Sz=(M-1)/2 and n_beta unpaired beta electrons has
        <S**2> = Sz(Sz+1) + n_beta. Archived link-1 guesses: 3.97-4.03 against 4.0
        for Fe(0,5)/Fe(0,-5), and 5.005-5.009 against 5.0 for Fe(+1,6)/Fe(+1,-6).
        """
        sz = (self.multiplicity - 1) / 2
        return sz * (sz + 1) + self.n_beta_unpaired

    @property
    def guess_s2_tolerance(self) -> float:
        return 0.15 + 0.05 * self.open_shell_fragments

    def charge_multiplicity_line(self) -> str:
        return " ".join([f"{self.charge} {self.multiplicity}"] +
                        [f"{f.charge} {f.multiplicity}" for f in self.fragments])


def _strict_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    return value


def validate_fragment_plan(
    symbols: Sequence[str], charge: int, multiplicity: int, fragments: Iterable[Fragment],
) -> FragmentPlan:
    """Reject any fragment guess that Gaussian could not build as specified.

    N_e(fragment) = sum(Z_i) - q_fragment. Fragment charges set the electron
    count of the initial guess only; they are not oxidation states, population
    charges, or constraints on the converged wavefunction.
    """
    charge = _strict_int(charge, "total charge")
    multiplicity = _strict_int(multiplicity, "total multiplicity")
    frags = tuple(fragments)
    if len(frags) < 2:
        raise ValueError("a fragment guess needs at least two fragments")
    unknown = sorted({s for s in symbols if s not in SYMBOL_TO_Z})
    if unknown:
        raise ValueError(f"unknown element symbols {unknown}")
    n_atoms = len(symbols)
    total_electrons = sum(SYMBOL_TO_Z[s] for s in symbols) - charge
    if total_electrons <= 0:
        raise ValueError(f"total charge {charge} leaves {total_electrons} electrons")
    if multiplicity < 1:
        raise ValueError("total multiplicity must be >= 1 (only fragment multiplicities carry a sign)")
    if multiplicity > total_electrons + 1 or (total_electrons + multiplicity - 1) % 2:
        raise ValueError(f"total multiplicity {multiplicity} is impossible for {total_electrons} electrons")
    seen: dict[int, int] = {}
    electrons = []
    for k, frag in enumerate(frags, start=1):
        q = _strict_int(frag.charge, f"fragment {k} charge")
        m = _strict_int(frag.multiplicity, f"fragment {k} multiplicity")
        if not frag.atoms:
            raise ValueError(f"fragment {k} has no atoms")
        for atom in frag.atoms:
            atom = _strict_int(atom, f"fragment {k} atom index")
            if not 1 <= atom <= n_atoms:
                raise ValueError(f"fragment {k}: atom index {atom} outside 1..{n_atoms}")
            if atom in seen:
                raise ValueError(f"atom {atom} is in fragments {seen[atom]} and {k}")
            seen[atom] = k
        if m == 0:
            raise ValueError(f"fragment {k}: multiplicity 0 is meaningless")
        if m == -1:
            raise ValueError(f"fragment {k}: a singlet has no spin orientation; write 1, not -1")
        n_e = sum(SYMBOL_TO_Z[symbols[a - 1]] for a in frag.atoms) - q
        if n_e < 0:
            raise ValueError(f"fragment {k}: charge {q} leaves {n_e} electrons")
        if abs(m) - 1 > n_e:
            raise ValueError(f"fragment {k}: |multiplicity| {abs(m)} needs more than its {n_e} electrons")
        if (n_e + abs(m) - 1) % 2:
            parity = "odd" if n_e % 2 else "even"
            raise ValueError(
                f"fragment {k}: multiplicity {m} has the wrong parity for {n_e} ({parity}) electrons"
            )
        electrons.append(n_e)
    missing = sorted(set(range(1, n_atoms + 1)) - set(seen))
    if missing:
        raise ValueError(f"atoms not assigned to any fragment: {missing}")
    if sum(f.charge for f in frags) != charge:
        raise ValueError(
            f"fragment charges sum to {sum(f.charge for f in frags)}, total charge is {charge}"
        )
    signed = sum(f.signed_unpaired for f in frags)
    if signed != multiplicity - 1:
        raise ValueError(
            f"signed fragment spins sum to {signed} unpaired electrons, but M - 1 = {multiplicity - 1}"
        )
    return FragmentPlan(tuple(symbols), charge, multiplicity, frags, tuple(electrons))


def fragments_from_spec(specification: dict) -> list[Fragment]:
    """Convert the --fragment-spec JSON form (unsigned multiplicity + orientation)."""
    out = []
    for item in specification["fragments"]:
        orientation = item.get("orientation", "alpha")
        sign = {1: 1, "+": 1, "alpha": 1, "up": 1, -1: -1, "-": -1, "beta": -1, "down": -1}.get(orientation)
        if sign is None:
            raise ValueError(f"unknown orientation {orientation!r}")
        m = _strict_int(item["multiplicity"], "fragment multiplicity")
        if m < 1:
            raise ValueError("spec multiplicities are unsigned; use orientation for beta")
        out.append(Fragment(tuple(item["atoms"]), item["charge"], sign * m))
    return out


# ----------------------------------------------------------------------- render


def _header(checkpoint: str, memory: str, nproc: int) -> list[str]:
    return [f"%chk={checkpoint}", f"%mem={memory}", f"%nprocshared={nproc}"]


def _coordinate_lines(symbols: Sequence[str], coords: Sequence[Sequence[float]],
                      atom_fragment: dict[int, int] | None) -> list[str]:
    lines = []
    for i, (symbol, xyz) in enumerate(zip(symbols, coords), start=1):
        label = symbol if atom_fragment is None else f"{symbol}(Fragment={atom_fragment[i]})"
        lines.append(f"{label:18s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}")
    return lines


@dataclass(frozen=True)
class TandemJob:
    name: str
    symbols: tuple[str, ...]
    coords: tuple[tuple[float, float, float], ...]
    charge: int
    multiplicity: int
    plan: FragmentPlan | None  # None = default-guess control (no fragment link)
    level: LevelOfTheory = ARCHIVED_LEVEL
    force_stage: bool = True
    guess_only: bool = False  # Guess=(Fragment=N,Only): the documented G16 variant, not the archive's
    title: str = ""
    memory: str = "24GB"
    nproc: int = 12
    # (link 0, link 1) routes built by the caller; link 1 is a campaign job, and
    # force_stage/guess_only/level do not apply. See spin.tandem_campaign_routes.
    campaign_routes: tuple[str, str] | None = None

    @property
    def checkpoint(self) -> str:
        return f"{self.name}.chk"

    def stage_kinds(self) -> list[str]:
        if self.campaign_routes is not None:
            return [STAGE_FRAGMENT, STAGE_CAMPAIGN]
        kinds = [STAGE_CONTROL_STABLE] if self.plan is None else [STAGE_FRAGMENT, STAGE_STABLE]
        return kinds + ([STAGE_FORCE] if self.force_stage else [])

    def routes(self) -> list[str]:
        if self.campaign_routes is not None:
            return list(self.campaign_routes)
        routes = []
        for kind in self.stage_kinds():
            if kind == STAGE_FRAGMENT:
                routes.append(fragment_route(self.level, len(self.plan.fragments), self.guess_only))
            elif kind == STAGE_STABLE:
                routes.append(stable_route(self.level))
            elif kind == STAGE_CONTROL_STABLE:
                routes.append(stable_route(self.level, from_checkpoint=False))
            else:
                routes.append(force_route(self.level))
        return routes


def render_tandem(job: TandemJob) -> str:
    if job.campaign_routes is not None and job.plan is None:
        raise ValueError("a campaign tandem needs a fragment plan")
    if job.plan is not None:
        if (job.plan.charge, job.plan.multiplicity) != (job.charge, job.multiplicity):
            raise ValueError("fragment plan and job disagree on total charge/multiplicity")
        if tuple(job.plan.symbols) != tuple(job.symbols):
            raise ValueError("fragment plan was validated for a different atom list")
    if len(job.symbols) != len(job.coords):
        raise ValueError("symbols and coordinates differ in length")
    lines: list[str] = []
    for index, (kind, route) in enumerate(zip(job.stage_kinds(), job.routes())):
        if index:
            lines.append("--Link1--")
        lines += _header(job.checkpoint, job.memory, job.nproc)
        lines += [route, "", f"{job.title or job.name}; stage={index} {kind}", ""]
        if kind == STAGE_FRAGMENT:
            lines.append(job.plan.charge_multiplicity_line())
            lines += _coordinate_lines(job.symbols, job.coords, job.plan.atom_fragment)
        elif kind == STAGE_CONTROL_STABLE:
            lines.append(f"{job.charge} {job.multiplicity}")
            lines += _coordinate_lines(job.symbols, job.coords, None)
        else:
            lines.append(f"{job.charge} {job.multiplicity}")
        lines.append("")
    text = "\n".join(lines) + "\n"
    problems = inspect_tandem_input(text, job).problems
    if problems:
        raise ValueError("generated input failed inspection:\n  " + "\n  ".join(problems))
    return text


# ---------------------------------------------------------------------- inspect

_JOB_TYPES = {"sp", "opt", "freq", "irc", "ircmax", "force", "stable", "scan", "polar", "admp", "bomd",
              "td", "volume", "nmr"}


def route_tokens(route: str) -> list[str]:
    """Whitespace tokens of a route, keeping parenthesised option lists whole."""
    body = route.strip()
    if not body.startswith("#"):
        raise ValueError("route must start with '#'")
    body = re.sub(r"^#[pPnNtT]?", "", body)
    tokens, depth, current = [], 0, ""
    for ch in body:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch.isspace() and depth == 0:
            if current:
                tokens.append(current)
            current = ""
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens


def _keyword(token: str) -> str:
    return re.split(r"[=(]", token, maxsplit=1)[0].lower()


def _value(tokens: list[str], key: str) -> str | None:
    for token in tokens:
        if _keyword(token) == key:
            parts = re.split(r"\s*=\s*", token, maxsplit=1)
            return parts[1].strip("()").lower() if len(parts) > 1 else ""
    return None


@dataclass
class StageText:
    link0: list[str]
    route: str
    title: str
    charge_multiplicity: list[int]
    atom_lines: list[str]
    trailing: list[str]


@dataclass
class Inspection:
    problems: list[str] = field(default_factory=list)
    stages: list[StageText] = field(default_factory=list)


def split_input(text: str) -> list[StageText]:
    if "\r" in text:
        raise ValueError("input contains CR characters; Gaussian inputs must use LF line endings")
    chunks = re.split(r"(?im)^[ \t]*--link1--[ \t]*$", text)
    stages = []
    for chunk in chunks:
        lines = chunk.strip("\n").split("\n")
        i = 0
        link0 = []
        while i < len(lines) and lines[i].startswith("%"):
            link0.append(lines[i].strip())
            i += 1
        route_lines = []
        while i < len(lines) and lines[i].strip():
            route_lines.append(lines[i].strip())
            i += 1
        i += 1  # blank after route
        title_lines = []
        while i < len(lines) and lines[i].strip():
            title_lines.append(lines[i].strip())
            i += 1
        i += 1  # blank after title
        cm = lines[i].replace(",", " ").split() if i < len(lines) else []
        i += 1
        atoms = []
        while i < len(lines) and lines[i].strip():
            atoms.append(lines[i])
            i += 1
        trailing = [l for l in lines[i:] if l.strip()]
        try:
            cm_ints = [int(v) for v in cm]
        except ValueError:
            cm_ints = []
        stages.append(StageText(link0, " ".join(route_lines), " ".join(title_lines), cm_ints, atoms, trailing))
    return stages


_ATOM_RE = re.compile(
    r"^\s*([A-Z][a-z]?)(?:\(Fragment=(\d+)\))?\s+([-+]?\d+\.\d*(?:[EeDd][-+]?\d+)?)\s+"
    r"([-+]?\d+\.\d*(?:[EeDd][-+]?\d+)?)\s+([-+]?\d+\.\d*(?:[EeDd][-+]?\d+)?)\s*$"
)


def inspect_tandem_input(text: str, job: TandemJob | None = None, coord_tol: float = 1e-9) -> Inspection:
    """Independent structural audit of a tandem input, run on every file before it is written.

    With ``job`` the audit also checks stage count, routes, geometry and the fragment
    plan against the request; without it only the generic tandem rules apply.
    """
    report = Inspection()
    p = report.problems
    try:
        stages = split_input(text)
    except ValueError as exc:
        return Inspection([str(exc)])
    report.stages = stages
    if not text.endswith("\n"):
        p.append("input does not end with a newline")
    kinds = job.stage_kinds() if job else None
    if kinds and len(stages) != len(kinds):
        p.append(f"expected {len(kinds)} Link1 stages, found {len(stages)}")
    first_has_fragments = bool(stages) and "fragment" in stages[0].route.lower()
    if kinds is None:
        campaign = (first_has_fragments and len(stages) > 1
                    and "stable" not in [_keyword(t) for t in _safe_tokens(stages[1].route)])
        if campaign:
            kinds = [STAGE_FRAGMENT, STAGE_CAMPAIGN]
            if len(stages) != 2:
                p.append(f"a fragment + campaign-job tandem has 2 Link1 stages, found {len(stages)}")
        else:
            kinds = ([STAGE_FRAGMENT, STAGE_STABLE] if first_has_fragments else [STAGE_CONTROL_STABLE])
            kinds += [STAGE_FORCE] * max(0, len(stages) - len(kinds))
    chk_names = []
    level_signature = None
    total_cm = None
    for index, (stage, kind) in enumerate(zip(stages, kinds)):
        tag = f"stage {index} ({kind})"
        chks = [l.split("=", 1)[1] for l in stage.link0 if l.lower().startswith("%chk=")]
        if len(chks) != 1:
            p.append(f"{tag}: needs exactly one %chk line, found {len(chks)}")
        chk_names += chks
        if any(l.lower().startswith("%oldchk") for l in stage.link0):
            p.append(f"{tag}: %oldchk breaks single-checkpoint continuity")
        try:
            tokens = route_tokens(stage.route)
        except ValueError as exc:
            p.append(f"{tag}: {exc}")
            continue
        keys = [_keyword(t) for t in tokens]
        job_types = sorted({k for k in keys if k in _JOB_TYPES})
        guess = _value(tokens, "guess") or ""
        geom = _value(tokens, "geom")
        # IOP(a/b=c,...) has no "=" before its parentheses, so _value would split inside it.
        iop = ",".join(re.findall(r"(?i)\biop\(([^)]*)\)", stage.route)).replace(" ", "")
        method = next((t for t in tokens if "/" in t and "=" not in t.split("/")[0]), "")
        signature = (method.lower(), (_value(tokens, "scf") or ""), (_value(tokens, "int") or ""),
                     "nosymm" in keys)
        if level_signature is None:
            level_signature = signature
        elif signature != level_signature:
            p.append(f"{tag}: level of theory differs from stage 0: {signature} vs {level_signature}")
        if CONTINUE_SCF_IOP not in iop.split(","):
            p.append(f"{tag}: IOP(5/13=1) is missing; the archived routes carry it in every link")
        if kind == STAGE_FRAGMENT:
            m = re.fullmatch(r"fragment=(\d+)(,only)?", guess)
            if job_types != ["sp"] or not m:
                p.append(f"{tag}: must be 'SP Guess=(Fragment=N)' and nothing else; got jobs {job_types}, guess {guess!r}")
            if geom is not None:
                p.append(f"{tag}: the fragment stage must carry explicit coordinates, not Geom={geom}")
            if "stable" in keys:
                p.append(f"{tag}: Stable=Opt belongs to the next stage only")
            n_frag = int(m.group(1)) if m else 0
            if job and job.plan and n_frag != len(job.plan.fragments):
                p.append(f"{tag}: Fragment={n_frag} but the plan has {len(job.plan.fragments)} fragments")
            if job and m and bool(m.group(2)) != job.guess_only:
                p.append(f"{tag}: ',Only' present={bool(m.group(2))} but requested={job.guess_only}")
            cm = stage.charge_multiplicity
            if len(cm) != 2 + 2 * n_frag:
                p.append(f"{tag}: charge/multiplicity line has {len(cm)} integers, expected {2 + 2 * n_frag}")
            parsed = [_ATOM_RE.match(l) for l in stage.atom_lines]
            if not stage.atom_lines or not all(parsed):
                p.append(f"{tag}: unreadable or missing fragment-labelled coordinates")
                continue
            if any(mt.group(2) is None for mt in parsed):
                p.append(f"{tag}: every atom needs a (Fragment=k) label")
                continue
            symbols = [mt.group(1) for mt in parsed]
            labels = [int(mt.group(2)) for mt in parsed]
            if len(cm) == 2 + 2 * n_frag:
                total_cm = (cm[0], cm[1])
                frags = [Fragment(tuple(i + 1 for i, k in enumerate(labels) if k == f),
                                  cm[2 * f], cm[2 * f + 1]) for f in range(1, n_frag + 1)]
                try:
                    plan = validate_fragment_plan(symbols, cm[0], cm[1], frags)
                except ValueError as exc:
                    p.append(f"{tag}: invalid fragment guess: {exc}")
                    plan = None
                if plan is not None and job and job.plan is not None and plan != job.plan:
                    p.append(f"{tag}: written fragment guess differs from the requested plan")
            if job:
                _compare_geometry(p, tag, parsed, job, coord_tol)
        elif kind == STAGE_CONTROL_STABLE:
            if job_types != ["stable"] or "opt" not in (_value(tokens, "stable") or ""):
                p.append(f"{tag}: must be Stable=Opt only; got {job_types}")
            if "fragment" in guess or geom is not None:
                p.append(f"{tag}: a default-guess control takes no fragment guess and explicit coordinates")
            parsed = [_ATOM_RE.match(l) for l in stage.atom_lines]
            if not stage.atom_lines or not all(parsed) or any(mt.group(2) for mt in parsed):
                p.append(f"{tag}: needs plain coordinates")
            elif job:
                _compare_geometry(p, tag, parsed, job, coord_tol)
            total_cm = tuple(stage.charge_multiplicity[:2]) if len(stage.charge_multiplicity) == 2 else None
            if total_cm is None:
                p.append(f"{tag}: charge/multiplicity line must be 'Q M'")
        elif kind == STAGE_CAMPAIGN:
            # The archived link 1: the campaign job on the read guess and geometry.
            if not job_types or job_types == ["sp"]:
                p.append(f"{tag}: needs the campaign job (Opt, Opt(TS), Freq, ...); got {job_types or 'none'}")
            if "stable" in keys:
                p.append(f"{tag}: Stable is not part of the campaign tandem")
            if guess != "read":
                p.append(f"{tag}: must read orbitals with Guess=Read, got Guess={guess or 'default'}")
            if geom not in ("checkpoint", "check"):
                p.append(f"{tag}: must reuse the geometry with Geom=Checkpoint, got Geom={geom}")
            if stage.atom_lines:
                p.append(f"{tag}: Geom=Checkpoint stage must not repeat coordinates")
            cm = stage.charge_multiplicity
            if len(cm) != 2:
                p.append(f"{tag}: charge/multiplicity line must be 'Q M', got {cm}")
            elif total_cm is not None and tuple(cm) != total_cm:
                p.append(f"{tag}: charge/multiplicity {tuple(cm)} changes across Link1 (stage 0: {total_cm})")
        else:
            want = "stable" if kind == STAGE_STABLE else "force"
            if job_types != [want]:
                p.append(f"{tag}: job type must be {want} only (fixed geometry); got {job_types}")
            if kind == STAGE_STABLE and "opt" not in (_value(tokens, "stable") or ""):
                p.append(f"{tag}: needs Stable=Opt")
            if kind == STAGE_FORCE and "stable" in keys:
                p.append(f"{tag}: Stable belongs to the stability stage only")
            if guess != "read":
                p.append(f"{tag}: must read orbitals with Guess=Read, got Guess={guess or 'default'}")
            if geom not in ("checkpoint", "check"):
                p.append(f"{tag}: must reuse the geometry with Geom=Checkpoint, got Geom={geom}")
            if stage.atom_lines:
                p.append(f"{tag}: Geom=Checkpoint stage must not repeat coordinates")
            cm = stage.charge_multiplicity
            if len(cm) != 2:
                p.append(f"{tag}: charge/multiplicity line must be 'Q M', got {cm}")
            elif total_cm is not None and tuple(cm) != total_cm:
                p.append(f"{tag}: charge/multiplicity {tuple(cm)} changes across Link1 (stage 0: {total_cm})")
        if stage.trailing:
            p.append(f"{tag}: unexpected trailing input sections: {stage.trailing[:2]}")
        if job and total_cm is not None and total_cm != (job.charge, job.multiplicity):
            p.append(f"{tag}: total charge/multiplicity {total_cm} != requested {(job.charge, job.multiplicity)}")
    if len(set(chk_names)) > 1:
        p.append(f"checkpoint names differ across stages: {sorted(set(chk_names))}")
    if job and chk_names and chk_names[0] != job.checkpoint:
        p.append(f"checkpoint {chk_names[0]} != {job.checkpoint}")
    if job:
        for index, (stage, route) in enumerate(zip(stages, job.routes())):
            if stage.route != route:
                p.append(f"stage {index}: route differs from the fixed template: {stage.route!r}")
    return report


def _safe_tokens(route: str) -> list[str]:
    try:
        return route_tokens(route)
    except ValueError:
        return []


def plan_from_input(text: str) -> tuple[FragmentPlan | None, int]:
    """(fragment plan or None for a control job, number of Link1 stages) of a written input."""
    stages = split_input(text)
    first = stages[0]
    m = re.search(r"fragment=(\d+)", first.route.lower())
    if not m:
        return None, len(stages)
    n = int(m.group(1))
    parsed = [_ATOM_RE.match(l) for l in first.atom_lines]
    symbols = [mt.group(1) for mt in parsed]
    labels = [int(mt.group(2)) for mt in parsed]
    cm = first.charge_multiplicity
    frags = [Fragment(tuple(i + 1 for i, k in enumerate(labels) if k == f), cm[2 * f], cm[2 * f + 1])
             for f in range(1, n + 1)]
    return validate_fragment_plan(symbols, cm[0], cm[1], frags), len(stages)


def _compare_geometry(p: list[str], tag: str, parsed, job: TandemJob, tol: float) -> None:
    if [mt.group(1) for mt in parsed] != list(job.symbols):
        p.append(f"{tag}: element order differs from the source geometry")
        return
    worst = max(abs(float(mt.group(c + 3)) - xyz[c]) for mt, xyz in zip(parsed, job.coords) for c in range(3))
    if worst > tol:
        p.append(f"{tag}: coordinates differ from the source geometry by {worst:.2e} Å")


# ------------------------------------------------------------------------- logs

_TERMINATION_RE = re.compile(r"^ (?:Normal termination of Gaussian|Error termination)[^\n]*\n", re.M)
# G16 prints "a.u." for the SCF that Stable=Opt re-optimises, "A.U." elsewhere
_SCF_RE = re.compile(r"SCF Done:\s+E\((\S+)\)\s+=\s+(-?\d+\.\d+)\s+[Aa]\.[Uu]\. after\s+(\d+) cycles")
_S2_RE = re.compile(r"S\*\*2 before annihilation\s+(-?\d+\.\d+)")
_GUESS_S2_RE = re.compile(r"Initial guess <Sx>=.*?<S\*\*2>=\s*(-?\d+\.\d+)")
_CM_RE = re.compile(r"Charge =\s*(-?\d+) Multiplicity =\s*(-?\d+)\b(?! in fragment)")
_FRAG_CM_RE = re.compile(r"Charge =\s*(-?\d+) Multiplicity =\s*(-?\d+) in fragment\s+(\d+)")
_ROUTE_RE = re.compile(r"\n -{10,}\n( #.*?)\n -{10,}\n", re.S)
_ORIENT_RE = re.compile(r"Input orientation:.*?-{20,}\n.*?-{20,}\n(.*?)\n -{20,}", re.S)
_STABLE = "The wavefunction is stable under the perturbations considered"
_UNSTABLE_RE = re.compile(r"The wavefunction has an? (internal|RHF -> UHF|RHF -> RHF)[^\n]*instability")
_MULLIKEN_RE = re.compile(r"Mulliken charges and spin densities:\n\s+1\s+2\n(.*?)\n Sum of Mulliken", re.S)
_CAP_RE = re.compile(r"within\s+(\d+) cycles")


def split_log_stages(text: str) -> list[tuple[str, str]]:
    """(stage text, termination) per Link1 job; the last may be unterminated ('')."""
    out, start = [], 0
    for m in _TERMINATION_RE.finditer(text):
        out.append((text[start:m.end()], "normal" if "Normal" in m.group(0) else "error"))
        start = m.end()
    tail = text[start:]
    if _SCF_RE.search(tail) or _ROUTE_RE.search(tail):
        out.append((tail, ""))
    return out


def _route(stage: str) -> str:
    m = _ROUTE_RE.search(stage)
    return " ".join(l.strip() for l in m.group(1).splitlines()) if m else ""


def _orientation(stage: str) -> list[tuple[float, float, float]]:
    blocks = _ORIENT_RE.findall(stage)
    if not blocks:
        return []
    rows = []
    for line in blocks[-1].splitlines():
        parts = line.split()
        if len(parts) == 6:
            rows.append(tuple(float(v) for v in parts[3:]))
    return rows


def _mulliken_spins(stage: str) -> list[float]:
    blocks = _MULLIKEN_RE.findall(stage)
    if not blocks:
        return []
    return [float(line.split()[3]) for line in blocks[-1].splitlines() if len(line.split()) == 4]


_FRAG_SCF_RE = re.compile(r"Fragment guess: doing (?:MCBS )?calculation for fragment")


def parse_tandem_log(text: str, plan: FragmentPlan | None, force_stage: bool = True,
                     energy_tol_hartree: float = 1e-5, coord_tol: float = 1e-5) -> dict:
    """Electronic-state evidence from a tandem log, stage by stage.

    A normal termination is not enough: the result also records whether the
    fragment guess reached link 1 intact, what Stable=Opt changed, whether the
    force SCF reproduced the stable state, and whether the geometry stayed fixed.
    """
    stages = split_log_stages(text)
    expected = (1 if plan is None else 2) + (1 if force_stage else 0)
    r: dict = {"stages_found": len(stages), "stages_expected": expected,
               "terminations": [t for _, t in stages], "issues": []}
    issues = r["issues"]
    if len(stages) < expected or any(t != "normal" for _, t in stages[:expected]):
        issues.append(f"terminations {r['terminations']} (expected {expected} normal)")
    idx = 0
    first_geometry: list = []
    if plan is not None and stages:
        s1 = stages[0][0]
        idx = 1
        scf = _SCF_RE.findall(s1)
        # G09 D.01 prints "doing MCBS calculation for fragment"; G16 C.01 prints "doing calculation for
        # fragment" after a "doing full-system calculation" preamble that runs no SCF.
        n_mcbs = len(_FRAG_SCF_RE.findall(s1))
        frag_cm = _FRAG_CM_RE.findall(s1)
        total = _CM_RE.search(s1.replace(" in supermolecule", ""))
        r.update({
            "s1_route": _route(s1),
            "s1_fragment_scf": n_mcbs,
            "s1_fragment_scf_energies": [float(e) for _, e, _ in scf[:n_mcbs]],
            "s1_fragment_scf_cycles": [int(c) for _, _, c in scf[:n_mcbs]],
            "s1_fragment_s2": [float(v) for v in _S2_RE.findall(s1)[:n_mcbs]],
            "s1_unconverged_scf": s1.count("Convergence criterion not met"),
            "s1_supermolecule_scf": max(0, len(scf) - n_mcbs),
            "s1_fragments_echoed": [f"{q} {m}" for q, m, _ in frag_cm],
        })
        want = [f"{f.charge} {f.multiplicity}" for f in plan.fragments]
        if frag_cm and r["s1_fragments_echoed"] != want:
            issues.append(f"stage 0 echoed fragments {r['s1_fragments_echoed']} != plan {want}")
        if r["s1_supermolecule_scf"]:
            issues.append("stage 0 ran a supermolecule SCF (archived G09 D.01 runs did not); "
                          "link 1 may not start from the pure fragment guess")
        first_geometry = _orientation(s1)
    if len(stages) > idx:
        s2 = stages[idx][0]
        scf = _SCF_RE.findall(s2)
        s2s = [float(v) for v in _S2_RE.findall(s2)]
        guess = _GUESS_S2_RE.search(s2)
        cm = _CM_RE.search(s2)
        r.update({
            "s2_route": _route(s2),
            "s2_charge_multiplicity": f"{cm.group(1)} {cm.group(2)}" if cm else "",
            "s2_guess_from_checkpoint": "Initial guess from the checkpoint file" in s2,
            "s2_guess_s2": float(guess.group(1)) if guess else None,
            "s2_scf_energies": [float(e) for _, e, _ in scf],
            "s2_scf_cycles": [int(c) for _, _, c in scf],
            "s2_scf_s2": s2s,
            "s2_cycle_cap": sorted({int(v) for v in _CAP_RE.findall(s2)}),
            "s2_unconverged_scf": s2.count("Convergence criterion not met") + s2.count("Convergence failure"),
            "s2_instabilities": [m.group(1) for m in _UNSTABLE_RE.finditer(s2)],
            "s2_final_stable": s2.rfind(_STABLE) > max((m.start() for m in _UNSTABLE_RE.finditer(s2)), default=-1),
            "s2_mulliken_spins": _mulliken_spins(s2),
            "s2_hirshfeld": parse_hirshfeld_table(s2),
        })
        if not first_geometry:
            first_geometry = _orientation(s2)
        if plan is not None:
            r["ideal_guess_s2"] = plan.ideal_guess_s2
            if not r["s2_guess_from_checkpoint"]:
                issues.append("link 1 did not read its guess from the checkpoint")
            elif r["s2_guess_s2"] is None or abs(r["s2_guess_s2"] - plan.ideal_guess_s2) > plan.guess_s2_tolerance:
                issues.append(f"link 1 guess <S**2>={r['s2_guess_s2']} is not the fragment pattern's "
                              f"{plan.ideal_guess_s2:.3f} (tol {plan.guess_s2_tolerance:.2f})")
            if cm and (int(cm.group(1)), int(cm.group(2))) != (plan.charge, plan.multiplicity):
                issues.append(f"link 1 charge/multiplicity {r['s2_charge_multiplicity']} differs from "
                              f"stage 0 ({plan.charge} {plan.multiplicity})")
        if r["s2_unconverged_scf"]:
            issues.append("an SCF in the stability stage did not converge")
        if not r["s2_final_stable"]:
            issues.append("the stability stage did not end with a stable wavefunction")
        if scf:
            e_first, e_last = float(scf[0][1]), float(scf[-1][1])
            r["stable_opt_dE_meV"] = (e_last - e_first) * 27211.386
            r["stable_opt_dS2"] = (s2s[-1] - s2s[0]) if s2s else None
            r["stable_opt_changed_state"] = bool(r["s2_instabilities"]) or abs(e_last - e_first) > energy_tol_hartree
        idx += 1
    if force_stage and len(stages) > idx:
        s3 = stages[idx][0]
        scf = _SCF_RE.findall(s3)
        s2s = [float(v) for v in _S2_RE.findall(s3)]
        frames = parse_force_frames(s3, Path("stage.log"))
        r.update({
            "s3_route": _route(s3),
            "s3_scf_energy": float(scf[-1][1]) if scf else None,
            "s3_scf_cycles": int(scf[-1][2]) if scf else None,
            "s3_s2": s2s[-1] if s2s else None,
            "s3_unconverged_scf": s3.count("Convergence criterion not met") + s3.count("Convergence failure"),
            "s3_mulliken_spins": _mulliken_spins(s3),
            "s3_hirshfeld": parse_hirshfeld_table(s3),
            "forces_ev_ang": frames[-1].forces_ev_ang if frames else None,
            "energy_ev": frames[-1].energy_ev if frames else None,
        })
        if r["s3_unconverged_scf"]:
            issues.append("the force-stage SCF did not converge")
        e2 = r.get("s2_scf_energies") or []
        if scf and e2:
            r["s3_minus_s2_hartree"] = float(scf[-1][1]) - e2[-1]
            if abs(r["s3_minus_s2_hartree"]) > energy_tol_hartree:
                issues.append(f"force-stage energy differs from the stable state by "
                              f"{r['s3_minus_s2_hartree']:.2e} Ha: the state did not survive Link1")
        geometry = _orientation(s3)
        if first_geometry and geometry:
            worst = max(abs(a - b) for p, q in zip(first_geometry, geometry) for a, b in zip(p, q))
            r["geometry_max_shift_ang"] = worst
            if worst > coord_tol or len(first_geometry) != len(geometry):
                issues.append(f"geometry moved by {worst:.2e} Å between stages")
        if not frames:
            issues.append("no force table in the force stage")
    r["status"] = "ok" if not issues else "check"
    return r


# ------------------------------------------------------------------ comparisons


@dataclass(frozen=True)
class StateTolerances:
    """Reproducibility tolerances, not physical thresholds.

    Two solutions that match within all of them are one electronic state reached
    from different guesses. Energy differences larger than ``energy_ev`` are kept
    as physical differences between states; a higher-energy stable solution is a
    metastable BS state, never a rejected one.
    """
    energy_ev: float = 1e-3
    site_spin: float = 0.2
    s2: float = 0.05
    force_rms_ev_ang: float = 5e-3


def compare_states(a: dict, b: dict, tol: StateTolerances = StateTolerances()) -> dict:
    """Classify two converged, stable solutions as same / distinct / ambiguous."""
    de = b["energy_ev"] - a["energy_ev"]
    sa = [v[2] for v in (a.get("s3_hirshfeld") or {}).get("atomic_spins_hirshfeld", [])]
    sb = [v[2] for v in (b.get("s3_hirshfeld") or {}).get("atomic_spins_hirshfeld", [])]
    ds = max((abs(x - y) for x, y in zip(sa, sb)), default=None) if sa and sb else None
    sign_flips = sum(1 for x, y in zip(sa, sb) if abs(x) > 0.5 and abs(y) > 0.5 and x * y < 0) if ds is not None else None
    ds2 = (b["s3_s2"] - a["s3_s2"]) if a.get("s3_s2") is not None and b.get("s3_s2") is not None else None
    df = None
    if a.get("forces_ev_ang") and b.get("forces_ev_ang"):
        diffs = [x - y for fa, fb in zip(a["forces_ev_ang"], b["forces_ev_ang"]) for x, y in zip(fa, fb)]
        df = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    votes_same = [abs(de) <= tol.energy_ev]
    if ds is not None:
        votes_same.append(ds <= tol.site_spin)
    if ds2 is not None:
        votes_same.append(abs(ds2) <= tol.s2)
    if df is not None:
        votes_same.append(df <= tol.force_rms_ev_ang)
    if all(votes_same):
        verdict = "same"
    elif not any(votes_same):
        verdict = "distinct"
    elif ds is not None and ds > tol.site_spin and abs(de) > tol.energy_ev:
        verdict = "distinct"
    else:
        verdict = "ambiguous"
    return {"verdict": verdict, "dE_meV": 1000 * de, "max_dspin_hirshfeld": ds,
            "hirshfeld_sign_flips": sign_flips, "dS2": ds2, "force_rms_diff_meV_ang": None if df is None else 1000 * df}
