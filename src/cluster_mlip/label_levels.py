"""Which level of theory labeled each Gaussian force frame.

A force label is only interchangeable with another if the Hamiltonian is the
same: functional, basis set (including 5D/6D, 7F/10F), integration grid and
anything else that changes the energy surface (dispersion, solvation, ECPs,
Hamiltonian-altering IOps). SCF convergence aids, optimizer settings, output
control and symmetry handling do not change the converged energy and are
ignored.

Raw route strings are a poor proxy: the same level is written many ways
(``6-311++G*`` vs ``6-311++G(d)``; ``Geom=Checkpoint Guess=Read`` added to a
Link1 stage), and Gaussian wraps its route echo at a fixed width *inside*
words, so naive line joining yields tokens like ``NOS YMM``. For output files
the level is therefore read from what Gaussian reports it did -- the
functional in each ``SCF Done: E(...)`` line, the ``Standard basis:`` line
with its d/f function types, and the archive entry -- with the route used
only for what nothing else records (grid, extras).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dataset import write_labeled_extxyz
from .gaussian import (
    _FORCE_HEADER_RE, _SCF_RE, gaussian_job_complete, parse_force_frames, rms_force,
)
from .io import _decode_bytes, source_tree
from .local_spins import local_spin_summary
from .models import LabeledFrame, geometry_signature
from .routes import (
    keyword_name, keyword_options, route_has_frequency, route_has_irc, route_optimizes,
    route_search_kind, route_tokens,
)

LEVEL_CLASSES = ("same_level", "grid_differs", "different_level", "unverifiable")

# Keywords that do not change the converged energy surface.
_NEUTRAL_KEYWORDS = {
    "opt", "freq", "force", "irc", "ircmax", "sp", "scan", "stable", "nosymm", "nosym",
    "symmetry", "symm", "geom", "geometry", "guess", "pop", "population", "scf", "test",
    "units", "maxdisk", "output", "punch", "gfinput", "gfprint", "density", "nmr",
    "polar", "volume", "prop", "iop", "int", "integral", "5d", "6d", "7f", "10f",
    "pure", "cartesian", "chkbasis", "chkbas", "checkbasis", "genchk", "transformation",
    "fchk", "formcheck", "counterpoise", "nosymmetry",
}
# IOps whose only effect is to continue past an SCF failure; the frame-level
# SCF check (gaussian._SCF_NOT_CONVERGED_RE) judges the label instead.
_NEUTRAL_IOPS = {"5/13"}
_GRID_NAMES = {
    "ultrafine": "ultrafine", "ultrafinegrid": "ultrafine", "superfine": "superfine",
    "superfinegrid": "superfine", "fine": "fine", "finegrid": "fine", "sg1": "sg1",
    "sg1grid": "sg1", "coarse": "coarse", "coarsegrid": "coarse",
}
_PROGRAM_RE = re.compile(r"Gaussian\s+(03|09|16)\b|G(03|09|16)Rev", re.I)
_STANDARD_BASIS_RE = re.compile(
    r"^\s*Standard basis:\s*(\S+)\s*\((\d+)D,\s*(\d+)F\)", re.I | re.M
)
_GENERAL_BASIS_RE = re.compile(
    r"^\s*General basis read from cards:\s*\((\d+)D,\s*(\d+)F\)", re.I | re.M
)
_SCF_METHOD_RE = re.compile(r"SCF Done:\s+E\(([^)]*)\)", re.I)
_ROUTE_ECHO_RE = re.compile(r"^ -{10,}[ \t]*\r?\n((?: #[^\n]*\n)(?: [^\n]*\n){0,12}?) -{10,}", re.M)


@dataclass(frozen=True)
class LabelLevel:
    functional: str
    basis: str
    basis_functions: str
    grid: str
    extras: tuple[str, ...] = ()
    evidence: tuple[str, ...] = field(default=(), compare=False)

    @property
    def key(self) -> str:
        extras = ",".join(self.extras) or "-"
        return (
            f"{self.functional}/{self.basis}|{self.basis_functions or 'df=?'}"
            f"|grid={self.grid}|extras={extras}"
        )

    @property
    def short_id(self) -> str:
        return hashlib.sha1(self.key.encode()).hexdigest()[:8]


def normalize_functional(name: str) -> str:
    """``UBPW91``, ``UB-PW91`` (SCF Done label) and ``BPW91`` are one level."""
    value = name.strip().upper().replace("-", "")
    if not value:
        return "unknown"
    if not value.startswith("REV"):
        if value.startswith("RO") and len(value) > 3:
            value = value[2:]
        elif value[0] in "UR" and len(value) > 2:
            value = value[1:]
    return value


def normalize_basis(name: str) -> str:
    value = name.strip().lower()
    if value in {"chkbas", "chkbasis", "checkbasis"}:
        return "from_checkpoint"
    value = value.replace("**", "(d,p)").replace("*", "(d)")
    return value or "unknown"


def _default_basis_functions(basis: str) -> str:
    """Gaussian's default d/f function type for a route-named basis."""
    if basis in {"gen", "genecp", "from_checkpoint", "unknown"}:
        return ""
    if basis.startswith("6-31") and not basis.startswith("6-311"):
        return "6D7F"  # 6-31G-family polarization functions default to Cartesian d
    return "5D7F"


def gaussian_program(text: str) -> str:
    match = _PROGRAM_RE.search(text[:20000])
    if not match:
        return ""
    return "g" + (match.group(1) or match.group(2))


def _default_grid(program: str) -> str:
    if program in {"g09", "g03"}:
        return "fine"
    if program == "g16":
        return "ultrafine"
    return "program_default_unknown"


def _grid_and_extras(route: str, program: str) -> tuple[str, list[str], dict[str, str]]:
    """Grid, Hamiltonian-changing extras, and explicit d/f overrides."""
    grid = ""
    extras: list[str] = []
    functions = {"d": "", "f": ""}
    for token in route_tokens(route):
        name = keyword_name(token).lstrip("#")
        if not name or name in {"p", "n", "t"}:
            continue
        options = [option.lower() for option in keyword_options(token)]
        if name in {"int", "integral"}:
            for option in options:
                bare = option.replace(" ", "")
                if bare.startswith("grid="):
                    grid = _GRID_NAMES.get(bare[5:], bare[5:])
                elif bare in _GRID_NAMES:
                    grid = _GRID_NAMES[bare]
                elif bare.startswith("acc2e"):
                    extras.append(f"int:{bare}")
            continue
        if name == "iop":
            for option in options:
                if option.split("=")[0].strip() not in _NEUTRAL_IOPS:
                    extras.append(f"iop:{option.replace(' ', '')}")
            continue
        if name in {"5d", "6d"}:
            functions["d"] = name.upper()
        elif name in {"7f", "10f"}:
            functions["f"] = name.upper()
        elif name == "cartesian":
            functions = {"d": "6D", "f": "10F"}
        elif name == "pure":
            functions = {"d": "5D", "f": "7F"}
        if "/" in token or name in _NEUTRAL_KEYWORDS:
            continue
        extras.append(token.lower().replace(" ", ""))
    return grid or _default_grid(program), sorted(set(extras)), functions


def _method_basis(route: str) -> tuple[str, str]:
    for token in route_tokens(route):
        cleaned = token.lstrip("#")
        if "/" not in cleaned or keyword_name(cleaned) in _NEUTRAL_KEYWORDS:
            continue
        method, _, basis = cleaned.partition("/")
        return normalize_functional(method), normalize_basis(basis)
    return "unknown", "unknown"


def route_label_level(route: str, program: str = "") -> LabelLevel:
    """The level a route requests, with Gaussian's defaults filled in."""
    functional, basis = _method_basis(route)
    grid, extras, overrides = _grid_and_extras(route, program)
    default = _default_basis_functions(basis)
    functions = ""
    if default:
        functions = (overrides["d"] or default[:2]) + (overrides["f"] or default[2:])
    return LabelLevel(functional, basis, functions, grid, tuple(extras), ("route",))


def compare_levels(candidate: LabelLevel, reference: LabelLevel) -> str:
    unknown = {"unknown", "from_checkpoint", "program_default_unknown", ""}
    if candidate.functional in unknown or candidate.basis in unknown or not candidate.basis_functions:
        return "unverifiable"
    if (candidate.functional, candidate.basis, candidate.basis_functions, candidate.extras) != (
        reference.functional, reference.basis, reference.basis_functions, reference.extras
    ):
        return "different_level"
    if candidate.grid in unknown:
        return "unverifiable"
    return "same_level" if candidate.grid == reference.grid else "grid_differs"


def route_echoes(text: str) -> list[tuple[int, str]]:
    """Route echoes rejoined as Gaussian wrote them.

    Each echoed line carries a one-space prefix and the route is cut at a
    fixed width, mid-word; dropping the prefix and concatenating restores it.
    """
    echoes = []
    for match in _ROUTE_ECHO_RE.finditer(text):
        lines = match.group(1).split("\n")
        route = "".join(line.rstrip("\r")[1:] for line in lines if line)
        echoes.append((match.start(), route.strip()))
    return echoes


def archive_entries(text: str) -> list[tuple[int, dict[str, str]]]:
    """Method, basis, route and version from each Gaussian archive entry."""
    entries = []
    for match in re.finditer(r"^ 1[\\|]1[\\|]", text, re.M):
        end = text.find("@", match.start())
        if end < 0:
            continue
        block = "".join(line[1:] for line in text[match.start():end].split("\n")).replace("\r", "")
        sep = "\\" if "\\" in block[:6] else "|"
        sections = block.split(sep * 2)
        header = sections[0].split(sep)
        entry = {
            "method": header[4] if len(header) > 4 else "",
            "basis": header[5] if len(header) > 5 else "",
            "route": sections[1] if len(sections) > 1 else "",
        }
        version = re.search(r"Version=([^\\|]+)", block)
        entry["version"] = version.group(1) if version else ""
        entries.append((match.start(), entry))
    return entries


def _last_before(items: list, position: int):
    chosen = None
    for item in items:
        if item[0] < position:
            chosen = item
        else:
            break
    return chosen


def frame_label_levels(text: str, frames: list[LabeledFrame]) -> list[tuple[LabelLevel, str]]:
    """(level, route) for each parsed frame, from what the output reports."""
    program = gaussian_program(text)
    headers = list(_FORCE_HEADER_RE.finditer(text))
    energies = list(_SCF_RE.finditer(text))
    echoes = route_echoes(text)
    archives = archive_entries(text)
    bases = sorted(
        [(m.start(), (normalize_basis(m.group(1)), f"{m.group(2)}D{m.group(3)}F"))
         for m in _STANDARD_BASIS_RE.finditer(text)]
        + [(m.start(), ("gen", f"{m.group(1)}D{m.group(2)}F"))
           for m in _GENERAL_BASIS_RE.finditer(text)]
    )
    results = []
    for frame in frames:
        header = headers[frame.record.metadata["force_frame_index"]]
        energy = _last_before([(m.start(), m) for m in energies], header.start())
        position = energy[0] if energy else header.start()
        # A Link1 section echoes its route once, before its first SCF.
        echo = _last_before(echoes, position)
        section_start = echo[0] if echo else 0
        section_end = next((pos for pos, _ in echoes if pos > position), len(text))
        route = echo[1] if echo else ""
        archive = next(
            (entry for pos, entry in archives if position < pos < section_end), None
        )
        if archive and archive.get("route"):
            route = archive["route"]
        level = route_label_level(route, program)
        evidence = ["archive_route" if archive and archive.get("route") else "route_echo"]
        functional, basis, functions = level.functional, level.basis, level.basis_functions
        method = _SCF_METHOD_RE.search(energy[1].group(0)) if energy else None
        if method:
            functional = normalize_functional(method.group(1))
            evidence.append("scf_done")
        basis_line = _last_before(bases, position)
        if basis_line and basis_line[0] > section_start:
            basis, functions = basis_line[1]
            evidence.append("basis_line")
        elif archive and archive.get("basis"):
            basis = normalize_basis(archive["basis"])
            functions = _default_basis_functions(basis) if basis == level.basis else ""
            evidence.append("archive_basis")
        results.append((
            LabelLevel(functional, basis, functions, level.grid, level.extras, tuple(evidence)),
            route,
        ))
    return results


def calculation_kind(route: str) -> str:
    if route_has_irc(route):
        return "irc"
    if route_optimizes(route):
        return "saddle_optimization" if route_search_kind(route) == "saddle" else "minimum_optimization"
    if route_has_frequency(route):
        return "frequency"
    if re.search(r"(?:^|[\s#])force\b", route, re.I):
        return "force"
    return "single_point"


LEVEL_COLUMNS = [
    "source", "force_frame_index", "program", "formula", "charge", "multiplicity", "n_atoms",
    "energy_ev", "rms_force_ev_ang", "max_force_ev_ang", "calculation_kind", "functional",
    "basis", "basis_functions", "grid", "extras", "level_key", "level_class", "level_evidence",
    "scf_convergence_warning", "job_normal_termination", "duplicate_of", "geometry_sha1",
    "local_spin_available", "mulliken_spin_sum", "fe_moment_pattern", "route",
]


def _gaussian_outputs(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("._") or path.stat().st_size < 200:
            continue
        suffix = path.suffix.lower()
        if suffix in {".log", ".out"} or re.fullmatch(r"\.o\d+", suffix):
            yield path
            continue
        if suffix in {".com", ".gjf", ".chk", ".fchk", ".txt", ".zip", ".pdf", ".docx", ".doc"}:
            continue
        with path.open("rb") as handle:
            if b"Entering Gaussian System" in handle.read(65536):
                yield path


def audit_label_levels(
    sources: list[Path],
    output: Path,
    reference: LabelLevel,
    formulas: set[str] | None = None,
) -> dict[str, object]:
    """Classify every force frame in the Gaussian outputs under ``sources``."""
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    by_class: dict[str, list[LabeledFrame]] = {}
    seen_geometries: dict[tuple, str] = {}
    files_seen = files_used = 0
    failures: list[tuple[str, str]] = []
    for source in sources:
        with source_tree(source) as root:
            for path in _gaussian_outputs(root):
                files_seen += 1
                relative = f"{source.name}/{path.relative_to(root).as_posix()}"
                try:
                    text = _decode_bytes(path.read_bytes())
                    frames = parse_force_frames(text, Path(relative))
                    if not frames:
                        continue
                    if formulas and frames[0].record.formula not in formulas:
                        continue
                    files_used += 1
                    levels = frame_label_levels(text, frames)
                    complete = gaussian_job_complete(text)
                    program = gaussian_program(text)
                    for frame, (level, route) in zip(frames, levels):
                        record = frame.record
                        klass = compare_levels(level, reference)
                        signature = (record.formula, record.charge, record.multiplicity,
                                     geometry_signature(record.atoms))
                        geometry_sha = hashlib.sha1(signature[3].encode()).hexdigest()[:16]
                        frame_id = f"{relative}#{record.metadata['force_frame_index']}"
                        duplicate_of = seen_geometries.get(signature, "")
                        if not duplicate_of:
                            seen_geometries[signature] = frame_id
                        summary = local_spin_summary(record)
                        record.record_id = frame_id
                        record.source = relative
                        record.config_type = "legacy_" + calculation_kind(route)
                        record.metadata.update({
                            "parent_record_id": "legacy:" + hashlib.sha1(relative.encode()).hexdigest()[:16],
                            "gaussian_output": relative,
                            "first_route": route,
                            "label_level": level.key,
                            "label_level_class": klass,
                            "calculation_kind": calculation_kind(route),
                            "duplicate_of": duplicate_of,
                            **summary,
                        })
                        forces = [math.sqrt(x * x + y * y + z * z) for x, y, z in frame.forces_ev_ang]
                        rows.append({
                            "source": relative,
                            "force_frame_index": record.metadata["force_frame_index"],
                            "program": program,
                            "formula": record.formula,
                            "charge": record.charge,
                            "multiplicity": record.multiplicity,
                            "n_atoms": len(record.atoms),
                            "energy_ev": f"{frame.energy_ev:.8f}",
                            "rms_force_ev_ang": f"{rms_force(frame):.6f}",
                            "max_force_ev_ang": f"{max(forces, default=0.0):.6f}",
                            "calculation_kind": calculation_kind(route),
                            "functional": level.functional,
                            "basis": level.basis,
                            "basis_functions": level.basis_functions,
                            "grid": level.grid,
                            "extras": ",".join(level.extras),
                            "level_key": level.key,
                            "level_class": klass,
                            "level_evidence": "+".join(level.evidence),
                            "scf_convergence_warning": bool(record.metadata.get("scf_convergence_warning")),
                            "job_normal_termination": complete,
                            "duplicate_of": duplicate_of,
                            "geometry_sha1": geometry_sha,
                            "local_spin_available": summary["local_spin_available"],
                            "mulliken_spin_sum": summary.get("mulliken_spin_sum", ""),
                            "fe_moment_pattern": summary.get("fe_moment_pattern", ""),
                            "route": route,
                        })
                        usable = not record.metadata.get("scf_convergence_warning") and not duplicate_of
                        if usable:
                            bucket = klass if klass != "different_level" else f"different_level__{level.short_id}"
                            by_class.setdefault(bucket, []).append(frame)
                except Exception as exc:  # one unreadable file must not end the audit
                    failures.append((relative, f"{type(exc).__name__}: {exc}"))
    with (output / "label_levels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEVEL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    frames_dir = output / "frames"
    frames_dir.mkdir(exist_ok=True)
    for bucket, members in sorted(by_class.items()):
        write_labeled_extxyz(members, frames_dir / f"{bucket}.extxyz")
    with (output / "failed_outputs.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        for name, message in failures:
            handle.write(f"{name}\t{message}\n")
    summary = summarize_levels(rows, reference, files_seen, files_used, len(failures))
    (output / "label_levels_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (output / "label_levels.md").write_text(
        render_levels_report(summary), encoding="utf-8", newline="\n"
    )
    return summary


def summarize_levels(
    rows: list[dict[str, object]], reference: LabelLevel,
    files_seen: int, files_used: int, failed: int,
) -> dict[str, object]:
    def usable(row) -> bool:
        return not row["scf_convergence_warning"] and not row["duplicate_of"]

    by_class_kind: Counter = Counter()
    usable_by_class_kind: Counter = Counter()
    files_by_class: dict[str, set[str]] = {}
    levels: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = (str(row["level_class"]), str(row["calculation_kind"]))
        by_class_kind[key] += 1
        if usable(row):
            usable_by_class_kind[key] += 1
        files_by_class.setdefault(str(row["level_class"]), set()).add(str(row["source"]))
        level = levels.setdefault(str(row["level_key"]), {
            "class": row["level_class"], "frames": 0, "usable_frames": 0, "files": set(),
            "example_route": row["route"],
        })
        level["frames"] += 1
        level["usable_frames"] += int(usable(row))
        level["files"].add(str(row["source"]))
    same = [row for row in rows if row["level_class"] == "same_level" and usable(row)]
    same_opt = [row for row in same if row["calculation_kind"] == "minimum_optimization"]
    return {
        "reference_level": reference.key,
        "files_scanned": files_seen,
        "files_with_matching_force_frames": files_used,
        "files_failed": failed,
        "frames_total": len(rows),
        "frames_scf_not_converged": sum(bool(row["scf_convergence_warning"]) for row in rows),
        "frames_exact_duplicates": sum(bool(row["duplicate_of"]) for row in rows),
        "same_level_usable_frames": len(same),
        "same_level_usable_minimum_optimization_frames": len(same_opt),
        "same_level_source_files": len({row["source"] for row in same}),
        "same_level_minimum_optimization_source_files": len({row["source"] for row in same_opt}),
        "same_level_frames_by_multiplicity": dict(sorted(Counter(
            str(row["multiplicity"]) for row in same).items(), key=lambda item: int(item[0]))),
        "same_level_frames_with_local_spins": sum(bool(row["local_spin_available"]) for row in same),
        "frames_by_class_and_kind": {
            f"{klass}|{kind}": count for (klass, kind), count in sorted(by_class_kind.items())
        },
        "usable_frames_by_class_and_kind": {
            f"{klass}|{kind}": count for (klass, kind), count in sorted(usable_by_class_kind.items())
        },
        "source_files_by_class": {klass: len(files) for klass, files in sorted(files_by_class.items())},
        "levels": {
            key: {**value, "files": len(value["files"])} for key, value in sorted(levels.items())
        },
    }


def render_levels_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Force-label level audit",
        "",
        f"Reference level: `{summary['reference_level']}`",
        "",
        f"- Gaussian outputs scanned: {summary['files_scanned']}; with matching force frames: "
        f"{summary['files_with_matching_force_frames']}; unreadable: {summary['files_failed']}",
        f"- Force frames: {summary['frames_total']} "
        f"(SCF not converged: {summary['frames_scf_not_converged']}; exact duplicates: "
        f"{summary['frames_exact_duplicates']})",
        f"- **Usable frames at the reference level: {summary['same_level_usable_frames']}** "
        f"from {summary['same_level_source_files']} source files; of these, "
        f"**{summary['same_level_usable_minimum_optimization_frames']} are minimum-optimization "
        f"frames** from {summary['same_level_minimum_optimization_source_files']} files",
        f"- Of the usable reference-level frames, {summary['same_level_frames_with_local_spins']} "
        "carry a Mulliken spin table from their own SCF",
        "",
        "Usable = SCF converged and not an exact geometry/state duplicate of an earlier frame.",
        "Frames at other levels are written to separate `frames/<class>.extxyz` files and must not",
        "be mixed into the reference-level training set.",
        "",
        "## Frames by level class and calculation kind",
        "",
        "| Class | Kind | Frames | Usable |",
        "|---|---|---:|---:|",
    ]
    usable = summary["usable_frames_by_class_and_kind"]
    for key, count in summary["frames_by_class_and_kind"].items():
        klass, kind = key.split("|")
        lines.append(f"| {klass} | {kind} | {count} | {usable.get(key, 0)} |")
    lines += ["", "## Distinct levels", "", "| Class | Level | Frames | Usable | Files | Example route |",
              "|---|---|---:|---:|---:|---|"]
    for key, value in summary["levels"].items():
        route = str(value["example_route"]).replace("|", "\\|")
        lines.append(
            f"| {value['class']} | `{key}` | {value['frames']} | {value['usable_frames']} | "
            f"{value['files']} | `{route[:140]}` |"
        )
    lines += ["", "## Reference-level frames by multiplicity", "", "| M | Frames |", "|---:|---:|"]
    for multiplicity, count in summary["same_level_frames_by_multiplicity"].items():
        lines.append(f"| {multiplicity} | {count} |")
    return "\n".join(lines) + "\n"
