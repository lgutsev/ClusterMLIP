from __future__ import annotations

import csv
import json
import re
import shutil
import shlex
import subprocess
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree

from .models import Record


SUPPORTED_SUFFIXES = {
    ".log", ".out", ".txt", ".doc", ".docx", ".com", ".gjf", ".fchk", ".fch", ".chk"
}

DEFAULT_NESTED_ZIP_DEPTH = 4
DEFAULT_MAX_UNCOMPRESSED_BYTES = 100 * 1024**3


def _decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "cp1252", "mac_roman", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1", errors="ignore")


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".chk":
        formchk = shutil.which("formchk")
        if formchk is None:
            raise RuntimeError(
                "Gaussian formchk is required to read .chk files; load Gaussian or convert them to .fchk first"
            )
        with tempfile.TemporaryDirectory(prefix="cluster_mlip_formchk_") as tmp:
            output = Path(tmp) / f"{path.stem}.fchk"
            subprocess.run([formchk, str(path), str(output)], check=True, capture_output=True)
            return _decode_bytes(output.read_bytes())
    if suffix == ".docx":
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
        paragraphs: list[str] = []
        for paragraph in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
            text = "".join(
                node.text or ""
                for node in paragraph.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
            )
            paragraphs.append(text)
        return "\n".join(paragraphs)
    if suffix == ".doc":
        # Legacy Word files in this archive retain long ASCII Gaussian excerpts.
        # `strings` removes the OLE container without requiring Word libraries.
        result = subprocess.run(
            ["strings", "-n", "4", str(path)],
            check=True,
            capture_output=True,
        )
        return _decode_bytes(result.stdout)
    return _decode_bytes(path.read_bytes())


def _safe_extract(
    archive: zipfile.ZipFile,
    destination: Path,
    budget: list[int] | None = None,
) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        if budget is not None:
            budget[0] -= member.file_size
            if budget[0] < 0:
                raise ValueError("ZIP expansion exceeds the configured uncompressed-size limit")
        target = (destination / member.filename).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f"Unsafe ZIP member: {member.filename}")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open("wb") as sink:
            while chunk := source.read(1024 * 1024):
                sink.write(chunk)


def _expand_nested_zips(root: Path, max_depth: int, budget: list[int]) -> None:
    """Extract nested ZIPs alongside their archive without following unsafe paths."""
    frontier = sorted(root.rglob("*.zip"))
    seen: set[Path] = set()
    for depth in range(max_depth):
        next_frontier: list[Path] = []
        for archive_path in frontier:
            archive_path = archive_path.resolve()
            if archive_path in seen:
                continue
            seen.add(archive_path)
            destination = archive_path.with_name(f"{archive_path.name}.contents")
            destination.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    _safe_extract(archive, destination, budget)
            except (OSError, zipfile.BadZipFile):
                # A mislabeled/corrupt nested archive remains in the inventory;
                # it must not abort an otherwise useful scan.
                continue
            next_frontier.extend(sorted(destination.rglob("*.zip")))
        frontier = next_frontier
        if not frontier:
            return
    if frontier:
        raise ValueError(f"nested ZIP depth exceeds configured limit ({max_depth})")


@contextmanager
def source_tree(
    source: Path,
    nested_zip_depth: int = DEFAULT_NESTED_ZIP_DEPTH,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES,
) -> Iterator[Path]:
    if source.is_dir():
        yield source
        return
    if source.suffix.lower() != ".zip":
        yield source.parent
        return
    with tempfile.TemporaryDirectory(prefix="cluster_mlip_") as tmp:
        root = Path(tmp)
        budget = [max_uncompressed_bytes]
        with zipfile.ZipFile(source) as archive:
            _safe_extract(archive, root, budget)
        if nested_zip_depth:
            _expand_nested_zips(root, nested_zip_depth, budget)
        yield root


def iter_documents(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("._"):
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def quote_extxyz(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def write_extxyz(records: list[Record], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            fields = [
                "Properties=species:S:1:pos:R:3",
                f"record_id={quote_extxyz(rec.record_id)}",
                f"source={quote_extxyz(rec.source)}",
                f"formula={quote_extxyz(rec.formula)}",
                f"config_type={quote_extxyz(rec.config_type)}",
                f"charge={rec.charge}",
                f"spin={rec.multiplicity}",
                f"multiplicity={rec.multiplicity}",
                f"total_spin={rec.total_spin}",
                'pbc="F F F"',
            ]
            if rec.legacy_energy_hartree is not None:
                fields.append(f"legacy_energy_hartree={rec.legacy_energy_hartree:.14g}")
            if rec.imaginary_frequencies is not None:
                fields.append(f"imaginary_frequencies={rec.imaginary_frequencies}")
            if rec.irc_path is not None:
                fields.append(f"irc_path={rec.irc_path}")
            if rec.irc_point is not None:
                fields.append(f"irc_point={rec.irc_point}")
            if rec.route:
                fields.append(f"legacy_route={quote_extxyz(rec.route)}")
            handle.write(f"{len(rec.atoms)}\n{' '.join(fields)}\n")
            for atom in rec.atoms:
                handle.write(f"{atom.symbol:3s} {atom.x: .12f} {atom.y: .12f} {atom.z: .12f}\n")


def write_manifest(records: list[Record], path: Path) -> None:
    columns = [
        "record_id", "source", "formula", "n_atoms", "charge", "multiplicity",
        "total_spin", "config_type", "legacy_energy_hartree",
        "imaginary_frequencies", "irc_path", "irc_point", "route",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "record_id": rec.record_id,
                "source": rec.source,
                "formula": rec.formula,
                "n_atoms": len(rec.atoms),
                "charge": rec.charge,
                "multiplicity": rec.multiplicity,
                "total_spin": rec.total_spin,
                "config_type": rec.config_type,
                "legacy_energy_hartree": rec.legacy_energy_hartree,
                "imaginary_frequencies": rec.imaginary_frequencies,
                "irc_path": rec.irc_path,
                "irc_point": rec.irc_point,
                "route": rec.route,
            })


_INFO_RE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')


def read_extxyz(path: Path) -> list[Record]:
    from .models import Atom

    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[Record] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n_atoms = int(lines[i].strip())
        info_line = lines[i + 1]
        info: dict[str, str] = {}
        for key, raw in _INFO_RE.findall(info_line):
            info[key] = json.loads(raw) if raw.startswith('"') else raw
        atoms = []
        for line in lines[i + 2:i + 2 + n_atoms]:
            parts = line.split()
            atoms.append(Atom(parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
        records.append(Record(
            record_id=info["record_id"],
            source=info.get("source", ""),
            atoms=atoms,
            charge=int(info.get("charge", 0)),
            multiplicity=int(info.get("multiplicity", info.get("spin", 1))),
            config_type=info.get("config_type", "unknown"),
            route=info.get("legacy_route", ""),
            legacy_energy_hartree=float(info["legacy_energy_hartree"]) if "legacy_energy_hartree" in info else None,
            imaginary_frequencies=int(info["imaginary_frequencies"]) if "imaginary_frequencies" in info else None,
            irc_path=int(info["irc_path"]) if "irc_path" in info else None,
            irc_point=int(info["irc_point"]) if "irc_point" in info else None,
        ))
        i += n_atoms + 2
    return records
