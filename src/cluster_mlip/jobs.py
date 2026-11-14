from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import replace
from pathlib import Path

from .basis import render_gen_basis
from .io import write_text_lf
from .models import Atom, Record
from .routes import (
    corrected_minimum_route,
    corrected_saddle_route,
    force_only_route,
    intended_stationary_point,
    resolve_geometry_role,
    route_optimizes,
    route_policy,
    route_search_kind,
)


_LEGACY_SCF = "SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc)"
_LEGACY_IOP = "IOP(5/13=1,5/36=1,8/11=1)"

# Keep the historical Gaussian 09-era BPW91 protocol used for the source
# calculations.  The unperturbed records may be reoptimized and checked by a
# frequency calculation, whereas rattled records must retain their displaced
# geometry so that they contribute useful nonzero forces.
#
# The basis is Gen (see basis.py): 6-31G* for light elements, an explicit
# def2-TZVP-without-f contraction for Fe. Every route below reads it from
# the input body rather than naming a basis keyword.
DEFAULT_ROUTE = (
    f"#p UBPW91/Gen {_LEGACY_SCF} NoSymm Opt Freq {_LEGACY_IOP} Int=UltraFine"
)
# A saddle point needs its own search. DEFAULT_ROUTE's plain Opt walks downhill
# to the nearest minimum, which for a transition-state seed silently discards
# the stationary point the record exists for: the job terminates normally and
# yields a converged geometry that is simply the wrong one. TS selects the
# saddle search, CalcFC supplies a starting Hessian with usable curvature, and
# NoEigenTest keeps Gaussian from aborting a guess whose estimated Hessian does
# not already have exactly one negative eigenvalue.
DEFAULT_SADDLE_ROUTE = (
    f"#p UBPW91/Gen {_LEGACY_SCF} NoSymm Opt=(TS,CalcFC,NoEigenTest) Freq "
    f"{_LEGACY_IOP} Int=UltraFine"
)
DEFAULT_RATTLE_ROUTE = (
    f"#p UBPW91/Gen SP {_LEGACY_SCF} NoSymm {_LEGACY_IOP} Int=UltraFine"
)
DEFAULT_LINK1_ROUTE = (
    f"#p UBPW91/Gen Force {_LEGACY_SCF} NoSymm "
    f"Guess=Read Geom=Checkpoint {_LEGACY_IOP} Int=UltraFine"
)


def _geometry_sha256(record: Record) -> str:
    from .models import geometry_signature

    return hashlib.sha256(geometry_signature(record.atoms).encode()).hexdigest()


def _rattle(record: Record, sigma: float, seed: int, variant: int) -> Record:
    seed_text = f"{seed}|{record.record_id}|rattle|{variant}|{sigma:.12g}"
    resolved_seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big")
    rng = random.Random(resolved_seed)
    atoms = [
        Atom(a.symbol, a.x + rng.gauss(0, sigma), a.y + rng.gauss(0, sigma), a.z + rng.gauss(0, sigma))
        for a in record.atoms
    ]
    suffix = hashlib.sha1(seed_text.encode()).hexdigest()[:8]
    return Record(
        record_id=f"{record.record_id}-r{variant:02d}-{suffix}",
        source=record.source,
        atoms=atoms,
        charge=record.charge,
        multiplicity=record.multiplicity,
        config_type=f"{record.config_type}_rattled",
        route=record.route,
        legacy_energy_hartree=record.legacy_energy_hartree,
        imaginary_frequencies=record.imaginary_frequencies,
        irc_path=record.irc_path,
        irc_point=record.irc_point,
        electronic_state=record.electronic_state,
        metadata={
            **record.metadata,
            "parent_record_id": record.record_id,
            "source_record_id": record.record_id,
            "parent_geometry_sha256": _geometry_sha256(record),
            "campaign_seed": str(seed),
            "rattle_index": str(variant),
            "rattle_sigma": str(sigma),
            "resolved_rattle_seed": str(resolved_seed),
            "variant": f"r{variant:02d}",
        },
    )


def expanded_records(records: list[Record], rattles_per_seed: int, sigma: float, seed: int) -> list[Record]:
    expanded: list[Record] = []
    for record in records:
        expanded.append(
            replace(
                record,
                metadata={
                    **record.metadata,
                    "source_record_id": record.record_id,
                    "campaign_seed": str(seed),
                    "variant": "reference",
                },
            )
        )
        for variant in range(1, rattles_per_seed + 1):
            expanded.append(_rattle(record, sigma, seed, variant))
    return expanded


def _slug(value: str, limit: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return (cleaned or "unknown")[:limit].rstrip("-")


def human_job_stem(record: Record) -> str:
    """Readable, collision-safe filename stem without replacing machine identity."""
    source = _slug(Path(record.source).stem, 64)
    config_type = _slug(record.config_type.removesuffix("_rattled"), 32)
    variant = _slug(record.metadata.get("variant", "reference"), 16)
    charge = "0" if record.charge == 0 else f"{record.charge:+d}"
    state = f"q{charge}-m{record.multiplicity}"
    identity = hashlib.sha1(record.record_id.encode()).hexdigest()[:10]
    return "__".join(
        (source, _slug(record.formula, 32), config_type, state, variant, identity)
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


HIGHER_ORDER_POLICIES = ("force", "saddle-search")


def record_geometry(record: Record) -> dict[str, str]:
    """The geometry-role provenance columns for one record."""
    resolution = resolve_geometry_role(
        record.config_type, record.metadata, source=record.source, route=record.route
    )
    policy = route_policy(
        record.config_type, resolution["geometry_role"],
        resolution["source_calculation_type"],
    )
    meta = record.metadata

    def text(key: str, default: object = "") -> str:
        value = meta.get(key, default)
        return "" if value is None else str(value)

    return {
        "geometry_role": resolution["geometry_role"],
        "geometry_role_source": resolution["geometry_role_source"],
        "source_calculation_type": resolution["source_calculation_type"],
        "route_policy": policy,
        "requested_saddle_order": "",
        "irc_direction": text("irc_direction"),
        "irc_point_index": "" if record.irc_point is None else str(record.irc_point),
        "irc_path_position": text("irc_path_position"),
        "irc_parent_record_id": text("irc_parent_record_id"),
        "original_charge": text("original_charge", record.charge),
        "original_multiplicity": text("original_multiplicity", record.multiplicity),
    }


def higher_order_saddle_route(record: Record, route: str) -> tuple[str, int]:
    """``Opt=(Saddle=N,...)`` for a higher-order candidate whose N is known.

    N is the record's own archived imaginary-mode count; a candidate without
    one cannot be searched for, because Gaussian needs the order up front.
    """
    order = record.imaginary_frequencies
    if order is None or order < 2:
        raise ValueError(
            f"{record.record_id}: --higher-order-policy saddle-search needs the record's "
            f"imaginary-mode count (>= 2); it has {order!r}. Label it with Force instead"
        )
    searched = corrected_saddle_route(corrected_minimum_route(route), order)
    if route_search_kind(searched) != "saddle":
        raise ValueError(
            f"{record.record_id}: cannot build an Opt=(Saddle={order}) search from a route "
            f"that does not optimize: {route!r}"
        )
    return searched, order


def first_stage_route(
    record: Record,
    route: str,
    rattle_route: str,
    saddle_route: str,
    higher_order_policy: str = "force",
) -> tuple[str, str]:
    """Pick the first-stage route that matches what the record actually is.

    Returns the route and the intent it serves, which goes into jobs.csv so an
    audit can tell a deliberate choice from an accident. Rattled variants keep
    the single-point route whatever their parent was: their value is the
    displaced geometry, so nothing about them may be optimized. Reaction-path
    frames, and higher-order candidates unless a saddle search is requested,
    get the seed route rewritten to a fixed-geometry Force job.
    """
    if "rattle_index" in record.metadata:
        return rattle_route, "displaced_single_point"
    policy = record_geometry(record)["route_policy"]
    if policy == "fixed_geometry":
        return force_only_route(route), "fixed_geometry_force"
    if policy == "higher_order":
        if higher_order_policy == "saddle-search":
            return higher_order_saddle_route(record, saddle_route)[0], "higher_order_saddle"
        return force_only_route(route), "fixed_geometry_force"
    if intended_stationary_point(record.config_type) == "saddle":
        return saddle_route, "saddle"
    return route, "minimum" if route_optimizes(route) else "single_point"


def write_gaussian_jobs(
    records: list[Record],
    output: Path,
    route: str = DEFAULT_ROUTE,
    memory: str = "16GB",
    nproc: int = 16,
    rattle_route: str = DEFAULT_RATTLE_ROUTE,
    link1_route: str = DEFAULT_LINK1_ROUTE,
    saddle_route: str = DEFAULT_SADDLE_ROUTE,
    higher_order_policy: str = "force",
) -> None:
    if higher_order_policy not in HIGHER_ORDER_POLICIES:
        raise ValueError(f"higher-order policy must be one of {HIGHER_ORDER_POLICIES}")
    saddle_records = [
        record for record in records
        if "rattle_index" not in record.metadata
        and record_geometry(record)["route_policy"] == "saddle"
    ]
    if saddle_records and route_search_kind(saddle_route) != "saddle":
        raise ValueError(
            f"{len(saddle_records)} seed(s) are labeled saddle points, but the saddle route "
            f"requests a {route_search_kind(saddle_route)} search: {saddle_route!r}; a plain Opt "
            "relaxes a transition state to the nearest minimum. Supply a route with "
            "Opt=(TS,CalcFC,NoEigenTest), or exclude those types with --types."
        )
    # Every route is decided before anything is written: a record that cannot
    # be given its search (a saddle-search request without a known order)
    # must fail the whole preparation, not leave half a campaign on disk.
    planned = {
        record.record_id: first_stage_route(
            record, route, rattle_route, saddle_route, higher_order_policy
        )
        for record in records
    }
    output.mkdir(parents=True, exist_ok=True)
    protected_suffixes = {".log", ".out", ".chk", ".status", ".rc", ".started", ".finished"}
    protected = [path for path in output.rglob("*") if path.is_file() and path.suffix.lower() in protected_suffixes]
    if protected:
        preview = ", ".join(str(path.relative_to(output)) for path in protected[:5])
        raise RuntimeError(
            "refusing to regenerate a campaign that already has calculation state; "
            f"use a fresh output directory or archive the existing campaign first: {preview}"
        )
    manifest = output / "jobs.csv"
    rows: list[dict[str, object]] = []
    used_stems: set[str] = set()
    for record in records:
        stem = human_job_stem(record)
        if stem in used_stems:
            raise ValueError(f"human-readable job filename collision: {stem}")
        used_stems.add(stem)
        filename = f"{stem}.gjf"
        path = output / filename
        parent = record.metadata.get("parent_record_id", record.record_id)
        first_route, route_intent = planned[record.record_id]
        geometry = record_geometry(record)
        if route_intent == "higher_order_saddle":
            geometry["requested_saddle_order"] = str(
                higher_order_saddle_route(record, saddle_route)[1]
            )
        # Every stage of a fixed-geometry job is a Force job at that geometry.
        stage_link1_route = (
            force_only_route(link1_route) if route_intent == "fixed_geometry_force"
            else link1_route
        )
        gen_basis = render_gen_basis({a.symbol for a in record.atoms})
        lines = [
            f"%chk={stem}.chk",
            f"%mem={memory}",
            f"%nprocshared={nproc}",
            first_route,
            "",
            (
                f"MLIP label human_id={stem}; job_id={record.record_id}; "
                f"parent={parent}; source={record.source}; type={record.config_type}"
            ),
            "",
            f"{record.charge} {record.multiplicity}",
        ]
        lines.extend(f"{a.symbol:3s} {a.x: .12f} {a.y: .12f} {a.z: .12f}" for a in record.atoms)
        lines.append(gen_basis.rstrip("\n"))
        lines.extend(
            [
                "",
                "--Link1--",
                f"%chk={stem}.chk",
                f"%mem={memory}",
                f"%nprocshared={nproc}",
                stage_link1_route,
                "",
                f"MLIP diffuse-basis force label human_id={stem}; job_id={record.record_id}",
                "",
                f"{record.charge} {record.multiplicity}",
                gen_basis.rstrip("\n"),
                "",
            ]
        )
        write_text_lf(path, "\n".join(lines))
        rows.append(
            {
                "human_id": stem,
                "job_id": record.record_id,
                "source_record_id": record.metadata.get("source_record_id", parent),
                "parent_record_id": parent,
                "source": record.source,
                "source_basename": Path(record.source).name,
                "config_type": record.config_type,
                "formula": record.formula,
                "n_atoms": len(record.atoms),
                "charge": record.charge,
                "multiplicity": record.multiplicity,
                "variant": record.metadata.get("variant", "reference"),
                "rattle_index": record.metadata.get("rattle_index", ""),
                "rattle_sigma_angstrom": record.metadata.get("rattle_sigma", ""),
                "campaign_seed": record.metadata.get("campaign_seed", ""),
                "resolved_rattle_seed": record.metadata.get("resolved_rattle_seed", ""),
                "parent_geometry_sha256": record.metadata.get(
                    "parent_geometry_sha256", _geometry_sha256(record)
                ),
                "input_geometry_sha256": _geometry_sha256(record),
                "legacy_energy_hartree": (
                    "" if record.legacy_energy_hartree is None else record.legacy_energy_hartree
                ),
                "legacy_route": record.route,
                "state_inference": record.metadata.get("state_inference", ""),
                "intended_stationary_point": intended_stationary_point(record.config_type),
                **geometry,
                "route_intent": route_intent,
                "route_search_kind": route_search_kind(first_route),
                "first_route": first_route,
                "link1_route": stage_link1_route,
                "input": filename,
                "input_sha256": _file_sha256(path),
                "output": f"{stem}.log",
            }
        )

    with manifest.open("w", newline="", encoding="utf-8") as table:
        columns = list(rows[0]) if rows else ["human_id", "job_id", "source"]
        writer = csv.DictWriter(table, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    campaign = {
        "schema_version": 1,
        "job_count": len(rows),
        "source_record_count": len({str(row["source_record_id"]) for row in rows}),
        "campaign_seeds": sorted({str(row["campaign_seed"]) for row in rows}),
        "memory": memory,
        "nprocshared": nproc,
        "seed_route": route,
        "saddle_route": saddle_route,
        "higher_order_policy": higher_order_policy,
        "rattle_route": rattle_route,
        "link1_route": link1_route,
        "jobs_csv_sha256": _file_sha256(manifest),
    }
    (output / "campaign_manifest.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    runner = output / "run_one.sh"
    write_text_lf(
        runner,
        "#!/usr/bin/env bash\nset -euo pipefail\ninput=$1\noutput=${input%.gjf}.log\ng16 \"$input\" > \"$output\"\n",
    )
    runner.chmod(0o755)
