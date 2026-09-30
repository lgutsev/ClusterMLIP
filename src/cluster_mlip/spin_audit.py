"""Read-only tests of whether (geometry, charge, multiplicity) identifies E/F.

numpy is optional and imported only when this command is used. Reports describe
label evidence, not magnetic ground states or validated SCF stability.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import read_labeled_extxyz
from .gaussian import parse_force_frames
from .io import parse_extxyz_info_line


def coordinates(frame):
    return np.array([(a.x, a.y, a.z) for a in frame.record.atoms], dtype=float)


def spins(frame):
    """Reject partial/misindexed populations; never sort spin values by magnitude."""
    rows = frame.record.metadata.get("atomic_spins")
    if not isinstance(rows, list) or len(rows) != len(frame.record.atoms):
        return None
    try:
        by_id = {int(row[0]): (row[1], float(row[2])) for row in rows}
        if len(by_id) != len(rows):
            return None
        values = []
        for i, atom in enumerate(frame.record.atoms, 1):
            symbol, value = by_id[i]
            if symbol != atom.symbol or not np.isfinite(value):
                return None
            values.append(value)
        return np.array(values)
    except (KeyError, ValueError, TypeError, IndexError):
        return None


def fingerprint(frame):
    x = coordinates(frame)
    d = np.linalg.norm(x[:, None] - x[None, :], axis=2)
    symbols = [a.symbol for a in frame.record.atoms]
    pairs = defaultdict(list)
    for i in range(len(x)):
        for j in range(i):
            pairs[tuple(sorted((symbols[i], symbols[j])))].append(d[i, j])
    return np.array([v for key in sorted(pairs) for v in sorted(pairs[key])]), d


def alignments(left, right, tolerance, budget):
    """Exhaust bounded distance-graph isomorphisms and verify by proper Kabsch.

    Fingerprints only prune candidates. A mapping must have maximum atomic
    displacement <= tolerance after alignment. All symmetry-equivalent mappings
    are considered; capped searches are explicitly inconclusive.
    """
    x, y = coordinates(left), coordinates(right)
    x, y = x - x.mean(axis=0), y - y.mean(axis=0)
    _, dx = fingerprint(left)
    _, dy = fingerprint(right)
    sx = [a.symbol for a in left.record.atoms]
    sy = [a.symbol for a in right.record.atoms]
    elements = sorted(set(sx))
    descriptors = lambda d, s: [np.concatenate([np.sort(d[i, np.array(s) == e])
                                                for e in elements]) for i in range(len(s))]
    ax, ay = descriptors(dx, sx), descriptors(dy, sy)
    candidates = [[j for j in range(len(y)) if sx[i] == sy[j]
                   and np.max(np.abs(ax[i] - ay[j])) <= 2 * tolerance + 1e-12]
                  for i in range(len(x))]
    order = sorted(range(len(x)), key=lambda i: len(candidates[i]))
    mapping: dict[int, int] = {}
    used: set[int] = set()
    results: list[Any] = []
    visited, capped = 0, False

    def visit(depth):
        nonlocal visited, capped
        visited += 1
        if visited > budget:
            capped = True
            return
        if depth == len(x):
            p = [mapping[i] for i in range(len(x))]
            u, _, vt = np.linalg.svd(y[p].T @ x)
            correction = np.eye(3)
            correction[-1, -1] = np.linalg.det(u @ vt)
            rotation = u @ correction @ vt
            delta = y[p] @ rotation - x
            maximum = float(np.max(np.linalg.norm(delta, axis=1)))
            if maximum <= tolerance + 1e-12:
                results.append((p, rotation, maximum,
                                float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))))
            return
        i = order[depth]
        for j in candidates[i]:
            if j in used or any(abs(dx[i, k] - dy[j, v]) > 2*tolerance + 1e-12
                                for k, v in mapping.items()):
                continue
            mapping[i] = j
            used.add(j)
            visit(depth + 1)
            used.remove(j)
            del mapping[i]
            if capped:
                return
    visit(0)
    return results, capped


def compare(left, right, exact, near, budget):
    matches, capped = alignments(left, right, exact, budget)
    kind = "exact"
    if not matches and not capped:
        matches, capped = alignments(left, right, near, budget)
        kind = "near"
    if not matches:
        return {"geometry": "unresolved", "search_capped": True} if capped else None
    ls, rs = spins(left), spins(right)
    force_min, spin_min = float("inf"), float("inf")
    spin_force, best_mapping = None, None
    # Linear/single-atom geometries have an undetermined rotation around an axis.
    # Do not make force-conflict claims using an arbitrary SVD rotation.
    force_identifiable = np.linalg.matrix_rank(coordinates(left) - coordinates(left).mean(0), tol=1e-8) >= 2
    for p, rotation, _, _ in matches:
        f = float(np.sqrt(np.mean((np.array(left.forces_ev_ang) -
                                  np.array(right.forces_ev_ang)[p] @ rotation)**2)))
        force_min = min(force_min, f)
        if ls is not None and rs is not None:
            # Global reversal is physically equivalent in zero field.
            s = min(float(np.sqrt(np.mean((ls - sign*rs[p])**2))) for sign in (1, -1))
            if s < spin_min:
                spin_min, spin_force, best_mapping = s, f, p
    return {"geometry": kind, "search_capped": capped,
            "geometry_rms_A": min(m[3] for m in matches),
            "geometry_max_A": min(m[2] for m in matches),
            "mappings": len(matches), "delta_energy_eV": abs(left.energy_ev-right.energy_ev),
            "force_component_rms_eV_A": force_min if force_identifiable else None,
            "spin_rms": spin_min if np.isfinite(spin_min) else None,
            "force_at_best_spin_mapping_eV_A": spin_force if force_identifiable else None,
            "best_spin_mapping_right_indices_0based": best_mapping}


def verify_raw(frame, root, cache):
    """Reparse the original force index; no nearest-frame or basename fallback."""
    m = frame.record.metadata
    if "gaussian_output" not in m or "force_frame_index" not in m:
        return "missing_provenance"
    base = root if root is not None else Path(m.get("collection_campaign", "."))
    path = base / m["gaussian_output"]
    if path not in cache:
        if not path.is_file():
            cache[path] = None
        else:
            cache[path] = {f.record.metadata["force_frame_index"]: f
                           for f in parse_force_frames(path.read_text(errors="replace"), path)}
    if cache[path] is None:
        return "missing_log"
    raw = cache[path].get(int(m["force_frame_index"]))
    if raw is None:
        return "missing_raw_frame"
    if (raw.record.charge != frame.record.charge or raw.record.multiplicity != frame.record.multiplicity
            or [a.symbol for a in raw.record.atoms] != [a.symbol for a in frame.record.atoms]
            or not np.allclose(coordinates(raw), coordinates(frame), atol=1e-7, rtol=0)
            or abs(raw.energy_ev - frame.energy_ev) > 1e-6
            or not np.allclose(raw.forces_ev_ang, frame.forces_ev_ang, atol=1e-7, rtol=0)):
        return "label_mismatch"
    a, b = spins(frame), spins(raw)
    if b is None:
        return "raw_spin_missing" if a is None else "spin_not_supported_by_raw_frame"
    if a is None:
        return "dataset_spin_missing"
    return "verified" if np.allclose(a, b, atol=1e-6, rtol=0) else "spin_mismatch"


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_input(path):
    """Do not let the legacy reader silently assume state or column semantics."""
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i])
        if n < 1 or i + n + 2 > len(lines):
            raise ValueError(f"Truncated/empty frame in {path} at line {i+1}")
        info = parse_extxyz_info_line(lines[i+1])
        if info.get("Properties") != "species:S:1:pos:R:3:REF_forces:R:3":
            raise ValueError(f"{path}: expected ClusterMLIP species/pos/REF_forces columns")
        if "charge" not in info or not ({"multiplicity", "spin"} & info.keys()):
            raise ValueError(f"{path}: explicit charge and multiplicity/spin required")
        multiplicity = int(info["multiplicity"] if "multiplicity" in info else info["spin"])
        if multiplicity < 1 or ("spin" in info and int(info["spin"]) != multiplicity):
            raise ValueError(f"{path}: invalid/inconsistent multiplicity")
        if any(v.lower() not in {"f", "false", "0"} for v in info.get("pbc", "F F F").split()):
            raise ValueError("This audit supports nonperiodic clusters only")
        if any(len(line.split()) != 7 for line in lines[i+2:i+n+2]):
            raise ValueError(f"{path}: malformed atom/force columns")
        i += n + 2


def load_sources(args):
    paths = [Path(p).resolve() for p in args.datasets]
    if len(set(paths)) != len(paths):
        raise ValueError("Repeated input path")
    frames, origins, hashes = [], [], {}
    inventory: list[dict[str, Any]] = []
    if args.gaussian:
        if args.verify_raw or args.raw_root:
            raise ValueError("--gaussian reads raw logs directly; omit --verify-raw/--raw-root")
        expanded: set[Path] = set()
        for path in paths:
            if path.is_dir():
                for pattern in args.file_glob or ["*.log", "*.out"]:
                    expanded.update(p.resolve() for p in path.rglob(pattern) if p.is_file() and not p.is_symlink())
            elif path.is_file():
                expanded.add(path)
            else:
                raise FileNotFoundError(path)
        paths = sorted(expanded)
    for path in paths:
        content = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
        if args.gaussian:
            text = content.decode("utf-8", errors="replace")
            try:
                loaded = parse_force_frames(text, path)
            except (ValueError, IndexError) as exc:
                inventory.append({"path": str(path), "status": "parse_error", "detail": str(exc)})
                continue
            selected = [f for f in loaded if (not args.formula or f.record.formula == args.formula)
                        and f.record.metadata.get("explicit_charge_multiplicity")]
            if args.gaussian_frames == "last-per-section":
                last = {f.record.metadata["gaussian_section_index"]: f for f in selected}
                selected = list(last.values())
            inventory.append({"path": str(path), "status": "parsed" if selected else "no_selected_force_frames",
                              "force_frames": len(loaded), "selected_frames": len(selected),
                              "missing_explicit_state": sum(not f.record.metadata.get("explicit_charge_multiplicity") for f in loaded),
                              "normal_terminations": text.lower().count("normal termination of gaussian"),
                              "error_terminations": text.lower().count("error termination"),
                              "detail": ""})
            for f in selected:
                index = f.record.metadata["force_frame_index"]
                f.record.record_id = f"{path.name}__frame{index:06d}"
                f.record.metadata.update(gaussian_output=path.name, collection_campaign=str(path.parent))
            loaded = selected
        else:
            validate_input(path)
            loaded = read_labeled_extxyz(path)
        for index, f in enumerate(loaded):
            if args.formula and f.record.formula != args.formula:
                continue
            if not f.record.atoms or not np.isfinite(f.energy_ev) or not np.isfinite(coordinates(f)).all() or not np.isfinite(f.forces_ev_ang).all():
                raise ValueError(f"Nonfinite or empty frame: {path}:{index}")
            if len(f.forces_ev_ang) != len(f.record.atoms):
                raise ValueError(f"Force length mismatch: {path}:{index}")
            frames.append(f)
            origins.append((str(path), f.record.metadata["force_frame_index"] if args.gaussian else index))
        if args.gaussian and len(inventory) % 25 == 0:
            print(f"Read {len(inventory)} logs; selected {len(frames)} force frames", flush=True)
    return frames, origins, hashes, inventory


def run_audit(args):
    for key in ("exact_tolerance", "near_tolerance", "energy_threshold", "force_threshold",
                "spin_threshold", "spin_sum_tolerance"):
        value = getattr(args, key)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if args.near_tolerance < args.exact_tolerance or min(args.mapping_budget, args.max_pairs) < 1:
        raise ValueError("near tolerance must be >= exact tolerance; budgets must be positive")
    if args.file_glob and not args.gaussian:
        raise ValueError("--file-glob requires --gaussian")
    if args.raw_root and not args.verify_raw:
        raise ValueError("--raw-root requires --verify-raw")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new audit directory")
    frames, origins, hashes, inventory = load_sources(args)
    output.mkdir(parents=True, exist_ok=True)
    if args.gaussian:
        write_csv(output / "input_logs.csv", inventory, ["path", "status", "force_frames", "selected_frames",
                  "missing_explicit_state", "normal_terminations", "error_terminations", "detail"])
    if not frames:
        raise ValueError("No frames selected; see input_logs.csv for raw-mode coverage")
    cache: dict[Path, Any] = {}
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    fingerprints = []
    for i, frame in enumerate(frames):
        r, s = frame.record, spins(frame)
        m = r.metadata
        status = verify_raw(frame, Path(args.raw_root) if args.raw_root else None, cache) if args.verify_raw else "not_checked"
        if args.gaussian:
            status = "direct_raw" if s is not None else "raw_spin_missing"
        row = {"frame": i, "dataset": origins[i][0], "dataset_frame": origins[i][1],
               "record_id": r.record_id, "formula": r.formula, "charge": r.charge,
               "multiplicity": r.multiplicity, "energy_eV": frame.energy_ev,
               "force_component_rms_eV_A": float(np.sqrt(np.mean(np.array(frame.forces_ev_ang)**2))),
               "spin_complete": s is not None,
               "spin_sum_error": None if s is None else abs(float(s.sum()) - (r.multiplicity-1)),
               "label_route": str(m.get("force_route") or m.get("first_route") or m.get("link1_route") or ""),
               "s2_before": m.get("s2_before"), "s2_after": m.get("s2_after"),
               "raw_verification": status, "gaussian_output": m.get("gaussian_output", ""),
               "force_frame_index": m.get("force_frame_index"),
               "gaussian_section_index": m.get("gaussian_section_index"),
               "section_normal_termination": m.get("section_normal_termination"),
               "section_error_termination": m.get("section_error_termination"),
               "section_optimized": m.get("section_optimized"),
               "scf_warning": m.get("scf_convergence_warning", False)}
        rows.append(row)
        groups[(r.formula, r.charge)].append(i)
        fingerprints.append(fingerprint(frame)[0])
    cache.clear()
    pairs: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    examined, incomplete = 0, False
    for members in groups.values():
        # Diameter window is a necessary condition, never proof of equivalence.
        members.sort(key=lambda i: float(np.max(fingerprints[i])) if len(fingerprints[i]) else 0)
        for offset, i in enumerate(members):
            for j in members[offset+1:]:
                if len(fingerprints[i]) and np.max(fingerprints[j])-np.max(fingerprints[i]) > 2*args.near_tolerance:
                    break
                examined += 1
                if examined % 100000 == 0:
                    print(f"Examined {examined} candidate pairs; {len(pairs)} matches", flush=True)
                if examined > args.max_pairs:
                    incomplete = True
                    break
                if len(fingerprints[i]) and np.max(np.abs(fingerprints[i]-fingerprints[j])) > 2*args.near_tolerance:
                    continue
                result = compare(frames[i], frames[j], args.exact_tolerance, args.near_tolerance, args.mapping_budget)
                if result is None:
                    continue
                same = frames[i].record.multiplicity == frames[j].record.multiplicity
                result.update(left=i, right=j, same_multiplicity=same)
                routes = [rows[k]["label_route"] for k in (i, j)]
                result["route_comparison"] = ("missing" if not all(routes) else
                                               "same" if routes[0] == routes[1] else "different_review_required")
                result["evidence"] = "near_geometry_only"
                if result["search_capped"]:
                    result["evidence"] = "mapping_inconclusive"
                elif not same:
                    result["evidence"] = "different_multiplicity_control"
                elif result["geometry"] == "exact":
                    conflict = (result["delta_energy_eV"] > args.energy_threshold or
                                (result["force_component_rms_eV_A"] is not None and
                                 result["force_component_rms_eV_A"] > args.force_threshold))
                    different = result["spin_rms"] is not None and result["spin_rms"] > args.spin_threshold
                    result["evidence"] = ("label_conflict_with_distinct_spins" if conflict and different else
                                          "label_conflict_spin_unresolved" if conflict else
                                          "distinct_spins_without_large_EF_difference" if different else
                                          "consistent_with_thresholds")
                # Bad/missing sums and raw mismatches cannot support a spin attribution.
                invalid_spin = any(rows[k]["spin_sum_error"] is not None and
                                   rows[k]["spin_sum_error"] > args.spin_sum_tolerance for k in (i, j))
                raw_bad = args.verify_raw and any(rows[k]["raw_verification"] != "verified" for k in (i, j))
                result["spin_attribution_valid"] = not invalid_spin and not raw_bad and result.get("spin_rms") is not None
                if result["evidence"] == "label_conflict_with_distinct_spins" and not result["spin_attribution_valid"]:
                    result["evidence"] = "label_conflict_spin_unresolved"
                if any(rows[k]["scf_warning"] or rows[k]["raw_verification"] == "label_mismatch" for k in (i, j)):
                    result["evidence"] = "label_quality_problem"
                result["evidence_before_route_review"] = result["evidence"]
                if result["route_comparison"] == "different_review_required":
                    result["evidence"] = "route_difference_review_required"
                result["sections_normally_terminated"] = all(rows[k]["section_normal_termination"] is True and
                                                              rows[k]["section_error_termination"] is False for k in (i, j))
                pairs.append(result)
                counts[result["evidence"]] += 1
            if incomplete:
                break
        if incomplete:
            break
    # Changes between adjacent force indices within one output and multiplicity.
    trajectories = defaultdict(list)
    for i, f in enumerate(frames):
        m = f.record.metadata
        if "gaussian_output" in m and isinstance(m.get("force_frame_index"), int):
            trajectories[(m.get("collection_campaign", ""), m["gaussian_output"],
                          f.record.charge, f.record.multiplicity)].append(i)
    jumps = []
    for members in trajectories.values():
        members.sort(key=lambda i: rows[i]["force_frame_index"])
        for i, j in zip(members, members[1:]):
            if rows[j]["force_frame_index"] != rows[i]["force_frame_index"] + 1:
                continue
            a, b = spins(frames[i]), spins(frames[j])
            if a is None or b is None or [v.symbol for v in frames[i].record.atoms] != [v.symbol for v in frames[j].record.atoms]:
                continue
            change = min(float(np.sqrt(np.mean((a-sign*b)**2))) for sign in (1, -1))
            if change > args.spin_threshold:
                jumps.append({"left": i, "right": j, "spin_rms": change,
                              "delta_energy_eV": frames[j].energy_ev-frames[i].energy_ev,
                              "interpretation": "possible_root_change_not_proof"})
    summary = {"schema_version": 1, "frames": len(frames), "input_sha256": hashes,
               "settings": {k: v for k, v in vars(args).items() if k != "func"},
               "raw_logs": len(inventory),
               "raw_log_status": dict(Counter(r["status"] for r in inventory)),
               "frames_in_normally_terminated_sections": sum(r["section_normal_termination"] is True and r["section_error_termination"] is False for r in rows),
               "frames_with_complete_spins": sum(r["spin_complete"] for r in rows),
               "frames_with_bad_spin_sum": sum(r["spin_sum_error"] is not None and r["spin_sum_error"] > args.spin_sum_tolerance for r in rows),
               "label_routes": dict(Counter(r["label_route"] or "MISSING" for r in rows)),
               "raw_verification": dict(Counter(r["raw_verification"] for r in rows)),
               "pairs_examined": min(examined, args.max_pairs), "pair_budget_exhausted": incomplete,
               "pair_evidence": dict(counts), "possible_trajectory_root_changes": len(jumps),
               "limitations": ["No proof of SCF stability, method/basis consistency, or a ground-state envelope.",
                               "No conflicts found does not establish that total spin suffices; coverage may be sparse.",
                               "Near geometries are not conflicting identical inputs. Thresholds are user choices.",
                               "Local moments are population-analysis descriptors, not exact atomic spins.",
                               "Raw spin verification only accepts populations between the matched SCF and force table.",
                               "Trajectory flags assume Gaussian center ordering is retained; changes may be physical.",
                               "No model inference or force-coverage adequacy test is performed."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    write_csv(output / "frames.csv", rows, list(rows[0]))
    fields = ["left", "right", "geometry", "same_multiplicity", "evidence", "evidence_before_route_review", "spin_attribution_valid",
              "search_capped", "sections_normally_terminated", "route_comparison", "geometry_rms_A", "geometry_max_A", "mappings", "delta_energy_eV",
              "force_component_rms_eV_A", "spin_rms", "force_at_best_spin_mapping_eV_A",
              "best_spin_mapping_right_indices_0based"]
    write_csv(output / "pairs.csv", pairs, fields)
    write_csv(output / "trajectory_flags.csv", jumps, ["left", "right", "spin_rms", "delta_energy_eV", "interpretation"])
    text = ["# Local-spin ambiguity audit", "", f"Frames: {len(frames)}",
            f"Raw-log coverage: {summary['raw_log_status']}",
            f"Frames in normally terminated sections: {summary['frames_in_normally_terminated_sections']}",
            f"Complete local-spin arrays: {summary['frames_with_complete_spins']}",
            f"Inconsistent spin sums: {summary['frames_with_bad_spin_sum']}",
            f"Pair search truncated: {incomplete}", "", "## Pair evidence", ""]
    text += [f"- {key}: {value}" for key, value in sorted(counts.items())] or ["- No matched pairs found."]
    text += ["", "## Interpretation", "",
             "Exact same-Q/multiplicity E/F conflicts challenge the current model inputs. Distinct local spins support a missing-state hypothesis, subject to label/method quality. Conflicts without distinct spins require parser/method/root investigation. Different-multiplicity controls are distinguishable by the current model. All counts are pair counts, not independent electronic states.",
             "", f"Possible trajectory root changes: {len(jumps)}. These are flags, not confirmed discontinuities.",
             "", f"Raw verification: {summary['raw_verification']}", "",
             "Frame indices in pairs.csv and trajectory_flags.csv refer to frames.csv. Energy thresholds are eV per cluster; force RMS is over Cartesian components. See summary.json for input hashes and all thresholds.", "", "## Limitations", ""]
    text += [f"- {v}" for v in summary["limitations"]]
    (output / "report.md").write_text("\n".join(text)+"\n")
    print(f"Audited {len(frames)} frames: {output / 'report.md'}", flush=True)
    return 2 if incomplete or counts["mapping_inconclusive"] or any(r["status"] == "parse_error" for r in inventory) else 0
