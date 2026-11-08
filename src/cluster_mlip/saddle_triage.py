"""Why each saddle search in a campaign failed, and which rescue fits it.

Transition-state searches fail far more often than minimizations, and for
reasons that need different fixes. A plain relaunch (``relaunch-routes``) only
repairs a wrong route; it does nothing for a TS search that ran out of steps,
broke its internal coordinates, lost the SCF, or slid off the saddle into a
minimum. This module reads every started saddle job, works out how it ended and
where along its trajectory it came closest to a first-order saddle, and
recommends one of three rescues:

``restart_best_frame``
    Restart Opt=(TS,CalcFC,NoEigenTest) from the frame with exactly one
    negative Hessian eigenvalue and the smallest force -- the search was near
    the saddle before it diverged or stopped.
``restart_best_frame_cartesian``
    The same, but the internal coordinates broke (FormBX/Tors), so the restart
    must use Opt=Cartesian (or a Z-matrix with dummy atoms).
``mode_displacement``
    The search never held exactly one negative eigenvalue but did see extra
    ones, or converged with the wrong number of imaginary modes: displace along
    the surplus modes rather than rattling at random, which would push the
    geometry toward a minimum.
``mlip_ts_guess``
    No DFT step ever showed index-1 curvature (or no Hessian information
    survives): generate a TS guess on the MLIP surface (Sella / CI-NEB between
    the IRC endpoints) at the target charge and multiplicity, then verify with
    DFT.
``scf_rescue``
    The SCF failed before any force was computed: an electronic-structure
    problem (guess, spin state), not a saddle-search problem.

The Hessian information is Berny's own: each optimization step prints the
eigenvalues of its current (updated) second-derivative matrix as
``Eigenvalues ---`` lines. Those are approximate after the first step, which is
why the restart recipes recompute the Hessian (CalcFC).

Read-only: nothing in the campaign is modified.
"""
from __future__ import annotations

import collections
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gaussian import _FORCE_HEADER_RE, parse_force_frames, rms_force
from .route_audit import audit_campaign_routes
from .routes import (
    expected_imaginary_modes,
    failed_torsion_atoms,
    internal_coordinate_failure,
    log_routes,
    route_optimizes,
    route_uses_cartesian,
)

TRIAGE_COLUMNS = [
    "input", "batch", "config_type", "state", "expected_imaginary_modes",
    "termination", "error_link", "final_imaginary_modes",
    "charge", "multiplicity", "frames", "section_frames",
    "best_frame_index", "best_max_force_ev_ang", "best_rms_force_ev_ang",
    "best_negative_eigenvalues", "best_energy_above_lowest_ev",
    "last_negative_eigenvalues", "index1_frames", "max_negative_eigenvalues",
    "cartesian", "internal_coordinate_marker", "failed_torsion_atoms",
    "recommendation", "reason", "output",
]

RECOMMENDATIONS = {
    "restart_best_frame": "restart Opt=(TS,CalcFC,NoEigenTest) from the best index-1 frame",
    "restart_best_frame_cartesian": "restart from the best index-1 frame in Cartesians",
    "mode_displacement": "displace along the surplus imaginary modes, then TS search",
    "mlip_ts_guess": "TS guess on the MLIP surface (Sella/CI-NEB), verify with DFT",
    "scf_rescue": "fix the SCF (guess/spin state) before any saddle search",
    "freq_check": "converged without a frequency analysis; run Freq to confirm the order",
    "none": "converged to the expected saddle order; nothing to rescue",
}

_EIGEN_LINE_RE = re.compile(r"^\s*Eigenvalues ---(.*)$", re.M)
_ERROR_LINK_RE = re.compile(r"Error termination via Lnk1e in \S*?(l\d+)\.exe", re.I)
_STEPS_EXCEEDED_RE = re.compile(
    r"Number of steps exceeded|Optimization stopped\.", re.I
)
_SCF_FAILED_RE = re.compile(r"Convergence failure -- run terminated", re.I)
_CLOSE_CONTACT_RE = re.compile(r"Small interatomic distances encountered", re.I)


def _float(token: str) -> float | None:
    try:
        value = float(token.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def step_negative_eigenvalues(text: str) -> list[tuple[int, int]]:
    """``(position, negatives)`` for each Berny Hessian printout in ``text``.

    One printout is a run of consecutive ``Eigenvalues ---`` lines; its
    position is that of its first line.
    """
    blocks: list[tuple[int, int]] = []
    previous_end = -1
    for match in _EIGEN_LINE_RE.finditer(text):
        values = [v for v in (_float(t) for t in match.group(1).split()) if v is not None]
        negatives = sum(1 for value in values if value < 0)
        # Continuation lines follow the previous one directly (only a newline
        # between them); anything else starts a new printout.
        if blocks and text[previous_end:match.start()].strip() == "":
            position, count = blocks[-1]
            blocks[-1] = (position, count + negatives)
        else:
            blocks.append((match.start(), negatives))
        previous_end = match.end()
    return blocks


def termination(text: str, state: str) -> tuple[str, str]:
    """How the job ended, and the failing Gaussian link where one is named."""
    link_match = None
    for link_match in _ERROR_LINK_RE.finditer(text):
        pass
    link = link_match.group(1) if link_match else ""
    if state == "complete":
        return "normal", link
    if internal_coordinate_failure(text):
        return "internal_coordinate_failure", link
    if _SCF_FAILED_RE.search(text) or link == "l502":
        return "scf_failure", link
    if _STEPS_EXCEEDED_RE.search(text) or link == "l9999":
        return "steps_exhausted", link
    if _CLOSE_CONTACT_RE.search(text) or link == "l202":
        return "close_contact", link
    if "Error termination" in text:
        return "other_error", link
    # Started, never terminated: wall-time, preemption, or still running.
    return "no_termination", link


def _max_force(frame: Any) -> float:
    return max(math.sqrt(x * x + y * y + z * z) for x, y, z in frame.forces_ev_ang)


def triage_output(text: str, source: Path, config_type: str, state: str,
                  final_imaginary: int | None) -> dict[str, Any]:
    """Triage one saddle job's output text."""
    expected = expected_imaginary_modes(config_type)
    ending, link = termination(text, state)
    routes = [route for route in log_routes(text) if route_optimizes(route)]
    cartesian = bool(routes) and all(route_uses_cartesian(route) for route in routes)
    frames = [
        frame for frame in parse_force_frames(text, source)
        if not frame.record.metadata.get("scf_unconverged")
    ]
    # Each frame's Hessian printout is the first one after its force table and
    # before the next force table.
    headers = [match.start() for match in _FORCE_HEADER_RE.finditer(text)]
    hessians = step_negative_eigenvalues(text)
    negatives_by_frame: dict[int, int] = {}
    for frame in frames:
        index = frame.record.metadata["force_frame_index"]
        start = headers[index]
        end = headers[index + 1] if index + 1 < len(headers) else len(text)
        found = next((count for position, count in hessians if start < position < end), None)
        if found is not None:
            negatives_by_frame[index] = found

    # The search to judge is the last Link1 section that optimized (printed a
    # Hessian). Generated jobs end in a plain Force stage, and a ladder runs
    # one search per multiplicity: the final section is often a single point.
    def section_of(frame: Any) -> Any:
        return frame.record.metadata.get("gaussian_section_index")

    searching = [
        frame for frame in frames
        if frame.record.metadata["force_frame_index"] in negatives_by_frame
    ]
    anchor = searching[-1] if searching else (frames[-1] if frames else None)
    section_frames = (
        [frame for frame in frames if section_of(frame) == section_of(anchor)]
        if anchor is not None else []
    )

    def index_of(frame: Any) -> int:
        return frame.record.metadata["force_frame_index"]

    index1 = [frame for frame in section_frames if negatives_by_frame.get(index_of(frame)) == 1]
    pool = index1 or section_frames
    best = min(pool, key=_max_force) if pool else None
    lowest = min((frame.energy_ev for frame in section_frames), default=None)
    last_negatives = (
        negatives_by_frame.get(index_of(section_frames[-1])) if section_frames else None
    )
    most_negatives = max(negatives_by_frame.values(), default=None)

    marker = internal_coordinate_failure(text)
    if state == "complete" and final_imaginary is not None and final_imaginary == expected:
        recommendation, reason = "none", "converged with the expected imaginary-mode count"
    elif state == "complete" and final_imaginary is None:
        recommendation, reason = "freq_check", "converged, but no frequency analysis in the log"
    elif not frames:
        if ending == "scf_failure":
            recommendation, reason = "scf_rescue", "SCF failed before the first force evaluation"
        else:
            recommendation, reason = "mlip_ts_guess", "no usable DFT frame to restart from"
    elif state == "complete" and final_imaginary is not None and expected is not None \
            and final_imaginary > expected:
        recommendation = "mode_displacement"
        reason = f"converged with {final_imaginary} imaginary modes, expected {expected}"
    elif index1:
        if ending == "internal_coordinate_failure" and not cartesian:
            recommendation = "restart_best_frame_cartesian"
            reason = f"internal coordinates broke ({marker}); {len(index1)} index-1 frame(s)"
        elif ending == "internal_coordinate_failure":
            recommendation = "mode_displacement"
            reason = ("coordinates broke even in Cartesians; nudge the geometry "
                      "along the index-1 mode before restarting")
        else:
            recommendation = "restart_best_frame"
            drift = (
                f"; ended with {last_negatives} negative eigenvalue(s)"
                if last_negatives not in (None, 1) else ""
            )
            reason = f"{ending}; {len(index1)} index-1 frame(s){drift}"
    elif most_negatives is not None and most_negatives >= 2:
        recommendation = "mode_displacement"
        reason = f"never index 1; up to {most_negatives} negative Hessian eigenvalues"
    else:
        recommendation = "mlip_ts_guess"
        reason = (
            "no Hessian printout to locate the saddle region"
            if most_negatives is None
            else "search never showed a negative Hessian eigenvalue (minimum-like region)"
        )
        if state == "complete" and final_imaginary == 0:
            reason = "converged to a minimum; " + reason

    return {
        "expected_imaginary_modes": "" if expected is None else expected,
        "termination": ending,
        "error_link": link,
        "final_imaginary_modes": "" if final_imaginary is None else final_imaginary,
        "charge": section_frames[-1].record.charge if section_frames else "",
        "multiplicity": section_frames[-1].record.multiplicity if section_frames else "",
        "frames": len(frames),
        "section_frames": len(section_frames),
        "best_frame_index": index_of(best) if best else "",
        "best_max_force_ev_ang": f"{_max_force(best):.4f}" if best else "",
        "best_rms_force_ev_ang": f"{rms_force(best):.4f}" if best else "",
        "best_negative_eigenvalues": (
            negatives_by_frame.get(index_of(best), "") if best else ""
        ),
        "best_energy_above_lowest_ev": (
            f"{best.energy_ev - lowest:.4f}" if best and lowest is not None else ""
        ),
        "last_negative_eigenvalues": "" if last_negatives is None else last_negatives,
        "index1_frames": len(index1),
        "max_negative_eigenvalues": "" if most_negatives is None else most_negatives,
        "cartesian": "true" if cartesian else "false",
        "internal_coordinate_marker": marker,
        "failed_torsion_atoms": failed_torsion_atoms(text),
        "recommendation": recommendation,
        "reason": reason,
    }


def triage_campaign_saddles(
    campaign: Path,
    destination: Path | None = None,
    *,
    include_converged: bool = False,
) -> dict[str, Any]:
    """Triage every started saddle job; write ``saddle_triage.csv``/``.json``."""
    campaign = campaign.resolve()
    audit = audit_campaign_routes(campaign, write_reports=False)
    rows: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    for audit_row in audit["rows"]:
        if audit_row["intent"] != "saddle" or audit_row["state"] == "not_started":
            continue
        output = campaign / audit_row["output"] if audit_row["output"] else None
        if output is None or not output.is_file():
            unreadable.append({"input": audit_row["input"], "reason": "output not found"})
            continue
        text = output.read_text(encoding="utf-8", errors="replace")
        final = audit_row["final_imaginary_modes"]
        verdict = triage_output(
            text, output, audit_row["config_type"], audit_row["state"],
            int(final) if str(final).strip() not in ("", "None") else None,
        )
        if verdict["recommendation"] == "none" and not include_converged:
            continue
        rows.append({
            "input": audit_row["input"],
            "batch": audit_row["batch"],
            "config_type": audit_row["config_type"],
            "state": audit_row["state"],
            "output": audit_row["output"],
            **verdict,
        })

    summary: dict[str, Any] = {
        "campaign": str(campaign),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "saddles_started": sum(
            1 for row in audit["rows"]
            if row["intent"] == "saddle" and row["state"] != "not_started"
        ),
        "saddles_not_started": sum(
            1 for row in audit["rows"]
            if row["intent"] == "saddle" and row["state"] == "not_started"
        ),
        "triaged": len(rows),
        "by_state": dict(collections.Counter(row["state"] for row in rows)),
        "by_termination": dict(collections.Counter(row["termination"] for row in rows)),
        "by_recommendation": dict(collections.Counter(row["recommendation"] for row in rows)),
        "unreadable": unreadable,
    }
    destination = (destination or campaign / "monitoring").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "saddle_triage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAGE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (destination / "saddle_triage.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"summary": summary, "rows": rows, "destination": destination}
