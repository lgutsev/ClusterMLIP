"""Restart failed first-order saddle searches from their best frame.

A transition-state search that ran out of steps, hit the wall time or broke
its internal coordinates has usually passed close to the saddle on the way.
``triage-saddles`` finds that point: the step with exactly one negative
Hessian eigenvalue and the smallest force. A plain checkpoint restart
continues from the *last* geometry instead, which is often where the search
had already drifted off.

For every started first-order saddle job that triage marks
``restart_best_frame`` or ``restart_best_frame_cartesian``, this builds a
restart through the ordinary restart machinery (archived log, seed checkpoint,
manifest rows, batch listing, backups) with one difference: the first stage
starts from the best frame's coordinates with ``Opt=(TS,CalcFC,NoEigenTest)``,
a smaller ``MaxStep``, ``Cartesian`` where the coordinates broke, and
``Guess=Read`` from the failed attempt's checkpoint. Method, basis, SCF and IOP
settings and the later spin-flip stages are untouched.

A frame is used only if it was printed in the input orientation: the orbitals
read from the checkpoint are tied to that frame.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .gaussian import parse_force_frames
from .restart import SaddleStage, prepare_spin_restarts
from .route_audit import audit_campaign_routes
from .routes import expected_imaginary_modes
from .saddle_triage import triage_output

RESTARTABLE = ("restart_best_frame", "restart_best_frame_cartesian")


def restart_saddles(
    campaign: Path,
    *,
    start: int = 1,
    end: int | None = None,
    max_step: int | None = 10,
    assume_stopped: bool = False,
    dry_run: bool = False,
    exclude_inputs: set[str] | None = None,
) -> dict[str, Any]:
    """Plan (and unless ``dry_run`` apply) best-frame restarts; see the module doc."""
    campaign = campaign.resolve()
    audit = audit_campaign_routes(campaign, write_reports=False)
    stages: dict[str, SaddleStage] = {}
    candidates: list[dict[str, Any]] = []
    not_chosen: list[dict[str, str]] = []
    recommendations: Counter[str] = Counter()
    for row in audit["rows"]:
        if row["intent"] != "saddle" or row["state"] == "not_started" or not row["output"]:
            continue
        if expected_imaginary_modes(row["config_type"]) != 1:
            continue
        output = campaign / row["output"]
        if not output.is_file():
            continue
        name = Path(row["input"]).name
        text = output.read_text(encoding="utf-8", errors="replace")
        final = row["final_imaginary_modes"]
        verdict = triage_output(
            text, output, row["config_type"], row["state"],
            int(final) if str(final).strip() not in ("", "None") else None,
        )
        recommendations[verdict["recommendation"]] += 1
        if verdict["recommendation"] not in RESTARTABLE:
            continue
        if exclude_inputs and name in exclude_inputs:
            not_chosen.append({"input": name, "batch": row["batch"], "reason": "excluded"})
            continue
        frame = next(
            (item for item in parse_force_frames(text, output)
             if item.record.metadata["force_frame_index"] == verdict["best_frame_index"]),
            None,
        )
        if frame is None:
            not_chosen.append({"input": name, "batch": row["batch"],
                               "reason": "best frame could not be re-read from the log"})
            continue
        orientation = str(frame.record.metadata.get("orientation", ""))
        if not orientation.lower().startswith("input"):
            not_chosen.append({"input": name, "batch": row["batch"], "reason": (
                f"best frame is in the {orientation or 'unknown'} orientation; the checkpoint "
                "orbitals would not match its coordinates")})
            continue
        stages[name] = SaddleStage(
            atoms=[(atom.symbol, atom.x, atom.y, atom.z) for atom in frame.record.atoms],
            charge=frame.record.charge, multiplicity=frame.record.multiplicity,
            frame_index=int(verdict["best_frame_index"]),
            cartesian=verdict["recommendation"].endswith("cartesian"),
            max_step=max_step,
        )
        candidates.append({
            "input": name, "batch": row["batch"], "config_type": row["config_type"],
            "recommendation": verdict["recommendation"], "termination": verdict["termination"],
            "frame_index": verdict["best_frame_index"],
            "max_force_ev_ang": verdict["best_max_force_ev_ang"],
            "index1_frames": verdict["index1_frames"],
            "charge": frame.record.charge, "multiplicity": frame.record.multiplicity,
        })

    result: dict[str, Any] = {
        "campaign": str(campaign), "dry_run": dry_run,
        "saddles_by_recommendation": dict(recommendations),
        "candidates": candidates, "not_chosen": not_chosen,
        "input_count": 0, "stage_count": 0, "plan": [], "skipped": [], "backup": "",
    }
    if not stages:
        return result
    try:
        restart = prepare_spin_restarts(
            campaign, start=start, end=end, assume_stopped=assume_stopped, dry_run=dry_run,
            only_inputs=set(stages), saddle_stages=stages,
        )
    except RuntimeError as error:
        # Raised when every candidate was skipped; report the reasons instead.
        if "no interrupted inputs to recover" not in str(error):
            raise
        restart = prepare_spin_restarts(
            campaign, start=start, end=end, assume_stopped=assume_stopped, dry_run=True,
            only_inputs=set(stages), saddle_stages=stages,
        )
    relevant = [item for item in restart["skipped"] if item["reason"] not in ("complete",)]
    result.update({
        "input_count": restart["input_count"], "stage_count": restart["stage_count"],
        "plan": restart["plan"], "skipped": relevant, "backup": restart["backup"],
    })
    return result
