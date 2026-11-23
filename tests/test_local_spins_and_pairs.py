"""Local-spin provenance, verified spin-ladder pairs, source-geometry split
groups, trajectory collection, and the label-level audit.

The synthetic logs follow Gaussian 09's layout closely where it matters:
the route echo wraps mid-word, a Mulliken table is printed at the first
optimization step and again in the final population analysis *after* the
last force table, and each Link1 stage reads its geometry from the previous
stage's checkpoint.
"""
from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from cluster_mlip.cli import main
from cluster_mlip.dataset import read_labeled_extxyz
from cluster_mlip.gaussian import parse_force_frames
from cluster_mlip.label_levels import (
    compare_levels, frame_label_levels, route_echoes, route_label_level,
)
from cluster_mlip.ladder_pairs import source_geometry_groups
from cluster_mlip.local_spins import local_spin_summary
from cluster_mlip.models import Atom
from cluster_mlip.spin import SPIN_MANIFEST_COLUMNS
from cluster_mlip.training import TrainingConfig, scan_dataset, write_training_campaign

ROUTE = ("#p UBPW91/6-311++G* SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) "
         "NoSymm Opt IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine")
CHECKPOINT_ROUTE = ROUTE + " Geom=Checkpoint Guess=Read"
Z = {"Fe": 26, "O": 8}

TRIANGLE = [("Fe", 0.0, 0.0, 0.0), ("Fe", 2.3, 0.0, 0.0), ("Fe", 1.15, 1.99, 0.0)]


def _moved(geometry, step):
    """A distinct geometry per optimization step."""
    return [(s, x + 0.01 * step, y - 0.005 * step, z) for s, x, y, z in geometry]


def _distorted(geometry, amount):
    """A genuinely different structure: the apex atom moves off the triangle."""
    return [(s, x, y + (amount if i == len(geometry) - 1 else 0.0), z)
            for i, (s, x, y, z) in enumerate(geometry)]


def _wrap(text: str, width: int = 70) -> str:
    """Gaussian's echo: fixed-width chunks, each line with a 1-space prefix."""
    return "\n".join(" " + text[i:i + width] for i in range(0, len(text), width))


def _orientation(geometry) -> str:
    rows = "\n".join(
        f"{i:7d}{Z[s]:11d}           0    {x:12.6f}{y:12.6f}{z:12.6f}"
        for i, (s, x, y, z) in enumerate(geometry, 1)
    )
    rule = " " + "-" * 69
    return (f"                          Input orientation:\n{rule}\n"
            " Center     Atomic      Atomic             Coordinates (Angstroms)\n"
            " Number     Number       Type             X           Y           Z\n"
            f"{rule}\n{rows}\n{rule}\n")


def _forces(geometry, scale=0.001) -> str:
    rows = "\n".join(
        f"{i:7d}{Z[s]:9d}   {scale * i: .9f}  {-scale * i: .9f}  {0.0: .9f}"
        for i, (s, *_rest) in enumerate(geometry, 1)
    )
    rule = " " + "-" * 67
    return (f"{rule}\n Center     Atomic                   Forces (Hartrees/Bohr)\n"
            " Number     Number              X              Y              Z\n"
            f"{rule}\n{rows}\n{rule}\n")


def _mulliken(geometry, spins, charges=None) -> str:
    charges = charges or [0.01 * (-1) ** i for i in range(len(geometry))]
    rows = "\n".join(
        f"{i:6d}  {s:<3s}{q:12.6f}{m:12.6f}"
        for i, ((s, *_), q, m) in enumerate(zip(geometry, charges, spins), 1)
    )
    return (" Mulliken charges and spin densities:\n               1          2\n"
            f"{rows}\n Sum of Mulliken charges =   0.00000  {sum(spins):9.5f}\n")


def g09_stage(route, charge, mult, geometries, energies, spins_first, spins_final,
              *, optimized=True, scf_fail_step=None, program="09",
              basis="6-311++G(d) (5D, 7F)", title="ladder stage") -> str:
    """One Link1 section of a G09 optimization."""
    s = mult - 1
    parts = [
        " Entering Gaussian System, Link 0=g09\n",
        f" Gaussian {program}:  ES64L-G{program}RevD.01 24-Apr-2013\n",
        " " + "-" * 70 + "\n", _wrap(route) + "\n", " " + "-" * 70 + "\n",
        " 1/18=20,19=15,38=1/1,3;\n", f" {title}\n",
        f" Charge = {charge} Multiplicity = {mult}\n",
    ]
    for step, (geometry, energy) in enumerate(zip(geometries, energies)):
        parts.append(_orientation(geometry))
        parts.append(f" Standard basis: {basis}\n")
        if step == scf_fail_step:
            parts.append(" >>>>>>>>>> Convergence criterion not met.\n")
        parts.append(f" SCF Done:  E(UB-PW91) =  {energy:.10f}     A.U. after   20 cycles\n")
        parts.append(f" <Sx>= 0.0000 <Sy>= 0.0000 <Sz>= {s / 2:.4f} <S**2>= {s / 2 * (s / 2 + 1) + 0.01:.4f}\n")
        parts.append(f" S**2 before annihilation {s / 2 * (s / 2 + 1) + 0.01:.4f},   after {s / 2 * (s / 2 + 1):.4f}\n")
        if step == 0 and spins_first is not None:
            parts.append(_mulliken(geometry, spins_first))
        parts.append(_forces(geometry, 0.001 * (len(geometries) - step)))
    if optimized:
        final = geometries[-1]
        parts.append(" Optimization completed.\n    -- Stationary point found.\n")
        parts.append(_orientation(final))
        parts.append("\n Population analysis using the SCF density.\n\n")
        if spins_final is not None:
            parts.append(_mulliken(final, spins_final))
        method = "UBPW91"
        archive = (f"1\\1\\GINC-QB3\\FOpt\\{method}\\6-311++G(d)\\Fe3\\USER\\29-Sep-2026\\0\\\\"
                   f"{route}\\\\{title}\\\\{charge},{mult}\\Fe,0.,0.,0.\\\\Version=ES64L-G09RevD.01"
                   f"\\State={mult}-A\\HF={energies[-1]}\\\\@")
        parts.append(_wrap(archive) + "\n")
        parts.append(" Normal termination of Gaussian 09 at Mon Sep 29 10:00:00 2026.\n")
    return "".join(parts)


def ladder_text(multiplicities, stage_geometries, *, handoff_break=False,
                last_stage_optimized=True, scf_fail=None, spins=True):
    """A multi-stage ladder; each stage starts on its predecessor's final geometry."""
    sections = []
    previous_final = None
    for stage, mult in enumerate(multiplicities):
        start = stage_geometries[stage]
        geometries = [_moved(start, step) for step in range(3)]
        if stage and not handoff_break:
            # Geom=Checkpoint: the first geometry is the previous final one.
            geometries[0] = previous_final
        energies = [-3790.1 - 0.001 * stage - 0.0001 * step for step in range(3)]
        first = [(mult - 1) / 3] * 3 if spins else None
        final = [(mult - 1) / 3 + 0.0] * 3 if spins else None
        sections.append(g09_stage(
            ROUTE if stage == 0 else CHECKPOINT_ROUTE, 0, mult, geometries, energies,
            first, final,
            optimized=last_stage_optimized or stage < len(multiplicities) - 1,
            scf_fail_step=scf_fail if stage == 0 else None,
        ))
        previous_final = geometries[-1]
    return "".join(sections)


def _coordinates(geometry):
    return "\n".join(f"{s:4s} {x: .12f} {y: .12f} {z: .12f}" for s, x, y, z in geometry)


class Campaign:
    """A spin campaign on disk: spin_jobs.csv, inputs/, slurm_batches/."""

    def __init__(self, root: Path):
        self.root = root
        (root / "inputs").mkdir(parents=True)
        self.rows: list[dict[str, str]] = []

    def ladder(self, name, record_id, multiplicities, geometry, *, batch=1, log=None,
               extra_rows=None, source_sha=None):
        chain = f"{record_id}-ladder-m{multiplicities[0]}-m{multiplicities[-1]}"
        stages = []
        for stage, mult in enumerate(multiplicities):
            route = ROUTE if stage == 0 else CHECKPOINT_ROUTE
            head = (f"%chk={chain}-s{stage:02d}-m{mult}.chk\n{route}\n\nstage {stage}\n\n0 {mult}\n")
            stages.append(head + (_coordinates(geometry) + "\n" if stage == 0 else "") + "\n")
        (self.root / "inputs" / f"{name}.gjf").write_text("--Link1--\n".join(stages), encoding="utf-8")
        for stage, mult in enumerate(multiplicities):
            row = {column: "" for column in SPIN_MANIFEST_COLUMNS}
            row.update({
                "job_id": f"{chain}-s{stage:02d}", "chain_id": chain, "stage_index": str(stage),
                "pathway": "multiplicity_ladder", "parent_record_id": record_id,
                "source": f"warehouse/{name}.out", "formula": "Fe3", "config_type": "minimum",
                "route_search_kind": "minimum",
                "first_route": ROUTE if stage == 0 else CHECKPOINT_ROUTE,
                "source_geometry_sha256": source_sha or f"sha-{record_id}",
                "high_spin_multiplicity": str(multiplicities[0]),
                "final_target_multiplicity": str(multiplicities[-1]),
                "intended_charge": "0", "intended_multiplicity": str(mult),
                "spin_flip_index": str(stage),
                "predecessor_job_id": f"{chain}-s{stage - 1:02d}" if stage else "",
                "predecessor_multiplicity": str(multiplicities[stage - 1]) if stage else "",
                "checkpoint": f"{chain}-s{stage:02d}-m{mult}.chk",
                "input": f"inputs/{name}.gjf", "output": f"{name}.log",
            })
            row.update(extra_rows or {})
            self.rows.append(row)
        if log is not None:
            batch_dir = self.root / "slurm_batches" / f"batch_{batch:04d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            (batch_dir / f"{name}.log").write_text(log, encoding="utf-8")
        return chain

    def write(self):
        columns = list(SPIN_MANIFEST_COLUMNS) + sorted(
            {key for row in self.rows for key in row} - set(SPIN_MANIFEST_COLUMNS))
        with (self.root / "spin_jobs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class LocalSpinParsingTests(unittest.TestCase):
    def test_spins_attach_to_their_own_scf_only(self):
        geometries = [_moved(TRIANGLE, step) for step in range(3)]
        text = g09_stage(ROUTE, 0, 5, geometries, [-3790.1, -3790.2, -3790.3],
                         [1.4, 1.3, 1.3], [1.5, 1.2, 1.3])
        frames = parse_force_frames(text, Path("x.log"))
        self.assertEqual(len(frames), 3)
        first, middle, final = (frame.record.metadata for frame in frames)
        self.assertEqual(first["atomic_spin_association"], "scf_step")
        self.assertEqual([row[2] for row in first["atomic_spins"]], [1.4, 1.3, 1.3])
        self.assertEqual(len(first["atomic_charges"]), 3)
        # No table was printed for the middle step; none may be borrowed.
        self.assertNotIn("atomic_spins", middle)
        # The converged frame gets the final population analysis.
        self.assertEqual(final["atomic_spin_association"], "final_population_analysis")
        self.assertEqual([row[2] for row in final["atomic_spins"]], [1.5, 1.2, 1.3])

    def test_final_analysis_is_refused_after_a_different_geometry(self):
        geometries = [_moved(TRIANGLE, step) for step in range(2)]
        text = g09_stage(ROUTE, 0, 5, geometries, [-3790.1, -3790.2], None, [1.5, 1.2, 1.3])
        text = text.replace(" Optimization completed.\n    -- Stationary point found.\n"
                            + _orientation(geometries[-1]),
                            " Optimization completed.\n    -- Stationary point found.\n"
                            + _orientation(_moved(TRIANGLE, 9)))
        final = parse_force_frames(text, Path("x.log"))[-1].record.metadata
        self.assertNotIn("atomic_spins", final)

    def test_spin_table_does_not_cross_a_link1_boundary(self):
        text = ladder_text([5, 3], [TRIANGLE, TRIANGLE], spins=False)
        text = text.replace(" Charge = 0 Multiplicity = 3\n",
                            _mulliken(TRIANGLE, [9.0, 9.0, 9.0]) + " Charge = 0 Multiplicity = 3\n")
        frames = parse_force_frames(text, Path("x.log"))
        self.assertFalse(any("atomic_spins" in frame.record.metadata
                             for frame in frames if frame.record.multiplicity == 5))

    def test_older_spin_only_table(self):
        text = g09_stage(ROUTE, 0, 5, [TRIANGLE], [-3790.1], None, None, optimized=False)
        text = text.replace(" Center     Atomic                   Forces",
                            " Mulliken atomic spin densities:\n              1\n"
                            "     1  Fe    1.500000\n     2  Fe    1.300000\n     3  Fe    1.200000\n"
                            " Sum of Mulliken spin densities=   4.00000\n"
                            " Center     Atomic                   Forces", 1)
        metadata = parse_force_frames(text, Path("x.log"))[0].record.metadata
        self.assertEqual([row[2] for row in metadata["atomic_spins"]], [1.5, 1.3, 1.2])
        self.assertNotIn("atomic_charges", metadata)

    def test_iop_5_13_continuation_is_flagged(self):
        text = ladder_text([5], [TRIANGLE], scf_fail=1)
        flags = [frame.record.metadata["scf_convergence_warning"]
                 for frame in parse_force_frames(text, Path("x.log"))]
        self.assertEqual(flags, [False, True, False])

    def test_summary_checks_spin_sum_against_multiplicity(self):
        frame = parse_force_frames(ladder_text([5], [TRIANGLE]), Path("x.log"))[0]
        summary = local_spin_summary(frame.record)
        self.assertTrue(summary["local_spin_consistent"])
        self.assertEqual(summary["fe_moment_pattern"], "ferro_like")
        frame.record.multiplicity = 7  # a table attached to the wrong state
        self.assertFalse(local_spin_summary(frame.record)["local_spin_consistent"])


class CompletedStagesCollectionTests(unittest.TestCase):
    def _collect(self, campaign: Path, destination: Path, *extra) -> int:
        with redirect_stdout(io.StringIO()):
            return main(["collect", str(campaign), "-o", str(destination),
                         "--frames", "completed-stages", "--stratify-by", "", *extra])

    def test_trajectory_spins_pairs_and_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            campaign = Campaign(root)
            other = _distorted(TRIANGLE, 0.3)
            # 1. A clean two-stage ladder.
            clean = campaign.ladder("a-fe3-1-5", "rec-a", [5, 3], TRIANGLE,
                                    log=ladder_text([5, 3], [TRIANGLE, TRIANGLE]))
            # 2. The same warehouse geometry under a second record ID (a twin).
            campaign.ladder("a-fe3-1-5-twin", "rec-a-twin", [5], TRIANGLE,
                            log=ladder_text([5], [TRIANGLE]), source_sha="sha-rec-a")
            # 3. A ladder whose second stage did not start where the first ended.
            broken = campaign.ladder("a-fe3-2-5", "rec-b", [5, 3], other,
                                     log=ladder_text([5, 3], [other, _moved(other, 50)],
                                                     handoff_break=True))
            # 4. A second stage that never converged, and an IOP(5/13) SCF.
            campaign.ladder("a-fe3-3-5", "rec-c", [5, 3], _distorted(TRIANGLE, 0.6),
                            log=ladder_text([5, 3], [_distorted(TRIANGLE, 0.6)] * 2,
                                            last_stage_optimized=False, scf_fail=1), batch=2)
            # 5. Outside the requested batch range.
            campaign.ladder("a-fe3-4-5", "rec-d", [5], _distorted(TRIANGLE, 0.9),
                            log=ladder_text([5], [_distorted(TRIANGLE, 0.9)]), batch=7)
            campaign.write()
            destination = Path(tmp) / "data"
            self.assertEqual(self._collect(root, destination, "--allow-partial",
                                           "--batch-range", "1-6", "--formula", "Fe3"), 0)

            frames = read_labeled_extxyz(destination / "all.extxyz")
            sources = {frame.record.metadata["parent_record_id"] for frame in frames}
            self.assertNotIn("rec-d", sources)
            # Whole trajectories: 3 frames per finished stage. rec-c keeps its
            # finished first stage minus the unconverged-SCF frame.
            per_stage: dict[str, int] = {}
            for frame in frames:
                per_stage[frame.record.metadata["job_id"]] = per_stage.get(
                    frame.record.metadata["job_id"], 0) + 1
            self.assertEqual(per_stage[f"{clean}-s00"], 3)
            self.assertEqual(per_stage["rec-c-ladder-m5-m3-s00"], 2)
            self.assertNotIn("rec-c-ladder-m5-m3-s01", per_stage)
            summary = json.loads((destination / "dataset_summary.json").read_text())
            self.assertEqual(summary["frames_dropped_scf_not_converged"], 1)
            self.assertEqual(summary["frames_in_unfinished_stages"], 3)
            # The twin's first stage is the clean ladder's first stage again.
            self.assertGreater(summary["frames_dropped_exact_duplicate"], 0)

            # Frames and independent geometries are reported separately; the
            # twin records share one split group.
            self.assertEqual(summary["frames"], len(frames))
            self.assertEqual(summary["source_records"], 3)  # the twin's frames were all duplicates
            groups = {frame.record.metadata["parent_record_id"]: frame.record.metadata["split_group"]
                      for frame in frames}
            self.assertEqual(len(set(groups.values())), 3)
            for split in ("train", "valid", "test"):
                members = read_labeled_extxyz(destination / f"{split}.extxyz")
                for group in {m.record.metadata["split_group"] for m in members}:
                    self.assertTrue(all(f.record.metadata["split_group"] != group
                                        for other in ("train", "valid", "test") if other != split
                                        for f in read_labeled_extxyz(destination / f"{other}.extxyz")))

            # Local spins: per-atom columns in the extxyz, and the long table.
            header = (destination / "all.extxyz").read_text().splitlines()[1]
            self.assertIn("mulliken_spin:R:1:mulliken_charge:R:1", header)
            local = _csv(destination / "local_spins.csv")
            self.assertTrue(local)
            self.assertEqual({row["atomic_spin_association"] for row in local},
                             {"scf_step", "final_population_analysis"})
            provenance = _csv(destination / "frame_provenance.csv")
            self.assertEqual(len(provenance), len(frames))
            first = next(row for row in provenance if row["job_id"] == f"{clean}-s01"
                         and row["stage_frame_position"] == "first")
            self.assertEqual(first["predecessor_job_id"], f"{clean}-s00")
            self.assertEqual(first["parent_record_id"], "rec-a")

            # Pairs: verified by geometry, never by job order alone.
            pairs = _csv(destination / "spin_pairs.csv")
            verified = [p for p in pairs if p["paired"] == "True"]
            self.assertTrue(any(p["child_job_id"] == f"{clean}-s01"
                                and p["pair_basis"] == "manifest+geometry" for p in verified))
            refused = [p for p in pairs if p["child_job_id"] == f"{broken}-s01"]
            self.assertEqual([p["pair_basis"] for p in refused], ["manifest_claim_not_verified"])
            self.assertEqual(refused[0]["paired"], "False")
            gap = next(p for p in verified if p["child_job_id"] == f"{clean}-s01")
            self.assertAlmostEqual(
                float(gap["vertical_gap_ev"]),
                float(gap["child_energy_ev"]) - float(gap["parent_energy_ev"]), places=6)
            tagged = [f for f in frames if f.record.metadata.get("spin_pairs")]
            self.assertEqual(
                {entry["role"] for f in tagged for entry in f.record.metadata["spin_pairs"]},
                {"high_spin_parent", "low_spin_child"})

    def test_restart_seeded_from_parent_is_found_by_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            campaign = Campaign(root)
            parent_log = ladder_text([5], [TRIANGLE])
            campaign.ladder("a-fe3-1-5", "rec-a", [5], TRIANGLE, log=parent_log)
            parent_final = parse_force_frames(parent_log, Path("p.log"))[-1].record.atoms
            final_geometry = [(a.symbol, a.x, a.y, a.z) for a in parent_final]
            # A restart at M=3 that started from the M=5 final geometry; its
            # manifest row names no predecessor.
            restart_log = g09_stage(CHECKPOINT_ROUTE, 0, 3,
                                    [final_geometry, _moved(final_geometry, 1)],
                                    [-3790.5, -3790.6], [0.7, 0.7, 0.6], [0.7, 0.7, 0.6])
            campaign.ladder("a-fe3-1-3__restart01-from-m3", "rec-a", [3], TRIANGLE,
                            log=restart_log, extra_rows={"restart_of_job_id": "old-m3"})
            campaign.write()
            destination = Path(tmp) / "data"
            self.assertEqual(self._collect(root, destination), 0)
            pairs = _csv(destination / "spin_pairs.csv")
            self.assertEqual([(p["pair_basis"], p["pair_kind"], p["paired"]) for p in pairs],
                             [("geometry_only", "restart_seeded_from_parent", "True")])


class EvaluationPairTests(unittest.TestCase):
    def test_gap_errors_and_per_frame_predictions(self):
        from cluster_mlip.evaluate import write_evaluation_report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            campaign = Campaign(root)
            campaign.ladder("a-fe3-1-5", "rec-a", [5, 3], TRIANGLE,
                            log=ladder_text([5, 3], [TRIANGLE, TRIANGLE]))
            # A twin ladder: every frame duplicates rec-a's and is dropped.
            campaign.ladder("a-fe3-1-5-twin", "rec-a-twin", [5, 3], TRIANGLE,
                            log=ladder_text([5, 3], [TRIANGLE, TRIANGLE]), source_sha="sha-rec-a")
            campaign.write()
            with redirect_stdout(io.StringIO()):
                main(["collect", str(root), "-o", str(Path(tmp) / "data"),
                      "--frames", "completed-stages", "--stratify-by", ""])
            pair_rows = _csv(Path(tmp) / "data" / "spin_pairs.csv")
            # Two manifest hand-offs, plus two cross-ladder identities the
            # geometry search finds because the twins are the same structure.
            self.assertEqual(sorted(row["pair_basis"] for row in pair_rows if row["paired"] == "True"),
                             ["geometry_only", "geometry_only",
                              "manifest+geometry", "manifest+geometry"])
            self.assertTrue(all(row["parent_frame_in_dataset"] == "True"
                                and row["child_frame_in_dataset"] == "True" for row in pair_rows))
            frames = read_labeled_extxyz(Path(tmp) / "data" / "all.extxyz")
            # A model that is exact except for a per-multiplicity offset has a
            # gap error equal to the difference of the offsets.
            offset = {5: 0.10, 3: 0.35}
            predictions = [(f.energy_ev + offset[f.record.multiplicity], f.forces_ev_ang)
                           for f in frames]
            out = Path(tmp) / "eval"
            write_evaluation_report(frames, predictions, out)
            gaps = _csv(out / "spin_pair_errors.csv")
            self.assertEqual(len(gaps), 1)
            self.assertAlmostEqual(float(gaps[0]["gap_error_ev"]), 0.25, places=6)
            rows = _csv(out / "predictions.csv")
            self.assertEqual(len(rows), len(frames))
            self.assertEqual({row["split_group"] for row in rows}, {"geom:rec-a"})
            self.assertIn("Vertical spin-flip gaps", (out / "report.md").read_text())


class SourceGeometryGroupTests(unittest.TestCase):
    def test_twins_merge_and_distinct_geometries_do_not(self):
        base = [Atom(s, x, y, z) for s, x, y, z in TRIANGLE]
        rotated = [Atom(a.symbol, -a.y, a.x, a.z) for a in base]  # rigid rotation
        other = [Atom(s, x, y, z) for s, x, y, z in _moved(TRIANGLE, 30)]
        stretched = [Atom("Fe", 0, 0, 0), Atom("Fe", 2.6, 0, 0), Atom("Fe", 1.3, 2.2, 0)]
        groups = source_geometry_groups({"a": base, "b": rotated, "c": stretched, "d": other})
        self.assertEqual(groups["a"], groups["b"])
        self.assertEqual(groups["a"], groups["d"])  # a pure translation of a
        self.assertNotEqual(groups["a"], groups["c"])
        exact = source_geometry_groups({"a": base, "c": stretched}, {"a": "h", "c": "h"}, 0.0)
        self.assertEqual(exact["a"], exact["c"])


class LabelLevelTests(unittest.TestCase):
    def test_route_echo_rejoins_mid_word_wraps(self):
        text = " " + "-" * 70 + "\n" + _wrap(ROUTE) + "\n " + "-" * 70 + "\n"
        self.assertEqual(route_echoes(text), [(0, ROUTE)])

    def test_equivalent_routes_are_one_level(self):
        reference = route_label_level(ROUTE, "g09")
        for route in (CHECKPOINT_ROUTE,
                      ROUTE.replace("Opt", "Opt=(TS,CalcFC,NoEigenTest)") + " Freq",
                      ROUTE.replace("6-311++G*", "6-311++G(d)")):
            self.assertEqual(compare_levels(route_label_level(route, "g09"), reference),
                             "same_level", route)
        self.assertEqual(compare_levels(
            route_label_level(ROUTE.replace("++G*", "+G*"), "g09"), reference), "different_level")
        self.assertEqual(compare_levels(
            route_label_level(ROUTE.replace(" Int=UltraFine", ""), "g09"), reference),
            "grid_differs")
        self.assertEqual(compare_levels(
            route_label_level(ROUTE.replace("6-311++G*", "ChkBas"), "g09"), reference),
            "unverifiable")

    def test_output_level_comes_from_what_gaussian_reports(self):
        # The route asks for ChkBas; the basis line says what was used.
        route = ROUTE.replace("6-311++G*", "ChkBas").replace("Opt", "Freq")
        text = g09_stage(route, 0, 5, [TRIANGLE], [-3790.1], None, None, optimized=False)
        frames = parse_force_frames(text, Path("x.log"))
        level, _ = frame_label_levels(text, frames)[0]
        self.assertEqual(compare_levels(level, route_label_level(ROUTE, "g09")), "same_level")
        self.assertIn("basis_line", level.evidence)
        self.assertIn("scf_done", level.evidence)

    def test_audit_separates_incompatible_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "Out_Files"
            legacy.mkdir()
            (legacy / "A_fe3_1_5_GF.out").write_text(ladder_text([5], [TRIANGLE]))
            (legacy / "A_fe3_2_5_GF.out").write_text(g09_stage(
                ROUTE.replace("++G*", "+G*"), 0, 5,
                [_moved(TRIANGLE, 5 + i) for i in range(2)], [-3790.1, -3790.2],
                None, None, basis="6-311+G(d) (5D, 7F)"))
            (legacy / "A_fe3_3_5_GF.out").write_text(g09_stage(
                ROUTE.replace(" Int=UltraFine", ""), 0, 5,
                [_moved(TRIANGLE, 20 + i) for i in range(2)], [-3790.1, -3790.2], None, None))
            (legacy / "A_fe2o_1_3.out").write_text(g09_stage(
                ROUTE, 0, 3, [[("Fe", 0, 0, 0), ("O", 1.6, 0, 0)]], [-1338.0], None, None))
            out = Path(tmp) / "levels"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["label-levels", str(legacy), "-o", str(out),
                                       "--formula", "Fe3"]), 0)
            summary = json.loads((out / "label_levels_summary.json").read_text())
            self.assertEqual(summary["files_with_matching_force_frames"], 3)
            self.assertEqual(summary["same_level_usable_minimum_optimization_frames"], 3)
            self.assertEqual(summary["same_level_source_files"], 1)
            kinds = summary["usable_frames_by_class_and_kind"]
            self.assertEqual(kinds["different_level|minimum_optimization"], 2)
            self.assertEqual(kinds["grid_differs|minimum_optimization"], 2)
            files = sorted(path.name for path in (out / "frames").iterdir())
            self.assertIn("same_level.extxyz", files)
            self.assertIn("grid_differs.extxyz", files)
            self.assertTrue(any(name.startswith("different_level__") for name in files))
            self.assertEqual(len(read_labeled_extxyz(out / "frames" / "same_level.extxyz")), 3)


class TrainingLevelGuardTests(unittest.TestCase):
    def test_ladder_stage_routes_are_one_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            campaign = Campaign(root)
            campaign.ladder("a-fe3-1-5", "rec-a", [5, 3], TRIANGLE,
                            log=ladder_text([5, 3], [TRIANGLE, TRIANGLE]))
            campaign.ladder("b-fe3-1-5", "rec-b", [5, 3], _distorted(TRIANGLE, 0.4),
                            log=ladder_text([5, 3], [_distorted(TRIANGLE, 0.4)] * 2))
            campaign.write()
            destination = Path(tmp) / "data"
            with redirect_stdout(io.StringIO()):
                main(["collect", str(root), "-o", str(destination), "--frames",
                      "completed-stages", "--stratify-by", "", "--valid-fraction", "0.3",
                      "--test-fraction", "0.3"])
            facts = scan_dataset(destination)
            self.assertEqual(len(facts.label_routes), 2)
            self.assertEqual(len(facts.label_levels), 1)
            plan = write_training_campaign(
                TrainingConfig(dataset_dir=destination, output_dir=Path(tmp) / "model"))
            self.assertEqual(len(plan["label_levels"]), 1)


if __name__ == "__main__":
    unittest.main()
