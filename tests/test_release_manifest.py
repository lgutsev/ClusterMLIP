"""release-manifest: per-frame qualification of collected labels for a release."""
from __future__ import annotations

import csv
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from cluster_mlip.cli import main
from cluster_mlip.dataset import read_labeled_extxyz, write_labeled_extxyz
from cluster_mlip.release_manifest import (
    _mark_split_leaks, chemistry_family, nearest_multiplicity,
)
from test_local_spins_and_pairs import TRIANGLE, Campaign, _distorted, ladder_text


def _run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(io.StringIO()):
        code = main(list(argv))
    return code, out.getvalue()


def _manifest(path: Path) -> list[dict[str, str]]:
    with (path / "release_manifest.tsv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _edit_frames(directory: Path, edit) -> int:
    """Apply *edit* to every frame of collect's split files; count the changed ones."""
    changed = 0
    for name in ("train", "valid", "test"):
        path = directory / f"{name}.extxyz"
        if not path.exists():
            continue
        frames = read_labeled_extxyz(path)
        changed += sum(bool(edit(frame)) for frame in frames)
        write_labeled_extxyz(frames, path)
    return changed


def _contaminated(text: str) -> str:
    """Every <S^2> after annihilation of the M=3 stage reads 6.0 (an S=2 value)."""
    head, tail = text.split("Multiplicity = 3", 1)
    tail = re.sub(r"after \d+\.\d+", "after 6.0000", tail)
    return head + "Multiplicity = 3" + tail


class HelperTests(unittest.TestCase):
    def test_nearest_multiplicity(self):
        self.assertEqual(nearest_multiplicity(2.0, 3), 3)
        self.assertEqual(nearest_multiplicity(2.9, 3), 3)
        self.assertEqual(nearest_multiplicity(4.1, 3), 5)
        self.assertEqual(nearest_multiplicity(0.4, 1), 1)
        self.assertEqual(nearest_multiplicity(1.2, 1), 3)
        self.assertEqual(nearest_multiplicity(0.75, 2), 2)
        self.assertEqual(nearest_multiplicity(210.5, 29), 29)

    def test_chemistry_family(self):
        self.assertEqual(chemistry_family(["Fe", "Fe"]), "Fe")
        self.assertEqual(chemistry_family(["Fe", "O"]), "FeO")
        self.assertEqual(chemistry_family(["Fe", "O", "N"]), "FeNO")
        self.assertEqual(chemistry_family(["Fe", "H", "O"]), "FeHO")


    def test_split_leaks_follow_parent_group_and_restart_lineage(self):
        def row(run, split, parent, group="", restart_of=""):
            return {"run_id": run, "split": split, "parent_record_id": parent,
                    "split_group": group, "restart_of_job_id": restart_of, "_reasons": []}

        rows = [
            row("a-s00", "train", "rec-a"), row("a-s01", "train", "rec-a"),
            # same parent, other split
            row("b-s00", "train", "rec-b"), row("b-s01", "test", "rec-b"),
            # restart of a job in another split, under a new parent id
            row("c-s00", "valid", "rec-c"), row("c-s00-r01", "train", "rec-c2",
                                                  restart_of="c-s00"),
            # twins merged into one split group
            row("d", "train", "rec-d", group="g1"), row("e", "valid", "rec-e", group="g1"),
        ]
        _mark_split_leaks(rows)
        leaked = {r["run_id"] for r in rows if "split_leak" in r["_reasons"]}
        self.assertEqual(leaked, {"b-s00", "b-s01", "c-s00", "c-s00-r01", "d", "e"})


class ReleaseManifestTests(unittest.TestCase):
    def _campaign(self, root: Path) -> Path:
        campaign = Campaign(root / "campaign")
        campaign.ladder("a-fe3-1-5", "rec-a", [5, 3], TRIANGLE,
                        log=ladder_text([5, 3], [TRIANGLE, TRIANGLE]))
        other = _distorted(TRIANGLE, 0.4)
        campaign.ladder("a-fe3-2-5", "rec-b", [5, 3], other,
                        log=_contaminated(ladder_text([5, 3], [other, other])))
        third = _distorted(TRIANGLE, 0.8)
        campaign.ladder("a-fe3-3-5", "rec-c", [5], third,
                        log=ladder_text([5], [third], scf_fail=1), batch=2)
        campaign.write()
        return campaign.root

    def _collect(self, campaign: Path, destination: Path, *extra: str) -> None:
        code, _ = _run("collect", str(campaign), "-o", str(destination), "--frames",
                       "completed-stages", "--stratify-by", "", "--allow-unconverged-scf",
                       *extra)
        self.assertEqual(code, 0)

    def test_decisions_reasons_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = self._campaign(root)
            self._collect(campaign, root / "data")

            # collect drops unconverged-SCF frames itself; the release check must
            # still refuse one that reaches it (an older or permissive collect).
            def flag(frame):
                metadata = frame.record.metadata
                if metadata.get("parent_record_id") == "rec-c" and                         metadata.get("force_frame_index") == 0:
                    metadata["scf_unconverged"] = True
                    return True
                return False

            self.assertEqual(_edit_frames(root / "data", flag), 1)
            code, out = _run("release-manifest", str(root / "data"), "-o", str(root / "rel"))
            self.assertEqual(code, 0, out)
            rows = _manifest(root / "rel")
            self.assertTrue(rows)
            by_parent: dict[str, list[dict[str, str]]] = {}
            for row in rows:
                by_parent.setdefault(row["parent_record_id"], []).append(row)
                self.assertEqual(row["energy_unit"], "eV")
                self.assertEqual(row["force_unit"], "eV/Angstrom")
                self.assertEqual(len(row["source_sha256"]), 64)
                self.assertEqual(row["chemistry"], "Fe")
                self.assertEqual(row["method"].upper(), "BPW91")
                self.assertNotIn("route_level_mismatch", row["reasons"])
                self.assertIn(row["link1_section"], {"0", "1"})
            # rec-a: clean, everything included; only stage finals claim a minimum.
            self.assertEqual({row["decision"] for row in by_parent["rec-a"]}, {"include"})
            claims = {row["stationary_claim"] for row in by_parent["rec-a"]}
            self.assertLessEqual(claims, {"none", "minimum"})
            # rec-b: the M=3 frames read <S^2> = 6 (S=2) -> held; M=5 frames included.
            for row in by_parent["rec-b"]:
                if row["multiplicity"] == "3":
                    self.assertEqual(row["decision"], "hold")
                    self.assertIn("s2_nearer_other_state", row["reasons"])
                    self.assertEqual(row["s2_nearest_multiplicity"], "5")
                else:
                    self.assertEqual(row["decision"], "include")
            # rec-c: the frame with an unconverged SCF is rejected, the others kept.
            rejected = [row for row in by_parent["rec-c"] if row["decision"] == "reject"]
            self.assertEqual(len(rejected), 1)
            self.assertIn("scf_not_converged", rejected[0]["reasons"])
            self.assertEqual(rejected[0]["scf_converged"], "False")

            summary = json.loads((root / "rel" / "release_summary.json").read_text())
            self.assertEqual(summary["gate"], "release")
            # Rerunning gives the same release hash.
            _run("release-manifest", str(root / "data"), "-o", str(root / "rel2"))
            again = json.loads((root / "rel2" / "release_summary.json").read_text())
            self.assertEqual(summary["release_sha256"], again["release_sha256"])
            included = sum(1 for row in rows if row["decision"] == "include")
            written = sum(
                (root / "rel" / name).read_text().count("Properties=")
                for name in summary["release_files"])
            self.assertEqual(written, included)

    def test_missing_source_duplicates_conflicts_and_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = self._campaign(root)
            self._collect(campaign, root / "a")
            self._collect(campaign, root / "b")
            # A second collect of the same outputs adds nothing new.
            code, _ = _run("release-manifest", str(root / "a"), str(root / "b"),
                           "-o", str(root / "rel"))
            rows = _manifest(root / "rel")
            from_b = [row for row in rows if row["collect_dir"] == str(root / "b")]
            self.assertTrue(from_b)
            finite_b = [row for row in from_b if row["label_sha256"]]
            self.assertTrue(all("duplicate_label" in row["reasons"] for row in finite_b))

            # Change one energy in b: that state is quarantined in both copies.
            done = []

            def shift(frame):
                if not done:
                    done.append(frame)
                    frame.energy_ev += 0.5
                    return True
                return False

            self.assertEqual(_edit_frames(root / "b", shift), 1)
            _run("release-manifest", str(root / "a"), str(root / "b"), "-o", str(root / "rel2"))
            conflicts = [row for row in _manifest(root / "rel2")
                         if "conflict_same_state" in row["reasons"]]
            self.assertEqual(len(conflicts), 2)
            self.assertEqual({row["decision"] for row in conflicts}, {"hold"})

            # A source output that is gone: its frames are held, not released.
            log = campaign / "slurm_batches" / "batch_0001" / "a-fe3-1-5.log"
            log.unlink()
            code, out = _run("release-manifest", str(root / "a"), "-o", str(root / "rel3"),
                             "--require-family", "Fe,FeO,FeNO")
            self.assertEqual(code, 3, out)
            rows = _manifest(root / "rel3")
            gone = [row for row in rows if row["parent_record_id"] == "rec-a"]
            self.assertTrue(gone)
            self.assertTrue(all(row["decision"] == "hold" and
                                "source_unavailable" in row["reasons"] for row in gone))
            summary = json.loads((root / "rel3" / "release_summary.json").read_text())
            self.assertEqual(summary["gate"], "hold_retraining")
            self.assertTrue(any(gap.startswith("FeO:") for gap in summary["gaps"]))
            self.assertTrue(any(gap.startswith("FeNO:") for gap in summary["gaps"]))

    def test_level_filter_holds_other_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = self._campaign(root)
            self._collect(campaign, root / "data")
            _run("release-manifest", str(root / "data"), "-o", str(root / "rel"),
                 "--level", "B3LYP/def2-TZVP|df=?|grid=?|extras=-")
            rows = _manifest(root / "rel")
            self.assertTrue(all("other_level" in row["reasons"] for row in rows))
            self.assertNotIn("include", {row["decision"] for row in rows})


if __name__ == "__main__":
    unittest.main()
