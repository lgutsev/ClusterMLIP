from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.dataset import write_labeled_extxyz
from cluster_mlip.models import Atom, LabeledFrame, Record
from cluster_mlip.io import parse_extxyz_info_line
from cluster_mlip.training import (
    BLIND_LABELS_EXIT_CODE,
    TrainingConfig,
    scan_dataset,
    write_training_campaign,
)


def _frame(
    record_id: str,
    charge: int = 0,
    multiplicity: int = 1,
    *,
    link1_route: str = "#p UBPW91/Gen Force",
) -> LabeledFrame:
    record = Record(
        record_id,
        "src.log",
        [Atom("Fe", 0.0, 0.0, 0.0), Atom("O", 1.5, 0.0, 0.0)],
        charge,
        multiplicity,
        "minimum",
        metadata={"parent_record_id": record_id, "link1_route": link1_route},
    )
    return LabeledFrame(record, -10.0, [(0.1, 0.0, 0.0), (-0.1, 0.0, 0.0)], Path("src.log"))


def _dataset(tmp: Path, frames: list[LabeledFrame]) -> Path:
    dataset = tmp / "dataset"
    dataset.mkdir()
    write_labeled_extxyz(frames, dataset / "all.extxyz")
    write_labeled_extxyz(frames, dataset / "train.extxyz")
    write_labeled_extxyz(frames[:1], dataset / "valid.extxyz")
    write_labeled_extxyz(frames[:1], dataset / "test.extxyz")
    return dataset


class ScanDatasetTests(unittest.TestCase):
    def test_collects_ranges_and_routes(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(
                tmp,
                [_frame("a", -1, 6), _frame("b", 0, 1), _frame("c", 2, 11)],
            )
            facts = scan_dataset(dataset)
            self.assertEqual(min(facts.charges), -1)
            self.assertEqual(max(facts.multiplicities), 11)
            self.assertEqual(facts.label_routes, {"#p UBPW91/Gen Force"})

    def test_missing_split_is_reported(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = tmp / "dataset"
            dataset.mkdir()
            write_labeled_extxyz([_frame("a")], dataset / "train.extxyz")
            with self.assertRaises(FileNotFoundError):
                scan_dataset(dataset)


class ScratchCampaignTests(unittest.TestCase):
    def test_scratch_run_has_locked_gaussian_args_and_embedding(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(tmp, [_frame("a", 0, 1), _frame("b", 0, 5)])
            output = tmp / "models" / "run"
            plan = write_training_campaign(
                TrainingConfig(dataset_dir=dataset, output_dir=output, seeds=(11, 23))
            )
            self.assertEqual(plan["model"], "ScaleShiftMACE")
            self.assertEqual(len(plan["seed_runs"]), 2)
            argv = plan["seed_runs"][0]["argv"]
            self.assertIn("--default_dtype=float64", argv)
            self.assertIn("--stress_weight=0", argv)
            self.assertIn("--energy_key=REF_energy", argv)
            self.assertIn("--use_embedding_readout=True", argv)
            self.assertNotIn("--use_embedding_readout", argv)
            self.assertTrue(any(a.startswith("--embedding_specs=") for a in argv))
            self.assertTrue(any(a.startswith("--seed=11") for a in argv))
            self.assertTrue((output / "seed_11" / "run.sh").is_file())
            self.assertTrue((output / "run_all_seeds.sh").is_file())
            manifest = json.loads((output / "train_manifest.json").read_text())
            self.assertEqual(manifest["seeds"], [11, 23])
            # The embedding_specs JSON contains {}" and must be single-quoted
            # in the rendered script so the shell does not mangle it.
            script_text = (output / "seed_11" / "run.sh").read_text()
            self.assertIn("--embedding_specs='{", script_text)
            self.assertIn("--hidden_irreps='128x0e + 128x1o + 128x2e'", script_text)

    def test_embedding_specs_read_the_info_keys_collect_writes(self):
        # mace-torch reads each graph-level embedding from
        # atoms.info[spec.get("key", name)], overriding --total_spin_key. Without
        # an explicit key it reads the absent info["total_spin"] and trains
        # spin-blind silently -- so the key must name what the dataset carries.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(tmp, [_frame("a", -1, 1), _frame("b", 0, 5)])
            plan = write_training_campaign(
                TrainingConfig(dataset_dir=dataset, output_dir=tmp / "run")
            )
            argv = plan["seed_runs"][0]["argv"]
            (spec_arg,) = [a for a in argv if a.startswith("--embedding_specs=")]
            specs = json.loads(spec_arg.partition("=")[2])
            self.assertEqual(specs["total_spin"]["key"], "spin")
            self.assertEqual(specs["total_charge"]["key"], "charge")
            self.assertIn("--total_spin_key=spin", argv)
            self.assertIn("--total_charge_key=charge", argv)

            header = (dataset / "train.extxyz").read_text().splitlines()[1]
            info = parse_extxyz_info_line(header)
            for spec in specs.values():
                self.assertIn(spec["key"], info)

    def test_multiplicity_outside_embedding_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(tmp, [_frame("a", 0, 250)])
            with self.assertRaises(ValueError):
                write_training_campaign(
                    TrainingConfig(dataset_dir=dataset, output_dir=tmp / "out")
                )

    def test_mixed_label_routes_are_refused_without_override(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(
                tmp,
                [
                    _frame("a", link1_route="#p UBPW91/Gen Force"),
                    _frame("b", link1_route="#p wB97XD/Gen Force"),
                ],
            )
            with self.assertRaises(ValueError):
                write_training_campaign(
                    TrainingConfig(dataset_dir=dataset, output_dir=tmp / "out")
                )
            plan = write_training_campaign(
                TrainingConfig(
                    dataset_dir=dataset, output_dir=tmp / "out2", allow_mixed_method=True
                )
            )
            self.assertEqual(len(plan["label_routes"]), 2)

    def test_refuses_nonempty_output_without_force(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(tmp, [_frame("a")])
            output = tmp / "out"
            output.mkdir()
            (output / "stale").write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_training_campaign(
                    TrainingConfig(dataset_dir=dataset, output_dir=output)
                )


class OptimizerDefaultsTests(unittest.TestCase):
    """Scratch defaults come from the Fe16 v1 diagnosis (lr 0.005 plateaued, and
    forces_weight 100 left energy ~1% of the loss); finetune keeps its defaults."""

    def _argv(self, **overrides):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(tmp, [_frame("a", 0, 1), _frame("b", 0, 5)])
            plan = write_training_campaign(
                TrainingConfig(dataset_dir=dataset, output_dir=tmp / "run", **overrides)
            )
            return plan["seed_runs"][0]["argv"]

    def test_scratch_defaults(self):
        argv = self._argv()
        self.assertIn("--lr=0.001", argv)
        self.assertIn("--forces_weight=10.0", argv)
        self.assertIn("--energy_weight=1.0", argv)

    def test_finetune_defaults_unchanged(self):
        argv = self._argv(mode="finetune", foundation_model="medium")
        self.assertIn("--lr=0.0001", argv)
        self.assertIn("--forces_weight=100.0", argv)

    def test_explicit_values_win(self):
        argv = self._argv(lr=0.002, forces_weight=50.0)
        self.assertIn("--lr=0.002", argv)
        self.assertIn("--forces_weight=50.0", argv)
        self.assertNotIn("--forces_weight=10.0", argv)


class FinetuneCampaignTests(unittest.TestCase):
    def test_polar_foundation_uses_polarmace_and_no_custom_embedding(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(tmp, [_frame("a", 0, 1), _frame("b", -1, 2)])
            output = tmp / "ft"
            plan = write_training_campaign(
                TrainingConfig(
                    dataset_dir=dataset,
                    output_dir=output,
                    mode="finetune",
                    foundation_model="polar-1-m",
                )
            )
            self.assertEqual(plan["model"], "PolarMACE")
            argv = plan["seed_runs"][0]["argv"]
            self.assertIn("--foundation_model=polar-1-m", argv)
            self.assertIn("--total_charge_key=charge", argv)
            self.assertFalse(any(a.startswith("--embedding_specs=") for a in argv))
            self.assertTrue((output / "PREFLIGHT.md").is_file())

    def test_generic_foundation_keeps_custom_embedding(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(tmp, [_frame("a", 0, 1)])
            plan = write_training_campaign(
                TrainingConfig(
                    dataset_dir=dataset,
                    output_dir=tmp / "ft",
                    mode="finetune",
                    foundation_model="medium",
                )
            )
            argv = plan["seed_runs"][0]["argv"]
            self.assertEqual(plan["model"], "ScaleShiftMACE")
            self.assertIn("--foundation_model=medium", argv)
            self.assertTrue(any(a.startswith("--embedding_specs=") for a in argv))
            self.assertIn("--amsgrad", argv)


def _git_bash() -> str | None:
    bash = shutil.which("bash")
    # On Windows, System32\bash.exe is WSL, which cannot see these temp paths.
    if bash is None or "system32" in bash.lower():
        return None
    return bash


_FAKE_MACE = """#!/usr/bin/env bash
for a in "$@"; do
  case "$a" in --name=*) n="${a#*=}" ;; --seed=*) s="${a#*=}" ;; esac
done
mkdir -p logs
log="logs/${n}_run-${s}.log"
c="$FAKE_COUNT"
echo "INFO: Total Training set [energy: 2, forces: 2, total_charge: $c, total_spin: $c]" >> "$log"
echo "INFO: Total Validation set [energy: 1, forces: 1, total_charge: $c, total_spin: $c]" >> "$log"
echo "INFO: Started training, reporting errors on validation set" >> "$log"
exec sleep "${FAKE_SLEEP:-0}"
"""


@unittest.skipIf(_git_bash() is None, "needs a POSIX bash")
class BlindLabelGuardTests(unittest.TestCase):
    """run.sh kills a MACE run whose log shows zero charge/spin labels."""

    def _run(self, tmp: Path, count: int, sleep: int = 0) -> subprocess.CompletedProcess:
        bindir = tmp / "bin"
        bindir.mkdir(exist_ok=True)
        fake = bindir / "mace_run_train"
        fake.write_bytes(_FAKE_MACE.encode())
        fake.chmod(0o755)
        output = tmp / "run"
        if not output.exists():
            dataset = _dataset(tmp, [_frame("a", 0, 1), _frame("b", 0, 5)])
            write_training_campaign(
                TrainingConfig(dataset_dir=dataset, output_dir=output, seeds=(7,))
            )
        env = dict(os.environ, FAKE_COUNT=str(count), FAKE_SLEEP=str(sleep))
        env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
        return subprocess.run(
            [_git_bash(), str(output / "seed_7" / "run.sh")],
            env=env, capture_output=True, text=True, timeout=60,
        )

    def test_zero_counts_fail_loudly(self):
        with tempfile.TemporaryDirectory() as raw:
            result = self._run(Path(raw), count=0)
            self.assertEqual(result.returncode, BLIND_LABELS_EXIT_CODE, result.stderr)
            self.assertIn("FATAL", result.stderr)
            self.assertIn("total_spin: 0", result.stderr)

    def test_zero_counts_kill_a_running_job(self):
        with tempfile.TemporaryDirectory() as raw:
            result = self._run(Path(raw), count=0, sleep=45)
            self.assertEqual(result.returncode, BLIND_LABELS_EXIT_CODE, result.stderr)

    def test_nonzero_counts_pass_and_stale_log_is_ignored(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            self.assertEqual(self._run(tmp, count=0).returncode, BLIND_LABELS_EXIT_CODE)
            # Rerun in the same directory: MACE appends to the old log, whose
            # zero counts must not fail the corrected run.
            result = self._run(tmp, count=14452)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("FATAL", result.stderr)



def _add_moment_column(path: Path, values=(4.0, -1.0), skip_frames: int = 0) -> None:
    """Append a signed per-atom `local_moment` column to an extxyz (test helper)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out, i, frame = [], 0, 0
    while i < len(lines):
        n = int(lines[i]); header = lines[i + 1]
        if frame >= skip_frames:
            header = header.replace("REF_forces:R:3", "REF_forces:R:3:local_moment:R:1")
            atoms = [f"{line} {values[k % len(values)]:.6f}" for k, line in enumerate(lines[i + 2:i + 2 + n])]
        else:
            atoms = lines[i + 2:i + 2 + n]
        out += [lines[i], header, *atoms]
        i += n + 2; frame += 1
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _level_frame(record_id: str, multiplicity: int = 1, level: str = "BPW91/6-311++g(d)") -> LabeledFrame:
    frame = _frame(record_id, 0, multiplicity)
    frame.record.metadata = {"parent_record_id": record_id, "label_level": level}
    return frame


class LabelLevelFallbackTests(unittest.TestCase):
    def test_label_level_stands_in_for_a_missing_route(self):
        with tempfile.TemporaryDirectory() as raw:
            dataset = _dataset(Path(raw), [_level_frame("a"), _level_frame("b", 5)])
            facts = scan_dataset(dataset)
            self.assertEqual(facts.label_routes, {"label_level:BPW91/6-311++g(d)"})

    def test_mixed_label_levels_are_still_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = _dataset(tmp, [_level_frame("a"), _level_frame("b", 5, level="B3LYP/def2tzvp")])
            with self.assertRaises(ValueError):
                write_training_campaign(TrainingConfig(dataset_dir=dataset, output_dir=tmp / "run"))


class LocalMomentInputTests(unittest.TestCase):
    def _dataset_with_moments(self, tmp: Path, skip_frames: int = 0) -> Path:
        dataset = _dataset(tmp, [_frame("a", 0, 1), _frame("b", 0, 5)])
        for name in ("all.extxyz", "train.extxyz", "valid.extxyz", "test.extxyz"):
            _add_moment_column(dataset / name, skip_frames=skip_frames if name == "train.extxyz" else 0)
        return dataset

    def test_per_atom_spec_has_explicit_key_and_manifest_lists_it(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            plan = write_training_campaign(TrainingConfig(
                dataset_dir=self._dataset_with_moments(tmp), output_dir=tmp / "run",
                local_moment_key="local_moment"))
            argv = plan["seed_runs"][0]["argv"]
            (spec_arg,) = [a for a in argv if a.startswith("--embedding_specs=")]
            spec = json.loads(spec_arg.partition("=")[2])["local_moment"]
            self.assertEqual((spec["type"], spec["per"], spec["key"]), ("continuous", "atom", "local_moment"))
            self.assertIn("local_moment", plan["required_inputs"]["arrays"])

    def test_frame_without_the_column_is_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset = self._dataset_with_moments(tmp, skip_frames=1)
            with self.assertRaisesRegex(ValueError, "required per-atom input missing"):
                write_training_campaign(TrainingConfig(
                    dataset_dir=dataset, output_dir=tmp / "run", local_moment_key="local_moment"))

    def test_baseline_ignores_the_column(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            plan = write_training_campaign(TrainingConfig(
                dataset_dir=self._dataset_with_moments(tmp), output_dir=tmp / "run"))
            (spec_arg,) = [a for a in plan["seed_runs"][0]["argv"] if a.startswith("--embedding_specs=")]
            self.assertNotIn("local_moment", json.loads(spec_arg.partition("=")[2]))
            self.assertEqual(plan["required_inputs"]["arrays"], {})

    def test_polar_finetune_refuses_a_local_moment_input(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            with self.assertRaisesRegex(ValueError, "silently ignored"):
                write_training_campaign(TrainingConfig(
                    dataset_dir=self._dataset_with_moments(tmp), output_dir=tmp / "run",
                    mode="finetune", foundation_model="polar-1-m", local_moment_key="local_moment"))


if __name__ == "__main__":
    unittest.main()
