"""Checks for the bounded, read-only seminar evidence verifier."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    'grace_archive_verifier',
    Path(__file__).resolve().parents[1] / 'experiments/foundation_transfer_pilot/verify_archive.py',
)
assert SPEC and SPEC.loader
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class ArchiveVerifierTests(unittest.TestCase):
    def test_tied_ranks_use_average(self):
        self.assertEqual(verifier.ranks([2.0, 1.0, 2.0, 4.0]), [2.5, 1.0, 2.5, 4.0])

    def test_reanalysis_and_changed_dataset_detection(self):
        with tempfile.TemporaryDirectory() as raw:
            archive = Path(raw)
            diagnostic = archive / 'fe16_v1/13_seminar_checks/calib_longB_heldout'
            diagnostic.mkdir(parents=True)
            (diagnostic / 'calibration.csv').write_text(
                'disagreement_ev_ang,committee_error_ev_ang\n1,3\n2,2\n3,1\n', encoding='utf-8',
            )
            (diagnostic / 'calibration.json').write_text(json.dumps({
                'n_frames': 3, 'n_models': 2, 'top_fraction': 1/3,
                'spearman_rho': -1.0, 'top_error_recall': 0.0,
            }), encoding='utf-8')
            dataset = archive / 'fe16_v1/02d_fe16_tol05'
            dataset.mkdir()
            (dataset / 'dataset_summary.json').write_text(json.dumps({
                'frames': 3, 'independent_source_geometry_groups': 3, 'splits': {},
            }), encoding='utf-8')
            data_path = dataset / 'train.extxyz'
            data_path.write_bytes(b'fixture')
            model = archive / 'fe16_v1/04_model'
            model.mkdir()
            (model / 'train_manifest.json').write_text(json.dumps({'dataset_files': {
                'train.extxyz': hashlib.sha256(b'fixture').hexdigest(),
            }}), encoding='utf-8')
            result = verifier.check_archive(archive)
            self.assertEqual(result['spearman_rho_from_rounded_csv'], -1.0)
            self.assertEqual(result['top_error_recall'], 0.0)
            data_path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hashes changed'):
                verifier.check_archive(archive)
