"""Recheck the existing seminar diagnostic without model inference or data writes.

Only aggregate diagnostics and hashes are emitted; no geometries or weights.
Usage: python verify_archive.py --archive D:/MLIP_Work_Folder/Cluster_MLIP
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for index in ordered[start:end]:
            result[index] = (start + 1 + end) / 2
        start = end
    return result


def check_archive(archive: Path) -> dict:
    directory = archive / 'fe16_v1/13_seminar_checks/calib_longB_heldout'
    csv_path = directory / 'calibration.csv'
    with csv_path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    scores = [float(row['disagreement_ev_ang']) for row in rows]
    errors = [float(row['committee_error_ev_ang']) for row in rows]
    if len(scores) < 3 or not all(math.isfinite(x) for x in scores + errors):
        raise ValueError('Need at least three finite diagnostic rows')
    summary_path = directory / 'calibration.json'
    recorded = json.loads(summary_path.read_text(encoding='utf-8'))
    top_n = math.ceil(len(rows) * recorded['top_fraction'])
    top_error = set(sorted(range(len(rows)), key=errors.__getitem__, reverse=True)[:top_n])
    top_score = set(sorted(range(len(rows)), key=scores.__getitem__, reverse=True)[:top_n])
    rho = statistics.correlation(ranks(scores), ranks(errors))
    recall = len(top_error & top_score) / top_n
    # CSV stores six decimals; the JSON used unrounded predictions.
    if not math.isclose(rho, recorded['spearman_rho'], abs_tol=0.001):
        raise ValueError('Archived rank correlation not reproduced within rounding tolerance')
    if recall != recorded['top_error_recall'] or len(rows) != recorded['n_frames']:
        raise ValueError('Archived recall or frame count mismatch')
    dataset = archive / 'fe16_v1/02d_fe16_tol05'
    dataset_summary = json.loads((dataset / 'dataset_summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((archive / 'fe16_v1/04_model/train_manifest.json').read_text(encoding='utf-8'))
    hash_checks = {}
    for name, expected in manifest['dataset_files'].items():
        actual = hashlib.sha256((dataset / name).read_bytes()).hexdigest()
        hash_checks[name] = actual == expected
    if not all(hash_checks.values()):
        raise ValueError('Archived training dataset hashes changed')
    evidence_paths = [csv_path, summary_path, dataset / 'dataset_summary.json']
    return {
        'check_type': 'archived_outputs_recalculated_no_inference',
        'n_frames': len(rows),
        'n_models_recorded': recorded['n_models'],
        'top_n': top_n,
        'spearman_rho_from_rounded_csv': rho,
        'spearman_rho_recorded_unrounded': recorded['spearman_rho'],
        'top_error_recall': recall,
        'dataset_frames_recorded': dataset_summary['frames'],
        'independent_source_geometry_groups_recorded': dataset_summary['independent_source_geometry_groups'],
        'dataset_splits_recorded': dataset_summary['splits'],
        'training_manifest_dataset_hash_checks': hash_checks,
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in evidence_paths},
        'limitations': [
            'Archived CSV reanalysis, not a fresh prediction run',
            'Twin-heavy Fe16 diagnostic, not independent supported-cluster validation',
            'Split counts read from archived summary; transitive split independence not certified',
            'No novelty-class statistics recomputed',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check_archive(args.archive), indent=2))
