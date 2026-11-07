#!/usr/bin/env python3
"""Read-only legacy extraction and conservative cross-warehouse overlap audit."""
import argparse
import collections
import csv
import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_write(path, columns, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def coords(rec):
    return np.array([[a.x, a.y, a.z] for a in rec.atoms], dtype=float)


def descriptor(rec):
    """Sorted distances by element-pair: invariant candidate filter, not identity."""
    x = coords(rec)
    groups = collections.defaultdict(list)
    for i, a in enumerate(rec.atoms):
        for j in range(i):
            groups[tuple(sorted((a.symbol, rec.atoms[j].symbol)))].append(
                float(np.linalg.norm(x[i] - x[j])))
    return np.array([d for key in sorted(groups) for d in sorted(groups[key])] or [0.])


def aligned_distance(a, b):
    """Element-preserving assignment followed by proper Kabsch rotation.

    Assignment uses per-atom sorted distances to each element. Symmetric or
    homometric cases can be unresolved; never label those as novel or delete.
    """
    x, y = coords(a), coords(b)
    x, y = x - x.mean(0), y - y.mean(0)
    sa, sb = [v.symbol for v in a.atoms], [v.symbol for v in b.atoms]
    elements = sorted(set(sa))
    def fingerprints(z, symbols):
        distances = np.linalg.norm(z[:, None] - z[None, :], axis=2)
        return np.array([np.concatenate([np.sort(distances[i, np.array(symbols) == e])
                                         for e in elements]) for i in range(len(z))])
    fa, fb = fingerprints(x, sa), fingerprints(y, sb)
    cost = np.linalg.norm(fa[:, None] - fb[None, :], axis=2)
    cost[np.array(sa)[:, None] != np.array(sb)[None, :]] = 1e10
    _, perm = linear_sum_assignment(cost)
    y = y[perm]
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    residual = np.linalg.norm(x @ (u @ correction @ vt) - y, axis=1)
    return float(np.sqrt(np.mean(residual**2))), float(residual.max())


def self_test():
    from cluster_mlip.models import Atom, Record
    def rec(x, symbols):
        return Record('x', 'test', [Atom(s, *v) for s, v in zip(symbols, x)], 0, 3, 'unknown')
    x = np.array([[0., 0., 0.], [2., 0., 0.], [.3, 1.7, .2], [.2, .4, 1.3]])
    s = ['Fe', 'Fe', 'O', 'O']
    a = rec(x, s)
    rot = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    p = [1, 0, 3, 2]
    b = rec((x @ rot + 5)[p], [s[i] for i in p])
    assert np.allclose(descriptor(a), descriptor(b))
    assert aligned_distance(a, b)[1] < 1e-10
    altered = x.copy()
    altered[2] += .3
    assert aligned_distance(a, rec(altered, s))[1] > .02
    # A reflected nonplanar, distinguishable geometry cannot use an improper rotation.
    distinct = ['Fe', 'O', 'N', 'H']
    assert aligned_distance(rec(x, distinct), rec(x * [-1, 1, 1], distinct))[1] > .02
    print('PASS: rotation, translation, element-preserving permutation, distortion, reflection')


def main():
    from cluster_mlip.gaussian import extract_document_records, parse_force_frames
    from cluster_mlip.io import SUPPORTED_SUFFIXES, read_document, source_tree, read_extxyz, write_extxyz, write_manifest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--campaigns', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-deviation', type=float, default=.02)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.source, args.campaigns, args.output)) or args.max_deviation <= 0:
        parser.error('source, campaigns, output and positive max-deviation required')
    source, campaigns, out = args.source.resolve(), args.campaigns.resolve(), args.output.resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if out == source or source in out.parents:
        raise ValueError('output must be outside the source collection')
    references = []
    reference_inputs = []
    # Require all three stores before beginning heavy extraction.
    for name in ('FenOm_Warehouse', 'FenOm_Warehouse2', 'General_Warehouse'):
        root = campaigns / name / 'extracted'
        manifest, seeds = root / 'manifest.csv', root / 'seeds.extxyz'
        if not manifest.is_file() or not seeds.is_file():
            raise FileNotFoundError(f'Required reference pair missing: {root}')
        with manifest.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        rs = read_extxyz(seeds)
        if collections.Counter(r['record_id'] for r in rows) != collections.Counter(r.record_id for r in rs):
            raise ValueError(f'Manifest/geometry record correspondence differs: {root}')
        references.extend((name, r) for r in rs)
        reference_inputs.append({'warehouse': name, 'manifest': str(manifest), 'manifest_sha256': sha(manifest),
                                 'seeds': str(seeds), 'seeds_sha256': sha(seeds), 'records': len(rs)})
    out.mkdir(parents=True, exist_ok=False)
    extracted = out / 'extracted'
    extracted.mkdir()
    files, incoming, provenance = [], [], []
    # Hash every source file. Nested archives are read only in temporary storage.
    for file in sorted(source.rglob('*')):
        if not file.is_file():
            continue
        rel = file.relative_to(source).as_posix()
        entry = {'source': rel, 'sha256': '', 'bytes': '', 'status': 'unsupported', 'records': 0,
                 'force_frames': 0, 'error': ''}
        try:
            before = file.stat()
            entry.update(sha256=sha(file), bytes=before.st_size)
            if file.suffix.lower() not in SUPPORTED_SUFFIXES | {'.zip'}:
                files.append(entry)
                continue
            with source_tree(file) as tree:
                docs = [file] if file.suffix.lower() != '.zip' else sorted(
                    p for p in tree.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
                for doc in docs:
                    docrel = rel if doc == file else rel + '::' + doc.relative_to(tree).as_posix()
                    try:
                        text = read_document(doc)
                        records = extract_document_records(text, docrel)
                        try:
                            force_count = len(parse_force_frames(text, Path(docrel)))
                            force_error = ''
                        except Exception as exc:
                            force_count, force_error = 0, str(exc)
                        entry['force_frames'] += force_count
                        for index, r in enumerate(records):
                            original = r.record_id
                            r.record_id = hashlib.sha256(f'{docrel}|{index}|{original}'.encode()).hexdigest()[:24]
                            r.metadata.update(original_record_id=original, source_root=str(source),
                                              source_file_sha256=entry['sha256'], document_record_index=index)
                            incoming.append(r)
                            provenance.append({'record_id': r.record_id, 'original_record_id': original,
                                               'source': docrel, 'document_record_index': index,
                                               'source_file_sha256': entry['sha256'], 'document_force_frames': force_count,
                                               'force_parse_error': force_error,
                                               'force_label_for_this_seed': 'not_assigned'})
                        entry['records'] += len(records)
                    except Exception as exc:
                        entry['error'] += f'{docrel}: {exc}\n'
                entry['status'] = 'partial' if entry['error'] else ('parsed' if entry['records'] else 'no_records')
            after = file.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('source changed during read; rerun in a new output directory')
        except Exception as exc:
            entry.update(status='error', error=entry['error'] + str(exc))
        files.append(entry)
    write_extxyz(incoming, extracted / 'seeds.extxyz')
    write_manifest(incoming, extracted / 'manifest.csv')
    csv_write(out / 'files.csv', list(files[0]) if files else ['source','status'], files)
    csv_write(out / 'record_provenance.csv', list(provenance[0]) if provenance else ['record_id'], provenance)
    # Keep every record; matches are proposals for review, never destructive filtering.
    groups = collections.defaultdict(list)
    for warehouse, r in references:
        groups[r.formula].append((warehouse, r))
    matches, inclusion = [], []
    new_file_hashes = collections.defaultdict(list)
    for f in files:
        if f['sha256']:
            new_file_hashes[f['sha256']].append(f['source'])
    file_duplicates = [{'sha256': h, 'sources': json.dumps(v)} for h,v in new_file_hashes.items() if len(v)>1]
    csv_write(out / 'incoming_file_duplicates.csv', ['sha256','sources'], file_duplicates)
    incoming_by_formula = collections.defaultdict(list)
    for r in incoming:
        incoming_by_formula[r.formula].append(r)
    for formula, new_records in incoming_by_formula.items():
        old = groups[formula]
        combined = old + [('incoming', r) for r in new_records]
        tree = cKDTree(np.array([descriptor(r) for _, r in combined]))
        offset = len(old)
        for i, r in enumerate(new_records):
            candidates = tree.query_ball_point(descriptor(r), 2 * args.max_deviation, p=np.inf)
            found = 0
            for j in candidates:
                if j >= offset + i:
                    continue  # self and future incoming pairs, each internal pair once
                warehouse, other = combined[j]
                rmsd, maxdev = aligned_distance(r, other)
                route_equal = bool(r.route and other.route and r.route == other.route)
                same_state = r.charge == other.charge and r.multiplicity == other.multiplicity
                if maxdev <= args.max_deviation:
                    kind = 'geometry_match'
                else:
                    kind = 'distance_candidate_unresolved'
                matches.append({'incoming_record': r.record_id, 'other_warehouse': warehouse,
                                'other_record': other.record_id, 'incoming_source': r.source,
                                'other_source': other.source, 'match_type': kind,
                                'aligned_rmsd_A': rmsd, 'aligned_max_deviation_A': maxdev,
                                'same_charge_multiplicity': same_state, 'identical_raw_route': route_equal,
                                'decision': 'retain_distinct_global_state' if not same_state else 'retain_pending_electronic_method_review'})
                found += 1
            inclusion.append({'record_id': r.record_id, 'formula': formula, 'candidate_count': found,
                              'decision': 'retain_pending_review' if found else 'no_overlap_candidate_at_tolerance',
                              'training_ready': False})
    columns = ['incoming_record','other_warehouse','other_record','incoming_source','other_source','match_type',
               'aligned_rmsd_A','aligned_max_deviation_A','same_charge_multiplicity','identical_raw_route','decision']
    csv_write(out / 'overlap.csv', columns, matches)
    csv_write(out / 'inclusion_proposals.csv', ['record_id','formula','candidate_count','decision','training_ready'], inclusion)
    summary = {'source': str(source), 'output': str(out), 'reference_inputs': reference_inputs,
               'python': platform.python_version(), 'script_sha256': sha(Path(__file__)),
               'files': len(files), 'records': len(incoming),
               'file_status': dict(collections.Counter(f['status'] for f in files)),
               'formulas': dict(collections.Counter(r.formula for r in incoming)),
               'bare_fe': dict(collections.Counter(r.formula for r in incoming if re.fullmatch(r'Fe\d*',r.formula))),
               'overlap_pairs': len(matches), 'max_deviation_A': args.max_deviation,
               'limitations': ['No automatic deduplication or training inclusion',
                 'Force counts are per document, not labels assigned to extracted seeds',
                 'Raw route equality does not establish electronic-root or numerical equivalence',
                 'Isotope metadata is not verified by the legacy Record schema',
                 'Symmetric/permuted distance candidates may require further alignment review',
                 'Existing raw source-file hashes unavailable: cross-warehouse file identity not claimed',
                 'Records are not independent structural families']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'report.md').write_text('# Legacy inventory and overlap audit\n\n'
        f"Files: {len(files)}; extracted records: {len(incoming)}; overlap candidates: {len(matches)}.\n\n"
        'All three manifests and geometry stores were checked. All records retained.\n\n' +
        '\n'.join('- ' + v for v in summary['limitations']) + '\n')
    complete = bool(incoming) and not any(f['status'] in ('partial','error') for f in files)
    (out / 'RUN_STATUS.json').write_text(json.dumps({'status': 'complete' if complete else 'partial',
                                                   'summary_written': True}) + '\n')
    print(json.dumps({'status': 'complete' if complete else 'partial', 'records': len(incoming), 'output': str(out)}))


if __name__ == '__main__':
    main()
