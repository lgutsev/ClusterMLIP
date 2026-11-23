"""IRC frames and higher-order saddle candidates are labeled, never optimized.

An IRC point is a fixed point along a reaction path, not a stationary point:
an MLIP needs its energy and forces at exactly the archived geometry. These
tests pin the whole chain -- extraction, generation, audit, in-place repair of
an existing campaign, campaign-status stage counting, and collection.
"""
import csv
import hashlib
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.batch_progress import write_batch_progress
from cluster_mlip.cli import main
from cluster_mlip.gaussian import extract_document_records
from cluster_mlip.io import read_extxyz, write_extxyz
from cluster_mlip.jobs import write_gaussian_jobs
from cluster_mlip.models import Atom, LabeledFrame, Record
from cluster_mlip.physical_checks import stationary_point_check
from cluster_mlip.relaunch import prepare_route_relaunch
from cluster_mlip.restart import prepare_spin_restarts
from cluster_mlip.route_audit import audit_campaign_routes
from cluster_mlip.routes import (
    force_only_route,
    forbidden_fixed_geometry_keywords,
    input_stage_routes,
    inspect_job,
    resolve_geometry_role,
    route_is_force_only,
    route_optimizes,
    route_policy,
)
from cluster_mlip.spin import SPIN_MANIFEST_COLUMNS, write_spin_jobs
from cluster_mlip.stratify import classify_record

# The Warehouse 2 protocol, as its inputs carry it.
W2_ROUTE = (
    '#p UBPW91/6-311++G* SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm Opt '
    'IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine Pop=Regular'
)
W2_LATER = W2_ROUTE.replace(' Pop=Regular', ' Geom=Checkpoint Guess=Read Pop=Regular')
# What the assignment specifies for the repaired stages, verbatim.
FORCE_ROOT = (
    '#p UBPW91/6-311++G* SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm Force '
    'IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine Pop=Regular'
)
FORCE_LATER = (
    '#p UBPW91/6-311++G* SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc) NoSymm Force '
    'IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine Geom=Checkpoint Guess=Read Pop=Regular'
)

FE2O2 = [('Fe', 0.0, 0.0, 0.0), ('Fe', 2.2, 0.0, 0.0), ('O', 1.1, 1.1, 0.0), ('O', 1.1, -1.1, 0.0)]
_Z = {'Fe': 26, 'O': 8, 'N': 7, 'H': 1}
_RULE = ' ' + '-' * 69


def _orientation(atoms):
    lines = [' Input orientation:', _RULE,
             ' Center     Atomic     Atomic              Coordinates (Angstroms)',
             ' Number     Number      Type              X           Y           Z', _RULE]
    for index, (symbol, x, y, z) in enumerate(atoms, 1):
        lines.append(f' {index:6d} {_Z[symbol]:10d} {0:12d} {x:15.6f} {y:11.6f} {z:11.6f}')
    lines.append(_RULE)
    return '\n'.join(lines) + '\n'


def _forces(atoms, energy):
    lines = [f' SCF Done:  E(UBPW91) =  {energy:.10f}     A.U.', ' Forces (Hartrees/Bohr)',
             _RULE, ' Center     Atomic                   Forces (Hartrees/Bohr)',
             ' Number     Number              X              Y              Z', _RULE]
    for index, (symbol, *_) in enumerate(atoms, 1):
        lines.append(f' {index:6d} {_Z[symbol]:8d}    0.001000000   -0.002000000    0.000500000')
    lines.append(_RULE)
    return '\n'.join(lines) + '\n'


def ladder_log(routes, mults, atoms=FE2O2, *, optimized=False, stop_after=None, moved=False):
    """A g09 log of a Link1 ladder; ``stop_after`` cuts it off inside that stage."""
    parts = []
    for stage, (route, mult) in enumerate(zip(routes, mults)):
        if stage:
            parts.append(f' Link1:  Proceeding to internal job step number  {stage + 1}.\n')
        # Displace one atom: a rigid shift or rotation is the same geometry.
        geometry = [(s, x + (0.2 if moved and stage and index == 0 else 0.0), y, z)
                    for index, (s, x, y, z) in enumerate(atoms)]
        parts.append(f' {route}\n Charge =  0 Multiplicity = {mult}\n')
        parts.append(_orientation(geometry) + _forces(geometry, -2660.0 - stage))
        if stop_after == stage:
            break
        if optimized:
            parts.append(' Optimization completed.\n    -- Stationary point found.\n')
        parts.append(' Normal termination of Gaussian 09 at Wed Sep 23 12:00:00 2026.\n')
    return ''.join(parts)


def ladder_input(chain, routes, mults, atoms=FE2O2):
    sections, previous = [], ''
    for stage, (route, mult) in enumerate(zip(routes, mults)):
        chk = f'{chain}-s{stage:02d}-m{mult}.chk'
        header = (f'%oldchk={previous}\n' if previous else '') + (
            f'%chk={chk}\n%mem=24GB\n%nprocshared=12\n')
        body = f'\nClusterMLIP spin pathway; stage={stage}\n\n0 {mult}\n'
        if not stage:
            body += ''.join(f'{s:18s} {x: .12f} {y: .12f} {z: .12f}\n' for s, x, y, z in atoms)
        sections.append(f'{header}{route}\n{body}\n')
        previous = chk
    return '--Link1--\n'.join(sections)


def ladder_routes(n):
    return [W2_ROUTE] + [W2_LATER] * (n - 1)


class W2Campaign:
    """A legacy spin campaign shaped like Warehouse 2: config_type in the
    manifest, no geometry-role columns, fixed batch folders, Opt everywhere."""

    MULTS = (11, 9, 7)

    def build(self, root, jobs):
        campaign = root / 'w2'
        (campaign / 'inputs').mkdir(parents=True)
        rows = []
        listings: dict[str, list[str]] = {}
        for job in jobs:
            name = job['name'] + '.gjf'
            routes = job.get('routes') or ladder_routes(len(self.MULTS))
            text = ladder_input(job['name'], routes, self.MULTS)
            (campaign / 'inputs' / name).write_text(text, encoding='utf-8', newline='\n')
            digest = hashlib.sha256((campaign / 'inputs' / name).read_bytes()).hexdigest()
            batch = campaign / 'slurm_batches' / job['batch']
            batch.mkdir(parents=True, exist_ok=True)
            shutil.copy2(campaign / 'inputs' / name, batch / name)
            listings.setdefault(job['batch'], []).append(name)
            previous = ''
            for stage, mult in enumerate(self.MULTS):
                chk = f"{job['name']}-s{stage:02d}-m{mult}.chk"
                row = {column: '' for column in SPIN_MANIFEST_COLUMNS}
                row.update({
                    'job_id': f"{job['name']}-s{stage:02d}", 'chain_id': job['name'],
                    'stage_index': str(stage), 'pathway': 'multiplicity_ladder',
                    'config_type': job['config_type'], 'source': job['source'],
                    'parent_record_id': job.get('parent', job['name']), 'formula': 'Fe2O2',
                    'intended_charge': '0', 'intended_multiplicity': str(mult),
                    'high_spin_multiplicity': str(self.MULTS[0]),
                    'final_target_multiplicity': str(self.MULTS[-1]),
                    'checkpoint': chk, 'predecessor_checkpoint': previous,
                    'first_route': routes[stage], 'route_search_kind': 'minimum',
                    'input': f'inputs/{name}', 'output': f"{job['name']}.log",
                    'input_sha256': digest,
                })
                rows.append(row)
                previous = chk
            log = job.get('log')
            if log == 'complete':
                (batch / f"{job['name']}.log").write_text(
                    ladder_log(routes, self.MULTS, optimized=True), encoding='utf-8')
                (batch / f"{job['name']}.rc").write_text('0')
                (batch / f"{job['name']}.started").write_text('t0')
                (batch / f"{job['name']}.finished").write_text('t1')
            elif log == 'partial':
                (batch / f"{job['name']}.log").write_text(
                    ladder_log(routes, self.MULTS, optimized=True, stop_after=1),
                    encoding='utf-8')
            if job.get('running'):
                (batch / f"{job['name']}.started").write_text('now')
            for stage, mult in enumerate(self.MULTS):
                if job.get('checkpoints', log is not None):
                    (batch / f"{job['name']}-s{stage:02d}-m{mult}.chk").write_bytes(b'opt')
        for batch, names in listings.items():
            (campaign / 'slurm_batches' / batch / 'inputs.txt').write_text(
                '\n'.join(names) + '\n', encoding='utf-8', newline='\n')
        with (campaign / 'spin_jobs.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=SPIN_MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        (campaign / 'slurm_plan.json').write_text(
            json.dumps({'batch_count': 7, 'config': {'cpus_per_job': 12},
                        'batches': {'batch_0006': listings.get('batch_0006', []),
                                    'batch_0007': listings.get('batch_0007', [])}},
                       sort_keys=True) + '\n', encoding='utf-8')
        return campaign

    def standard(self, root):
        return self.build(root, [
            # IRC-derived frames that the legacy extraction mislabeled as
            # higher-order saddles; the source name is the only evidence left.
            {'name': 'irc-fwd__fe2o2__higher-order-saddle__q0-m11__reference__aaa',
             'config_type': 'higher_order_saddle', 'batch': 'batch_0006',
             'source': 'warehouse2/IRC/Fe2O2_irc_fwd_pt5.log', 'log': 'complete'},
            {'name': 'irc-rev__fe2o2__higher-order-saddle__q0-m11__reference__bbb',
             'config_type': 'higher_order_saddle', 'batch': 'batch_0006',
             'source': 'warehouse2/IRC/Fe2O2_irc_rev_pt3.log'},
            # A genuine higher-order candidate from a frequency analysis.
            {'name': 'hos__fe2o2__higher-order-saddle__q0-m11__reference__ccc',
             'config_type': 'higher_order_saddle', 'batch': 'batch_0007',
             'source': 'warehouse2/Fe2O2_hos_7.log', 'log': 'complete'},
            # A correctly launched, finished minimum that must stay untouched.
            {'name': 'min__fe2o2__minimum__q0-m11__reference__ddd',
             'config_type': 'minimum', 'batch': 'batch_0007',
             'source': 'warehouse2/Fe2O2_min_1.log', 'log': 'complete'},
        ])


def _rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _snapshot(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob('*')) if path.is_file()
    }


def _stages(text):
    """(route, oldchk, chk, has_coordinates) per Link1 stage."""
    out = []
    for section in re.split(r'^\s*--Link1--\s*$', text, flags=re.M):
        lines = [line.strip() for line in section.splitlines()]
        route = next((line for line in lines if line.startswith('#')), '')

        def directive(key):
            return next((line.split('=', 1)[1] for line in lines
                         if line.lower().startswith(key)), '')
        coordinates = any(re.match(r'^(Fe|O)\s+-?\d', line) for line in lines)
        out.append((route, directive('%oldchk'), directive('%chk'), coordinates))
    return out


# ---------------------------------------------------------------------------
# 1-2. Extraction: IRC frames carry path metadata and are never saddles
# ---------------------------------------------------------------------------

def _irc_log(route=' #p ubpw91/6-311+G* irc=(calcfc,maxpoints=2)', freq_block=''):
    atoms = [('Fe', 0.0, 0.0, 0.0), ('N', 1.7, 0.0, 0.0), ('O', 2.8, 0.0, 0.0)]

    def frame(dx, energy):
        moved = [(s, x + dx * i, y, z) for i, (s, x, y, z) in enumerate(atoms)]
        return _orientation(moved) + f' SCF Done:  E(UBPW91) =  {energy:.8f}     A.U.\n'

    return (
        f'{route}\n ' + '-' * 70 + '\n Charge = 0 Multiplicity = 2\n'
        + frame(0.0, -1412.0)                                   # the TS the path starts from
        + ' Point Number:   1          Path Number:   1\n' + frame(0.05, -1412.01) + freq_block
        + ' Point Number:   2          Path Number:   1\n' + frame(0.10, -1412.02)
        + ' Calculation of FORWARD path complete.\n'
        + ' Point Number:   1          Path Number:   2\n' + frame(-0.05, -1412.015)
        + ' Point Number:   2          Path Number:   2\n' + frame(-0.10, -1412.03)
        + ' Reaction path calculation complete.\n'
    )


def _g09_irc_log():
    """A Gaussian 09 IRC log in its own layout.

    The Point/Path summary is printed *after* a point has converged and is
    followed by "# OF POINTS ALONG THE PATH" / "# OF STEPS" lines -- which is
    exactly what the Warehouse 2 seeds recorded as the "route" of IRC frames
    extracted by the old parser. Each point takes several constrained
    optimization steps; the reverse path's second point never finished; the
    archive entry repeats the last energy as HF=.
    """
    atoms = [('Fe', 0.0, 0.0, 0.0), ('N', 1.7, 0.0, 0.0), ('O', 2.8, 0.0, 0.0)]

    def frame(dx, energy):
        moved = [(s, x + dx * i, y, z) for i, (s, x, y, z) in enumerate(atoms)]
        return _orientation(moved) + f' SCF Done:  E(UBPW91) =  {energy:.8f}     A.U.\n'

    def summary(point, path):
        return (f' Optimization completed.\n    -- Optimized point #   {point} Found.\n'
                ' IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC\n'
                f' Point Number:   {point}          Path Number:   {path}\n'
                '   CHANGE IN THE REACTION COORDINATE =    0.10007\n'
                '   NET REACTION COORDINATE UP TO THIS POINT =    0.10007\n'
                f'  # OF POINTS ALONG THE PATH =   {point}\n  # OF STEPS =   2\n\n'
                ' Calculating another point on the path.\n')

    return (
        ' ' + '-' * 70 + '\n #p ubpw91/6-311+G* irc=(calcfc,maxpoints=2)\n ' + '-' * 70 + '\n'
        ' Charge = 0 Multiplicity = 2\n'
        + frame(0.00, -1412.000)                                  # TS
        + frame(0.03, -1412.004) + frame(0.05, -1412.006) + summary(1, 1)
        + frame(0.08, -1412.010) + frame(0.10, -1412.012) + summary(2, 1)
        + ' Calculation of FORWARD path complete.\n'
        + ' Beginning calculation of the REVERSE path.\n'
        + frame(-0.03, -1412.005) + frame(-0.05, -1412.007) + summary(1, 2)
        + frame(-0.08, -1412.009)                                 # cut off here
        + ' 1\\1\\GINC-QB\\IRC\\UBPW91\\6-311+G\\Fe1N1O1\\\\HF=-1412.009\\\n'
    )


class IrcExtractionTests(unittest.TestCase):
    def test_gaussian09_layout_numbers_each_geometry_by_the_marker_that_closes_it(self):
        records = extract_document_records(_g09_irc_log(), 'IRC/A_feno_IRC_2_00001.out')
        self.assertEqual(len(records), 8)          # TS + 2 + 2 + 2 + 1 cut-off step
        for record in records:
            self.assertFalse(record.route.startswith('# OF'), record.route)
            self.assertEqual(record.metadata['irc_marker_layout'], 'marker_after_point')
            self.assertNotIn(record.config_type, ('higher_order_saddle', 'unknown'))
        summary = [(r.metadata.get('irc_direction'), r.irc_point, r.metadata['irc_frame_kind'],
                    r.metadata['geometry_role']) for r in records]
        self.assertEqual(summary, [
            ('', None, 'ts', 'transition_state'),
            ('forward', 1, 'optimization_step', 'irc_point'),
            ('forward', 1, 'converged_point', 'irc_point'),
            ('forward', 2, 'optimization_step', 'irc_point'),
            ('forward', 2, 'converged_point', 'reaction_path_endpoint'),
            ('reverse', 1, 'optimization_step', 'irc_point'),
            ('reverse', 1, 'converged_point', 'irc_point'),
            ('reverse', 2, 'unconverged_step', 'irc_point'),
        ])
        # The converged frame of a point is the geometry printed right before
        # its marker (dx = 0.05 for forward point 1).
        converged = next(r for r in records if r.irc_point == 1 and r.irc_path == 1
                         and r.metadata['irc_frame_kind'] == 'converged_point')
        self.assertAlmostEqual(converged.atoms[2].x, 2.9, places=6)
        ts = records[0]
        self.assertTrue(all(r.metadata['irc_parent_record_id'] == ts.record_id
                            for r in records[1:]))


    def test_irc_frames_carry_direction_point_parent_and_position(self):
        records = extract_document_records(_irc_log(), 'archive/FeNO_path.log')
        by_key = {(r.metadata.get('irc_direction'), r.irc_point): r for r in records}
        ts = by_key[('', None)]
        self.assertEqual(ts.config_type, 'transition_state')
        self.assertEqual(ts.metadata['geometry_role'], 'transition_state')
        self.assertEqual(ts.metadata['irc_path_position'], 'ts')
        self.assertEqual(ts.metadata['source_calculation_type'], 'irc')
        forward_mid, forward_end = by_key[('forward', 1)], by_key[('forward', 2)]
        reverse_end = by_key[('reverse', 2)]
        self.assertEqual(forward_mid.config_type, 'irc_forward')
        self.assertEqual(forward_mid.metadata['geometry_role'], 'irc_point')
        self.assertEqual(forward_mid.metadata['irc_path_position'], 'intermediate')
        self.assertEqual(forward_end.metadata['geometry_role'], 'reaction_path_endpoint')
        self.assertEqual(reverse_end.config_type, 'irc_reverse')
        self.assertEqual(reverse_end.metadata['irc_path_position'], 'endpoint')
        for record in (forward_mid, forward_end, reverse_end):
            self.assertEqual(record.metadata['irc_parent_record_id'], ts.record_id)
            self.assertEqual(record.metadata['irc_parent_source'], 'archive/FeNO_path.log')
            self.assertEqual(record.metadata['original_charge'], 0)
            self.assertEqual(record.metadata['original_multiplicity'], 2)
            self.assertEqual(record.metadata['geometry_role_source'], 'irc_point_marker')

    def test_a_cut_short_path_has_no_endpoint(self):
        text = _irc_log().split(' Calculation of FORWARD')[0]
        records = extract_document_records(text, 'archive/FeNO_path.log')
        positions = {r.irc_point: r.metadata['irc_path_position']
                     for r in records if r.irc_point}
        self.assertEqual(positions, {1: 'intermediate', 2: 'intermediate'})

    def test_one_directional_irc_names_its_single_path(self):
        text = _irc_log(route=' #p ubpw91/6-311+G* irc=(calcfc,reverse)')
        records = [r for r in extract_document_records(text, 'a.log') if r.irc_point]
        self.assertEqual({r.metadata['irc_direction'] for r in records}, {'reverse'})
        self.assertEqual({r.config_type for r in records}, {'irc_reverse'})

    def test_extracted_metadata_survives_the_seeds_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = extract_document_records(_irc_log(), 'archive/FeNO_path.log')
            write_extxyz(records, Path(tmp) / 'seeds.extxyz')
            again = {r.record_id: r for r in read_extxyz(Path(tmp) / 'seeds.extxyz')}
            for record in records:
                self.assertEqual(again[record.record_id].metadata['geometry_role'],
                                 record.metadata['geometry_role'])
                self.assertEqual(again[record.record_id].irc_point, record.irc_point)

    def test_multiple_imaginary_modes_at_an_irc_point_do_not_make_a_saddle(self):
        freq = (' Harmonic frequencies (cm**-1)\n'
                ' Frequencies --  -512.0000  -230.0000  -80.0000\n')
        records = extract_document_records(_irc_log(freq_block=freq), 'archive/x.log')
        types = {r.config_type for r in records}
        self.assertNotIn('higher_order_saddle', types)
        self.assertNotIn('first_order_saddle', types)
        point = next(r for r in records if r.irc_point == 1 and r.irc_path == 1)
        self.assertEqual(point.metadata['geometry_role'], 'irc_point')

    def test_irc_route_wins_even_without_nearby_point_markers(self):
        # A long constrained-optimization block leaves the energy >100k
        # characters from its Point Number marker; the IRC route still says
        # what the calculation was.
        atoms = [('Fe', 0.0, 0.0, 0.0), ('N', 1.7, 0.0, 0.0), ('O', 2.8, 0.0, 0.0)]
        freq = ' Harmonic frequencies (cm**-1)\n Frequencies --  -512.0  -230.0  -80.0\n'
        text = (' #p ubpw91/6-311+G* irc=(calcfc)\n Charge = 0 Multiplicity = 2\n'
                + _orientation(atoms) + ' SCF Done:  E(UBPW91) =  -1412.0     A.U.\n'
                + ' Point Number:   1          Path Number:   1\n' + ' filler\n' * 30_000
                + _orientation([(s, x + 0.1, y, z) for s, x, y, z in atoms])
                + ' SCF Done:  E(UBPW91) =  -1412.1     A.U.\n' + freq)
        records = extract_document_records(text, 'archive/run7.log')
        self.assertEqual(len(records), 2)
        later = next(r for r in records if r.metadata['irc_path_position'] != 'ts')
        self.assertEqual(later.config_type, 'irc_point')
        self.assertEqual(later.metadata['geometry_role'], 'irc_point')
        self.assertEqual(later.metadata['source_calculation_type'], 'irc')

    def test_filename_evidence_is_an_explicit_reported_fallback(self):
        atoms = [('Fe', 0.0, 0.0, 0.0), ('N', 1.7, 0.0, 0.0), ('O', 2.8, 0.0, 0.0)]
        freq = ' Harmonic frequencies (cm**-1)\n Frequencies --  -512.0  -230.0  80.0\n'
        body = (' #p ubpw91/6-311+G* opt freq\n Charge = 0 Multiplicity = 2\n'
                + _orientation(atoms) + ' SCF Done:  E(UBPW91) =  -1412.0     A.U.\n')
        irc_named = extract_document_records(body + freq, 'W2/IRC_runs/FeNO_irc_fwd_5.log')[0]
        self.assertEqual(irc_named.config_type, 'irc_forward')
        self.assertEqual(irc_named.metadata['geometry_role_source'], 'filename_fallback')
        self.assertEqual(irc_named.metadata['geometry_role_evidence'], 'IRC_runs')
        # Without the name, the same log is still what it always was.
        plain = extract_document_records(body + freq, 'W2/FeNO_5.log')[0]
        self.assertEqual(plain.config_type, 'higher_order_saddle')
        # A name never overrides an optimization that established a minimum,
        # and "irc" inside a word is not evidence.
        minimum_freq = freq.replace('-512.0  -230.0', '512.0  230.0')
        minimum = extract_document_records(body + minimum_freq, 'W2/irc_endpoint_opt.log')[0]
        self.assertEqual(minimum.config_type, 'minimum')
        circle = extract_document_records(body + freq, 'W2/circle_5.log')[0]
        self.assertEqual(circle.config_type, 'higher_order_saddle')

    def test_legacy_rows_resolve_through_the_same_ranked_evidence(self):
        fallback = resolve_geometry_role(
            'higher_order_saddle', source='warehouse2/IRC/Fe2O2_irc_fwd_pt5.log')
        self.assertEqual(fallback['geometry_role'], 'irc_point')
        self.assertEqual(fallback['geometry_role_source'], 'filename_fallback')
        route = resolve_geometry_role('higher_order_saddle', route='#p ubpw91 irc=(calcfc)')
        self.assertEqual(route['geometry_role_source'], 'irc_route')
        genuine = resolve_geometry_role('higher_order_saddle', source='w2/Fe2O2_hos_7.log')
        self.assertEqual(genuine['geometry_role'], 'higher_order_candidate')
        # A name never overturns an explicit TS or an established minimum --
        # nor re-flags a TS search an earlier relaunch already corrected.
        for label in ('transition_state', 'minimum'):
            kept = resolve_geometry_role(label, source='w2/IRC/Fe2O2_ts_for_irc.log')
            self.assertNotEqual(kept['geometry_role_source'], 'filename_fallback', label)
        for config_type in ('irc_forward', 'irc_reverse', 'irc_point', 'irc_input_seed'):
            self.assertEqual(route_policy(config_type), 'fixed_geometry', config_type)
        self.assertEqual(route_policy('higher_order_saddle'), 'higher_order')
        self.assertEqual(route_policy('higher_order_saddle', 'irc_point'), 'fixed_geometry')
        self.assertEqual(route_policy('transition_state'), 'saddle')
        self.assertEqual(route_policy('transition_state_rattled'), 'unconstrained')

    def test_path_frames_are_not_counted_as_stationary_points(self):
        def frame(config_type, role):
            record = Record('r', 's', [Atom('Fe', 0, 0, 0), Atom('O', 1.6, 0, 0)], 0, 5,
                            config_type, metadata={'geometry_role': role} if role else {})
            return LabeledFrame(record, 0.0, [(0.5, 0.0, 0.0), (-0.5, 0.0, 0.0)], Path('x'))

        irc_as_hos = frame('higher_order_saddle', 'irc_point')
        self.assertEqual(classify_record(irc_as_hos.record)['pes_region'], 'irc')
        predictions = [(0.0, [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])]
        self.assertEqual(
            stationary_point_check([irc_as_hos], predictions)['n_frames_considered'], 0)
        verified = frame('minimum', 'stationary_minimum')
        self.assertEqual(
            stationary_point_check([verified], predictions)['n_frames_considered'], 1)


# ---------------------------------------------------------------------------
# Route rules
# ---------------------------------------------------------------------------

class RouteRuleTests(unittest.TestCase):
    def test_force_rewrite_reproduces_the_specified_routes(self):
        self.assertEqual(force_only_route(W2_ROUTE), FORCE_ROOT)
        self.assertEqual(force_only_route(W2_LATER), FORCE_LATER)
        self.assertEqual(force_only_route(FORCE_ROOT), FORCE_ROOT)

    def test_force_rewrite_removes_everything_that_moves_or_reinterprets(self):
        route = ('#p UBPW91/Gen NoSymm Opt=(TS,CalcFC,NoEigenTest) Freq=NoRaman Stable=Opt '
                 'Guess=(Read,Always) Geom=Checkpoint IOP(5/13=1)')
        fixed = force_only_route(route)
        self.assertEqual(forbidden_fixed_geometry_keywords(fixed), [])
        self.assertTrue(route_is_force_only(fixed))
        self.assertIn('Guess=Read', fixed)
        self.assertIn('Geom=Checkpoint', fixed)
        # A checkpoint stage always reads its predecessor's orbitals.
        self.assertIn('Guess=Read', force_only_route('#p UBPW91/Gen Opt Geom=Checkpoint'))
        # Stable=Opt is a wavefunction option, not a geometry optimization.
        self.assertFalse(route_optimizes('#p UBPW91/Gen Force Stable=Opt'))

    def test_audit_rules_for_path_points_and_higher_order_candidates(self):
        def verdict(config_type, route, **kwargs):
            text = f'%chk=a.chk\n{route}\n\ntitle\n\n0 11\nFe 0.0 0.0 0.0\n\n'
            return inspect_job(config_type, text, **kwargs)

        opt = verdict('irc_forward', W2_ROUTE)
        self.assertEqual(opt['findings'], ['path_point_launched_as_optimization'])
        self.assertTrue(opt['must_relaunch'])
        force = verdict('irc_forward', FORCE_ROOT)
        self.assertEqual((force['findings'], force['severity']), ([], 'ok'))
        freq = verdict('irc_forward', FORCE_ROOT + ' Freq')
        self.assertEqual(freq['findings'], ['path_point_route_has_extra_keywords'])
        self.assertFalse(freq['must_relaunch'])

        self.assertEqual(verdict('higher_order_saddle', W2_ROUTE)['findings'],
                         ['higher_order_launched_as_minimum_search'])
        self.assertEqual(verdict('higher_order_saddle', FORCE_ROOT)['findings'], [])
        search = W2_ROUTE.replace(' Opt ', ' Opt=(Saddle=2,CalcFC,NoEigenTest) Freq ')
        self.assertEqual(verdict('higher_order_saddle', search)['findings'],
                         ['higher_order_saddle_search_unrequested'])
        self.assertEqual(
            verdict('higher_order_saddle', search, requested_saddle_order=2)['findings'], [])
        self.assertEqual(
            verdict('higher_order_saddle', search, requested_saddle_order=3)['findings'],
            ['higher_order_saddle_search_unrequested'])
        ts_search = W2_ROUTE.replace(' Opt ', ' Opt=(TS,CalcFC,NoEigenTest) Freq ')
        self.assertIn('higher_order_saddle_search_unrequested',
                      verdict('higher_order_saddle', ts_search)['findings'])
        # A recorded role outranks the label.
        self.assertEqual(verdict('higher_order_saddle', W2_ROUTE,
                                 geometry_role='irc_point')['findings'],
                         ['path_point_launched_as_optimization'])

    def test_a_force_input_whose_log_optimized_is_still_an_error(self):
        text = f'%chk=a.chk\n{FORCE_ROOT}\n\nt\n\n0 11\nFe 0.0 0.0 0.0\n\n'
        log = ladder_log([W2_ROUTE], [11], optimized=True)
        self.assertIn('path_point_launched_as_optimization',
                      inspect_job('irc_forward', text, log)['findings'])


# ---------------------------------------------------------------------------
# 3-4. Generation: force-only ladders, checkpoint lineage, no forbidden keywords
# ---------------------------------------------------------------------------

def _record(config_type, record_id='rec1', multiplicity=11, **metadata):
    return Record(record_id=record_id, source='warehouse2/IRC/Fe2O2_path.log',
                  atoms=[Atom(*atom) for atom in FE2O2], charge=0,
                  multiplicity=multiplicity, config_type=config_type, metadata=metadata)


class GenerationTests(unittest.TestCase):
    def test_force_only_spin_ladder_keeps_the_checkpoint_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'spin'
            write_spin_jobs([_record('irc_forward')], output, 11, [7], route=W2_ROUTE)
            rows = _rows(output / 'spin_jobs.csv')
            text = (output / rows[0]['input']).read_text(encoding='utf-8')
            stages = _stages(text)
            self.assertEqual([s[0] for s in stages], [FORCE_ROOT, FORCE_LATER, FORCE_LATER])
            self.assertEqual(stages[0][1], '')
            self.assertTrue(stages[0][3])
            for previous, current in zip(stages, stages[1:]):
                self.assertEqual(current[1], previous[2])
                self.assertFalse(current[3])
            self.assertEqual([r['intended_multiplicity'] for r in rows], ['11', '9', '7'])
            for row in rows:
                self.assertEqual(row['geometry_role'], 'irc_point')
                self.assertEqual(row['route_policy'], 'fixed_geometry')
                self.assertEqual(row['route_search_kind'], 'none')
                self.assertEqual(row['original_multiplicity'], '11')
            # The generated campaign audits clean.
            audit = inspect_job(rows[0]['config_type'], text,
                                geometry_role=rows[0]['geometry_role'])
            self.assertEqual(audit['findings'], [])

    def test_no_generator_emits_opt_freq_stable_or_guess_always_for_path_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            texts = []
            spin = Path(tmp) / 'spin'
            freq_route = W2_ROUTE + ' Freq Stable=Opt'
            specification = {'record_id': 'rec1', 'name': 'afm', 'target_multiplicity': 1,
                             'fragments': [
                                 {'charge': 0, 'multiplicity': 5, 'atoms': [1, 3],
                                  'orientation': 'alpha'},
                                 {'charge': 0, 'multiplicity': 5, 'atoms': [2, 4],
                                  'orientation': 'beta'}]}
            write_spin_jobs([_record('higher_order_saddle', geometry_role='irc_point')], spin,
                            11, [9], route=freq_route, fragment_specifications=[specification],
                            strategy='both')
            texts += [path.read_text() for path in (spin / 'inputs').glob('*.gjf')]
            flat = Path(tmp) / 'flat'
            candidate = _record('higher_order_saddle', 'rec2')
            candidate.source = 'warehouse2/Fe2O2_hos_7.log'
            write_gaussian_jobs([_record('irc_reverse'), candidate], flat)
            texts += [path.read_text() for path in flat.glob('*.gjf')]
            self.assertEqual(len(texts), 4)
            for text in texts:
                for route in input_stage_routes(text):
                    self.assertEqual(forbidden_fixed_geometry_keywords(route), [], route)
                    self.assertTrue(route_is_force_only(route), route)
            fragment = next(t for t in texts if 'Fragment=' in t)
            self.assertIn('Guess=(Fragment=2)', fragment)
            flat_rows = {r['job_id']: r for r in _rows(flat / 'jobs.csv')}
            self.assertEqual(flat_rows['rec1']['route_intent'], 'fixed_geometry_force')
            self.assertEqual(flat_rows['rec2']['geometry_role'], 'higher_order_candidate')

    def test_a_higher_order_search_is_generated_only_on_request_with_its_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = _record('higher_order_saddle')
            record.source = 'warehouse2/Fe2O2_hos_7.log'
            record.imaginary_frequencies = 3
            output = Path(tmp) / 'spin'
            write_spin_jobs([record], output, 11, [9], route=W2_ROUTE,
                            higher_order_policy='saddle-search')
            rows = _rows(output / 'spin_jobs.csv')
            self.assertIn('Saddle=3', rows[0]['first_route'])
            self.assertEqual({r['requested_saddle_order'] for r in rows}, {'3'})
            record.imaginary_frequencies = None
            with self.assertRaises(ValueError):
                write_spin_jobs([record], Path(tmp) / 'unknown', 11, [9], route=W2_ROUTE,
                                higher_order_policy='saddle-search')


# ---------------------------------------------------------------------------
# 7, 8, 11, 12. In-place repair of an existing campaign
# ---------------------------------------------------------------------------

class InPlaceRepairTests(W2Campaign, unittest.TestCase):
    def repair(self, campaign, **kwargs):
        options = dict(start=6, path_point_policy='force', higher_order_policy='force')
        options.update(kwargs)
        return prepare_route_relaunch(campaign, **options)

    def test_dry_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self.standard(Path(tmp))
            before = _snapshot(Path(tmp))
            plan = self.repair(campaign, dry_run=True, assume_stopped=True)
            self.assertEqual(plan['input_count'], 3)
            self.assertEqual(_snapshot(Path(tmp)), before)
            # --plan-output is the one exception, and only where it is pointed.
            outside = Path(tmp) / 'plans' / 'plan.csv'
            self.repair(campaign, dry_run=True, plan_output=outside)
            after = _snapshot(Path(tmp))
            self.assertEqual(set(after) - set(before), {str(Path('plans') / 'plan.csv')})
            self.assertEqual({k: v for k, v in after.items() if k in before}, before)

    def test_repair_replaces_opt_with_force_in_the_existing_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self.standard(Path(tmp))
            slurm_plan = (campaign / 'slurm_plan.json').read_bytes()
            minimum = 'min__fe2o2__minimum__q0-m11__reference__ddd'
            untouched = {
                name: (campaign / 'slurm_batches/batch_0007' / name).read_bytes()
                for name in (f'{minimum}.gjf', f'{minimum}.log', f'{minimum}.rc')
            }
            result = self.repair(campaign)
            plan = {row['original_input'].split('/')[-1].split('__')[0]: row
                    for row in result['plan']}
            self.assertEqual(set(plan), {'irc-fwd', 'irc-rev', 'hos'})
            self.assertEqual(plan['irc-fwd']['geometry_role'], 'irc_point')
            self.assertEqual(plan['irc-fwd']['geometry_role_source'], 'filename_fallback')
            self.assertEqual(plan['hos']['geometry_role'], 'higher_order_candidate')
            for row in plan.values():
                self.assertEqual(row['repair'], 'fixed_geometry_force')
                self.assertEqual(row['route_before'], W2_ROUTE)
                self.assertEqual(row['route_after'], FORCE_ROOT)
                self.assertEqual(row['ladder_stages'], '3')
                self.assertEqual(row['input_stages'], '3')
                self.assertTrue(row['repair_reason'])
            # Completed invalid input: log and markers archived beside it.
            self.assertEqual(plan['irc-rev']['previous_state'], 'not_started')
            self.assertEqual(plan['irc-rev']['archived_output'], '')
            fwd = plan['irc-fwd']
            self.assertEqual(fwd['previous_state'], 'complete')
            archived = campaign / fwd['archived_output']
            self.assertEqual(archived.parent.name, 'batch_0006')
            self.assertTrue(archived.is_file())
            for suffix in ('.rc', '.started', '.finished'):
                self.assertTrue(archived.with_suffix(suffix).is_file(), suffix)

            rebuilt = (campaign / fwd['new_input']).read_text(encoding='utf-8')
            stages = _stages(rebuilt)
            self.assertEqual([s[0] for s in stages], [FORCE_ROOT, FORCE_LATER, FORCE_LATER])
            self.assertTrue(stages[0][3])
            for previous, current in zip(stages, stages[1:]):
                self.assertEqual(current[1], previous[2])
                self.assertIn('-routefix01', current[2])

            # Same batch folders, same order; the replacement took each slot.
            listing = (campaign / 'slurm_batches/batch_0006/inputs.txt').read_text().split()
            self.assertEqual([name.split('__')[0] for name in listing], ['irc-fwd', 'irc-rev'])
            self.assertTrue(all('__routefix01' in name for name in listing))
            listing7 = (campaign / 'slurm_batches/batch_0007/inputs.txt').read_text().split()
            self.assertEqual(listing7[1], f'{minimum}.gjf')
            self.assertEqual((campaign / 'slurm_plan.json').read_bytes(), slurm_plan)
            for name, content in untouched.items():
                self.assertEqual(
                    (campaign / 'slurm_batches/batch_0007' / name).read_bytes(), content)

            manifest = _rows(campaign / 'spin_jobs.csv')
            active = [r for r in manifest if r['submission_active'] == 'true']
            self.assertEqual(len(active), 9)
            for row in active:
                self.assertEqual(row['route_policy'], 'fixed_geometry' if row['geometry_role']
                                 == 'irc_point' else 'higher_order')
                self.assertEqual(row['route_search_kind'], 'none')
                self.assertIn('Force', row['first_route'])
            retired = [r for r in manifest if r['submission_active'] == 'false']
            self.assertEqual(len(retired), 9)
            self.assertTrue(all(r['route_invalidated'] for r in retired))
            self.assertEqual(
                [r['intended_multiplicity'] for r in active if 'irc-fwd' in r['input']],
                ['11', '9', '7'])

            # The campaign now audits clean for this range.
            audit = audit_campaign_routes(campaign)
            self.assertEqual(audit['summary']['must_relaunch'], 0)
            status = write_batch_progress(campaign, start=6, end=7, audit=True)
            self.assertEqual(status['summary']['route_relaunch_required'], 0)
            self.assertEqual(
                [i for i in status['issues'] if i['severity'] == 'error'], [])
            plan_csv = _rows(campaign / 'route_fix_plan.csv')
            self.assertEqual(len(plan_csv), 3)
            for column in ('original_input', 'new_input', 'geometry_role', 'route_before',
                           'route_after', 'archived_output', 'ladder_stages', 'repair_reason'):
                self.assertIn(column, plan_csv[0])

    def test_seed_routes_replace_the_filename_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self.standard(Path(tmp))
            seed = _record('higher_order_saddle', 'irc-fwd__fe2o2__higher-order-saddle__q0-'
                           'm11__reference__aaa')
            seed.route = '#p ubpw91/6-311+G* irc=(calcfc,forward)'
            write_extxyz([seed], Path(tmp) / 'seeds.extxyz')
            plan = self.repair(campaign, dry_run=True, seeds=Path(tmp) / 'seeds.extxyz')
            sources = {row['original_input'].split('/')[-1].split('__')[0]:
                       row['geometry_role_source'] for row in plan['plan']}
            self.assertEqual(sources['irc-fwd'], 'seed_record:irc_route')
            self.assertEqual(sources['irc-rev'], 'filename_fallback')

    def test_potentially_active_jobs_need_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self.build(Path(tmp), [
                {'name': 'irc-fwd__fe2o2__higher-order-saddle__q0-m11__reference__aaa',
                 'config_type': 'higher_order_saddle', 'batch': 'batch_0006',
                 'source': 'warehouse2/IRC/Fe2O2_irc_fwd_pt5.log', 'log': 'partial',
                 'running': True},
            ])
            before = _snapshot(Path(tmp))
            refused = self.repair(campaign, dry_run=True)
            self.assertEqual(refused['input_count'], 0)
            self.assertIn('activity_unconfirmed', refused['skipped'][0]['reason'])
            with self.assertRaises(RuntimeError):
                self.repair(campaign)
            self.assertEqual(
                {k: v for k, v in _snapshot(Path(tmp)).items() if k in before}, before)
            confirmed = self.repair(campaign, assume_stopped=True)
            self.assertEqual(confirmed['input_count'], 1)
            self.assertEqual(len(confirmed['overridden_active_attempts']), 1)

    def test_a_checkpoint_only_restart_is_rebuilt_from_its_coordinate_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self.build(Path(tmp), [
                {'name': 'irc-fwd__fe2o2__higher-order-saddle__q0-m11__reference__aaa',
                 'config_type': 'higher_order_saddle', 'batch': 'batch_0006',
                 'source': 'warehouse2/IRC/Fe2O2_irc_fwd_pt5.log', 'log': 'partial'},
            ])
            restart = prepare_spin_restarts(campaign, start=6, end=6, assume_stopped=True)
            continuation = campaign / restart['plan'][0]['new_input']
            self.assertIsNone(re.search(r'^Fe\s', continuation.read_text(), re.M))
            result = self.repair(campaign, assume_stopped=True)
            plan = result['plan'][0]
            self.assertEqual(plan['geometry_source'], 'checkpoint')
            self.assertEqual(plan['ladder_stages'], '3')
            rebuilt = (campaign / plan['new_input']).read_text()
            stages = _stages(rebuilt)
            self.assertEqual([s[0] for s in stages], [FORCE_ROOT, FORCE_LATER, FORCE_LATER])
            self.assertTrue(stages[0][3])
            seed = campaign / restart['plan'][0]['seed_checkpoint']
            self.assertNotIn(seed.name, rebuilt)
            manifest = _rows(campaign / 'spin_jobs.csv')
            active = [r for r in manifest if r['submission_active'] == 'true']
            self.assertEqual([r['intended_multiplicity'] for r in active], ['11', '9', '7'])
            self.assertEqual({r['input'] for r in active}, {plan['new_input']})
            listing = (campaign / 'slurm_batches/batch_0006/inputs.txt').read_text().split()
            self.assertEqual(listing, [Path(plan['new_input']).name])

    def test_cli_accepts_the_documented_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self.standard(Path(tmp))
            before = _snapshot(Path(tmp))
            code = main(['relaunch-routes', str(campaign), '--start', '6', '--end', '7',
                         '--path-point-policy', 'force', '--higher-order-policy', 'force',
                         '--assume-stopped', '--dry-run'])
            self.assertEqual(code, 0)
            self.assertEqual(_snapshot(Path(tmp)), before)


# ---------------------------------------------------------------------------
# 5, 6, 9. Collection
# ---------------------------------------------------------------------------

class CollectionTests(W2Campaign, unittest.TestCase):
    def _finish_relaunch(self, campaign, result, *, moved=False):
        """Write the logs the relaunched force-only ladders would produce."""
        for row in result['plan']:
            text = (campaign / row['new_input']).read_text()
            batch = campaign / 'slurm_batches' / row['batch']
            (batch / Path(row['new_output']).name).write_text(
                ladder_log(input_stage_routes(text), self.MULTS, moved=moved),
                encoding='utf-8')
            (batch / Path(row['new_output']).with_suffix('.rc').name).write_text('0')

    def _collect(self, campaign, destination, *extra):
        code = main(['collect', str(campaign), '-o', str(destination),
                     '--frames', 'converged', *extra])
        frames = []
        text = (destination / 'all.extxyz').read_text()
        for line in text.splitlines():
            if 'REF_energy=' in line:
                metadata = json.loads(json.loads(re.search(r'metadata=("(?:[^"\\]|\\.)*")',
                                                          line).group(1)))
                frames.append(metadata)
        failures = (destination / 'failed_outputs.tsv').read_text()
        return code, frames, failures

    def test_normally_terminated_force_stages_are_valid_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self.standard(Path(tmp))
            result = prepare_route_relaunch(campaign, start=6, end=7)
            self._finish_relaunch(campaign, result)
            code, frames, failures = self._collect(campaign, Path(tmp) / 'dataset')
            self.assertEqual(code, 0)
            force = [f for f in frames if f.get('force_label_kind') == 'fixed_geometry_force']
            # irc-fwd, irc-rev and hos: three stages each, all at the archived geometry.
            self.assertEqual(len(force), 9)
            self.assertEqual({f['fixed_geometry_check'] for f in force}, {'matched'})
            self.assertEqual({f['spin_stage_optimized'] for f in force}, {False})
            # 9. The archived plain-Opt results never come back, even when a
            # route mismatch is explicitly allowed.
            self.assertIn('invalidated by relaunch-routes', failures)
            _, forced, forced_failures = self._collect(
                campaign, Path(tmp) / 'forced', '--allow-route-mismatch')
            self.assertEqual(len(forced), len(frames))
            self.assertIn('invalidated by relaunch-routes', forced_failures)
            # 6. The untouched minimum still contributes only converged frames.
            optimized = [f for f in frames if f.get('force_label_kind') == 'optimization']
            self.assertEqual(len(optimized), 3)
            self.assertTrue(all(f['spin_stage_optimized'] for f in optimized))

    def test_a_force_stage_whose_atoms_moved_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self.standard(Path(tmp))
            result = prepare_route_relaunch(campaign, start=6, end=7)
            self._finish_relaunch(campaign, result, moved=True)
            _, frames, failures = self._collect(campaign, Path(tmp) / 'dataset')
            self.assertFalse(any(f.get('force_label_kind') == 'fixed_geometry_force'
                                 for f in frames))
            self.assertIn('differs from the archived input coordinates', failures)

    def test_optimization_frames_stay_converged_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self.build(Path(tmp), [
                {'name': 'min__fe2o2__minimum__q0-m11__reference__ddd',
                 'config_type': 'minimum', 'batch': 'batch_0006',
                 'source': 'warehouse2/Fe2O2_min_1.log'},
            ])
            name = 'min__fe2o2__minimum__q0-m11__reference__ddd'
            log = ladder_log(ladder_routes(3), self.MULTS, optimized=False)
            (campaign / 'slurm_batches/batch_0006' / f'{name}.log').write_text(log)
            _, frames, _ = self._collect(campaign, Path(tmp) / 'dataset')
            self.assertEqual(frames, [])

    def test_archived_attempts_keep_their_converged_stages_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = 'min__fe2o2__minimum__q0-m11__reference__ddd'
            campaign = self.build(Path(tmp), [
                {'name': name, 'config_type': 'minimum', 'batch': 'batch_0006',
                 'source': 'warehouse2/Fe2O2_min_1.log', 'log': 'partial',
                 'checkpoints': False},
            ])
            # No checkpoint survived, so the ladder is rerun from scratch in
            # place: the archived log's finished m11 stage is recomputed.
            prepare_spin_restarts(campaign, start=6, end=6, assume_stopped=True,
                                  rerun_missing_checkpoints=True)
            batch = campaign / 'slurm_batches/batch_0006'
            archived = next(batch.glob(f'{name}__before-rerun01.log'))
            self.assertTrue(archived.is_file())

            # Before the rerun finishes, the archived stage is recovered.
            _, frames, _ = self._collect(campaign, Path(tmp) / 'early')
            self.assertEqual([f['intended_multiplicity'] for f in frames], ['11'])
            self.assertEqual(frames[0]['submission_active'], 'false')

            (batch / f'{name}.log').write_text(
                ladder_log(ladder_routes(3), self.MULTS, optimized=True))
            (batch / f'{name}.rc').write_text('0')
            destination = Path(tmp) / 'late'
            _, frames, _ = self._collect(campaign, destination)
            self.assertEqual(sorted(f['intended_multiplicity'] for f in frames),
                             ['11', '7', '9'])
            self.assertTrue(all(f['submission_active'] == 'true' for f in frames))
            superseded = (destination / 'superseded_labels.tsv').read_text()
            self.assertIn(f'{name}-s00\t{name}-s00-rerun-r01', superseded)


# ---------------------------------------------------------------------------
# 10. campaign-status counts only the active attempt
# ---------------------------------------------------------------------------

class ActiveRowAuditTests(W2Campaign, unittest.TestCase):
    NAME = 'min__fe2o2__minimum__q0-m11__reference__ddd'

    def campaign(self, root):
        campaign = self.build(root, [
            {'name': self.NAME, 'config_type': 'minimum', 'batch': 'batch_0006',
             'source': 'warehouse2/Fe2O2_min_1.log', 'log': 'partial', 'checkpoints': False},
        ])
        prepare_spin_restarts(campaign, start=6, end=6, assume_stopped=True,
                              rerun_missing_checkpoints=True)
        return campaign

    def test_inactive_rows_are_not_stages_of_the_current_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self.campaign(Path(tmp))
            rows = _rows(campaign / 'spin_jobs.csv')
            self.assertEqual(len([r for r in rows if r['input'].endswith(self.NAME + '.gjf')]), 6)
            # Poison the archived rows' hash: it must not be compared.
            for row in rows:
                if row['submission_active'] == 'false':
                    row['input_sha256'] = '0' * 64
            with (campaign / 'spin_jobs.csv').open('w', newline='', encoding='utf-8') as h:
                writer = csv.DictWriter(h, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = write_batch_progress(campaign, start=6, end=6, audit=True)
            details = [issue['detail'] for issue in result['issues']]
            self.assertFalse(any('Link1 stages' in detail for detail in details), details)
            self.assertFalse(any('SHA256' in detail for detail in details), details)
            self.assertEqual(result['jobs'][0]['planned_stages'], 3)
            self.assertEqual(result['batches'][0]['planned_stages'], 3)

    def test_a_listed_input_with_only_inactive_rows_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self.campaign(Path(tmp))
            rows = [r for r in _rows(campaign / 'spin_jobs.csv')
                    if r['submission_active'] == 'false']
            with (campaign / 'spin_jobs.csv').open('w', newline='', encoding='utf-8') as h:
                writer = csv.DictWriter(h, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = write_batch_progress(campaign, start=6, end=6, audit=True)
            self.assertTrue(any('only inactive' in issue['detail'] for issue in result['issues']))


# ---------------------------------------------------------------------------
# Restarts: a Geom=Checkpoint stage never carries explicit coordinates
# ---------------------------------------------------------------------------

class StageZeroRestartTests(W2Campaign, unittest.TestCase):
    NAME = 'min__fe2o2__minimum__q0-m11__reference__ddd'

    def interrupted(self, root, routes, *, stop_after=0, finished_stage_optimized=True):
        campaign = self.build(root, [
            {'name': self.NAME, 'config_type': 'minimum', 'batch': 'batch_0006',
             'source': 'warehouse2/Fe2O2_min_1.log', 'routes': routes, 'checkpoints': False},
        ])
        batch = campaign / 'slurm_batches/batch_0006'
        (batch / f'{self.NAME}.log').write_text(ladder_log(
            routes, self.MULTS, optimized=finished_stage_optimized, stop_after=stop_after))
        for stage, mult in enumerate(self.MULTS[:stop_after + 1]):
            (batch / f'{self.NAME}-s{stage:02d}-m{mult}.chk').write_bytes(b'partial')
        restart = prepare_spin_restarts(campaign, start=6, end=6, assume_stopped=True)
        text = (campaign / restart['plan'][0]['new_input']).read_text(encoding='utf-8')
        return campaign, restart, text

    def assert_checkpoint_stages_are_bare(self, text, first_multiplicity):
        from cluster_mlip.routes import checkpoint_stages_with_coordinates

        self.assertEqual(checkpoint_stages_with_coordinates(text), [])
        stages = _stages(text)
        for route, oldchk, _, coordinates in stages:
            self.assertIn('Geom=Checkpoint', route)
            self.assertIn('Guess=Read', route)
            self.assertTrue(oldchk)
            self.assertFalse(coordinates)
        first = re.split(r'^\s*--Link1--\s*$', text, flags=re.M)[0]
        self.assertRegex(first, rf'(?m)^0 {first_multiplicity}\s*$')
        # The last molecule specification is terminated by a blank line too.
        self.assertTrue(text.endswith('\n\n'), repr(text[-20:]))
        return stages

    def test_optimization_ladder_interrupted_in_stage_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, restart, text = self.interrupted(Path(tmp), ladder_routes(3))
            self.assertEqual(restart['plan'][0]['restart_multiplicity'], '11')
            stages = self.assert_checkpoint_stages_are_bare(text, 11)
            self.assertEqual(len(stages), 3)
            self.assertTrue(all(route_optimizes(route) for route, *_ in stages))
            self.assertIn('-seed-r01', stages[0][1])
            for previous, current in zip(stages, stages[1:]):
                self.assertEqual(current[1], previous[2])

    def test_force_only_ladder_interrupted_in_stage_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            routes = [FORCE_ROOT, FORCE_LATER, FORCE_LATER]
            _, restart, text = self.interrupted(Path(tmp), routes)
            stages = self.assert_checkpoint_stages_are_bare(text, 11)
            for route, *_ in stages:
                self.assertTrue(route_is_force_only(route), route)

    def test_a_force_only_stage_counts_as_finished_without_an_optimization_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            routes = [FORCE_ROOT, FORCE_LATER, FORCE_LATER]
            _, restart, text = self.interrupted(
                Path(tmp), routes, stop_after=1, finished_stage_optimized=False)
            # Stage 0 terminated normally and is not redone.
            self.assertEqual(restart['plan'][0]['restart_multiplicity'], '9')
            self.assertEqual(restart['plan'][0]['completed_stages_in_attempt'], '1')
            self.assertEqual(len(self.assert_checkpoint_stages_are_bare(text, 9)), 2)

    def test_campaign_status_flags_a_checkpoint_stage_with_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = self.build(Path(tmp), [
                {'name': self.NAME, 'config_type': 'minimum', 'batch': 'batch_0006',
                 'source': 'warehouse2/Fe2O2_min_1.log'},
            ])
            listed = campaign / 'slurm_batches/batch_0006' / f'{self.NAME}.gjf'
            text = listed.read_text()
            listed.write_text(text.replace(' Int=UltraFine Pop=Regular\n',
                                           ' Int=UltraFine Geom=Checkpoint Guess=Read Pop=Regular\n', 1))
            result = write_batch_progress(campaign, start=6, end=6, audit=True)
            self.assertTrue(any('stage 0 reads Geom=Checkpoint' in issue['detail']
                                for issue in result['issues']))

    def test_coordinate_removal_keeps_a_following_basis_block(self):
        from cluster_mlip.routes import drop_molecule_coordinates

        body = '\ntitle\n\n0 11\nFe 0.0 0.0 0.0\nO 1.6 0.0 0.0\n\nFe 0\nS 1 1.0\n 1.0 1.0\n****\n\n'
        self.assertEqual(drop_molecule_coordinates(body),
                         '\ntitle\n\n0 11\n\nFe 0\nS 1 1.0\n 1.0 1.0\n****\n\n')


# ---------------------------------------------------------------------------
# Adversarial-review regressions (one test per finding)
# ---------------------------------------------------------------------------

def _rewrite_manifest(campaign, update):
    rows = _rows(campaign / 'spin_jobs.csv')
    for row in rows:
        update(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (campaign / 'spin_jobs.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ReviewFindingTests(W2Campaign, unittest.TestCase):
    SADDLE2 = W2_ROUTE.replace(' Opt ', ' Opt=(Saddle=2,CalcFC,NoEigenTest) Freq ')

    def test_f2_a_recorded_saddle_order_is_honoured_unless_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = 'hos__fe2o2__higher-order-saddle__q0-m11__reference__ccc'
            later = self.SADDLE2.replace(' Pop=Regular', ' Geom=Checkpoint Guess=Read Pop=Regular')
            campaign = self.build(Path(tmp), [
                {'name': name, 'config_type': 'higher_order_saddle', 'batch': 'batch_0006',
                 'source': 'warehouse2/Fe2O2_hos_7.log', 'routes': [self.SADDLE2, later, later]},
            ])
            _rewrite_manifest(campaign, lambda row: row.update(requested_saddle_order='2'))
            batch = campaign / 'slurm_batches/batch_0006'
            (batch / f'{name}.log').write_text(
                f' {self.SADDLE2}\n Charge =  0 Multiplicity = 11\n'
                ' Tors failed for dihedral     1 -     2 -     3 -     4\n FormBX had a problem.\n'
                ' Error termination via Lnk1e\n')
            plan = prepare_route_relaunch(campaign, dry_run=True)['plan'][0]
            self.assertEqual(plan['repair'], 'higher_order_saddle_search')
            self.assertEqual((plan['saddle_order'], plan['saddle_order_source']),
                             ('2', 'recorded_request'))
            self.assertIn('Saddle=2', plan['route_after'])
            self.assertIn('Cartesian', plan['route_after'])
            forced = prepare_route_relaunch(
                campaign, dry_run=True, higher_order_policy='force')['plan'][0]
            self.assertEqual(forced['repair'], 'fixed_geometry_force')
            self.assertIn('overrides the recorded requested_saddle_order=2',
                          forced['repair_reason'])

    def test_f3_a_running_saddle_search_is_not_reflagged_by_an_irc_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            ts = W2_ROUTE.replace(' Opt ', ' Opt=(TS,CalcFC,NoEigenTest) Freq ')
            later = ts.replace(' Pop=Regular', ' Geom=Checkpoint Guess=Read Pop=Regular')
            campaign = self.build(Path(tmp), [
                {'name': 'ts__fe2o2__first-order-saddle__q0-m11__reference__fff',
                 'config_type': 'first_order_saddle', 'batch': 'batch_0006',
                 'source': 'warehouse2/IRC/Fe2O2_irc_ts_4.log', 'routes': [ts, later, later]},
            ])
            audit = audit_campaign_routes(campaign, write_reports=False)
            row = audit['rows'][0]
            self.assertEqual(row['geometry_role'], 'transition_state')
            self.assertEqual(row['geometry_role_source'], 'running_saddle_search')
            self.assertEqual(row['must_relaunch'], 'false')
            self.assertEqual(audit['summary']['irc_named_saddle_searches_kept'], 1)

    def test_f4_f8_names_agree_between_extraction_and_legacy_rows(self):
        atoms = [('Fe', 0.0, 0.0, 0.0), ('N', 1.7, 0.0, 0.0), ('O', 2.8, 0.0, 0.0)]
        body = (' #p ubpw91/6-311+G* opt freq\n Charge = 0 Multiplicity = 2\n'
                + _orientation(atoms) + ' SCF Done:  E(UBPW91) =  -1412.0     A.U.\n'
                ' Harmonic frequencies (cm**-1)\n Frequencies --  -512.0  230.0  80.0\n')
        for source in ('w2/IRC/Fe2O2_ts.log', 'w2/Fe2O2_ts_for_irc.log'):
            record = extract_document_records(body, source)[0]
            self.assertEqual(record.config_type, 'transition_state', source)
            self.assertEqual(
                resolve_geometry_role('transition_state', source=source)['geometry_role'],
                record.metadata['geometry_role'], source)
        named = extract_document_records(
            body.replace('-512.0  230.0', '-512.0  -230.0'), 'W2/IRC_runs/FeNO_irc_fwd_5.log')[0]
        self.assertEqual((named.config_type, named.metadata['irc_direction']),
                         ('irc_forward', 'forward'))

    def test_f5_a_ladder_rerun_in_place_can_still_be_relaunched(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = 'irc-fwd__fe2o2__higher-order-saddle__q0-m11__reference__aaa'
            campaign = self.build(Path(tmp), [
                {'name': name, 'config_type': 'higher_order_saddle', 'batch': 'batch_0006',
                 'source': 'warehouse2/IRC/Fe2O2_irc_fwd_pt5.log', 'log': 'partial',
                 'checkpoints': False},
            ])
            prepare_spin_restarts(campaign, start=6, end=6, assume_stopped=True,
                                  rerun_missing_checkpoints=True)
            result = prepare_route_relaunch(campaign, assume_stopped=True)
            self.assertEqual(result['plan'][0]['ladder_stages'], '3')
            manifest = _rows(campaign / 'spin_jobs.csv')
            archived = [r for r in manifest if r['job_id'].endswith(('-s00', '-s01', '-s02'))]
            self.assertTrue(all(r['output'].endswith('__before-rerun01.log') for r in archived))
            rerun = [r for r in manifest if '-rerun-r01' in r['job_id']
                     and '-rf01' not in r['job_id']]
            # The rerun never ran, so it has no log to archive; it is retired as is.
            self.assertEqual({r['output'] for r in rerun}, {f'{name}.log'})
            self.assertTrue(all(r['route_invalidated'] for r in archived + rerun))
            active = [r for r in manifest if r['submission_active'] == 'true']
            self.assertEqual([r['intended_multiplicity'] for r in active], ['11', '9', '7'])

    def test_f9_a_force_label_that_cannot_be_checked_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            routes = [FORCE_ROOT, FORCE_LATER, FORCE_LATER]
            name = 'irc-fwd__fe2o2__irc-forward__q0-m11__reference__ggg'
            campaign = self.build(Path(tmp), [
                {'name': name, 'config_type': 'irc_forward', 'batch': 'batch_0006',
                 'source': 'warehouse2/IRC/Fe2O2_irc_fwd_pt5.log', 'routes': routes,
                 'checkpoints': False},
            ])
            batch = campaign / 'slurm_batches/batch_0006'
            (batch / f'{name}.log').write_text(ladder_log(routes, self.MULTS, stop_after=1))
            (batch / f'{name}-s00-m11.chk').write_bytes(b'done')
            restart = prepare_spin_restarts(campaign, start=6, end=6, assume_stopped=True)
            new_output = campaign / restart['plan'][0]['new_output']
            text = (campaign / restart['plan'][0]['new_input']).read_text()
            new_output.write_text(ladder_log(input_stage_routes(text), self.MULTS[1:]))
            (campaign / 'inputs' / f'{name}.gjf').unlink()     # the coordinate root is gone
            code = main(['collect', str(campaign), '-o', str(Path(tmp) / 'd'),
                         '--frames', 'converged'])
            failures = (Path(tmp) / 'd' / 'failed_outputs.tsv').read_text()
            self.assertIn('no archived coordinates to check against', failures)

    def test_f10_a_refused_saddle_search_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = _record('minimum', 'good')
            bad = _record('higher_order_saddle', 'bad')
            bad.source = 'warehouse2/Fe2O2_hos_7.log'
            output = Path(tmp) / 'flat'
            with self.assertRaises(ValueError):
                write_gaussian_jobs([good, bad], output, higher_order_policy='saddle-search')
            self.assertFalse(output.exists() and any(output.iterdir()))


class ReviewRouteParsingTests(unittest.TestCase):
    def test_f6_f7_separators_and_parenthesised_options(self):
        self.assertTrue(route_optimizes('#p B3LYP/6-31G*,Opt,Freq'))
        for route in ('#p B3LYP/6-31G*,Opt,Freq', '#p B3LYP/6-31G* Opt Guess(Always)',
                      '#p B3LYP/6-31G* Opt,Freq=NoRaman', '#p B3LYP/6-31G* Opt = (TS, CalcFC)'):
            fixed = force_only_route(route)
            self.assertEqual(forbidden_fixed_geometry_keywords(fixed), [], fixed)
            self.assertTrue(route_is_force_only(fixed), fixed)
        self.assertIn('Guess=Always',
                      forbidden_fixed_geometry_keywords('#p B3LYP/6-31G* Force Guess(Always)'))
        records = [r for r in extract_document_records(
            _irc_log(route=' #p ubpw91/6-311+G* irc(reverse,calcfc)'), 'a.log') if r.irc_point]
        self.assertEqual({r.metadata['irc_direction'] for r in records}, {'reverse'})


if __name__ == '__main__':
    unittest.main()
