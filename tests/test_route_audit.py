"""Transition states must be launched as saddle searches, not ordinary Opt."""
import csv
import hashlib
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.batch_progress import write_batch_progress
from cluster_mlip.cli import build_parser, main
from cluster_mlip.jobs import (
    DEFAULT_RATTLE_ROUTE,
    DEFAULT_ROUTE,
    DEFAULT_SADDLE_ROUTE,
    expanded_records,
    write_gaussian_jobs,
)
from cluster_mlip.io import write_extxyz
from cluster_mlip.models import Atom, Record
from cluster_mlip.relaunch import prepare_route_relaunch
from cluster_mlip.restart import prepare_spin_restarts
from cluster_mlip.route_audit import audit_campaign_routes
from cluster_mlip.routes import (
    config_type_from_stem,
    corrected_cartesian_route,
    geometry_source,
    input_is_zmatrix,
    route_uses_cartesian,
    corrected_minimum_route,
    corrected_saddle_route,
    inspect_job,
    intended_stationary_point,
    opt_options,
    route_search_kind,
    saddle_route_gaps,
)
from cluster_mlip.slurm import SlurmConfig, prepare_slurm_batches
from cluster_mlip.spin import (
    DEFAULT_SPIN_ROUTE,
    SPIN_MANIFEST_COLUMNS,
    write_spin_jobs,
)

ATOMS = [Atom('Fe', 0.0, 0.0, 0.0), Atom('N', 1.7, 0.0, 0.0), Atom('O', 2.85, 0.0, 0.0)]

# What a transition state relaxed by a plain Opt actually leaves behind: both
# Link1 stages terminate normally, the final force frame parses cleanly, and
# the only frequency analysis has no imaginary mode. Nothing but the route and
# that mode count reveals the geometry is a minimum, not the labeled saddle.
_GEOMETRY = """ Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic     Atomic              Coordinates (Angstroms)
 Number     Number      Type              X           Y           Z
 ---------------------------------------------------------------------
    1         26             0        0.000000    0.000000    0.000000
    2          7             0        1.700000    0.000000    0.000000
    3          8             0        2.850000    0.000000    0.000000
 ---------------------------------------------------------------------
"""

_FORCES = """ SCF Done:  E(UBPW91) =  -1400.1000000000     A.U.
 Forces (Hartrees/Bohr)
 -------------------------------------------------------------------
 Center     Atomic                   Forces (Hartrees/Bohr)
 Number     Number              X              Y              Z
 -------------------------------------------------------------------
      1       26           0.000100000    0.000000000    0.000000000
      2        7          -0.000100000    0.000000000    0.000000000
      3        8           0.000000000    0.000000000    0.000000000
 -------------------------------------------------------------------
"""

_OPT_STAGE = (
    " #p UBPW91/Gen SCF=(VShift=5) NoSymm Opt Freq IOP(5/13=1) Int=UltraFine\n"
    " Charge = 0 Multiplicity = 4\n"
    + _GEOMETRY
    + _FORCES
    + " Harmonic frequencies (cm**-1), IR intensities (KM/Mole)\n"
    " Frequencies --    120.0000   340.0000   890.0000\n"
    " Normal termination of Gaussian 09\n"
)

_FORCE_STAGE = (
    " Link1:  Proceeding to internal job step number  2.\n"
    " #p UBPW91/Gen Force SCF=(VShift=5) NoSymm Guess=Read Geom=Checkpoint\n"
    " Charge = 0 Multiplicity = 4\n"
    + _GEOMETRY
    + _FORCES
    + " Normal termination of Gaussian 09\n"
)

MINIMUM_LOG = _OPT_STAGE + _FORCE_STAGE

# The same job, run correctly: a TS search that converged on one imaginary mode.
SADDLE_LOG = MINIMUM_LOG.replace(
    "NoSymm Opt Freq", "NoSymm Opt=(TS,CalcFC,NoEigenTest) Freq"
).replace("Frequencies --    120.0000", "Frequencies --   -420.0000")


def _record(config_type, record_id='rec1'):
    return Record(
        record_id=record_id,
        source='warehouse/FeNO_4_12345.txt',
        atoms=list(ATOMS),
        charge=0,
        multiplicity=4,
        config_type=config_type,
    )


class RouteParsingTests(unittest.TestCase):
    def test_bare_opt_does_not_absorb_the_following_keyword(self):
        # "Opt Freq" is a minimization plus a frequency job, not Opt=Freq;
        # reading it as an option would make every plain Opt look configured.
        route = '#p UBPW91/Gen NoSymm Opt Freq IOP(5/13=1) Int=UltraFine'
        self.assertEqual(opt_options(route), set())
        self.assertEqual(route_search_kind(route), 'minimum')

    def test_saddle_searches_are_recognized_in_every_spelling(self):
        for route, gaps in [
            ('#p B3LYP/Gen Opt=(TS,CalcFC,NoEigenTest) Freq', []),
            ('#p B3LYP/Gen Opt=(TS,CalcFC,NoEigen) Freq', []),
            ('#p B3LYP/Gen Opt(TS,CalcAll) Freq', ['noeigentest']),
            ('#p B3LYP/Gen Opt=TS', ['noeigentest', 'hessian', 'freq']),
            ('#p B3LYP/Gen Opt=QST3 Freq', ['noeigentest', 'hessian']),
        ]:
            with self.subTest(route=route):
                self.assertEqual(route_search_kind(route), 'saddle')
                self.assertEqual(saddle_route_gaps(route), gaps)

    def test_non_optimizing_routes_have_no_search(self):
        for route in ('#p UBPW91/Gen SP NoSymm', '#p UBPW91/Gen Force Guess=Read Geom=Checkpoint'):
            self.assertEqual(route_search_kind(route), 'none')

    def test_correction_adds_only_what_is_missing_and_is_idempotent(self):
        corrected = corrected_saddle_route(DEFAULT_ROUTE)
        self.assertEqual(route_search_kind(corrected), 'saddle')
        self.assertEqual(saddle_route_gaps(corrected), [])
        self.assertEqual(corrected_saddle_route(corrected), corrected)
        # Unrelated protocol settings survive the rewrite untouched.
        for fragment in ('UBPW91/Gen', 'SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc)',
                         'NoSymm', 'IOP(5/13=1,5/36=1,8/11=1)', 'Int=UltraFine', 'Freq'):
            self.assertIn(fragment, corrected)

    def test_correction_preserves_existing_opt_options(self):
        corrected = corrected_saddle_route('#p B3LYP/Gen Opt=(ModRedundant,MaxCycles=100) Freq')
        self.assertIn('ModRedundant', corrected)
        self.assertIn('MaxCycles=100', corrected)
        self.assertEqual(route_search_kind(corrected), 'saddle')

    def test_non_optimizing_stage_is_never_turned_into_a_search(self):
        force = '#p UBPW91/Gen Force NoSymm Guess=Read Geom=Checkpoint'
        self.assertEqual(corrected_saddle_route(force), force)

    def test_minimum_correction_strips_saddle_keywords(self):
        corrected = corrected_minimum_route('#p B3LYP/Gen Opt=(TS,CalcFC,NoEigenTest) Freq')
        self.assertEqual(route_search_kind(corrected), 'minimum')
        self.assertIn('CalcFC', corrected)

    def test_intent_comes_from_the_label_and_ignores_rattled_variants(self):
        self.assertEqual(intended_stationary_point('transition_state'), 'saddle')
        self.assertEqual(intended_stationary_point('first_order_saddle'), 'saddle')
        self.assertEqual(intended_stationary_point('higher_order_saddle'), 'saddle')
        self.assertEqual(intended_stationary_point('minimum'), 'minimum')
        # Displaced single points carry no stationary-point claim, and neither
        # do types whose curvature was never established.
        self.assertEqual(intended_stationary_point('transition_state_rattled'), 'unconstrained')
        for config_type in ('warehouse_structure', 'optimized_unverified', 'irc_forward'):
            self.assertEqual(intended_stationary_point(config_type), 'unconstrained')

    def test_config_type_recovered_from_generated_filename(self):
        stem = 'wh__FeNO__transition-state__q0-m4__reference__abcdef1234'
        self.assertEqual(config_type_from_stem(stem + '.gjf'), 'transition_state')
        self.assertEqual(
            config_type_from_stem('wh__FeNO__first-order-saddle__q0-m4__r01__abcdef1234'),
            'first_order_saddle_rattled',
        )

    def test_inspect_job_flags_a_transition_state_run_as_plain_opt(self):
        verdict = inspect_job(
            'transition_state',
            f'%chk=a.chk\n{DEFAULT_ROUTE}\n\ntitle\n\n0 4\nFe 0. 0. 0.\n',
            MINIMUM_LOG,
        )
        self.assertIn('saddle_launched_as_minimum_search', verdict['findings'])
        self.assertIn('saddle_converged_to_minimum', verdict['findings'])
        self.assertEqual(verdict['final_imaginary_modes'], 0)
        self.assertEqual(verdict['severity'], 'error')
        self.assertTrue(verdict['must_relaunch'])

    def test_inspect_job_accepts_a_correct_saddle_search(self):
        verdict = inspect_job(
            'transition_state',
            f'%chk=a.chk\n{DEFAULT_SADDLE_ROUTE}\n\ntitle\n\n0 4\nFe 0. 0. 0.\n',
        )
        self.assertEqual(verdict['findings'], [])
        self.assertEqual(verdict['severity'], 'ok')
        self.assertFalse(verdict['must_relaunch'])

    def test_inspect_job_leaves_unconstrained_labels_alone(self):
        for config_type in ('warehouse_structure', 'minimum'):
            verdict = inspect_job(config_type, f'#route\n{DEFAULT_ROUTE}\n')
            self.assertEqual(verdict['findings'], [], config_type)

    def test_inspect_job_flags_a_minimum_run_as_a_saddle_search(self):
        verdict = inspect_job('minimum', f'%chk=a.chk\n{DEFAULT_SADDLE_ROUTE}\n')
        self.assertEqual(verdict['findings'], ['minimum_launched_as_saddle_search'])
        self.assertTrue(verdict['must_relaunch'])


class PreparationTests(unittest.TestCase):
    def test_prepare_gives_saddle_seeds_a_saddle_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'campaign'
            records = expanded_records(
                [_record('transition_state', 'ts1'), _record('minimum', 'min1')],
                rattles_per_seed=1, sigma=0.05, seed=7,
            )
            write_gaussian_jobs(records, output)
            with (output / 'jobs.csv').open(newline='') as handle:
                rows = {(r['config_type'], r['variant']): r for r in csv.DictReader(handle)}
            saddle = rows[('transition_state', 'reference')]
            self.assertEqual(saddle['route_intent'], 'saddle')
            self.assertEqual(saddle['route_search_kind'], 'saddle')
            self.assertEqual(saddle['first_route'], DEFAULT_SADDLE_ROUTE)
            self.assertEqual(route_search_kind((output / saddle['input']).read_text().split('\n')[3]),
                             'saddle')
            minimum = rows[('minimum', 'reference')]
            self.assertEqual(minimum['route_intent'], 'minimum')
            self.assertEqual(minimum['first_route'], DEFAULT_ROUTE)
            # A rattled transition state keeps the single-point route: its
            # value is the displaced geometry, which no search may move.
            rattled = rows[('transition_state_rattled', 'r01')]
            self.assertEqual(rattled['route_intent'], 'displaced_single_point')
            self.assertEqual(rattled['first_route'], DEFAULT_RATTLE_ROUTE)

    def test_prepare_refuses_a_minimizing_saddle_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as caught:
                write_gaussian_jobs(
                    [_record('first_order_saddle')], Path(tmp) / 'c', saddle_route=DEFAULT_ROUTE
                )
            self.assertIn('labeled saddle points', str(caught.exception))

    def test_prepare_spins_refuses_saddle_seeds_with_a_minimizing_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = _record('transition_state')
            record.multiplicity = 10
            with self.assertRaises(ValueError) as caught:
                write_spin_jobs([record], Path(tmp) / 'spin', 10, [8])
            self.assertIn('labeled saddle points', str(caught.exception))

    def test_prepare_spins_records_the_label_and_search_it_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = _record('minimum')
            record.multiplicity = 10
            output = Path(tmp) / 'spin'
            write_spin_jobs([record], output, 10, [8], route=DEFAULT_SPIN_ROUTE)
            with (output / 'spin_jobs.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual(row['config_type'], 'minimum')
                self.assertEqual(row['route_search_kind'], 'minimum')


class BrokenCampaign:
    """Build the campaign state that actually exists on disk today.

    ``write_gaussian_jobs`` now refuses to launch a labeled saddle with a
    minimizing route, so the historical mistake has to be reconstructed the
    way it was made: generate the campaign, then rewrite the first-stage route
    to the plain ``Opt`` the old generator used and drop the route-intent
    columns it did not record. That is exactly what the audit must recognize.
    """

    def campaign(self, root, config_type='transition_state', break_route=True,
                 finished=True, manifest_config_type=True):
        records = expanded_records([_record(config_type)], 0, 0.05, 7)
        write_gaussian_jobs(records, root)
        with (root / 'jobs.csv').open(newline='') as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        stem = Path(rows[0]['input']).stem
        job_input = root / rows[0]['input']
        if break_route:
            job_input.write_text(
                job_input.read_text().replace(DEFAULT_SADDLE_ROUTE, DEFAULT_ROUTE),
                encoding='utf-8',
            )
            for row in rows:
                row['first_route'] = DEFAULT_ROUTE
            fields = [
                f for f in fields
                if f not in ('route_intent', 'route_search_kind', 'intended_stationary_point')
            ]
        if not manifest_config_type:
            # Older campaigns have no config_type column; the label then has to
            # come from the generated filename or the mistake stays invisible.
            fields = [f for f in fields if f != 'config_type']
        with (root / 'jobs.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        prepare_slurm_batches(root, SlurmConfig(jobs_per_batch=4, cpus_per_job=16))
        batch = root / 'slurm_batches/batch_0001'
        if finished:
            # The log has to match the route the input actually carried.
            (batch / f'{stem}.log').write_text(MINIMUM_LOG if break_route else SADDLE_LOG)
            (batch / f'{stem}.rc').write_text('0')
        return root, batch, stem


class CampaignAuditTests(BrokenCampaign, unittest.TestCase):
    def test_audit_finds_a_finished_transition_state_run_as_plain_opt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, batch, stem = self.campaign(Path(tmp))
            result = audit_campaign_routes(root)
            summary = result['summary']
            self.assertEqual(summary['inputs_audited'], 1)
            self.assertEqual(summary['must_relaunch'], 1)
            self.assertEqual(summary['completed_but_invalid'], 1)
            row = result['rows'][0]
            self.assertEqual(row['intent'], 'saddle')
            self.assertEqual(row['search_kinds'], 'minimum')
            self.assertEqual(row['state'], 'complete')
            self.assertEqual(row['final_imaginary_modes'], 0)
            self.assertEqual(row['expected_imaginary_modes'], '1')
            self.assertIn('saddle_launched_as_minimum_search', row['findings'])
            self.assertEqual(row['config_type_source'], 'manifest')
            report = (result['destination'] / 'route_audit.md').read_text()
            self.assertIn('saddle_launched_as_minimum_search', report)
            self.assertTrue((result['destination'] / 'route_relaunch_candidates.csv').is_file())

    def test_audit_falls_back_to_the_filename_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = self.campaign(Path(tmp), manifest_config_type=False)
            result = audit_campaign_routes(root)
            row = result['rows'][0]
            self.assertEqual(row['config_type'], 'transition_state')
            self.assertEqual(row['config_type_source'], 'filename')
            self.assertTrue(row['must_relaunch'] == 'true')

    def test_audit_is_clean_for_a_correct_saddle_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, batch, stem = self.campaign(Path(tmp), break_route=False)
            result = audit_campaign_routes(root)
            self.assertEqual(result['summary']['must_relaunch'], 0)
            self.assertEqual(result['rows'][0]['findings'], '')
            self.assertEqual(result['rows'][0]['final_imaginary_modes'], 1)

    def test_audit_does_not_flag_unlabeled_geometries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = self.campaign(Path(tmp), config_type='warehouse_structure')
            result = audit_campaign_routes(root)
            self.assertEqual(result['summary']['must_relaunch'], 0)

    def test_cli_audit_routes_exits_two_when_work_is_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = self.campaign(Path(tmp))
            self.assertEqual(main(['audit-routes', str(root)]), 2)
            args = build_parser().parse_args(['audit-routes', 'c', '--include-inactive'])
            self.assertTrue(args.include_inactive)


class BatchAuditTests(BrokenCampaign, unittest.TestCase):
    def test_batch_audit_reports_the_wrong_search_on_a_clean_run(self):
        # The job is complete, its return code is 0, and its final force frame
        # parses -- every pre-existing audit check passes. Only the route
        # comparison catches it.
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = self.campaign(Path(tmp))
            result = write_batch_progress(root, audit=True)
            self.assertEqual(result['jobs'][0]['state'], 'complete')
            self.assertEqual(result['jobs'][0]['route_intent'], 'saddle')
            self.assertEqual(result['jobs'][0]['route_search_kind'], 'minimum')
            self.assertEqual(result['jobs'][0]['must_relaunch'], 'true')
            self.assertEqual(result['summary']['route_relaunch_required'], 1)
            details = '\n'.join(item['detail'] for item in result['issues'])
            self.assertIn('saddle_launched_as_minimum_search', details)
            self.assertIn('relaunch required', details)
            self.assertTrue(result['summary']['errors'])

    def test_batch_audit_stays_quiet_for_a_correct_saddle_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = self.campaign(Path(tmp), break_route=False)
            result = write_batch_progress(root, audit=True)
            self.assertEqual(result['summary']['route_relaunch_required'], 0)
            self.assertNotIn('saddle', '\n'.join(i['detail'] for i in result['issues']))


class RelaunchTests(BrokenCampaign, unittest.TestCase):
    def test_dry_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, batch, stem = self.campaign(Path(tmp))
            before = {
                path.relative_to(root): path.read_bytes()
                for path in sorted(root.rglob('*')) if path.is_file()
            }
            plan = prepare_route_relaunch(root, dry_run=True)
            self.assertEqual(plan['input_count'], 1)
            self.assertEqual(route_search_kind(plan['plan'][0]['route_after']), 'saddle')
            after = {
                path.relative_to(root): path.read_bytes()
                for path in sorted(root.rglob('*')) if path.is_file()
            }
            # The audit reports are the one thing a dry run may write.
            changed = {
                name for name in set(before) | set(after)
                if before.get(name) != after.get(name)
            }
            self.assertTrue(all(name.parts[0] == 'monitoring' for name in changed), changed)

    def test_relaunch_rebuilds_from_the_original_geometry_and_archives_the_bad_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, batch, stem = self.campaign(Path(tmp))
            original_input = (root / f'{stem}.gjf').read_text()
            result = prepare_route_relaunch(root)
            self.assertEqual(result['input_count'], 1)
            new_input = root / result['plan'][0]['new_input']
            text = new_input.read_text()

            # The corrected search, on the untouched original guess geometry.
            routes = [line for line in text.splitlines() if line.startswith('#')]
            self.assertEqual(route_search_kind(routes[0]), 'saddle')
            self.assertEqual(saddle_route_gaps(routes[0]), [])
            self.assertEqual(route_search_kind(routes[1]), 'none')  # Link1 Force stage
            coordinates = [line for line in text.splitlines() if line.startswith('Fe ')]
            self.assertEqual(coordinates,
                             [line for line in original_input.splitlines() if line.startswith('Fe ')])

            # Fresh checkpoints: nothing may read or overwrite the ones the
            # collapsed minimum wrote.
            self.assertIn('-routefix01.chk', text)
            self.assertNotIn(f'%chk={stem}.chk', text)

            # The invalid log is archived, not deleted, and the batch listing
            # now runs the replacement instead of the original.
            self.assertFalse((batch / f'{stem}.log').exists())
            archived = batch / f'{stem}__before-routefix01.log'
            self.assertTrue(archived.is_file())
            self.assertEqual(archived.read_text(), MINIMUM_LOG)
            self.assertTrue((batch / f'{stem}__before-routefix01.rc').is_file())
            listing = (batch / 'inputs.txt').read_text().split()
            self.assertEqual(listing, [new_input.name])
            self.assertTrue((batch / new_input.name).exists())
            self.assertTrue((root / 'route_fix_plan.csv').is_file())
            self.assertTrue(any((root / 'route_fix_backups').iterdir()))

            # The manifest retires the old row and activates the new one.
            with (root / 'jobs.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            old = [r for r in rows if r['submission_active'] == 'false']
            new = [r for r in rows if r['submission_active'] == 'true']
            self.assertEqual(len(old), 1)
            self.assertEqual(len(new), 1)
            self.assertIn('saddle_launched_as_minimum_search', old[0]['route_invalidated'])
            self.assertEqual(old[0]['output'], archived.name)
            self.assertEqual(new[0]['superseded_by_job_id'], '')
            self.assertEqual(new[0]['relaunch_of_job_id'], old[0]['job_id'])
            self.assertEqual(old[0]['superseded_by_job_id'], new[0]['job_id'])

            # And the campaign now audits clean, with nothing left to relaunch.
            audit = audit_campaign_routes(root)
            self.assertEqual(audit['summary']['must_relaunch'], 0)
            with self.assertRaises(RuntimeError):
                prepare_route_relaunch(root)

    def test_relaunch_records_the_hash_of_the_file_it_wrote(self):
        # write_text applies platform newline translation, so a hash taken from
        # the in-memory text would not match what an audit reads back.
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = self.campaign(Path(tmp))
            result = prepare_route_relaunch(root)
            new_input = root / result['plan'][0]['new_input']
            digest = hashlib.sha256(new_input.read_bytes()).hexdigest()
            with (root / 'jobs.csv').open(newline='') as handle:
                active = [r for r in csv.DictReader(handle) if r['submission_active'] == 'true']
            self.assertEqual([r['input_sha256'] for r in active], [digest])
            audited = write_batch_progress(root, audit=True)
            self.assertNotIn('SHA256', '\n'.join(i['detail'] for i in audited['issues']))

    def test_relaunch_leaves_the_molecular_specification_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, stem = self.campaign(Path(tmp))
            original = (root / f'{stem}.gjf').read_text()
            result = prepare_route_relaunch(root)
            rebuilt = (root / result['plan'][0]['new_input']).read_text()

            def body(text):
                return [
                    line.rstrip() for line in text.splitlines()
                    if not line.lstrip().startswith(('#', '%'))
                ]

            self.assertEqual(body(rebuilt), body(original))
            self.assertTrue(result['plan'][0]['preserved_body_sha256'])

    def test_relaunch_declines_a_correct_route_that_still_collapsed(self):
        # A genuine TS search that converged on a minimum needs a better guess,
        # not a rewritten route; a mechanical relaunch would change nothing.
        with tempfile.TemporaryDirectory() as tmp:
            root, batch, stem = self.campaign(Path(tmp), break_route=False, finished=False)
            (batch / f'{stem}.log').write_text(SADDLE_LOG.replace(
                'Frequencies --   -420.0000', 'Frequencies --    420.0000'))
            (batch / f'{stem}.rc').write_text('0')
            audit = audit_campaign_routes(root)
            self.assertEqual(audit['rows'][0]['findings'], 'saddle_converged_to_minimum')
            plan = prepare_route_relaunch(root, dry_run=True)
            self.assertEqual(plan['input_count'], 0)
            self.assertIn('better guess geometry', plan['skipped'][0]['reason'])

    def test_relaunch_waits_for_an_unconfirmed_attempt_to_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, batch, stem = self.campaign(Path(tmp))
            (batch / f'{stem}.started').write_text('now')
            plan = prepare_route_relaunch(root, dry_run=True)
            self.assertEqual(plan['input_count'], 0)
            self.assertIn('assume-stopped', plan['skipped'][0]['reason'])
            forced = prepare_route_relaunch(root, dry_run=True, assume_stopped=True)
            self.assertEqual(forced['input_count'], 1)

    def test_higher_order_saddle_order_comes_from_the_seed_record(self):
        # "higher_order_saddle" only means >1 imaginary mode. The real count is
        # in the seeds extxyz, so each record gets its own Saddle=N rather than
        # one blanket guess across records whose orders differ.
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = self.campaign(Path(tmp), config_type='higher_order_saddle')
            record = _record('higher_order_saddle')
            record.imaginary_frequencies = 3
            seeds = Path(tmp) / 'seeds.extxyz'
            write_extxyz([record], seeds)

            plan = prepare_route_relaunch(root, dry_run=True, saddle_order_from=seeds)
            self.assertEqual(plan['input_count'], 1)
            self.assertEqual(plan['plan'][0]['saddle_order'], '3')
            self.assertEqual(plan['plan'][0]['saddle_order_source'],
                             'seed_imaginary_frequencies')
            self.assertIn('Saddle=3', plan['plan'][0]['route_after'])
            # An explicit order still wins, for a deliberate override.
            forced = prepare_route_relaunch(
                root, dry_run=True, saddle_order=2, saddle_order_from=seeds)
            self.assertEqual(forced['plan'][0]['saddle_order_source'],
                             'explicit_saddle_order')
            self.assertIn('Saddle=2', forced['plan'][0]['route_after'])

    def test_seed_order_disagreeing_with_the_label_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = self.campaign(Path(tmp), config_type='higher_order_saddle')
            record = _record('higher_order_saddle')
            record.imaginary_frequencies = 1
            seeds = Path(tmp) / 'seeds.extxyz'
            write_extxyz([record], seeds)
            plan = prepare_route_relaunch(root, dry_run=True, saddle_order_from=seeds)
            self.assertEqual(plan['input_count'], 0)
            self.assertIn('disagree', plan['skipped'][0]['reason'])

    def test_first_order_labels_never_consult_the_seed_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = self.campaign(Path(tmp), config_type='transition_state')
            record = _record('transition_state')
            record.imaginary_frequencies = 4
            seeds = Path(tmp) / 'seeds.extxyz'
            write_extxyz([record], seeds)
            plan = prepare_route_relaunch(root, dry_run=True, saddle_order_from=seeds)
            self.assertEqual(plan['plan'][0]['saddle_order'], '1')
            self.assertEqual(plan['plan'][0]['saddle_order_source'], 'first_order')
            self.assertIn('TS', plan['plan'][0]['route_after'])
            self.assertNotIn('Saddle=', plan['plan'][0]['route_after'])

    def test_higher_order_saddles_need_an_explicit_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = self.campaign(Path(tmp), config_type='higher_order_saddle')
            plan = prepare_route_relaunch(root, dry_run=True)
            self.assertEqual(plan['input_count'], 0)
            self.assertIn('--saddle-order', plan['skipped'][0]['reason'])
            plan = prepare_route_relaunch(root, dry_run=True, saddle_order=2)
            self.assertEqual(plan['input_count'], 1)
            self.assertIn('Saddle=2', plan['plan'][0]['route_after'])

    def test_collect_refuses_labels_from_a_mislaunched_transition_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, batch, stem = self.campaign(Path(tmp))
            destination = Path(tmp) / 'dataset'
            self.assertEqual(
                main(['collect', str(root), '-o', str(destination)]), 2
            )
            rejected = (destination / 'failed_outputs.tsv').read_text()
            self.assertIn('saddle_launched_as_minimum_search', rejected)
            self.assertEqual((destination / 'all.extxyz').read_text(), '')
            # The escape hatch still exists for a deliberate decision.
            self.assertEqual(
                main(['collect', str(root), '-o', str(Path(tmp) / 'forced'),
                      '--allow-route-mismatch']), 0
            )

    def test_collect_refuses_an_archived_invalidated_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, batch, stem = self.campaign(Path(tmp))
            prepare_route_relaunch(root)
            destination = Path(tmp) / 'dataset'
            self.assertEqual(main(['collect', str(root), '-o', str(destination)]), 2)
            self.assertIn('saddle_launched_as_minimum_search',
                          (destination / 'failed_outputs.tsv').read_text())


class InternalCoordinateFailureTests(BrokenCampaign, unittest.TestCase):
    """FormBX/Tors failures are a coordinate-system problem, not a route one."""

    # The tail of a real g09 failure: forces printed, then the optimizer dies
    # building internal coordinates because a torsion is degenerate.
    FORMBX_LOG = (
        " #p UBPW91/Gen SCF=(VShift=5) NoSymm Opt=(TS,CalcFC,NoEigenTest) Freq"
        " IOP(5/13=1) Int=UltraFine\n"
        " Charge = 0 Multiplicity = 4\n"
        + _GEOMETRY
        + _FORCES
        + " Cartesian Forces:  Max     0.005072867 RMS     0.001994465\n"
        " Berny optimization.\n"
        " Using GEDIIS/GDIIS optimizer.\n"
        " Tors failed for dihedral     1 -     2 -     3 -     4\n"
        " FormBX had a problem.\n"
        " Error termination via Lnk1e in /g09/l103.exe\n"
    )

    def test_the_marker_is_detected_and_flagged_for_relaunch(self):
        verdict = inspect_job(
            'transition_state',
            f'%chk=a.chk\n{DEFAULT_SADDLE_ROUTE}\n\nt\n\n0 4\nFe 0. 0. 0.\n',
            self.FORMBX_LOG,
        )
        self.assertIn('internal_coordinate_failure', verdict['findings'])
        self.assertTrue(verdict['must_relaunch'])

    def test_a_minimization_can_break_the_coordinate_system_too(self):
        verdict = inspect_job(
            'minimum',
            f'%chk=a.chk\n{DEFAULT_ROUTE}\n\nt\n\n0 4\nFe 0. 0. 0.\n',
            self.FORMBX_LOG.replace('Opt=(TS,CalcFC,NoEigenTest)', 'Opt'),
        )
        self.assertEqual(verdict['findings'], ['internal_coordinate_failure'])

    def test_relaunch_switches_to_cartesian_keeping_the_saddle_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, batch, stem = self.campaign(Path(tmp), break_route=False, finished=False)
            (batch / f'{stem}.log').write_text(self.FORMBX_LOG)
            (batch / f'{stem}.rc').write_text('1')
            audit = audit_campaign_routes(root)
            self.assertEqual(audit['rows'][0]['state'], 'failed')
            self.assertEqual(audit['rows'][0]['findings'], 'internal_coordinate_failure')

            result = prepare_route_relaunch(root)
            plan = result['plan'][0]
            self.assertEqual(plan['coordinate_system'], 'cartesian')
            route = plan['route_after']
            self.assertTrue(route_uses_cartesian(route))
            # The search it already had correctly is preserved.
            self.assertEqual(route_search_kind(route), 'saddle')
            self.assertEqual(saddle_route_gaps(route), [])
            # And the non-optimizing Force stage keeps internal coordinates
            # irrelevant to it.
            rebuilt = (root / plan['new_input']).read_text()
            routes = [l.strip() for l in rebuilt.splitlines() if l.startswith('#')]
            self.assertTrue(route_uses_cartesian(routes[0]))
            self.assertEqual(route_search_kind(routes[1]), 'none')
            self.assertNotIn('Cartesian', routes[1])

    def test_a_mislaunched_saddle_that_also_broke_gets_both_fixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, batch, stem = self.campaign(Path(tmp), finished=False)
            (batch / f'{stem}.log').write_text(
                self.FORMBX_LOG.replace('Opt=(TS,CalcFC,NoEigenTest)', 'Opt'))
            (batch / f'{stem}.rc').write_text('1')
            plan = prepare_route_relaunch(root, dry_run=True)['plan'][0]
            route = plan['route_after']
            self.assertEqual(route_search_kind(route), 'saddle')
            self.assertTrue(route_uses_cartesian(route))
            self.assertEqual(saddle_route_gaps(route), [])

    def test_the_failing_torsion_atoms_are_carried_into_the_audit(self):
        # Whoever rebuilds a stuck structure by hand needs to know which
        # dihedral broke; Gaussian prints it, so do not make them reopen logs.
        with tempfile.TemporaryDirectory() as tmp:
            root, batch, stem = self.campaign(Path(tmp), break_route=False, finished=False)
            (batch / f'{stem}.log').write_text(self.FORMBX_LOG)
            (batch / f'{stem}.rc').write_text('1')
            row = audit_campaign_routes(root)['rows'][0]
            self.assertEqual(row['failed_torsion_atoms'], '1-2-3-4')

    def test_a_zmatrix_input_keeps_its_internal_coordinates(self):
        # A dummy-atom Z-matrix (what ChemCraft produces) is itself the fix for
        # a degenerate torsion. Bolting Opt=Cartesian onto it would discard
        # that construction, so the route correction must leave it in
        # internals.
        with tempfile.TemporaryDirectory() as tmp:
            root, batch, stem = self.campaign(Path(tmp), finished=False)
            job_input = root / f'{stem}.gjf'
            body = job_input.read_text()
            zmatrix = body.split('0 4')[0] + '\n'.join([
                '0 4',
                'Fe',
                'Fe   1    B1',
                'X    1    1.0000    2   90.0',
                'H    1    B2        3   A1     2   D1',
                '',
                'B1   2.2000',
                'B2   1.7000',
                'A1   90.000',
                'D1   180.000',
                '',
            ])
            job_input.write_text(zmatrix, encoding='utf-8')
            (batch / f'{stem}.gjf').write_text(zmatrix, encoding='utf-8')
            (batch / f'{stem}.log').write_text(
                self.FORMBX_LOG.replace('Opt=(TS,CalcFC,NoEigenTest)', 'Opt'))
            (batch / f'{stem}.rc').write_text('1')

            self.assertTrue(input_is_zmatrix(zmatrix))
            plan = prepare_route_relaunch(root, dry_run=True)['plan'][0]
            self.assertEqual(plan['coordinate_system'], 'zmatrix_preserved')
            route = plan['route_after']
            self.assertFalse(route_uses_cartesian(route))
            # The route is still corrected to a saddle search.
            self.assertEqual(route_search_kind(route), 'saddle')
            self.assertEqual(saddle_route_gaps(route), [])

    def test_a_zmatrix_geometry_is_not_mistaken_for_a_checkpoint_seed(self):
        # Before Z-matrices were recognized these fell through to "unknown"
        # and were skipped, so a hand-repaired input could not be corrected.
        zmatrix = (
            '%chk=a.chk\n#p UBPW91/Gen Opt\n\nt\n\n0 4\nFe\n'
            'Fe   1    2.2000\nX    1    1.0    2   90.0\n\n'
        )
        self.assertEqual(geometry_source(zmatrix), 'input_coordinates')

    def test_failing_in_cartesian_is_reported_as_needing_a_human(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, batch, stem = self.campaign(Path(tmp), break_route=False, finished=False)
            cartesian_route = corrected_cartesian_route(DEFAULT_SADDLE_ROUTE)
            job_input = root / f'{stem}.gjf'
            job_input.write_text(
                job_input.read_text().replace(DEFAULT_SADDLE_ROUTE, cartesian_route),
                encoding='utf-8',
            )
            (batch / f'{stem}.gjf').write_text(job_input.read_text(), encoding='utf-8')
            (batch / f'{stem}.log').write_text(
                self.FORMBX_LOG.replace(
                    'Opt=(TS,CalcFC,NoEigenTest)', 'Opt=(TS,CalcFC,NoEigenTest,Cartesian)'))
            (batch / f'{stem}.rc').write_text('1')
            audit = audit_campaign_routes(root)
            self.assertEqual(
                audit['rows'][0]['findings'], 'internal_coordinate_failure_in_cartesian')
            # Reported, but not claimed to be fixable by a route rewrite.
            self.assertEqual(audit['summary']['must_relaunch'], 0)


class SpinLadderRelaunchTests(unittest.TestCase):
    """A ladder's one-spin-flip-at-a-time chain must survive a route relaunch."""

    ROUTE = '#p UBPW91/6-311++G* NoSymm Opt IOP(5/13=1) Int=UltraFine'

    def campaign(self, root, mults=(11, 9, 7)):
        campaign = root / 'w2'
        inputs, batch = campaign / 'inputs', campaign / 'slurm_batches/batch_0001'
        inputs.mkdir(parents=True)
        batch.mkdir(parents=True)
        name = ('seed__fe2o2__transition-state__q0-m11__reference__abc1234567'
                '__spin-ladder-m11-to-m7.gjf')
        sections, rows, previous = [], [], ''
        for stage, mult in enumerate(mults):
            chk = f'chain-s{stage:02d}-m{mult}.chk'
            header = (f'%oldchk={previous}\n' if previous else '') + (
                f'%chk={chk}\n%mem=24GB\n%nprocshared=12\n')
            route = self.ROUTE + (' Geom=Checkpoint Guess=Read' if stage else '')
            body = f'\n\nstage {stage}\n\n0 {mult}\n'
            if not stage:
                body += ('Fe   0.000000  0.000000  0.000000\n'
                         'Fe   2.200000  0.000000  0.000000\n'
                         'O    1.100000  1.100000  0.000000\n'
                         'O    1.100000 -1.100000  0.000000\n')
            sections.append(f'{header}{route}{body}')
            row = {column: '' for column in SPIN_MANIFEST_COLUMNS}
            row.update({
                'job_id': f'chain-s{stage:02d}', 'chain_id': 'chain',
                'stage_index': str(stage), 'pathway': 'multiplicity_ladder',
                'config_type': 'transition_state', 'intended_charge': '0',
                'intended_multiplicity': str(mult), 'high_spin_multiplicity': str(mults[0]),
                'final_target_multiplicity': str(mults[-1]), 'checkpoint': chk,
                'predecessor_checkpoint': previous, 'parent_record_id': 'parent',
                'source': 'warehouse/Fe2O2_ts.txt', 'formula': 'Fe2O2',
                'input': f'inputs/{name}', 'output': f'{Path(name).stem}.log',
            })
            rows.append(row)
            previous = chk
        (inputs / name).write_text('\n--Link1--\n'.join(sections), encoding='utf-8')
        digest = hashlib.sha256((inputs / name).read_bytes()).hexdigest()
        for row in rows:
            row['input_sha256'] = digest
        with (campaign / 'spin_jobs.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=SPIN_MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        shutil.copy2(inputs / name, batch / name)
        (batch / 'inputs.txt').write_text(name + '\n', encoding='utf-8')
        for row in rows:
            (batch / row['checkpoint']).write_bytes(b'collapsed-minimum')
        return campaign, batch, name, rows

    def _stages(self, text):
        """(route, oldchk, chk) per stage of a rendered input."""
        out = []
        for section in re.split(r'^\s*--Link1--\s*$', text, flags=re.M):
            route = next((l.strip() for l in section.splitlines()
                          if l.strip().startswith('#')), '')
            def directive(key):
                return next((l.split('=', 1)[1].strip() for l in section.splitlines()
                             if l.strip().lower().startswith(key)), '')
            out.append((route, directive('%oldchk'), directive('%chk')))
        return out

    def test_ladder_chain_is_preserved_and_every_stage_becomes_a_ts_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batch, name, _ = self.campaign(Path(tmp))
            result = prepare_route_relaunch(campaign)
            rebuilt = (campaign / result['plan'][0]['new_input']).read_text()
            stages = self._stages(rebuilt)
            self.assertEqual(len(stages), 3)
            for route, _, _ in stages:
                self.assertEqual(route_search_kind(route), 'saddle')
                self.assertEqual(saddle_route_gaps(route), [])
            # Stage k must still read stage k-1's checkpoint: the spin-flip
            # pathway is the point of the ladder.
            self.assertEqual(stages[0][1], '')
            self.assertEqual(stages[1][1], stages[0][2])
            self.assertEqual(stages[2][1], stages[1][2])
            # And none of them may touch the collapsed originals.
            for _, oldchk, chk in stages:
                for reference in (oldchk, chk):
                    if reference:
                        self.assertIn('-routefix01', reference)
            # Later stages still take their geometry from the checkpoint, not
            # from coordinates that were never there.
            self.assertIn('Geom=Checkpoint', stages[1][0])
            self.assertNotIn('Geom=Checkpoint', stages[0][0])

    def test_a_checkpoint_seeded_restart_is_rebuilt_from_the_root_ladder(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batch, name, rows = self.campaign(Path(tmp))
            # Stage 0 finished (collapsed), stage 1 was cut off by wall time.
            (batch / f'{Path(name).stem}.log').write_text(
                f' {self.ROUTE}\n Charge = 0 Multiplicity = 11\n'
                ' SCF Done:  E(UBPW91) =  -100.0 A.U.\n Stationary point found\n'
                ' Optimization completed.\n Normal termination of Gaussian 09\n'
                ' Link1:  Proceeding to internal job step number  2.\n'
                f' {self.ROUTE} Geom=Checkpoint Guess=Read\n'
                ' Charge = 0 Multiplicity = 9\n SCF Done:  E(UBPW91) =  -99.0 A.U.\n'
            )
            restart = prepare_spin_restarts(campaign, assume_stopped=True)
            continuation = campaign / restart['plan'][0]['new_input']
            seed = campaign / restart['plan'][0]['seed_checkpoint']
            self.assertEqual(geometry_source(continuation.read_text()), 'checkpoint')

            audit = audit_campaign_routes(campaign)
            row = next(r for r in audit['rows'] if r['must_relaunch'] == 'true')
            self.assertEqual(row['geometry_source'], 'checkpoint')
            self.assertEqual(
                audit['summary']['must_relaunch_by_geometry_source'], {'checkpoint': 1})

            result = prepare_route_relaunch(campaign, assume_stopped=True)
            plan = result['plan'][0]
            self.assertEqual(plan['geometry_source'], 'checkpoint')
            self.assertEqual(plan['rebuilt_from'], f'inputs/{name}')
            # The complete ladder is restored, not the two-stage tail the
            # restart was reduced to.
            self.assertEqual(plan['ladder_stages'], '3')
            rebuilt = (campaign / plan['new_input']).read_text()
            stages = self._stages(rebuilt)
            self.assertEqual(len(stages), 3)
            self.assertIn('Fe   0.000000', rebuilt)
            self.assertEqual([s[1] for s in stages][0], '')

            # The poisoned seed checkpoint is neither renamed nor referenced.
            self.assertTrue(seed.is_file())
            self.assertNotIn(seed.name, rebuilt)
            self.assertNotIn(f'{seed.stem}-routefix01', rebuilt)

            # Both lineage inputs are retired, and only the rebuild is active.
            self.assertEqual(
                set(plan['retired_inputs'].split(';')),
                {f'inputs/{name}', restart['plan'][0]['new_input']},
            )
            with (campaign / 'spin_jobs.csv').open(newline='') as handle:
                manifest = list(csv.DictReader(handle))
            active = [r for r in manifest if r['submission_active'] == 'true']
            self.assertEqual(len(active), 3)
            self.assertEqual({r['input'] for r in active}, {plan['new_input']})
            self.assertEqual(
                [r['intended_multiplicity'] for r in active], ['11', '9', '7'])
            for retired in (r for r in manifest if r['submission_active'] == 'false'):
                self.assertTrue(retired['route_invalidated'])
            # The earlier attempt keeps the archived log it actually produced.
            root_rows = [r for r in manifest if r['relaunch_attempt'] == ''
                         and r['input'] == f'inputs/{name}']
            self.assertTrue(all('before-restart01' in r['output'] for r in root_rows))

            listing = (batch / 'inputs.txt').read_text().split()
            self.assertEqual(listing, [Path(plan['new_input']).name])
            self.assertEqual(audit_campaign_routes(campaign)['summary']['must_relaunch'], 0)


class LauncherSafetyTests(SpinLadderRelaunchTests):
    def test_preflight_refuses_a_checkpoint_name_that_already_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batch, name, rows = self.campaign(Path(tmp))
            # A leftover file with the name the rebuild would write to.
            (batch / 'chain-s01-m9-routefix01.chk').write_bytes(b'live job')
            plan = prepare_route_relaunch(campaign, dry_run=True)
            self.assertTrue(any('would be overwritten' in p
                                for p in plan['launcher_problems']))
            with self.assertRaises(RuntimeError) as caught:
                prepare_route_relaunch(campaign)
            self.assertIn('refusing to relaunch', str(caught.exception))
            # Nothing was touched.
            self.assertEqual((batch / 'inputs.txt').read_text().split(), [name])
            self.assertFalse(list((campaign / 'inputs').glob('*routefix*')))

    def test_preflight_refuses_a_stage_count_that_disagrees_with_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batch, name, rows = self.campaign(Path(tmp))
            # Drop a manifest row so the ladder would be misdescribed.
            with (campaign / 'spin_jobs.csv').open('w', newline='', encoding='utf-8') as h:
                writer = csv.DictWriter(h, fieldnames=SPIN_MANIFEST_COLUMNS)
                writer.writeheader()
                writer.writerows(rows[:2])
            plan = prepare_route_relaunch(campaign, dry_run=True)
            self.assertTrue(any('manifest rows' in p for p in plan['launcher_problems']))
            with self.assertRaises(RuntimeError):
                prepare_route_relaunch(campaign)

    def test_launchers_keep_a_consistent_inputs_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batch, name, _ = self.campaign(Path(tmp))
            result = prepare_route_relaunch(campaign)
            listing = [
                line.strip()
                for line in (batch / 'inputs.txt').read_text().splitlines()
                if line.strip()
            ]
            # Every listed input resolves, stems are unique (so .log files
            # cannot collide), and the retired input is gone.
            self.assertEqual(len(set(listing)), len(listing))
            self.assertEqual(len({Path(n).stem for n in listing}), len(listing))
            for entry in listing:
                self.assertTrue((batch / entry).exists(), entry)
            self.assertNotIn(name, listing)
            self.assertIn(Path(result['plan'][0]['new_input']).name, listing)
            self.assertEqual(result['launcher_problems'], [])
            # %nprocshared still matches what the batch scripts allocate.
            rebuilt = (campaign / result['plan'][0]['new_input']).read_text()
            self.assertEqual(set(re.findall(r'%nprocshared=(\d+)', rebuilt)), {'12'})

    def test_assume_stopped_overrides_are_counted_not_silent(self):
        # QB4 cannot see QB3's queue, so --assume-stopped is an unverifiable
        # claim. Applying it silently is how a live job gets its log archived
        # from under it.
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batch, name, _ = self.campaign(Path(tmp))
            (batch / f'{Path(name).stem}.started').write_text('now')
            blocked = prepare_route_relaunch(campaign, dry_run=True)
            self.assertEqual(blocked['input_count'], 0)
            self.assertEqual(blocked['overridden_active_attempts'], [])
            self.assertIn('assume-stopped', blocked['skipped'][0]['reason'])

            result = prepare_route_relaunch(campaign, assume_stopped=True)
            self.assertEqual(result['input_count'], 1)
            self.assertEqual(len(result['overridden_active_attempts']), 1)
            self.assertEqual(
                result['overridden_active_attempts'][0]['input'], f'inputs/{name}')
            self.assertTrue((campaign / 'route_fix_overrides.csv').is_file())

    def test_relaunch_range_confines_changes_to_the_selected_batches(self):
        # Matters because a submission range and a relaunch range have to
        # agree: batches outside the range must be left completely alone.
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batch, name, _ = self.campaign(Path(tmp))
            second = campaign / 'slurm_batches/batch_0002'
            second.mkdir()
            shutil.copy2(batch / name, second / name)
            (second / 'inputs.txt').write_text(name + '\n', encoding='utf-8')
            before = (second / 'inputs.txt').read_text()
            result = prepare_route_relaunch(campaign, start=2, end=2, dry_run=True)
            # batch_0001 holds the manifest's copy of this input, so a range of
            # 2 only ever touches what batch_0002 lists.
            self.assertTrue(all(row['batch'] == 'batch_0002' for row in result['plan']))
            self.assertEqual((second / 'inputs.txt').read_text(), before)
            with self.assertRaises(ValueError):
                prepare_route_relaunch(campaign, start=1, end=99, dry_run=True)

    def test_nproc_mismatch_is_reported_against_the_saved_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, batch, name, _ = self.campaign(Path(tmp))
            (campaign / 'slurm_plan.json').write_text(
                json.dumps({'batch_count': 1, 'config': {'cpus_per_job': 16}}))
            plan = prepare_route_relaunch(campaign, dry_run=True)
            self.assertTrue(any('%nprocshared disagrees' in p
                                for p in plan['launcher_problems']))


if __name__ == '__main__':
    unittest.main()
