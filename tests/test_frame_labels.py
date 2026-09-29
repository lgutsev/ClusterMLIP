import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from cluster_mlip.cli import build_parser
from cluster_mlip.dataset import read_labeled_extxyz
from cluster_mlip.routes import frame_label
from cluster_mlip.stratify import classify_record

FIXTURE = Path(__file__).parent / 'fixtures' / 'force.log'
NORMAL = '\n Normal termination of Gaussian 09\n'
CONVERGED = '\n Stationary point found.\n'


def _label(position, count, **overrides):
    kwargs = dict(position=position, count=count, search_kind='minimum', converged=True,
                  job_config_type='irc_reverse', job_geometry_role='irc_point')
    kwargs.update(overrides)
    return frame_label(**kwargs)


class FrameLabelTests(unittest.TestCase):
    def test_optimization_from_an_irc_point(self):
        roles = [_label(i, 3) for i in range(3)]
        self.assertEqual([r['frame_role'] for r in roles],
                         ['input_geometry', 'optimization_step', 'optimized_endpoint'])
        # Only the first frame is the IRC point; the search ended somewhere else.
        self.assertEqual([r['config_type'] for r in roles],
                         ['irc_reverse', 'optimization_path', 'optimized_unverified'])
        self.assertEqual(roles[0]['geometry_role'], 'irc_point')

    def test_unconverged_search_has_no_endpoint(self):
        self.assertEqual(_label(2, 3, converged=False)['frame_role'], 'optimization_step')

    def test_fixed_geometry_stage_keeps_the_job_label(self):
        label = _label(0, 1, search_kind='none')
        self.assertEqual((label['frame_role'], label['config_type'], label['geometry_role']),
                         ('fixed_geometry', 'irc_reverse', 'irc_point'))

    def test_checkpoint_seeded_first_frames(self):
        self.assertEqual(_label(0, 3, stage_index=2)['frame_role'], 'spin_flip_start')
        self.assertEqual(_label(0, 3, initialization='checkpoint_spin_flip')['frame_role'],
                         'spin_flip_start')
        restart = _label(0, 3, initialization='interrupted_checkpoint_restart')
        self.assertEqual((restart['frame_role'], restart['config_type']),
                         ('restart_start', 'optimization_path'))

    def test_a_minimum_is_only_re_found_in_its_own_state(self):
        own = _label(1, 2, job_config_type='minimum', job_geometry_role='stationary_minimum')
        self.assertEqual((own['config_type'], own['geometry_role']), ('minimum', 'stationary_minimum'))
        other = _label(1, 2, job_config_type='minimum', job_geometry_role='stationary_minimum',
                       same_state_as_label=False)
        self.assertEqual(other['config_type'], 'optimized_unverified')
        # Its starting frame is still that geometry, but not a stationary point at this M.
        start = _label(0, 2, job_config_type='minimum', job_geometry_role='stationary_minimum',
                       same_state_as_label=False)
        self.assertEqual((start['config_type'], start['geometry_role']),
                         ('minimum', 'unconstrained_geometry'))

    def test_saddle_search_endpoints(self):
        ts = _label(1, 2, search_kind='saddle', job_config_type='transition_state',
                    job_geometry_role='transition_state')
        self.assertEqual((ts['config_type'], ts['geometry_role']),
                         ('transition_state', 'transition_state'))
        higher = _label(1, 2, search_kind='saddle', job_config_type='higher_order_saddle',
                        job_geometry_role='higher_order_candidate')
        self.assertEqual((higher['config_type'], higher['geometry_role']),
                         ('higher_order_saddle', 'higher_order_candidate'))


class CollectFrameLabelTests(unittest.TestCase):
    """An old spin campaign (no config_type column) whose IRC point ran as a
    two-stage Opt ladder, M=3 then M=1."""

    def _campaign(self, root: Path) -> None:
        name = 'src__h2__irc-reverse__q0-m1__reference__abc__auto-spin-m3-to-m1.gjf'
        (root / 'inputs').mkdir()
        (root / 'inputs' / name).write_text(
            '%chk=m3.chk\n#p UBPW91/6-311++G* Opt\n\ntitle\n\n0 3\nH 0 0 0\nH 0.75 0 0\n\n'
            '--Link1--\n%chk=m1.chk\n#p UBPW91/6-311++G* Opt Geom=Checkpoint Guess=Read\n\n'
            'title\n\n0 1\n\n')
        with (root / 'spin_jobs.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                'job_id', 'output', 'input', 'parent_record_id', 'intended_charge',
                'intended_multiplicity', 'checkpoint', 'spin_plan_id', 'stage_index',
                'initialization', 'target_record_multiplicity'])
            writer.writeheader()
            for stage, (mult, init) in enumerate(
                    ((3, 'data_inferred_high_spin_direct_on_target_geometry'),
                     (1, 'checkpoint_spin_flip'))):
                writer.writerow(dict(job_id=f'm{mult}', output='ladder.log', input=f'inputs/{name}',
                                     parent_record_id='parent', intended_charge=0,
                                     intended_multiplicity=mult, checkpoint=f'm{mult}.chk',
                                     spin_plan_id='plan', stage_index=stage, initialization=init,
                                     target_record_multiplicity=1))
        text = FIXTURE.read_text()
        step = text.split('\n', 1)[1]  # one more optimization step, same section
        m3 = text.replace('Multiplicity = 1', 'Multiplicity = 3')
        (root / 'ladder.log').write_text(m3 + step * 2 + CONVERGED + NORMAL
                                         + text + step + CONVERGED + NORMAL)

    def _collect(self, root: Path, *extra: str):
        args = build_parser().parse_args(['collect', str(root), '-o', str(root / 'data'),
                                          '--frames', 'all', *extra])
        with redirect_stdout(io.StringIO()):
            code = args.func(args)
        frames = read_labeled_extxyz(root / 'data/all.extxyz')
        self.assertEqual(code, 0 if frames else 2)  # 2 = nothing collected
        return frames, (root / 'data/failed_outputs.tsv').read_text()

    def test_irc_point_run_as_opt_is_labelled_frame_by_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._campaign(root)
            frames, failures = self._collect(root)
            self.assertEqual(frames, [])
            self.assertIn('path_point_launched_as_optimization', failures)

            frames, failures = self._collect(root, '--allow-route-mismatch')
            self.assertEqual(failures, '')
            self.assertEqual([f.record.multiplicity for f in frames], [3, 3, 3, 1, 1])
            self.assertEqual([f.record.metadata['frame_role'] for f in frames], [
                'input_geometry', 'optimization_step', 'optimized_endpoint',
                'spin_flip_start', 'optimized_endpoint'])
            self.assertEqual([f.record.config_type for f in frames], [
                'irc_reverse', 'optimization_path', 'optimized_unverified',
                'optimization_path', 'optimized_unverified'])
            self.assertEqual([classify_record(f.record)['pes_region'] for f in frames],
                             ['irc', 'other', 'other', 'other', 'other'])
            first = frames[0].record.metadata
            # The job-level label survives next to the per-frame one.
            self.assertEqual((first['job_config_type'], first['config_type_source'],
                              first['job_geometry_role']), ('irc_reverse', 'filename', 'irc_point'))
            self.assertIn('path_point_launched_as_optimization', first['route_findings'])
            report = json.loads((root / 'data/label_report.json').read_text())
            self.assertEqual(report['frame_roles'], {'input_geometry': 1, 'optimization_step': 1,
                                                     'optimized_endpoint': 2, 'spin_flip_start': 1})


if __name__ == '__main__':
    unittest.main()
