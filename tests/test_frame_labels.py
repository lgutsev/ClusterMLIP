import csv
import io
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
                  job_config_type='irc_reverse')
    kwargs.update(overrides)
    return frame_label(**kwargs)


class FrameLabelTests(unittest.TestCase):
    def test_optimization_from_an_irc_point(self):
        labels = [_label(i, 3) for i in range(3)]
        self.assertEqual([x['frame_role'] for x in labels],
                         ['input_geometry', 'optimization_step', 'optimized_endpoint'])
        # Only the first frame is the IRC point; the search ended somewhere else.
        self.assertEqual([x['config_type'] for x in labels],
                         ['irc_reverse', 'optimization_path', 'optimized_unverified'])

    def test_unconverged_search_has_no_endpoint(self):
        self.assertEqual(_label(2, 3, converged=False)['frame_role'], 'optimization_step')

    def test_single_point_stage_keeps_the_job_label(self):
        self.assertEqual(_label(0, 1, search_kind='none', job_config_type='minimum_rattled'),
                         {'frame_role': 'fixed_geometry', 'config_type': 'minimum_rattled'})

    def test_checkpoint_seeded_first_frames(self):
        self.assertEqual(_label(0, 3, stage_index=2)['frame_role'], 'spin_flip_start')
        self.assertEqual(_label(0, 3, initialization='checkpoint_spin_flip')['frame_role'],
                         'spin_flip_start')
        self.assertEqual(_label(0, 3, initialization='interrupted_checkpoint_restart'),
                         {'frame_role': 'restart_start', 'config_type': 'optimization_path'})

    def test_a_minimum_is_only_stationary_in_its_own_state(self):
        self.assertEqual(_label(1, 2, job_config_type='minimum')['config_type'], 'minimum')
        self.assertEqual(_label(1, 2, job_config_type='minimum', same_state_as_label=False)
                         ['config_type'], 'optimized_unverified')
        self.assertEqual(_label(0, 2, job_config_type='minimum')['config_type'], 'minimum')
        # The same geometry at another multiplicity is not a stationary point there.
        self.assertEqual(_label(0, 2, job_config_type='minimum', same_state_as_label=False),
                         {'frame_role': 'input_geometry', 'config_type': 'optimization_path'})

    def test_saddle_search_endpoints(self):
        self.assertEqual(_label(1, 2, search_kind='saddle', job_config_type='first_order_saddle')
                         ['config_type'], 'first_order_saddle')
        self.assertEqual(_label(1, 2, search_kind='saddle', job_config_type='unknown')
                         ['config_type'], 'transition_state')


class CollectFrameLabelTests(unittest.TestCase):
    """An old spin campaign (no config_type column) whose IRC point ran as a
    two-stage Opt ladder, M=3 then M=1."""

    def test_irc_point_run_as_opt_is_labelled_frame_by_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
                    writer.writerow(dict(job_id=f'm{mult}', output='ladder.log',
                                         input=f'inputs/{name}', parent_record_id='parent',
                                         intended_charge=0, intended_multiplicity=mult,
                                         checkpoint=f'm{mult}.chk', spin_plan_id='plan',
                                         stage_index=stage, initialization=init,
                                         target_record_multiplicity=1))
            text = FIXTURE.read_text()
            step = text.split('\n', 1)[1]  # one more optimization step, same section
            m3 = text.replace('Multiplicity = 1', 'Multiplicity = 3')
            (root / 'ladder.log').write_text(m3 + step * 2 + CONVERGED + NORMAL
                                             + text + step + CONVERGED + NORMAL)
            args = build_parser().parse_args(['collect', str(root), '-o', str(root / 'data'),
                                              '--frames', 'all'])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(args.func(args), 0)
            frames = read_labeled_extxyz(root / 'data/all.extxyz')
            self.assertEqual((root / 'data/failed_outputs.tsv').read_text(), '')
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
            self.assertEqual((first['job_config_type'], first['config_type_source']),
                             ('irc_reverse', 'filename'))


if __name__ == '__main__':
    unittest.main()
