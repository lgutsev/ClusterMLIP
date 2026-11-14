import csv
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.saddle_modes import export_campaign_saddle_modes, frequency_analyses
from test_route_audit import BrokenCampaign


def _orientation(kind: str, x: float) -> str:
    rule = ' ' + '-' * 69
    return '\n'.join([
        f'                         {kind} orientation:', rule,
        ' Center     Atomic      Atomic             Coordinates (Angstroms)',
        ' Number     Number       Type             X           Y           Z', rule,
        '      1         26           0        0.000000    0.000000    0.000000',
        f'      2          8           0        {x:.6f}    0.000000    0.000000',
        '      3          1           0        0.000000    1.000000    0.000000', rule, ''])


def _freq_block(freqs, masses, vectors) -> str:
    """Three modes per group, in Gaussian's default layout."""
    out = [' Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering',
           ' and normal coordinates:']
    for start in range(0, len(freqs), 3):
        cols = range(start, min(start + 3, len(freqs)))
        out.append(''.join(f'{c + 1:23d}' for c in cols))
        out.append(''.join(f'{"A":>23}' for _ in cols))
        out.append(' Frequencies -- ' + ''.join(f'{freqs[c]:12.4f}           ' for c in cols))
        out.append(' Red. masses -- ' + ''.join(f'{masses[c]:12.4f}           ' for c in cols))
        out.append('  Atom  AN      X      Y      Z  ' + '      X      Y      Z  ' * (len(cols) - 1))
        for atom, number in enumerate((26, 8, 1)):
            out.append(f'{atom + 1:6d}{number:4d}  ' + '  '.join(
                ''.join(f'{v:7.2f}' for v in vectors[c][atom]) for c in cols))
    out.append(' -------------------\n - Thermochemistry -\n -------------------')
    return '\n'.join(out) + '\n'


V = [[(0.10, 0.00, 0.00), (-0.70, 0.00, 0.00), (0.00, 0.20, 0.00)],
     [(0.00, 0.30, 0.00), (0.00, 0.00, 0.40), (0.50, 0.00, 0.00)],
     [(0.00, 0.00, 0.60), (0.10, 0.10, 0.10), (0.00, 0.00, 0.90)]]


def _stage(mult, x, freqs, standard=False) -> str:
    text = f' Charge = 0 Multiplicity = {mult}\n' + _orientation('Input', x)
    if standard:
        text += _orientation('Standard', x + 5.0)
    text += ' SCF Done:  E(UBPW91) =  -10.0000000000     A.U. after   10 cycles\n'
    return text + _freq_block(freqs, [5.0, 6.0, 7.0], V)


class FrequencyAnalysisTests(unittest.TestCase):
    def test_reads_the_imaginary_mode_and_its_vectors(self):
        analyses = frequency_analyses(_stage(7, 1.6, [-520.5, 100.0, 300.0]))
        self.assertEqual(len(analyses), 1)
        analysis = analyses[0]
        self.assertEqual((analysis.charge, analysis.multiplicity), (0, 7))
        self.assertEqual(analysis.frequencies, [-520.5, 100.0, 300.0])
        self.assertEqual(len(analysis.imaginary), 1)
        mode = analysis.imaginary[0]
        self.assertEqual((mode.index, mode.frequency_cm1, mode.reduced_mass_amu), (1, -520.5, 5.0))
        self.assertEqual(mode.vectors, V[0])
        self.assertAlmostEqual(analysis.atoms[1].x, 1.6)

    def test_every_ladder_stage_gets_its_own_analysis_and_geometry(self):
        text = _stage(9, 1.6, [50.0, 100.0, 300.0]) + _stage(7, 1.7, [-80.0, -20.0, 300.0])
        first, second = frequency_analyses(text)
        self.assertEqual(first.imaginary, [])
        self.assertEqual(second.multiplicity, 7)
        self.assertEqual([m.index for m in second.imaginary], [1, 2])
        self.assertEqual(second.imaginary[1].vectors, V[1])
        self.assertAlmostEqual(second.atoms[1].x, 1.7)

    def test_standard_orientation_is_the_frame_of_the_modes(self):
        analysis = frequency_analyses(_stage(7, 1.6, [-10.0, 1.0, 2.0], standard=True))[0]
        self.assertTrue(analysis.orientation.lower().startswith('standard'))
        self.assertAlmostEqual(analysis.atoms[1].x, 6.6)

    def test_hpmodes_is_reported_not_misread(self):
        text = _stage(7, 1.6, [-10.0, 1.0, 2.0]).replace(
            '  Atom  AN      X      Y      Z  ', '   Coord Atom Element:')
        analysis = frequency_analyses(text)[0]
        self.assertIn('HPModes', analysis.unsupported)
        self.assertEqual(analysis.imaginary, [])


class CampaignExportTests(BrokenCampaign, unittest.TestCase):
    def test_export_writes_extxyz_and_csv_and_leaves_the_campaign_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, batch, stem = self.campaign(Path(tmp) / 'c', break_route=False, finished=False)
            (batch / f'{stem}.log').write_text(
                _stage(4, 1.6, [-520.5, 100.0, 300.0]) + ' Normal termination of Gaussian 09\n')
            before = sorted(p.name for p in root.rglob('*'))
            out = Path(tmp) / 'modes'
            summary = export_campaign_saddle_modes(root, out)
            self.assertEqual((summary['records'], summary['first_order']), (1, 1))
            lines = (out / 'saddle_modes.extxyz').read_text().splitlines()
            self.assertEqual(lines[0], '3')
            self.assertIn('Properties=species:S:1:pos:R:3:mode:R:3', lines[1])
            self.assertIn('frequency_cm1=-520.5000', lines[1])
            self.assertEqual(lines[3].split()[0], 'O')
            self.assertEqual([float(v) for v in lines[3].split()[4:]], [-0.70, 0.0, 0.0])
            with (out / 'saddle_modes.csv').open(newline='') as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual((row['formula'], row['largest_atom'], row['multiplicity']),
                             ('FeHO', 'O2', '4'))
            self.assertEqual(sorted(p.name for p in root.rglob('*')), before)


if __name__ == '__main__':
    unittest.main()
