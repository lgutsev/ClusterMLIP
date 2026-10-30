import unittest

import numpy as np

from cluster_mlip.legacy_doc_states import (
    HARTREE_TO_EV,
    deduplicate,
    document_level,
    document_text,
    normalize_level,
    parse_document_states,
)

TITLE = """###############################################
                   Fe_4      BPW91/6-311+G*
###############################################
"""

GEOM6 = """Input orientation:
 ---------------------------------------------------------------------
 Center     Atomic     Atomic              Coordinates (Angstroms)
 Number     Number      Type              X           Y           Z
 ---------------------------------------------------------------------
    1         26             0        0.000000    0.000000    0.000000
    2         26             0        2.200000    0.000000    0.000000
    3         26             0        1.100000    1.900000    0.000000
    4         26             0        1.100000    0.630000    1.800000
 ---------------------------------------------------------------------
"""

# G98-style table: no "orientation:" title and no "Atomic Type" column
GEOM5 = """ ----------------------------------------------------------
 Center     Atomic              Coordinates (Angstroms)
 Number     Number             X           Y           Z
 ----------------------------------------------------------
    1         26           0.180946    1.398777    0.000000
    2         26           1.313704   -0.508301    0.000000
    3         26          -0.747325   -0.445238    1.110455
    4         26          -0.747325   -0.445238   -1.110455
 ----------------------------------------------------------
"""


def spins(values, header="Mulliken atomic spin densities:"):
    rows = "\n".join(f"  {i}  Fe  {v: .6f}" for i, v in enumerate(values, 1))
    return f" {header}\n              1\n{rows}\n Sum of Mulliken spin densities=  {sum(values):.5f}\n"


def block(m, values, energy="HF=-5054.3127351\\S2=56.123\\S2-1=0.", q=0, geom=GEOM6, route="", extra=""):
    head = f"$$$$$$$$$  Fe_4    NEUTRAL    S = {m}\n"
    return head + route + f" Charge = {q:2d} Multiplicity ={m}\n" + geom + spins(values) + extra + "\n" + energy + "\n\n"


class LevelTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_level("UBPW91/6-311+G** scf=(VSHIFT=5)"), "BPW91/6-311+G*")
        self.assertEqual(normalize_level("BPW91/6-311+G*"), "BPW91/6-311+G*")
        self.assertEqual(normalize_level("UBLYP/6-311+G*"), "BLYP/6-311+G*")
        self.assertEqual(normalize_level("UB3LYP/6-311+G(d)"), "B3LYP/6-311+G*")
        self.assertEqual(normalize_level("UBPW91/6-311G*"), "BPW91/6-311G*")

    def test_document_title_variants(self):
        self.assertEqual(document_level(TITLE), "BPW91/6-311+G*")
        self.assertEqual(document_level("     Fe_4  ---   Fe_4-  BPW91/6-311+G*\n"), "BPW91/6-311+G*")
        self.assertEqual(document_level("        FE10      BPW91/6-311+G*\n"), "BPW91/6-311+G*")


class ParseTests(unittest.TestCase):
    def test_complete_block(self):
        text = TITLE + block(15, [3.5, 3.5, 3.5, 3.5])
        (s,) = parse_document_states(text, "FE4.doc")
        self.assertEqual((s.charge, s.multiplicity, s.symbols), (0, 15, ["Fe"] * 4))
        np.testing.assert_allclose(s.positions[1], [2.2, 0, 0])
        np.testing.assert_allclose(s.spins, [3.5] * 4)
        self.assertAlmostEqual(s.energy_ev, -5054.3127351 * HARTREE_TO_EV)
        self.assertAlmostEqual(s.s2, 56.123)
        self.assertEqual((s.level, s.level_source), ("BPW91/6-311+G*", "document"))
        self.assertEqual(s.scf_flags, [])

    def test_g98_table_and_total_spin_header(self):
        text = TITLE + block(1, [3.236, 3.236, -3.236, -3.236], geom=GEOM5).replace(
            "Mulliken atomic spin densities", "Total atomic spin densities")
        (s,) = parse_document_states(text)
        np.testing.assert_allclose(s.spins, [3.236, 3.236, -3.236, -3.236])
        self.assertTrue(s.genuine_antiparallel)
        self.assertEqual(s.artifact_sites, [])

    def test_word_wrapped_energy(self):
        text = TITLE + block(15, [3.5] * 4, energy="HF= -5054.31273 51\\S2=56.1\\S2-1=0.")
        (s,) = parse_document_states(text)
        self.assertAlmostEqual(s.energy_ev / HARTREE_TO_EV, -5054.3127351)

    def test_no_gluing_without_s2_field(self):
        text = TITLE + block(15, [3.5] * 4, energy="HF=-5054.3127351 58 eV")
        (s,) = parse_document_states(text)
        self.assertAlmostEqual(s.energy_ev / HARTREE_TO_EV, -5054.3127351)

    def test_dimer_from_zmatrix_bond_length(self):
        dimer = (" Charge = 0 Multiplicity = 1\n r=1.75\n ! r          2.2617   -DE/DX =    0.   !\n"
                 + spins([3.22, -3.22], "Total atomic spin densities:") + "\nHF=-2527.5041597\\S2=2.007\n")
        (s,) = parse_document_states("   Fe_2   BPW91/6-311+G*\n" + dimer)
        np.testing.assert_allclose(s.positions, [[0, 0, 0], [0, 0, 2.2617]])

    def test_flags_and_route_levels(self):
        route = " # UBLYP/6-311+G* scf=(VSHIFT=15) OPT IOP(5/13=1)\n"
        text = (TITLE + block(15, [3.5] * 4)
                + block(13, [3.0, 3.0, 3.0, 3.5], route=route)              # spin sum 12.5 != 12
                + block(13, [4.0, 4.0, 4.0, 0.0])                           # follows a BLYP route: ambiguous
                + block(5, [5.6, 2.0, 2.0, -5.6], route=" # UBPW91/6-311+G* OPT\n",
                        extra=" Frequencies --   -57.4397    131.5187\n"))
        a, b, c, d = parse_document_states(text)
        self.assertEqual((b.level, b.level_source), ("BLYP/6-311+G*", "route"))
        self.assertIn("spin_sum_mismatch", b.scf_flags)
        self.assertIn("iop_5_13", b.scf_flags)
        self.assertEqual(c.level_source, "ambiguous")
        self.assertEqual((d.level, d.level_source), ("BPW91/6-311+G*", "route"))
        self.assertEqual(d.artifact_sites, [0, 3])
        self.assertFalse(d.genuine_antiparallel)  # the only antiparallel site is an artifact site
        self.assertEqual(d.n_imaginary, 1)

    def test_block_title_names_the_level(self):
        text = TITLE + block(15, [3.5] * 4).replace("NEUTRAL    S = 15", "NEUTRAL    S = 15  C1  BLYP")
        text += block(13, [3.0] * 4).replace("S = 13", "S = 13  16-A  BPW91")
        text += block(11, [2.5] * 4, route=" # UBPW91/6-311+G* OPT\n").replace("S = 11", "S = 11  B3LYP")
        a, b, c = parse_document_states(text)
        self.assertEqual((a.level, a.level_source), ("BLYP/6-311+G*", "header"))
        self.assertEqual((b.level, b.level_source), ("BPW91/6-311+G*", "header"))
        self.assertEqual((c.level, c.level_source), ("BPW91/6-311+G*", "ambiguous"))  # title and route disagree

    def test_skips_are_counted(self):
        report = {}
        text = (TITLE + " Charge =  1 Multiplicity =14     IP =  5.675 eV\nHF=-5054.1\n"
                + block(15, [3.5] * 4)
                + " Charge = 0 Multiplicity = 3\n no geometry here\nHF=-1.0\n")
        self.assertEqual(len(parse_document_states(text, report=report)), 1)
        self.assertEqual(report["vertical_ion_summary"], 1)
        self.assertEqual(report["missing_geometry_spins"], 1)
        self.assertEqual(report["parsed"], 1)

    def test_spin_table_must_match_geometry(self):
        text = TITLE + block(15, [3.5, 3.5, 3.5])
        report = {}
        self.assertEqual(parse_document_states(text, report=report), [])
        self.assertEqual(report["spin_table_does_not_match_geometry"], 1)

    def test_deduplicate_repeated_block(self):
        text = TITLE + block(15, [3.5] * 4) + block(15, [3.5] * 4) + block(13, [3.0, 3.0, 3.0, 3.0])
        states = parse_document_states(text)
        self.assertEqual(len(states), 3)
        self.assertEqual([s.multiplicity for s in deduplicate(states)], [15, 13])

    def test_document_text_turns_word_marks_into_newlines(self):
        raw = b"\x00\x01Charge = 0 Multiplicity = 3\rHF=-1.5\x00\xff"
        self.assertEqual(document_text(raw), "Charge = 0 Multiplicity = 3\nHF=-1.5")


if __name__ == "__main__":
    unittest.main()
