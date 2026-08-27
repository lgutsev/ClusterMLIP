import unittest

from cluster_mlip.basis import FE_DEF2TZVP_NO_F, LIGHT_ELEMENT_BASIS, render_gen_basis


class RenderGenBasisTests(unittest.TestCase):
    def test_mixed_molecule_emits_both_groups_in_order(self):
        text = render_gen_basis({"Fe", "O"})
        lines = text.splitlines()
        self.assertEqual(lines[0], "O     0")
        self.assertEqual(lines[1], LIGHT_ELEMENT_BASIS)
        self.assertEqual(lines[2], "****")
        self.assertEqual(lines[3], "Fe     0")
        fe_lines = FE_DEF2TZVP_NO_F.splitlines()
        self.assertEqual(lines[4 : 4 + len(fe_lines)], fe_lines)
        self.assertEqual(lines[4 + len(fe_lines)], "****")
        self.assertTrue(text.endswith("\n"))

    def test_multiple_light_elements_share_one_group(self):
        text = render_gen_basis({"C", "H", "O", "N"})
        first_line = text.splitlines()[0]
        self.assertEqual(first_line, "C H N O     0")
        self.assertNotIn("Fe", text)

    def test_fe_only_fragment_has_no_light_group(self):
        text = render_gen_basis({"Fe"})
        self.assertNotIn(LIGHT_ELEMENT_BASIS, text)
        lines = text.splitlines()
        self.assertEqual(lines[0], "Fe     0")
        self.assertEqual(text.count("****"), 1)

    def test_light_only_fragment_has_no_fe_group(self):
        text = render_gen_basis({"O"})
        self.assertNotIn("Fe", text)
        self.assertEqual(text.count("****"), 1)

    def test_empty_elements_rejected(self):
        with self.assertRaises(ValueError):
            render_gen_basis(set())


if __name__ == "__main__":
    unittest.main()
