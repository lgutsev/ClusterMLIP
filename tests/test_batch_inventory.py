from pathlib import Path
import tempfile
import unittest
import zipfile

from cluster_mlip.batch_inventory import build_inventory, find_zips, known_formulas

FIXTURES = Path(__file__).parent / "fixtures"


def _build_warehouse_folder(root: Path) -> Path:
    folder = root / "warehouses"
    folder.mkdir()
    with zipfile.ZipFile(folder / "delivery1.zip", "w") as archive:
        archive.write(FIXTURES / "minimum.log", "minimum.log")
        archive.write(FIXTURES / "irc.log", "irc.log")
    with zipfile.ZipFile(folder / "delivery2.zip", "w") as archive:
        # Same structure as minimum.log under a different filename, to
        # exercise cross-ZIP merging of the same formula/charge/state.
        archive.write(FIXTURES / "minimum.log", "minimum_dup.log")
        archive.write(FIXTURES / "force.log", "force.log")
    return folder


class FindZipsTests(unittest.TestCase):
    def test_non_recursive_ignores_subfolders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = _build_warehouse_folder(root)
            nested = folder / "nested"
            nested.mkdir()
            with zipfile.ZipFile(nested / "delivery3.zip", "w") as archive:
                archive.write(FIXTURES / "minimum.log", "minimum.log")

            self.assertEqual(
                [p.name for p in find_zips(folder)], ["delivery1.zip", "delivery2.zip"]
            )
            self.assertEqual(
                [p.name for p in find_zips(folder, recursive=True)],
                ["delivery1.zip", "delivery2.zip", "delivery3.zip"],
            )


class BuildInventoryTests(unittest.TestCase):
    def test_merges_same_structure_across_zips_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = _build_warehouse_folder(root)
            result = build_inventory(folder, root / "out")

            self.assertEqual([z["source"] for z in result["zips"]], ["delivery1.zip", "delivery2.zip"])

            master_by_formula = {row["formula"]: row for row in result["master"]}
            self.assertIn("FeO", master_by_formula)
            # minimum.log's structure appears (under different filenames) in
            # both ZIPs -- the merged master list must show both sources.
            self.assertEqual(master_by_formula["FeO"]["sources"], ["delivery1.zip", "delivery2.zip"])
            self.assertEqual(master_by_formula["FeO"]["charge"], 0)
            self.assertEqual(master_by_formula["FeO"]["multiplicity"], 3)

            # force.log only appears in delivery2.
            other_only_in_delivery2 = [
                row for row in result["master"] if row["sources"] == ["delivery2.zip"]
            ]
            self.assertTrue(other_only_in_delivery2)

    def test_writes_per_zip_reports_and_top_level_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = _build_warehouse_folder(root)
            output = root / "out"
            build_inventory(folder, output)

            self.assertTrue((output / "inventory.json").exists())
            self.assertTrue((output / "inventory.md").exists())
            self.assertTrue((output / "by_source" / "delivery1" / "report.md").exists())
            self.assertTrue((output / "by_source" / "delivery2" / "records.csv").exists())

    def test_empty_folder_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, "no \\*.zip files"):
                build_inventory(empty, root / "out")


class KnownFormulasTests(unittest.TestCase):
    def test_known_formulas_is_the_coarse_formula_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = _build_warehouse_folder(root)
            result = build_inventory(folder, root / "out")
            formulas = known_formulas(result)
            self.assertIn("FeO", formulas)
            self.assertIsInstance(formulas, set)


if __name__ == "__main__":
    unittest.main()
