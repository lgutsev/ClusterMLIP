import json
import math
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.cli import main
from cluster_mlip.periodic import Structure, read_structures, write_structures
from cluster_mlip.vasp import (
    VaspSettings,
    _magmom_line,
    collect_vasp_campaign,
    parse_vasprun,
    prepare_vasp_campaign,
)

CELL = [(8.0, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, 0.0, 16.0)]


def _supported() -> Structure:
    # support first (C, O interleaved so POSCAR reordering is exercised), cluster last
    symbols = ["C", "O", "C", "O", "Fe", "Fe"]
    positions = [(0, 0, 1), (2, 0, 1), (0, 2, 1), (2, 2, 1), (1, 1, 3), (1, 1, 5.2)]
    return Structure(
        symbols, [tuple(map(float, p)) for p in positions], CELL, (True, True, True),
        {"structure_id": "fe2 on CO/x", "multiplicity": 7, "charge": 0, "support": "toy"},
        {"cluster": [0, 0, 0, 0, 1, 1]},
    )


def _force(role: str, m: int | None, original_index: int) -> tuple[float, float, float]:
    base = {"AB": 1.0, "A": 0.25, "B": 0.5}[role] + (m or 0) * 0.01
    return (base * (original_index + 1), -base, 0.1 * original_index)


def _energy(role: str, m: int | None) -> float:
    return {"AB": -100.0, "A": -40.0, "B": -55.0}[role] - (m or 0) * 0.1


def _fake_vasprun(job: Path, *, nelm: int = 300, n_sc: int = 20, mag: float | None = None,
                  truncated: bool = False) -> None:
    meta = json.loads((job / "job.json").read_text())
    role, m, order = meta["role"], meta["multiplicity"], meta["order"]
    forces = "".join(
        "<v>" + " ".join(f"{x:.8f}" for x in _force(role, m, i)) + "</v>" for i in order
    )
    scsteps = "<scstep><energy><i name=\"e_fr_energy\">1.0</i></energy></scstep>" * n_sc
    e = _energy(role, m)
    text = (
        "<?xml version=\"1.0\"?><modeling><parameters><separator name=\"electronic\">"
        f"<i type=\"int\" name=\"NELM\">{nelm}</i></separator></parameters>"
        f"<calculation>{scsteps}<varray name=\"forces\">{forces}</varray>"
        f"<energy><i name=\"e_fr_energy\">{e:.8f}</i><i name=\"e_wo_entrp\">{e - 0.002:.8f}</i>"
        f"<i name=\"e_0_energy\">{e - 0.001:.8f}</i></energy></calculation></modeling>"
    )
    if truncated:
        text = text[: len(text) // 2]
    (job / "vasprun.xml").write_text(text)
    expected = (m - 1) if m else 0
    value = expected if mag is None else mag
    (job / "OSZICAR").write_text(f"   1 F= {e:.5E} E0= {e:.5E}  d E =0.0  mag=    {value:.4f}\n")


class VaspPrepareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.structures = self.root / "supported.extxyz"
        write_structures([_supported()], self.structures)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_extxyz_round_trip_keeps_cell_pbc_and_columns(self) -> None:
        (s,) = read_structures(self.structures)
        self.assertEqual(s.cell, CELL)
        self.assertEqual(s.pbc, (True, True, True))
        self.assertEqual(s.arrays["cluster"], [0, 0, 0, 0, 1, 1])
        self.assertEqual(s.info["multiplicity"], 7)
        self.assertEqual(s.cluster_mask(), [False] * 4 + [True] * 2)

    def test_prepare_writes_three_roles_with_species_blocks_and_fixed_spin(self) -> None:
        out = self.root / "campaign"
        plan = prepare_vasp_campaign(self.structures, out, multiplicities=[5, 7])
        self.assertEqual(plan["n_jobs"], 1 + 2 * 2)
        jobs = (out / "jobs.txt").read_text().split()
        self.assertEqual(len(jobs), 5)
        sid = plan["entries"][0]["structure_id"]
        self.assertNotIn("/", sid)
        ab = out / "structures" / sid / "m7" / "AB"
        poscar = (ab / "POSCAR").read_text().splitlines()
        self.assertEqual(poscar[5].split(), ["C", "O", "Fe"])
        self.assertEqual(poscar[6].split(), ["2", "2", "2"])
        incar = (ab / "INCAR").read_text()
        self.assertIn("NUPDOWN = 6", incar)
        self.assertIn("MAGMOM = 4*0 2*3", incar)  # M-1 = 6 spread over the two Fe
        self.assertIn("IDIPOL = 3", incar)
        self.assertIn("IVDW = 12", incar)
        self.assertEqual((ab / "POTCAR.spec").read_text().split(), ["C", "O", "Fe_pv"])
        a_poscar = (out / "structures" / sid / "m7" / "A" / "POSCAR").read_text().splitlines()
        self.assertEqual(a_poscar[5].split(), ["Fe"])
        self.assertEqual(a_poscar[2:5], poscar[2:5])  # same cell for the frozen fragment
        b_incar = (out / "structures" / sid / "B" / "INCAR").read_text()
        self.assertIn("NUPDOWN = 0", b_incar)
        self.assertIn("MAGMOM = 4*0", b_incar)
        for script in ("run_vasp.sbatch", "submit.sh"):
            self.assertNotIn(b"\r\n", (out / script).read_bytes())
        self.assertIn("--array=0-4%10", (out / "submit.sh").read_text())

    def test_prepare_refuses_charged_cells_and_missing_split(self) -> None:
        s = _supported()
        s.info["charge"] = 1
        path = self.root / "charged.extxyz"
        write_structures([s], path)
        with self.assertRaisesRegex(ValueError, "charge"):
            prepare_vasp_campaign(path, self.root / "c1")
        s = _supported()
        s.arrays = {}
        path = self.root / "nosplit.extxyz"
        write_structures([s], path)
        with self.assertRaisesRegex(ValueError, "cluster"):
            prepare_vasp_campaign(path, self.root / "c2")
        prepare_vasp_campaign(path, self.root / "c3", cluster_elements={"Fe"})

    def test_explicit_moments_and_overrides(self) -> None:
        s = _supported()
        s.arrays["initial_magmoms"] = [0.5, 0.0, 0.0, 0.0, 3.8, -3.8]
        path = self.root / "moments.extxyz"
        write_structures([s], path)
        settings = VaspSettings(incar_overrides={"encut": "520"}, potcars={"Fe": "Fe"}, ivdw=0)
        plan = prepare_vasp_campaign(path, self.root / "c", settings=settings, multiplicities=[1])
        sid = plan["entries"][0]["structure_id"]
        incar = (self.root / "c" / "structures" / sid / "m1" / "AB" / "INCAR").read_text()
        self.assertIn("MAGMOM = 4*0 3.8 -3.8", incar)  # support moments zeroed
        self.assertIn("ENCUT = 520", incar)
        self.assertNotIn("IVDW", incar)

    def test_magmom_run_length_encoding(self) -> None:
        self.assertEqual(_magmom_line([0, 0, 0, 3.25, 3.25, -1.0]), "3*0 2*3.25 -1")


class VaspCollectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        structures = self.root / "supported.extxyz"
        write_structures([_supported()], structures)
        self.campaign = self.root / "campaign"
        self.plan = prepare_vasp_campaign(structures, self.campaign, multiplicities=[5, 7])
        self.entry = self.plan["entries"][0]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run_all(self, **overrides: dict) -> None:
        for job in (self.campaign / "jobs.txt").read_text().split():
            _fake_vasprun(self.campaign / job, **overrides.get(job, {}))

    def test_interaction_labels_are_differences_in_original_atom_order(self) -> None:
        self._run_all()
        summary = collect_vasp_campaign(self.campaign, self.root / "labels")
        self.assertEqual(summary["interaction_frames"], 2)
        self.assertEqual(summary["failed_jobs"], 0)
        frames = {f.info["multiplicity"]: f for f in read_structures(self.root / "labels" / "interaction.extxyz")}
        f7 = frames[7]
        self.assertAlmostEqual(f7.info["REF_energy"], _energy("AB", 7) - _energy("A", 7) - _energy("B", None))
        self.assertEqual(f7.info["config_type"], "support_interaction")
        self.assertEqual(f7.info["spin"], 7)
        self.assertIn("GGA=PE", f7.info["label_level"])
        for i, force in enumerate(f7.arrays["REF_forces"]):
            other = "A" if i >= 4 else "B"
            expected = [a - b for a, b in zip(_force("AB", 7, i), _force(other, 7 if other == "A" else None, i))]
            for got, want in zip(force, expected):
                self.assertAlmostEqual(got, want, places=6)
        clusters = read_structures(self.root / "labels" / "cluster_frozen.extxyz")
        self.assertEqual([c.symbols for c in clusters], [["Fe", "Fe"]] * 2)
        supports = read_structures(self.root / "labels" / "support_frozen.extxyz")
        self.assertEqual(len(supports), 1)
        self.assertEqual(supports[0].symbols, ["C", "O", "C", "O"])

    def test_unconverged_truncated_and_wrong_moment_jobs_are_rejected(self) -> None:
        jobs = self.entry["jobs"]
        self._run_all(**{
            jobs["AB_m5"]: {"nelm": 20, "n_sc": 20},
            jobs["A_m7"]: {"mag": 4.0},
        })
        summary = collect_vasp_campaign(self.campaign, self.root / "labels")
        self.assertEqual(summary["interaction_frames"], 0)
        self.assertEqual(summary["failed_jobs"], 2)
        failed = (self.root / "labels" / "failed_jobs.tsv").read_text()
        self.assertIn("NELM=20", failed)
        self.assertIn("mag 4.0", failed)

        _fake_vasprun(self.campaign / jobs["B"], truncated=True)
        self.assertFalse(parse_vasprun(self.campaign / jobs["B"] / "vasprun.xml").completed)

    def test_sigma0_energy_option(self) -> None:
        self._run_all()
        collect_vasp_campaign(self.campaign, self.root / "labels", energy="sigma0")
        f = read_structures(self.root / "labels" / "interaction.extxyz")[0]
        e = f.info["E_AB"] - f.info["E_A"] - f.info["E_B"]
        self.assertTrue(math.isclose(f.info["REF_energy"], e, abs_tol=1e-9))
        self.assertAlmostEqual(f.info["E_AB"], _energy("AB", f.info["multiplicity"]) - 0.001)

    def test_cli_round_trip(self) -> None:
        out = self.root / "cli_campaign"
        self.assertEqual(main(["vasp-prepare", str(self.root / "supported.extxyz"), "-o", str(out),
                               "--kpoints", "2,2,1", "--potcar", "Fe=Fe", "--incar", "ALGO=All"]), 0)
        incar = next(out.glob("structures/*/m7/AB/INCAR")).read_text()
        self.assertIn("ALGO = All", incar)
        self.assertIn("2 2 1", next(out.glob("structures/*/B/KPOINTS")).read_text())
        for job in (out / "jobs.txt").read_text().split():
            _fake_vasprun(out / job)
        self.assertEqual(main(["vasp-collect", str(out), "-o", str(self.root / "cli_labels")]), 0)


class AseInteropTests(unittest.TestCase):
    def test_reads_ase_written_extxyz(self) -> None:
        try:
            import numpy as np
            from ase import Atoms
            from ase.io import write
        except ImportError:
            self.skipTest("ASE not installed")
        atoms = Atoms("C2Fe", positions=[[0, 0, 0], [1.4, 0, 0], [0.7, 0, 2]], cell=[5, 5, 12], pbc=True)
        atoms.arrays["cluster"] = np.array([0, 0, 1])
        atoms.info.update(multiplicity=5, isomer="B0")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ase.extxyz"
            write(path, atoms)
            (s,) = read_structures(path)
        self.assertEqual(s.cluster_mask(), [False, False, True])
        self.assertEqual(s.info["multiplicity"], 5)
        self.assertEqual(s.info["isomer"], "B0")
        self.assertAlmostEqual(s.cell[2][2], 12.0)


if __name__ == "__main__":
    unittest.main()
