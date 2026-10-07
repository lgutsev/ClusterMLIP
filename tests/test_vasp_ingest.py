import csv
import json
import tempfile
import unittest
from pathlib import Path

from cluster_mlip.cli import main
from cluster_mlip.periodic import read_structures
from cluster_mlip.vasp_ingest import (
    find_job_dirs,
    ingest_vasp_runs,
    moment_pattern,
    parse_outcar_moment_tables,
    read_vasp_job,
)

# Two O support atoms (the first frozen) and two Fe cluster atoms.
SYMBOLS = ["O", "O", "Fe", "Fe"]
BASIS = [(6.0, 0.0, 0.0), (0.0, 6.0, 0.0), (0.0, 0.0, 12.0)]
ZVAL = {"O": 6.0, "Fe": 8.0}
POTCAR = {"O": "PAW_PBE O 08Apr2002", "Fe": "PAW_PBE Fe 06Sep2000"}


def _frac(step: int) -> list[tuple[float, float, float]]:
    return [(0.0, 0.0, 0.1), (0.5, 0.5, 0.1), (0.25, 0.25, 0.3 + 0.001 * step), (0.4, 0.25, 0.3)]


def _forces(step: int) -> list[tuple[float, float, float]]:
    return [(1.0, -1.0, 0.5), (0.1, 0.0, 0.0), (0.0, 0.0, -0.2 / (step + 1)), (0.01, 0.02, 0.03)]


def _vasprun(*, steps: int, nelm: int = 60, sc: list[int] | None = None, ispin: int = 2,
             nupdown: float = -1.0, isif: int = 2, ibrion: int = 2, nsw: int = 50, nelect: float | None = None,
             fe_potcar: str = POTCAR["Fe"], incar_extra: str = "", truncate: bool = False) -> str:
    sc = sc or [5] * steps
    valence = sum(ZVAL[s] for s in SYMBOLS)
    counts = [("O", 2), ("Fe", 2)]
    types = "".join(
        f"<rc><c>{n}</c><c>{el}</c><c>1.0</c><c>{ZVAL[el]}</c><c>  {fe_potcar if el == 'Fe' else POTCAR[el]}  </c></rc>"
        for el, n in counts
    )
    atoms = "".join(f"<rc><c>{s} </c><c>1</c></rc>" for s in SYMBOLS)
    basis = "".join("<v>" + " ".join(f"{x:.8f}" for x in row) + "</v>" for row in BASIS)

    def structure(step: int, name: str = "") -> str:
        pos = "".join("<v>" + " ".join(f"{x:.8f}" for x in p) + "</v>" for p in _frac(step))
        sel = ""
        if name == "initialpos":
            sel = '<varray name="selective" type="logical"><v type="logical"> F F F </v>' + \
                  '<v type="logical"> T T T </v>' * 3 + "</varray>"
        attr = f' name="{name}"' if name else ""
        return (f'<structure{attr}><crystal><varray name="basis">{basis}</varray></crystal>'
                f'<varray name="positions">{pos}</varray>{sel}</structure>')

    calcs = []
    for k in range(steps):
        forces = "".join("<v>" + " ".join(f"{x:.8f}" for x in f) + "</v>" for f in _forces(k))
        e = -50.0 - 0.01 * k
        calcs.append(
            "<calculation>" + "<scstep><energy><i name=\"e_fr_energy\">0.0</i></energy></scstep>" * sc[k]
            + structure(k)
            + f'<varray name="forces">{forces}</varray>'
            + '<varray name="stress"><v>10.0 0.0 0.0</v><v>0.0 10.0 0.0</v><v>0.0 0.0 -5.0</v></varray>'
            + f'<energy><i name="e_fr_energy">{e:.8f}</i><i name="e_wo_entrp">{e:.8f}</i>'
            + f'<i name="e_0_energy">{e + 0.001:.8f}</i></energy></calculation>'
        )
    text = (
        '<?xml version="1.0"?><modeling>'
        '<generator><i name="version" type="string">6.6.1 </i></generator>'
        f'<incar><i type="string" name="PREC">Accurate</i><i name="ENCUT">400.0</i>{incar_extra}</incar>'
        '<kpoints><generation param="Gamma"><v type="int" name="divisions"> 1 1 1 </v></generation></kpoints>'
        '<parameters><separator name="general"><i type="string" name="PREC">accura</i></separator>'
        f'<separator name="electronic"><i name="NELECT">{nelect if nelect is not None else valence}</i>'
        f'<i type="int" name="NELM">{nelm}</i><i type="int" name="ISPIN">{ispin}</i>'
        f'<i name="NUPDOWN">{nupdown}</i><i name="ENMAX">400.0</i>'
        '<i type="int" name="ISMEAR">0</i><i name="SIGMA">0.05</i></separator>'
        f'<separator name="ionic"><i type="int" name="IBRION">{ibrion}</i><i type="int" name="NSW">{nsw}</i>'
        f'<i type="int" name="ISIF">{isif}</i></separator><i type="logical" name="LDAU"> F </i></parameters>'
        f'<atominfo><array name="atoms"><set>{atoms}</set></array>'
        f'<array name="atomtypes"><set>{types}</set></array></atominfo>'
        + structure(0, "initialpos") + "".join(calcs) + "</modeling>\n"
    )
    return text[: len(text) // 2] if truncate else text


def _moment_table(moments: list[float]) -> str:
    rows = "".join(f"    {i}        0.000   0.000   0.000   {m:.3f}\n" for i, m in enumerate(moments, start=1))
    return (" magnetization (x)\n\n# of ion       s       p       d       tot\n"
            "------------------------------------------\n" + rows +
            "--------------------------------------------------\n"
            f"tot          0.000   0.000   0.000   {sum(moments):.3f}\n\n")


def _write_job(job: Path, *, steps: int, moments: list[list[float]] | None, total: list[float] | None,
               converged: bool = True, **kw) -> Path:
    job.mkdir(parents=True)
    (job / "vasprun.xml").write_text(_vasprun(steps=steps, **kw))
    outcar = ""
    if moments is not None:
        tables = moments + [moments[-1]]  # VASP repeats the last table at the end of the run
        outcar += "".join(_moment_table(t) for t in tables)
    if converged:
        outcar += " reached required accuracy - stopping structural energy minimisation\n"
    outcar += " General timing and accounting informations for this job:\n"
    (job / "OUTCAR").write_text(outcar)
    lines = []
    for k in range(steps):
        lines.append("RMM:   1    -0.5E+02   0.1E-05\n")
        mag = f"  mag=    {total[k]:.4f}" if total is not None else ""
        lines.append(f"  {k + 1} F= -.50000000E+02 E0= -.50000000E+02  d E =0.0{mag}\n")
    (job / "OSZICAR").write_text("".join(lines))
    return job


class VaspIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_every_ionic_step_with_aligned_moments_and_frozen_atoms(self) -> None:
        moments = [[0.0, 0.1, 3.0, -2.9], [0.0, 0.1, 3.1, -2.8], [0.0, 0.0, 3.2, -0.2]]
        _write_job(self.root / "runs" / "relax", steps=3, moments=moments, total=[0.2, 0.4, 3.0], nupdown=3.0)
        out = self.root / "out"
        summary = ingest_vasp_runs([self.root / "runs"], out, cluster_elements={"Fe"})
        self.assertEqual(summary["frames"], 3)
        self.assertEqual(summary["jobs_flagged"], {})
        frames = read_structures(out / "frames.extxyz")
        self.assertEqual([f.info["config_type"] for f in frames], ["relaxation_step"] * 2 + ["relaxed_minimum"])
        last = frames[-1]
        self.assertEqual(last.pbc, (True, True, True))
        self.assertEqual(last.cell, BASIS)
        self.assertAlmostEqual(last.positions[2][2], 12.0 * 0.302, places=8)
        self.assertEqual(last.arrays["REF_forces"][0], (1.0, -1.0, 0.5))  # frozen atom keeps its force
        self.assertEqual(last.arrays["fixed"], [True, False, False, False])
        self.assertEqual(last.arrays["cluster"], [0, 0, 1, 1])
        self.assertEqual(last.arrays["vasp_magmom"], [0.0, 0.0, 3.2, -0.2])  # step 3, not the repeat
        self.assertEqual(frames[0].arrays["vasp_magmom"], moments[0])
        self.assertEqual(last.info["moment_pattern"], "Fe:+0")
        self.assertEqual(frames[0].info["moment_pattern"], "Fe:+-")
        self.assertEqual(last.info["spin_constraint"], "NUPDOWN")
        self.assertEqual(last.info["multiplicity"], 4)
        self.assertEqual(last.info["charge"], 0)
        self.assertAlmostEqual(last.info["REF_energy"], -50.02)
        self.assertAlmostEqual(last.info["energy_sigma0"], -50.019)
        self.assertNotIn("REF_stress", last.info)  # ISIF 2: not a training target
        self.assertIn("ENCUT=400", last.info["label_level"])
        self.assertIn("PREC=accurate", last.info["label_level"])
        self.assertNotIn("POTCAR", last.info["label_level"])

    def test_released_spin_gets_multiplicity_only_near_an_integer(self) -> None:
        _write_job(self.root / "a", steps=1, moments=[[0, 0, 4.0, 4.0]], total=[8.0004])
        _write_job(self.root / "b", steps=1, moments=[[0, 0, 3.6, 3.7]], total=[7.31])
        out = self.root / "out"
        ingest_vasp_runs([self.root / "a", self.root / "b"], out)
        a, b = read_structures(out / "frames.extxyz")
        self.assertEqual((a.info["spin_constraint"], a.info["multiplicity"]), ("free", 9))
        self.assertEqual(b.info["spin_constraint"], "free_fractional")
        self.assertNotIn("multiplicity", b.info)
        self.assertAlmostEqual(b.info["total_magnetization"], 7.31)

    def test_scf_failures_dropped_and_final_failure_flagged(self) -> None:
        _write_job(self.root / "job", steps=3, moments=None, total=None, ispin=1, nelm=40, sc=[40, 5, 40])
        out = self.root / "out"
        summary = ingest_vasp_runs([self.root / "job"], out)
        self.assertEqual(summary["frames"], 1)
        self.assertIn("final SCF hit NELM", summary["jobs_flagged"]["job"])
        (row,) = csv.DictReader((out / "jobs.csv").open())
        self.assertEqual(row["scf_failed_steps"], "2")
        (frame,) = read_structures(out / "frames.extxyz")
        self.assertEqual(frame.info["ionic_step"], 1)
        self.assertEqual(frame.info["multiplicity"], 1)
        self.assertNotIn("vasp_magmom", frame.arrays)

    def test_misaligned_moment_tables_are_not_attached(self) -> None:
        job = _write_job(self.root / "job", steps=3, moments=[[0, 0, 3.0, 3.0]], total=[6.0] * 3)
        run = read_vasp_job(job)
        self.assertIsNone(run.atomic_moments)
        summary = ingest_vasp_runs([job], self.root / "out")
        self.assertIn("no per-step moment tables", summary["jobs_flagged"]["job"])

    def test_variable_cell_stress_in_ase_convention_and_charge_from_nelect(self) -> None:
        _write_job(self.root / "bulk", steps=1, moments=None, total=None, ispin=1, isif=3,
                   nelect=sum(ZVAL[s] for s in SYMBOLS) + 1)
        ingest_vasp_runs([self.root / "bulk"], self.root / "out")
        (frame,) = read_structures(self.root / "out" / "frames.extxyz")
        stress = [float(x) for x in str(frame.info["REF_stress"]).split()]
        self.assertAlmostEqual(stress[0], -1.0 / 160.21766208, places=9)  # 10 kB compressive -> negative
        self.assertAlmostEqual(stress[8], 0.5 / 160.21766208, places=9)
        self.assertEqual(frame.info["charge"], -1)

    def test_unfinished_relaxation_and_truncated_run_are_flagged(self) -> None:
        _write_job(self.root / "runs" / "short", steps=2, moments=None, total=None, ispin=1, converged=False)
        _write_job(self.root / "runs" / "cut", steps=2, moments=None, total=None, ispin=1, truncate=True)
        summary = ingest_vasp_runs([self.root / "runs"], self.root / "out")
        self.assertIn("relaxation did not reach EDIFFG", summary["jobs_flagged"]["short"])
        self.assertTrue(summary["jobs_flagged"]["cut"].startswith("unreadable"))
        frames = read_structures(self.root / "out" / "frames.extxyz")
        self.assertEqual([f.info["config_type"] for f in frames], ["relaxation_step"] * 2)

    def test_level_and_potcar_consistency_reported_across_jobs(self) -> None:
        _write_job(self.root / "r" / "a", steps=1, moments=None, total=None, ispin=1)
        _write_job(self.root / "r" / "b", steps=1, moments=None, total=None, ispin=1, fe_potcar="PAW_PBE Fe_pv 02Aug2007")
        _write_job(self.root / "r" / "c", steps=1, moments=None, total=None, ispin=1,
                   incar_extra='<i type="int" name="IVDW">12</i>')
        summary = ingest_vasp_runs([self.root / "r"], self.root / "out")
        self.assertTrue(summary["mixed_levels"])
        self.assertEqual(len(summary["levels_of_theory"]), 2)
        self.assertEqual(set(summary["potcar_conflicts"]), {"Fe"})

    def test_frame_selection_and_job_discovery(self) -> None:
        _write_job(self.root / "r" / "x" / "relax", steps=5, moments=None, total=None, ispin=1)
        self.assertEqual([n for n, _ in find_job_dirs([self.root / "r"])], ["x/relax"])
        s = ingest_vasp_runs([self.root / "r"], self.root / "o1", every=3)
        self.assertEqual(s["frames"], 3)  # steps 0, 3 and the final 4
        s = ingest_vasp_runs([self.root / "r"], self.root / "o2", final_only=True)
        self.assertEqual(s["frames"], 1)

    def test_outcar_tables_and_pattern_helpers(self) -> None:
        path = self.root / "OUTCAR"
        path.write_text(_moment_table([1.0, -1.0]) + _moment_table([0.5]))
        self.assertEqual(parse_outcar_moment_tables(path, 2), [[1.0, -1.0]])
        self.assertEqual(moment_pattern(["Fe", "Co", "Fe"], [2.0, -1.0, 0.1], [True, True, True]), "Fe:+0 Co:-")

    def test_cli(self) -> None:
        _write_job(self.root / "job", steps=2, moments=[[0, 0, 3, 3], [0, 0, 3, 3]], total=[6.0, 6.0])
        out = self.root / "cli_out"
        code = main(["vasp-ingest", str(self.root / "job"), "-o", str(out), "--cluster-elements", "Fe"])
        self.assertEqual(code, 0)
        summary = json.loads((out / "ingest_summary.json").read_text())
        self.assertEqual(summary["frames"], 2)
        self.assertEqual(summary["cluster_elements"], ["Fe"])


if __name__ == "__main__":
    unittest.main()
