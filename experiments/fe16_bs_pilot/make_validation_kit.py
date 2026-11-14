"""Small validation jobs for the tandem fragment workflow (Fe2 + archived replays).

    python make_validation_kit.py OUT_DIR            # write inputs, spin_jobs.csv, expected.json
    python make_validation_kit.py --check OUT_DIR [LOG_DIR]   # compare returned logs with expectations

Jobs (all fixed geometry, UBPW91/6-311++G*, three links unless noted):
  V1  Fe2 2.02 Å, Fe(0,5)/Fe(0,-5), M=1      archived link-0 syntax (fragment SCFs, no Only)
  V2  as V1 with Guess=(Fragment=2,Only)     Gaussian's documented variant (no fragment SCF)
  V3  Fe2, Fe(+1,6)/Fe(-1,-6), M=1           charge-separated initialization
  V4  Fe2, default guess, M=1, Stable=Opt     control: does Stable=Opt find the BS state alone?
  V5  identical to V1 under another name      reproducibility of the whole chain
  H1  replay of LG_Calcs Fe2O2N2_DimAd_AFM_1 (neutral Fe fragments)
  H2  replay of LG_Calcs Fe2O4H2_FeFe_TS_AFM_1 (Fe(+1) sextets, O2(2-) fragment)

For H1/H2 the archived run (G09 D.01) gives exact targets: the fragment-SCF energies
of link 0, the <S**2> of the guess read in link 1, and the energy of link 1's first
SCF. Link 1 is now Stable=Opt rather than Opt(TS), but its first SCF starts from the
same guess at the same geometry and should agree within ~1e-5 Ha.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

from cluster_mlip.fragment_tandem import (
    Fragment, TandemJob, parse_tandem_log, plan_from_input, render_tandem, validate_fragment_plan,
)
from cluster_mlip.io import write_text_lf

REPO = Path(__file__).resolve().parents[2]
REFS = json.loads((REPO / "tests/data/fragment_tandem/historical_reference.json").read_text(encoding="utf-8"))
FE2 = (("Fe", "Fe"), ((0.0, 0.0, 0.0), (0.0, 0.0, 2.02)))
MEMORY, NPROC = "8GB", 4


def fe2_job(name: str, fragments, guess_only=False, title=""):
    symbols, coords = FE2
    plan = None if fragments is None else validate_fragment_plan(symbols, 0, 1, fragments)
    return TandemJob(name, symbols, coords, 0, 1, plan, guess_only=guess_only,
                     title=title or name, memory=MEMORY, nproc=NPROC)


def replay_job(ref):
    symbols = tuple(a["symbol"] for a in ref["atoms"])
    frags = [Fragment(tuple(i + 1 for i, a in enumerate(ref["atoms"]) if a["fragment"] == k),
                      f["charge"], f["multiplicity"]) for k, f in enumerate(ref["fragments"], start=1)]
    plan = validate_fragment_plan(symbols, ref["charge"], ref["multiplicity"], frags)
    coords = tuple(tuple(float(v) for v in a["xyz"]) for a in ref["atoms"])
    return TandemJob(f"replay-{ref['name']}", symbols, coords, ref["charge"], ref["multiplicity"], plan,
                     title=f"replay of archived {ref['archive_path']}", memory=MEMORY, nproc=NPROC)


def jobs():
    neutral = [Fragment((1,), 0, 5), Fragment((2,), 0, -5)]
    return [
        ("V1", fe2_job("fe2-afm-neutral", neutral), None),
        ("V2", fe2_job("fe2-afm-neutral-guessonly", neutral, guess_only=True), None),
        ("V3", fe2_job("fe2-afm-chargesep", [Fragment((1,), 1, 6), Fragment((2,), -1, -6)]), None),
        ("V4", fe2_job("fe2-c0-default-guess", None), None),
        ("V5", fe2_job("fe2-afm-neutral-repeat", neutral), None),
        *[(f"H{i}", replay_job(ref), ref) for i, ref in enumerate(REFS, start=1)],
    ]


def write(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite {out}")
    (out / "inputs").mkdir(parents=True)
    rows, expected = [], {}
    for label, job, ref in jobs():
        name = f"{label.lower()}__{job.name}.gjf"
        text = render_tandem(job)
        write_text_lf(out / "inputs" / name, text)
        rows.append({"job_id": job.name, "chain_id": job.name, "stage_index": "0", "parent_record_id": label,
                     "bs_pattern": label, "fragment_label": label, "input": f"inputs/{name}",
                     "output": f"{Path(name).stem}.log",
                     "input_sha256": hashlib.sha256(text.encode()).hexdigest()})
        expected[label] = {"input": name, "ideal_guess_s2": None if job.plan is None else job.plan.ideal_guess_s2}
        if ref:
            expected[label].update({
                "archived_fragment_scf_energies": [f["energy"] for f in ref["stage1_fragment_scf"]],
                "archived_unconverged_fragment_scf": ref["stage1_unconverged_fragment_scf"],
                "archived_link1_guess_s2": ref["stage2_guess_s2"],
                "archived_link1_first_scf": ref["stage2_first_scf"],
            })
    with (out / "spin_jobs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_text_lf(out / "expected.json", json.dumps(expected, indent=1) + "\n")
    print(f"wrote {len(rows)} validation inputs to {out}")


def check(kit: Path, logs: Path) -> None:
    expected = json.loads((kit / "expected.json").read_text(encoding="utf-8"))
    found = {p.name: p for p in logs.rglob("*.log")}
    results = {}
    for label, exp in expected.items():
        plan, n = plan_from_input((kit / "inputs" / exp["input"]).read_text(encoding="utf-8"))
        log = found.get(Path(exp["input"]).stem + ".log")
        if log is None:
            print(f"{label}: missing log")
            continue
        r = parse_tandem_log(log.read_text(errors="replace"), plan, force_stage=True)
        results[label] = r
        msg = [f"{label}: {r['status']}", *(f"  issue: {i}" for i in r["issues"])]
        if plan is not None:
            msg.append(f"  link0: {r.get('s1_fragment_scf')} fragment SCFs, {r.get('s1_unconverged_scf')} "
                       f"unconverged, supermolecule SCFs {r.get('s1_supermolecule_scf')}")
            msg.append(f"  link1 guess <S^2> {r.get('s2_guess_s2')} (ideal {exp['ideal_guess_s2']:.3f})")
        msg.append(f"  link1 SCFs {r.get('s2_scf_energies')} instabilities {r.get('s2_instabilities')} "
                   f"final stable {r.get('s2_final_stable')}")
        msg.append(f"  link2 E {r.get('s3_scf_energy')} <S^2> {r.get('s3_s2')} "
                   f"Hirshfeld {[round(v[2], 2) for v in (r.get('s3_hirshfeld') or {}).get('atomic_spins_hirshfeld', [])]}")
        if "archived_link1_first_scf" in exp:
            arch = exp["archived_link1_first_scf"]
            first = (r.get("s2_scf_energies") or [None])[0]
            frag = r.get("s1_fragment_scf_energies") or []
            msg.append(f"  archived fragment SCF energies {exp['archived_fragment_scf_energies']} vs {frag}")
            msg.append(f"  archived link1 guess <S^2> {exp['archived_link1_guess_s2']} vs {r.get('s2_guess_s2')}")
            if first is not None:
                msg.append(f"  archived link1 first SCF {arch['energy']} vs {first}: "
                           f"diff {abs(first - arch['energy']):.2e} Ha")
        print("\n".join(msg))
    if "V1" in results and "V5" in results and results["V1"].get("s3_scf_energy") and results["V5"].get("s3_scf_energy"):
        print(f"reproducibility V1 vs V5: {abs(results['V1']['s3_scf_energy'] - results['V5']['s3_scf_energy']):.2e} Ha")


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        check(Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else Path(sys.argv[2]))
    else:
        write(Path(sys.argv[1]))
