"""Tandem fragment-guess workflow: validation, generated-input inspection, and log diagnostics.

The regression cases come from two archived G09 D.01 AFM jobs (LG_Calcs, 2019):
Fe2O2N2_DimAd_AFM_1 (neutral Fe fragments, one unconverged fragment SCF) and
Fe2O4H2_FeFe_TS_AFM_1 (Fe(+1) sextets with an O2(2-) fragment; completed TS).
"""
import json
import re
from pathlib import Path

import pytest

from cluster_mlip.fragment_tandem import (
    Fragment, StateTolerances, TandemJob, compare_states, fragment_route, inspect_tandem_input,
    parse_tandem_log, render_tandem, route_tokens, validate_fragment_plan, ARCHIVED_LEVEL,
)

DATA = Path(__file__).parent / "data" / "fragment_tandem"
REFS = {r["name"]: r for r in json.loads((DATA / "historical_reference.json").read_text(encoding="utf-8"))}


def historical_plan(ref):
    symbols = [a["symbol"] for a in ref["atoms"]]
    frags = [Fragment(tuple(i + 1 for i, a in enumerate(ref["atoms"]) if a["fragment"] == k),
                      f["charge"], f["multiplicity"]) for k, f in enumerate(ref["fragments"], start=1)]
    return validate_fragment_plan(symbols, ref["charge"], ref["multiplicity"], frags)


def historical_job(ref, **kw):
    plan = historical_plan(ref)
    coords = tuple(tuple(float(v) for v in a["xyz"]) for a in ref["atoms"])
    return TandemJob(ref["name"], plan.symbols, coords, ref["charge"], ref["multiplicity"], plan, **kw)


def norm_tokens(route):
    out = set()
    for tok in route_tokens(route.replace("# ", "#").replace(" #", " ")):
        tok = tok.lower()
        if tok.startswith("iop("):
            out |= {f"iop:{x}" for x in tok[4:-1].split(",")}
        else:
            out.add(tok)
    return out


# ------------------------------------------------------------------ validation

@pytest.mark.parametrize("name", sorted(REFS))
def test_archived_fragment_guesses_validate(name):
    plan = historical_plan(REFS[name])
    assert sum(f.charge for f in plan.fragments) == 0
    assert sum(f.signed_unpaired for f in plan.fragments) == REFS[name]["multiplicity"] - 1


@pytest.mark.parametrize("name", sorted(REFS))
def test_ideal_guess_s2_matches_archived_link1_guess(name):
    ref = REFS[name]
    plan = historical_plan(ref)
    assert abs(ref["stage2_guess_s2"] - plan.ideal_guess_s2) < plan.guess_s2_tolerance
    assert abs(ref["stage2_guess_s2"] - plan.ideal_guess_s2) < 0.01


FE2 = ["Fe", "Fe"]


@pytest.mark.parametrize("fragments, charge, mult, message", [
    ([Fragment((1,), 0, 5), Fragment((2,), 0, -5)], 0, 3, "signed fragment spins"),
    ([Fragment((1,), 0, 4), Fragment((2,), 0, -4)], 0, 1, "parity"),
    ([Fragment((1,), 1, 5), Fragment((2,), -1, -5)], 0, 1, "parity"),
    ([Fragment((1,), 1, 6), Fragment((2,), 0, -6)], 0, 1, "parity"),
    ([Fragment((1,), 1, 6), Fragment((2,), 1, -6)], 0, 1, "sum to 2"),
    ([Fragment((1,), 0, 5), Fragment((1,), 0, -5)], 0, 1, "in fragments 1 and 2"),
    ([Fragment((1,), 0, 5), Fragment((3,), 0, -5)], 0, 1, "outside"),
    ([Fragment((1,), 0, 5), Fragment((), 0, -5)], 0, 1, "no atoms"),
    ([Fragment((1,), 0, -1), Fragment((2,), 0, 1)], 0, 1, "no spin orientation"),
    ([Fragment((1,), 0, 29), Fragment((2,), 0, -29)], 0, 1, "needs more than"),
    ([Fragment((1,), 1.0, 6), Fragment((2,), -1, -6)], 0, 1, "integer"),
    ([Fragment((1,), True, 6), Fragment((2,), -1, -6)], 0, 1, "integer"),
    ([Fragment((1,), 0, 5)], 0, 5, "at least two"),
    ([Fragment((1,), 0, 5), Fragment((2,), 0, -5)], 0, 2, "impossible"),
])
def test_invalid_fragment_guesses_are_rejected(fragments, charge, mult, message):
    with pytest.raises(ValueError, match=message):
        validate_fragment_plan(FE2, charge, mult, fragments)


def test_missing_atom_is_rejected():
    with pytest.raises(ValueError, match="not assigned"):
        validate_fragment_plan(["Fe", "Fe", "O"], 0, 1, [Fragment((1,), 0, 5), Fragment((2,), 0, -5)])


def test_charge_separated_fe2_is_valid_and_not_an_oxidation_state():
    plan = validate_fragment_plan(FE2, 0, 1, [Fragment((1,), 1, 6), Fragment((2,), -1, -6)])
    assert plan.electrons == (25, 27)
    assert plan.ideal_guess_s2 == 5.0


# ------------------------------------------------------- historical regression

@pytest.mark.parametrize("name", sorted(REFS))
def test_generator_reproduces_archived_stage0_and_documents_stage1_deviations(name):
    ref = REFS[name]
    job = historical_job(ref)
    text = render_tandem(job)
    stages = text.split("--Link1--\n")
    assert len(stages) == 3
    # Stage 0: identical keyword set to the archived fragment stage.
    route0 = next(l for l in stages[0].splitlines() if l.startswith("#"))
    assert norm_tokens(route0) == norm_tokens(ref["routes"][0])
    # Fragment charge/multiplicity sequence and atom->fragment map as archived.
    cm = stages[0].split("\n\n")[2].splitlines()[0].split()
    assert [int(v) for v in cm] == [ref["charge"], ref["multiplicity"]] + [
        v for f in ref["fragments"] for v in (f["charge"], f["multiplicity"])]
    labels = [int(m) for m in re.findall(r"(?m)^[A-Z][a-z]?\(Fragment=(\d+)\)", stages[0])]
    assert labels == [a["fragment"] for a in ref["atoms"]]
    # One checkpoint for every link, as archived; no %oldchk.
    assert len(set(re.findall(r"%chk=(\S+)", text))) == 1 and "%oldchk" not in text
    # Link 1: archived Geom=Checkpoint/Guess=Read kept; the job-type change is the documented deviation.
    archived = norm_tokens(ref["routes"][1])
    ours = norm_tokens(next(l for l in stages[1].splitlines() if l.startswith("#")))
    assert {"geom=checkpoint", "guess=read"} <= archived & ours
    assert archived - ours == {"opt(ts,noeigentest,calcfc)", "freq"}
    assert ours - archived == {"stable=opt", "pop=hirshfeld"}
    assert stages[1].strip().splitlines()[-1] == f"{ref['charge']} {ref['multiplicity']}"


@pytest.mark.parametrize("name", sorted(REFS))
def test_archived_logs_show_fragment_only_stage0_and_intact_guess(name):
    ref = REFS[name]
    plan = historical_plan(ref)
    text = (DATA / f"{name}.log.excerpt").read_text(encoding="utf-8")
    result = parse_tandem_log(text, plan, force_stage=False)
    assert result["s1_fragment_scf"] == len(ref["fragments"])
    assert result["s1_supermolecule_scf"] == 0
    assert result["s1_unconverged_scf"] == ref["stage1_unconverged_fragment_scf"]
    assert result["s2_guess_from_checkpoint"]
    assert result["s2_guess_s2"] == pytest.approx(ref["stage2_guess_s2"])
    assert result["s2_scf_energies"][0] == pytest.approx(ref["stage2_first_scf"]["energy"])
    assert not any("guess" in issue for issue in result["issues"])
    # The archived link 1 was an optimization, not Stable=Opt: the parser must say so.
    assert any("stable wavefunction" in issue for issue in result["issues"])


def test_archived_multiplicity_change_across_link1_is_flagged():
    ref = REFS["Fe2O2N2_DimAd_AFM_1"]
    text = (DATA / f"{ref['name']}.log.excerpt").read_text(encoding="utf-8")
    # The pattern of New/Fe2O2N2_DimAd_AFM_1.out: stage 0 at M=1, link 1 at M=3.
    head, tail = text.split("Structure from the checkpoint file")
    tail = tail.replace("Charge =  0 Multiplicity = 1", "Charge =  0 Multiplicity = 3", 1)
    result = parse_tandem_log(head + "Structure from the checkpoint file" + tail, historical_plan(ref),
                              force_stage=False)
    assert any("differs from stage 0" in issue for issue in result["issues"])


# ------------------------------------------------------------ input inspection

def fe2_job(**kw):
    plan = validate_fragment_plan(FE2, 0, 1, [Fragment((1,), 0, 5), Fragment((2,), 0, -5)])
    return TandemJob("fe2-afm", plan.symbols, ((0.0, 0.0, 0.0), (0.0, 0.0, 2.02)), 0, 1, plan, **kw)


def test_rendered_fe2_input_passes_inspection_and_layout():
    job = fe2_job()
    text = render_tandem(job)
    assert inspect_tandem_input(text, job).problems == []
    assert inspect_tandem_input(text).problems == []
    assert text.count("--Link1--") == 2
    assert "Guess=(Fragment=2)" in text and "Fragment=2,Only" not in text
    s0, s1, s2 = text.split("--Link1--\n")
    assert "Stable" not in s0 and "Stable=Opt" in s1 and "Stable" not in s2
    assert all("IOP(5/13=1,5/36=1,8/11=1)" in s for s in (s0, s1, s2))
    assert not re.search(r"\bOpt\b(?!=)", text.replace("Stable=Opt", ""))
    assert s0.splitlines()[0] == s1.splitlines()[0] == s2.splitlines()[0] == "%chk=fe2-afm.chk"


def test_two_stage_and_guess_only_variants():
    job = fe2_job(force_stage=False, guess_only=True)
    text = render_tandem(job)
    assert text.count("--Link1--") == 1 and "Guess=(Fragment=2,Only)" in text
    assert inspect_tandem_input(text, job).problems == []


def test_control_job_has_no_fragment_stage():
    job = TandemJob("fe2-c0", ("Fe", "Fe"), ((0.0, 0.0, 0.0), (0.0, 0.0, 2.02)), 0, 1, None)
    text = render_tandem(job)
    assert "Fragment" not in text and text.count("--Link1--") == 1
    assert inspect_tandem_input(text, job).problems == []


MUTATIONS = {
    "stable in stage 0": (lambda t: t.replace(" Guess=(Fragment=2)", " Guess=(Fragment=2) Stable=Opt", 1), "Stable"),
    "no Guess=Read": (lambda t: t.replace("Stable=Opt Pop=Hirshfeld IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine Geom=Checkpoint Guess=Read",
                                           "Stable=Opt Pop=Hirshfeld IOP(5/13=1,5/36=1,8/11=1) Int=UltraFine Geom=Checkpoint"), "Guess=Read"),
    "allcheck": (lambda t: t.replace("Geom=Checkpoint", "Geom=AllCheck"), "Geom=Checkpoint"),
    "multiplicity changes": (lambda t: t[::-1].replace("1 0", "3 0", 1)[::-1], "changes across Link1"),
    "different chk": (lambda t: t.replace("%chk=fe2-afm.chk", "%chk=other.chk", 1), "differ"),
    "oldchk": (lambda t: t.replace("%chk=fe2-afm.chk\n%mem", "%oldchk=x.chk\n%chk=fe2-afm.chk\n%mem", 1), "%oldchk"),
    "5/13 dropped": (lambda t: t.replace("Stable=Opt Pop=Hirshfeld IOP(5/13=1,5/36", "Stable=Opt Pop=Hirshfeld IOP(5/36"), "5/13=1) is missing"),
    "nuclear optimization": (lambda t: t.replace("Force Pop", "Opt Force Pop"), "job type"),
    "coordinates repeated": (lambda t: t.rstrip("\n") + "\nFe 0.0 0.0 0.0\n\n", "repeat coordinates"),
    "level differs": (lambda t: t[::-1].replace("*G++113-6/19WPBU", "*G++113-6/PYL3BU", 1)[::-1], "level of theory"),
    "missing label": (lambda t: t.replace("Fe(Fragment=2)    ", "Fe                ", 1), "(Fragment=k)"),
    "fragment count": (lambda t: t.replace("Guess=(Fragment=2)", "Guess=(Fragment=3)"), "integers"),
    "geometry changed": (lambda t: t.replace(" 2.020000000000", " 2.030000000000"), "coordinates differ"),
    "crlf": (lambda t: t.replace("\n", "\r\n"), "CR"),
    "stage dropped": (lambda t: t.split("--Link1--")[0] + "--Link1--" + t.split("--Link1--")[2], "expected 3"),
}


@pytest.mark.parametrize("label", sorted(MUTATIONS))
def test_inspection_catches_structural_defects(label):
    mutate, message = MUTATIONS[label]
    job = fe2_job()
    bad = mutate(render_tandem(job))
    problems = inspect_tandem_input(bad, job).problems
    assert problems, label
    assert any(message in p for p in problems), problems


def test_route_tokens_keep_option_lists_whole():
    assert route_tokens(fragment_route(ARCHIVED_LEVEL, 16)) == [
        "UBPW91/6-311++G*", "SCF=(VShift=5,NoIncFock,MaxCyc=200,Tight,NoVarAcc)", "NoSymm", "SP",
        "IOP(5/13=1,5/36=1,8/11=1)", "Int=UltraFine", "Guess=(Fragment=16)"]


# ------------------------------------------------------------- log diagnostics

def _orient(z):
    return (" Input orientation:\n ---------------------------------------------------------------------\n"
            " Center     Atomic     Atomic              Coordinates (Angstroms)\n"
            " Number     Number      Type              X           Y           Z\n"
            " ---------------------------------------------------------------------\n"
            "    1         26             0        0.000000    0.000000    0.000000\n"
            f"    2         26             0        0.000000    0.000000    {z:.6f}\n"
            " ---------------------------------------------------------------------\n")


def _route_block(route):
    return f"\n ----------------------------------------------------------------------\n {route}\n" \
           " ----------------------------------------------------------------------\n"


def _scf(e, cycles, s2):
    return (f" SCF Done:  E(UB-PW91) =  {e:.8f}     A.U. after   {cycles} cycles\n"
            f" S**2 before annihilation     {s2:.4f},   after     {s2:.4f}\n")


def _pops(s):
    return (" Mulliken charges and spin densities:\n               1          2\n"
            f"     1  Fe   0.000000   {s:.6f}\n     2  Fe   0.000000  {-s:.6f}\n"
            " Sum of Mulliken charges =   0.00000   0.00000\n"
            " Hirshfeld charges, spin densities, dipoles, and CM5 charges using IRadAn=      4:\n"
            "              Q-H        S-H        Dx         Dy         Dz        Q-CM5\n"
            f"     1  Fe   0.000000   {s:.6f}   0.000000   0.000000   0.000000   0.000000\n"
            f"     2  Fe   0.000000  {-s:.6f}   0.000000   0.000000   0.000000   0.000000\n"
            "   Tot   0.000000   0.000000   0.000000   0.000000   0.000000   0.000000\n")


def _forces(f):
    return (" -------------------------------------------------------------------\n"
            " Center     Atomic                   Forces (Hartrees/Bohr)\n"
            " Number     Number              X              Y              Z\n"
            " -------------------------------------------------------------------\n"
            f"      1       26           0.000000000    0.000000000    {f:.9f}\n"
            f"      2       26           0.000000000    0.000000000   {-f:.9f}\n"
            " -------------------------------------------------------------------\n")


NORMAL = " Normal termination of Gaussian 09.\n"


def synthetic_log(job, *, e=-2526.5, guess_s2=4.0, supermolecule=False, unstable_first=False,
                  final_stable=True, e3=None, z3=2.02, s=3.0, f=0.01, unconverged_link1=False):
    s0 = (_route_block(job.routes()[0]) + " Charge =  0 Multiplicity = 1 in supermolecule\n"
          " Charge =  0 Multiplicity = 5 in fragment      1.\n Charge =  0 Multiplicity =-5 in fragment      2.\n"
          + _orient(2.02))
    for _ in range(2):
        s0 += " Fragment guess: doing MCBS calculation for fragment   1\n" + _scf(-1263.0, 30, 6.03)
    if supermolecule:
        s0 += _scf(e, 40, 4.0)
    s1 = (_route_block(job.routes()[1]) + " Charge =  0 Multiplicity = 1\n" + _orient(2.02)
          + ' Initial guess from the checkpoint file:  "fe2-afm.chk"\n'
          + f" Initial guess <Sx>= 0.0000 <Sy>= 0.0000 <Sz>= 0.0000 <S**2>= {guess_s2:.4f} S= 1.5\n")
    if unstable_first:
        s1 += _scf(e + 0.01, 40, 3.2) + " The wavefunction has an internal instability.\n"
    if unconverged_link1:  # what IOP(5/13=1) lets through: the SCF stops and the job goes on
        s1 += " >>>>>>>>>> Convergence criterion not met.\n"
    s1 += _scf(e, 20, 3.9)
    s1 += (" The wavefunction is stable under the perturbations considered.\n" if final_stable
           else " The wavefunction has an internal instability.\n") + _pops(s)
    s2 = (_route_block(job.routes()[2]) + " Charge =  0 Multiplicity = 1\n" + _orient(z3)
          + _scf(e if e3 is None else e3, 2, 3.9) + _pops(s) + _forces(f))
    return s0 + NORMAL + s1 + NORMAL + s2 + NORMAL


def test_parser_accepts_a_clean_tandem():
    job = fe2_job()
    r = parse_tandem_log(synthetic_log(job), job.plan)
    assert r["status"] == "ok", r["issues"]
    assert r["s1_fragment_scf"] == 2 and r["s1_supermolecule_scf"] == 0
    assert r["stable_opt_changed_state"] is False
    assert r["s3_hirshfeld"]["atomic_spins_hirshfeld"][0][2] == 3.0
    assert len(r["forces_ev_ang"]) == 2


@pytest.mark.parametrize("kwargs, message", [
    ({"guess_s2": 0.0}, "guess <S**2>"),
    ({"supermolecule": True}, "supermolecule SCF"),
    ({"final_stable": False}, "stable wavefunction"),
    ({"e3": -2526.4}, "did not survive"),
    ({"z3": 2.10}, "geometry moved"),
    ({"unconverged_link1": True}, "did not converge"),
])
def test_parser_flags_lost_or_changed_states(kwargs, message):
    job = fe2_job()
    r = parse_tandem_log(synthetic_log(job, **kwargs), job.plan)
    assert r["status"] == "check"
    assert any(message in issue for issue in r["issues"]), r["issues"]


def test_stable_opt_change_is_recorded_not_hidden():
    job = fe2_job()
    r = parse_tandem_log(synthetic_log(job, unstable_first=True), job.plan)
    assert r["status"] == "ok", r["issues"]
    assert r["stable_opt_changed_state"] is True
    assert r["s2_instabilities"] == ["internal"]
    assert r["stable_opt_dE_meV"] == pytest.approx(-0.01 * 27211.386, rel=1e-6)


def test_truncated_log_is_not_ok():
    job = fe2_job()
    text = synthetic_log(job)
    r = parse_tandem_log(text[: text.rindex("Normal termination")], job.plan)
    assert r["status"] == "check"


def test_state_comparison_separates_same_distinct_and_ambiguous():
    job = fe2_job()
    a = parse_tandem_log(synthetic_log(job), job.plan)
    same = parse_tandem_log(synthetic_log(job, e=-2526.5 + 1e-6), job.plan)
    other = parse_tandem_log(synthetic_log(job, e=-2526.49, s=1.0, f=0.05), job.plan)
    energy_only = parse_tandem_log(synthetic_log(job, e=-2526.49), job.plan)
    assert compare_states(a, same)["verdict"] == "same"
    distinct = compare_states(a, other)
    assert distinct["verdict"] == "distinct" and distinct["dE_meV"] > 200
    # Same spins and forces but 272 meV apart: not counted as a new state without review.
    assert compare_states(a, energy_only)["verdict"] == "ambiguous"
    loose = StateTolerances(energy_ev=1.0)
    assert compare_states(a, energy_only, loose)["verdict"] == "same"


@pytest.mark.parametrize("bad", [{"charge": 1.7}, {"charge": True}, {"multiplicity": 6.9}, {"atoms": [1.0]}])
def test_spin_fragment_validator_does_not_coerce(bad):
    from cluster_mlip.models import Atom, Record
    from cluster_mlip.spin import _validated_fragments
    record = Record("x", "x", [Atom("Fe", 0, 0, 0), Atom("Fe", 0, 0, 2)], 0, 1, "minimum")
    first = {"atoms": [1], "charge": 1, "multiplicity": 6, **bad}
    spec = {"target_multiplicity": 1, "fragments": [
        first, {"atoms": [2], "charge": -1, "multiplicity": 6, "orientation": "beta"}]}
    with pytest.raises(ValueError, match="integer"):
        _validated_fragments(record, spec)
