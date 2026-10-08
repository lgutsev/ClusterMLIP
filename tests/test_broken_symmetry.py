import csv
import itertools
import re

import pytest

from cluster_mlip.broken_symmetry import choose_sites, fragment_assignment, make_job, write_bs_jobs
from cluster_mlip.fragment_tandem import render_tandem
from cluster_mlip.gaussian import parse_hirshfeld_table
from cluster_mlip.models import Atom, Record
from cluster_mlip.spin import _validated_fragments


def fe16() -> Record:
    # 2.5 Å cubic grid fragment (3x3x2 minus 2 corners): one clear interior-ish site.
    points = [p for p in itertools.product((0, 1, 2), (0, 1, 2), (0, 1))][:16]
    atoms = [Atom("Fe", 2.5 * x + 0.01 * i, 2.5 * y, 2.5 * z) for i, (x, y, z) in enumerate(points)]
    return Record("toy", "toy.out", atoms, charge=0, multiplicity=49, config_type="minimum")


def stage0_pairs(text: str) -> list[tuple[int, int]]:
    line = next(l for l in text.splitlines() if l.startswith("0 49 "))
    values = [int(v) for v in line.split()[2:]]
    return list(zip(values[0::2], values[1::2]))


@pytest.mark.parametrize("pattern", ["P0", "P1", "P2", "P3"])
def test_neutral_patterns_validate_and_flip_the_chosen_site(pattern):
    record = fe16()
    sites = choose_sites(record)
    job, signed, charges = make_job(record, 49, pattern, sites)
    text = render_tandem(job)
    pairs = stage0_pairs(text)
    assert len(pairs) == 16 and charges == {}
    assert all(q == 0 for q, _ in pairs)
    assert all(abs(m) % 2 == 1 and abs(m) in (3, 5, 7) for _, m in pairs)  # neutral Fe: 26 e
    assert sum((1 if m > 0 else -1) * (abs(m) - 1) for _, m in pairs) == 48
    negative = [i for i, (_, m) in enumerate(pairs) if m < 0]
    assert negative == sites[pattern]
    if pattern == "P0":
        assert job.plan.n_beta_unpaired == 0 and job.plan.ideal_guess_s2 == 600.0
    else:
        assert pairs[sites[pattern][0]][1] == -5 and job.plan.ideal_guess_s2 == 604.0


@pytest.mark.parametrize("pattern", ["P0Q", "P1Q"])
def test_charge_separated_patterns_conserve_charge_and_parity(pattern):
    record = fe16()
    sites = choose_sites(record)
    job, signed, charges = make_job(record, 49, pattern, sites)
    pairs = stage0_pairs(render_tandem(job))
    central, surface = sites["_central"][0], sites["_surface_a"][0]
    assert charges == {central: -1, surface: +1}
    assert sum(q for q, _ in pairs) == 0
    assert pairs[central][0] == -1 and pairs[surface][0] == +1
    assert abs(pairs[central][1]) % 2 == 0 and abs(pairs[surface][1]) % 2 == 0  # 27 and 25 electrons
    assert (pairs[central][1] < 0) == (pattern == "P1Q")


def test_control_pattern_has_no_fragments():
    record = fe16()
    job, signed, _ = make_job(record, 49, "C0", choose_sites(record))
    text = render_tandem(job)
    assert signed is None and "Fragment" not in text and text.count("--Link1--") == 1


def test_assignment_rejects_impossible_and_wrong_parity():
    record = fe16()
    with pytest.raises(ValueError):
        fragment_assignment(record, 48, [0])  # even M has odd total unpaired -> impossible
    with pytest.raises(ValueError):
        fragment_assignment(record, 3, [0, 1, 2])  # too low for m>=3 fragments


def test_validator_rejects_a_quartet_fragment():
    record = fe16()
    spec = {"target_multiplicity": 49, "fragments": [
        {"atoms": [i + 1], "charge": 0, "multiplicity": 4 if i == 0 else 3, "orientation": "alpha"}
        for i in range(16)
    ]}
    with pytest.raises(ValueError, match="parity"):
        _validated_fragments(record, spec)


def test_three_link_tandem_routes():
    record = fe16()
    job, _, _ = make_job(record, 49, "P1", choose_sites(record))
    stages = render_tandem(job).split("--Link1--")
    assert len(stages) == 3
    assert "Guess=(Fragment=16)" in stages[0] and " SP " in stages[0] and "Stable" not in stages[0]
    assert re.search(r"Stable=Opt Pop=Hirshfeld .*Geom=Checkpoint Guess=Read", stages[1])
    assert re.search(r"Force Pop=Hirshfeld .*Geom=Checkpoint Guess=Read", stages[2])
    assert "5/13" not in stages[1] + stages[2]


def test_sites_are_distinct():
    sites = choose_sites(fe16())
    picked = [sites[p][0] for p in ("P1", "P2", "P3")]
    assert len(set(picked)) == 3


def test_write_bs_jobs_manifest(tmp_path):
    out = tmp_path / "bs"
    assert write_bs_jobs([fe16()], out, 49, reference_spins={"toy": {1: 3.5}}) == 4
    rows = list(csv.DictReader((out / "spin_jobs.csv").open(encoding="utf-8")))
    assert len(rows) == 12
    assert {r["bs_pattern"] for r in rows} == {"P0", "P1", "P2", "P3"}
    assert {r["stage_kind"] for r in rows} == {"fragment_init", "stable_opt", "force"}
    for r in rows:
        assert (out / r["input"]).is_file() and r["input_sha256"]
    sites = list(csv.DictReader((out / "sites.csv").open(encoding="utf-8")))
    assert len(sites) == 16 and sites[0]["reference_mulliken_spin"] == "3.500"
    assert sum(1 for s in sites if s["flipped_in"]) == 3
    with pytest.raises(RuntimeError):
        write_bs_jobs([fe16()], out, 49)


HIRSHFELD = """
 Hirshfeld charges, spin densities, dipoles, and CM5 charges using IRadAn=      4:
              Q-H        S-H        Dx         Dy         Dz        Q-CM5
     1  Fe   0.012345   3.123456   0.001000  -0.002000   0.003000   0.020000
     2  Fe  -0.012345  -3.500000   0.000000   0.000000   0.000000  -0.020000
   Tot   0.000000  -0.376544   0.001000  -0.002000   0.003000   0.000000
"""


def test_parse_hirshfeld_separates_spins_and_cm5_charges():
    table = parse_hirshfeld_table("noise\n" + HIRSHFELD)
    assert table["atomic_spins_hirshfeld"] == [[1, "Fe", 3.123456], [2, "Fe", -3.5]]
    assert table["atomic_charges_hirshfeld"][1][2] == -0.012345
    assert table["atomic_charges_cm5"] == [[1, "Fe", 0.02], [2, "Fe", -0.02]]
    assert parse_hirshfeld_table("no table here") is None
