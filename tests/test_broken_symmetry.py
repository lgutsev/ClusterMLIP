import csv
import itertools
import re

import pytest

from cluster_mlip.broken_symmetry import (
    choose_sites, fragment_assignment, render_bs_input, write_bs_jobs,
)
from cluster_mlip.gaussian import parse_hirshfeld_table
from cluster_mlip.models import Atom, Record
from cluster_mlip.spin import _validated_fragments


def fe16() -> Record:
    # 2.5 Å cubic grid fragment (3x3x2 minus 2 corners): one clear interior-ish site.
    points = [p for p in itertools.product((0, 1, 2), (0, 1, 2), (0, 1))][:16]
    atoms = [Atom("Fe", 2.5 * x + 0.01 * i, 2.5 * y, 2.5 * z) for i, (x, y, z) in enumerate(points)]
    return Record("toy", "toy.out", atoms, charge=0, multiplicity=49, config_type="minimum")


def fragment_multiplicities(text: str) -> list[int]:
    line = next(l for l in text.splitlines() if l.startswith("0 49 "))
    values = [int(v) for v in line.split()[2:]]
    return values[1::2]


@pytest.mark.parametrize("pattern", ["P1", "P2", "P3"])
def test_every_fragment_is_odd_and_validated(pattern):
    record = fe16()
    sites = choose_sites(record)
    text, rows, signed = render_bs_input(record, 49, pattern, sites[pattern])
    mults = fragment_multiplicities(text)
    assert len(mults) == 16
    assert all(abs(m) % 2 == 1 for m in mults), mults  # neutral Fe: 26 electrons -> odd multiplicity
    assert set(abs(m) for m in mults) <= {3, 5, 7}
    assert sum((1 if m > 0 else -1) * (abs(m) - 1) for m in mults) == 48
    assert sum(1 for m in mults if m < 0) == 1
    flipped = sites[pattern][0]
    assert mults[flipped] == -5


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


def test_two_stage_routes():
    record = fe16()
    text, rows, signed = render_bs_input(record, 49, "P0", [])
    assert signed is None
    assert "Fragment" not in text
    assert "5/13" not in text
    stages = text.split("--Link1--")
    assert len(stages) == 2
    assert "Stable=Opt" in stages[0] and "Pop=Hirshfeld" in stages[0] and " SP " in stages[0]
    assert re.search(r"Force .*Geom=Checkpoint Guess=Read", stages[1])
    assert [r["stage_index"] for r in rows] == ["0", "1"]


def test_sites_are_distinct():
    sites = choose_sites(fe16())
    picked = [sites[p][0] for p in ("P1", "P2", "P3")]
    assert len(set(picked)) == 3


def test_write_bs_jobs_manifest(tmp_path):
    out = tmp_path / "bs"
    assert write_bs_jobs([fe16()], out, 49) == 4
    rows = list(csv.DictReader((out / "spin_jobs.csv").open(encoding="utf-8")))
    assert len(rows) == 8
    assert {r["bs_pattern"] for r in rows} == {"P0", "P1", "P2", "P3"}
    for r in rows:
        assert (out / r["input"]).is_file() and r["input_sha256"]
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
