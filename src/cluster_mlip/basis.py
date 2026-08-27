"""Gaussian ``Gen`` (general/custom) basis set content shared by jobs.py and spin.py.

Every Gaussian input this pipeline writes uses a mixed basis: a simple
Pople-style keyword basis for the light/organic elements, and an explicit
def2-TZVP contraction (with the f-functions dropped -- not needed for this
application) for Fe. Both pieces of text live here, once, so jobs.py and
spin.py can't drift apart on what "the basis" actually is.
"""

from __future__ import annotations

from collections.abc import Iterable

# Current default for non-Fe elements. The user's stated intent is to
# upgrade this to "6-311G*" later once the smaller basis has served its
# purpose; diffuse functions (6-311+G*/++G*) are deliberately not used.
LIGHT_ELEMENT_BASIS = "6-31G*"

# def2-TZVP for Fe, all-electron, with the f-shell removed (generally not
# needed for this application). Pasted verbatim from the user's spec --
# do not reformat/reflow the exponents or coefficients.
FE_DEF2TZVP_NO_F = """\
S    8   1.00
 300784.8463700              0.22806273096D-03
  45088.9705570              0.17681788761D-02
  10262.5163170              0.91927083490D-02
   2905.2897293              0.37355495807D-01
    946.11487137             0.12151108426
    339.87832894             0.28818881468
    131.94425588             0.41126612677
     52.111494077            0.21518583573
S    4   1.00
    329.48839267            -0.24745216477D-01
    101.92332739            -0.11683089050
     16.240462745            0.55293621136
      6.8840675801           0.53601640182
S    2   1.00
     10.470693782           -0.22912708577
      1.7360039648           0.71159319984
S    1   1.00
      0.72577288979          1.0000000
S    1   1.00
      0.11595528203          1.0000000
S    1   1.00
      0.41968227746D-01      1.0000000
P    6   1.00
   1585.3959970              0.23793960179D-02
    375.38006499             0.19253154755D-01
    120.31816501             0.90021836536D-01
     44.788749031            0.25798172356
     17.829278584            0.41492649744
      7.2247153786           0.24207474784
P    3   1.00
     28.143219756           -0.29041755152D-01
      3.8743241412           0.55312260343
      1.5410752281           0.96771136842
P    1   1.00
      0.58285615250          1.0000000
P    1   1.00
      0.1349150              1.0000000
D    4   1.00
     61.996675034            0.11971972255D-01
     17.873732552            0.73210135410D-01
      6.2744782934           0.23103094314
      2.3552337175           0.39910706494
D    1   1.00
      0.85432239901           .41391589765
D    1   1.00
      0.27869254413           .21909269782
D    1   1.00
      0.0910000              1.0000000\
"""


def render_gen_basis(elements: Iterable[str], light_basis: str = LIGHT_ELEMENT_BASIS) -> str:
    """Render a Gaussian Gen-format basis section for the given elements.

    Only emits the element groups actually present: a pure-organic fragment
    gets just the light-element group, a pure-Fe fragment gets just the Fe
    group, and a mixed molecule gets both. Each group is terminated with
    ``****`` per Gaussian's Gen format, and the whole section ends with a
    blank line as Gaussian requires.
    """
    unique = set(elements)
    if not unique:
        raise ValueError("render_gen_basis requires at least one element")
    has_fe = "Fe" in unique
    light = sorted(unique - {"Fe"})

    lines: list[str] = []
    if light:
        lines.append(f"{' '.join(light)}     0")
        lines.append(light_basis)
        lines.append("****")
    if has_fe:
        lines.append("Fe     0")
        lines.extend(FE_DEF2TZVP_NO_F.splitlines())
        lines.append("****")
    lines.append("")
    return "\n".join(lines)
