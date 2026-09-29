"""Step 9: where does MACE-POLAR-1 break?  Compact Fe_n sub-clusters of the Fe16 global
minimum (the n atoms nearest its centre), evaluated at every odd multiplicity up to
the fully polarised limit. Validity: max |predicted atomic spin| <= 4 for Fe."""
import json
import warnings
from pathlib import Path

import numpy as np
from ase.io import read

warnings.filterwarnings("ignore")
import torch  # noqa: E402
from mace.calculators import mace_polar  # noqa: E402

OUT = Path(__file__).parent / "out"
calc = mace_polar("polar-1-m", device="cuda" if torch.cuda.is_available() else "cpu",
                  default_dtype="float64")
fe16 = read(OUT / "all_isomers.extxyz", 0)
order = np.argsort(np.linalg.norm(fe16.positions - fe16.positions.mean(0), axis=1))
rows = []
for n in range(2, 17):
    sub = fe16[order[:n]]
    for m in range(1, 4 * n + 2, 2):
        a = sub.copy()
        a.info = {"charge": 0, "spin": m}
        a.pbc = False
        a.calc = calc
        e = float(a.get_potential_energy())
        s = np.asarray(calc.results["spins"])
        rows.append(dict(n=n, mult=m, E=e, spin_max=float(np.abs(s).max()),
                         spin_sum=float(s.sum())))
    ok = [r for r in rows if r["n"] == n and r["spin_max"] <= 4.0]
    print(f"Fe{n:<2d} valid M: " + (f"{min(r['mult'] for r in ok)}..{max(r['mult'] for r in ok)}"
                                   if ok else "none")
          + f"  (of 1..{4 * n + 1});  ground M among valid: "
          + (str(min(ok, key=lambda r: r['E'])['mult']) if ok else "-"), flush=True)
(OUT / "polar_breakdown.json").write_text(json.dumps(rows, indent=1))
