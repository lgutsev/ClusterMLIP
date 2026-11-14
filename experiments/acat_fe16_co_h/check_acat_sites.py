"""Try ACAT's own cluster site finder on the pilot catalyst and compare with ours.

ACAT's ``ClusterAdsorptionSites`` relaxes a surrogate cluster with ``asap3.EMT`` and
classifies the surface with ``asap3.analysis.FullCNA``. asap3 has no Windows build, so
this check could not run on the laptop; run it on a Linux host with ``pip install acat asap3``.

    python check_acat_sites.py
"""
from __future__ import annotations

from collections import Counter

from common import load_catalyst, load_sites


def main() -> None:
    catalyst = load_catalyst()
    ours = load_sites(catalyst)
    print("built-in finder:", ours.counts(), "surface", ours.surface_ids, "interior", ours.interior_ids)
    try:
        from acat.adsorption_sites import ClusterAdsorptionSites
        import asap3  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"ACAT cluster sites unavailable here: {exc}")
    atoms = catalyst.copy()
    atoms.center(vacuum=8.0)
    try:
        cas = ClusterAdsorptionSites(atoms, allow_6fold=False, composition_effect=False)
    except Exception as exc:  # ACAT expects fcc/icosahedral/decahedral-like shapes
        raise SystemExit(f"ACAT could not classify this cluster: {type(exc).__name__}: {exc}")
    sites = cas.get_sites()
    print("ACAT:", dict(Counter((s["site"], s["surface"]) for s in sites)))
    mine = {tuple(sorted(s.indices)) for s in ours.sites}
    theirs = {tuple(sorted(int(i) for i in s["indices"])) for s in sites}
    print("in both:", len(mine & theirs), "only built-in:", len(mine - theirs), "only ACAT:", len(theirs - mine))
    for idx in sorted(theirs - mine):
        print("  only ACAT:", idx)
    for idx in sorted(mine - theirs):
        print("  only built-in:", idx)


if __name__ == "__main__":
    main()
