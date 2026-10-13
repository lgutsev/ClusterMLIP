"""Computed geometry for the landed Fe16 clusters, next to the stored s2 metrics.

Reads only committed outputs (gas_relaxed / supported_* extxyz, results.json)
and writes out/geometry_check.json. Nothing in results.json is changed: the
stored values keep their original definitions and are copied under "stored"
with those definitions spelled out; the recomputed values sit beside them.

    python examples/supported_fe16_tem/s12_geometry_check.py

Needs `pip install -e .[geometry]`. Geometry only: nothing here establishes
spin or electronic state, oxidation state or energy basin.
"""
from __future__ import annotations

import json
from pathlib import Path

from ase import Atoms
from ase.io import read

from cluster_mlip.geometry_checks import (
    MATCH_DEFAULTS, Partition, bond_change_sensitivity, cluster_internal_rmsds, match_atoms,
    structure_report, unwrapped_positions,
)

OUT = Path(__file__).parent / "out"
SUPPORTS = {"MgO100": (192, ["O"]), "graphene": (160, ["C"])}
TOLERANCES = (1.10, 1.15, 1.20, 1.25, 1.30)


def main() -> None:
    gas = [a for a in read(OUT / "gas_relaxed.extxyz", ":") if isinstance(a, Atoms)]
    stored = json.loads((OUT / "results.json").read_text())
    report: dict = {
        "definitions": {
            "stored.rmsd_vs_gas": "s2_relax.kabsch_rmsd: same atom order (index i -> i), proper Kabsch; "
                                  "= how far atoms moved from the landed gas basin",
            "stored.height": "min cluster z minus max z of ANY support atom (s2_relax)",
            "stored.n_contact": "cluster atoms with z < max support z + 2.8 A (s2_relax)",
            "trajectory_rmsd": "same as stored.rmsd_vs_gas, recomputed after unwrapping the cluster",
            "shape_rmsd_own_gas": "best element-preserving mapping (heuristic multi-start), proper Kabsch, "
                                  "vs the gas basin that was landed",
            "shape_rmsd_gas_B0": "same, vs gas basin B0",
            "local_min_height": "min over Fe of z minus mean z of surface atoms within 3 A laterally",
            "bonded": "d <= tol * (r_i + r_j), Cordero radii (Fe 1.52, O 0.66, C 0.76); default tol 1.2",
        },
        "matching": {"method": "multi-start Hungarian+Kabsch (heuristic, no error bound)", **MATCH_DEFAULTS},
        "caveat": "Geometry only; does not establish electronic state, spin, oxidation state or energy basin.",
    }
    for sname, (ns, site_elements) in SUPPORTS.items():
        block = report[sname] = {}
        frames = [a for a in read(OUT / f"supported_{sname}.extxyz", ":") if isinstance(a, Atoms)]
        for k, sup in enumerate(frames):
            name = sup.info["isomer"]
            part = Partition.from_specs(len(sup), substrate=f"0:{ns}")
            cluster = Atoms(sup.get_chemical_symbols()[ns:], unwrapped_positions(sup, part.cluster))
            internal = cluster_internal_rmsds(gas[k], cluster)
            m = match_atoms(gas[k], cluster)
            rep = structure_report(sup, part, metal="Fe", site_elements=site_elements)
            s = stored[sname]["isomers"][name]
            block[name] = {
                "gas_basin": f"B{k} ({gas[k].info['isomer']})",
                "stored": {"rmsd_vs_gas": round(s["rmsd_vs_gas"], 4), "height": round(s["height"], 4),
                           "n_contact": s["n_contact"]},
                "trajectory_rmsd": internal["cluster_trajectory_rmsd"],
                "shape_rmsd_own_gas": internal["cluster_shape_rmsd"],
                "shape_rmsd_gas_B0": round(match_atoms(gas[0], cluster)["rmsd"], 4),
                "local_min_height": rep["sites"]["min_height"],
                "surface_z_range": rep["sites"]["surface_layer"]["z_range"],
                "closest_fe_support": rep["sites"]["closest_contact"],
                "n_fe_within_bond_cutoff": rep["sites"]["n_in_contact"],
                "site_labels": rep["sites"]["site_counts"],
                "fe_fe_bonded": rep["cluster_coordination"]["Fe-Fe_bonded"],
                "connectivity_change_vs_own_gas": bond_change_sensitivity(gas[k], cluster, m["perm"], TOLERANCES),
            }
    (OUT / "geometry_check.json").write_text(json.dumps(report, indent=1) + "\n")
    for sname in SUPPORTS:
        for name, row in report[sname].items():
            sens = " ".join(f"{r['tolerance']:.2f}:+{r['formed']}/-{r['broken']}"
                            for r in row["connectivity_change_vs_own_gas"])
            print(f"{sname:8s} {name}  stored rmsd {row['stored']['rmsd_vs_gas']:.3f} h {row['stored']['height']:.2f} | "
                  f"shape own {row['shape_rmsd_own_gas']:.3f} B0 {row['shape_rmsd_gas_B0']:.3f} "
                  f"local h {row['local_min_height']['h']:.2f} closest {row['closest_fe_support']['d']:.3f}"
                  f"/{row['closest_fe_support']['bond_cutoff']:.3f} sites {row['site_labels']} | {sens}")


if __name__ == "__main__":
    main()
