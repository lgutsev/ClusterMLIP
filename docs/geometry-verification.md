# Numerical verification of cluster and adsorption geometry

2026-10-13. Tool: `cluster-mlip geometry-check` (`src/cluster_mlip/geometry_checks.py`,
`pip install -e .[geometry]`: numpy, scipy, ASE). Tests: `tests/test_geometry_checks.py`.

Why this exists: structural statements in this project (contacts, sites,
rearrangements, "relaxed into B0") must come from computed distances, not from
reading XYZ/POSCAR text. Alampara, Miret & Jablonka (*Nat. Comput. Sci.* 2026,
doi:10.1038/s43588-026-01045-2) found that language models fine-tuned on
text-encoded crystal structures learn category (element) information but barely
use coordinates ("geometric blindness"). That finding concerns language models
reading structures as text. It is **not** an explanation of the Fe16 MACE
force-learning failure (MACE is a geometric GNN) and is not cited as one.

## Existing coverage before this change

| Need | Already there | Gap |
|---|---|---|
| Bond graph / coordination | `stratify.bonding_graph` (Cordero radii, tol 1.2) | No periodic images; frame-level class only |
| Same-structure identity | `spin_audit.alignments`: exhaustive isomorphism + proper Kabsch, near-identical pairs | Not for relaxed/distorted pairs |
| Dedup of isolated clusters | `examples/legacy_inventory.aligned_distance` (fingerprint Hungarian + Kabsch) | Example-only, single start |
| Permutation-invariant distance | `spin.geometry_distance` (sorted pair distances) | No correspondence, no per-atom displacement |
| Supported cluster vs gas | `s2_relax.kabsch_rmsd`, `height`, `n_contact` | Assumes identity order and an unwrapped cluster; height measured from max substrate z (wrong on buckled graphene) |
| Adsorption site, adsorbate bonds/orientation, bond changes, close contacts | none | missing |

## What was added

`geometry-check STRUCTURE [--compare OTHER] [--substrate I] [--adsorbate I] [--site-elements O] [--metal Fe]` writes JSON with:

- **Contacts**: overall minimum distance, minimum per group pair (cluster/adsorbate/substrate) and per element pair; close contacts `d < 0.6 (r_i + r_j)`.
- **Coordination**: per-atom CN split by neighbour group and element; CN histogram; bonded M–M distance statistics. Bonded = `d <= 1.2 (r_i + r_j)`, the same radii and tolerance as `stratify`.
- **Cluster shape**: radius of gyration, principal moments, asphericity, connectivity (after unwrapping).
- **Sites** (supported): surface atoms = substrate atoms with no substrate atom above them within 1.0 Å laterally; site = number of bonded surface atoms of the `--site-elements` (top / bridge / hollow-3 / hollow-4 / n-fold), labelled with its composition (`top:O`); local height above surface atoms within 3 Å laterally; closest contact with its bond cutoff.
- **Adsorbate**: internal bonds, host contacts, tilt of the principal axis (bound end down) relative to the surface normal or to the host-centroid direction.
- **Comparisons**
  - Isolated: element-preserving Hungarian + proper Kabsch from fingerprint, identity, 4 principal-axis and 64 fixed-seed random starts; per-atom displacements and a mapping-free fingerprint RMS as an independent check. The search is heuristic: forward and reverse results can differ by ~1e-3 Å on near-spherical clusters. Exact-identity claims should still use `spin_audit.alignments`.
  - Supported: substrate frame, minimum image, substrate drift removed. Reports **trajectory RMSD** (same atom order: how far atoms moved) and **shape RMSD** (best mapping: how different the shapes are), bond changes and interface bond counts.

Every report carries the caveat that geometry does not establish oxidation
states, local spins, broken-symmetry character or barriers.

Two choices were corrected during verification:
- Oxide sites need `--site-elements O`. At tol 1.2 the Fe–Mg cutoff is 3.5 Å, so an Fe 2.0 Å above an O "bonds" to the four Mg²⁺ beside it (`5-fold:Mg4O` instead of `top:O`).
- The close-contact scale was lowered from 0.75 to 0.6. With the high-spin Fe radius, 0.75 flags ordinary 2.15–2.27 Å cluster Fe–Fe.

Invariance verified (tests): a distorted Fe13+NO rotated, translated and
atom-shuffled gives RMSD < 1e-6 Å, the recovered permutation, and identical
CN/contact/adsorbate reports; mirror images are not superposed. An Fe4 on
MgO(100) straddling the cell edge is unwrapped correctly, and its site, height
and closest contact are unchanged after a lateral shift + wrap + reordering of
all 68 atoms; that shift gives zero displacement.

## Worked comparison: Fe16 on MgO(100) and graphene (committed example data)

Data: `examples/supported_fe16_tem/out/{gas_relaxed,supported_MgO100,supported_graphene}.extxyz`
(MACE-MP-0 relaxations; cluster atoms 192–207 on MgO and 160–175 on graphene).

```bash
cluster-mlip geometry-check examples/supported_fe16_tem/out/supported_MgO100.extxyz@4 \
  --substrate 0:192 --site-elements O --metal Fe -o mgo_b4.json
```

**MgO(100), landed basin B4.** Closest Fe–O is 3.449 Å (Fe207–O81); the Fe–O bond
cutoff is 2.616 Å. Closest Fe–Mg is 3.755 Å. No Fe–substrate bonds, so no site is
assigned. The lowest Fe is 3.446 Å above the local surface plane (surface z
5.20–5.28 Å, rumpling 0.07 Å). Fe–Fe: 54 bonded pairs, 2.320–2.815 Å, mean 2.590 Å.
No close contacts. This puts numbers behind the example README's "3.4 Å, no Fe–O
bond" statement.

Atom correspondence explains what happened to the cluster:

| supported basin | trajectory RMSD vs own gas basin (identity order; = stored `rmsd_vs_gas`) | shape RMSD vs own gas basin | shape RMSD vs gas **B0** |
|---|---|---|---|
| B0 | 0.011 | 0.010 | 0.010 |
| B1 | 0.297 | 0.289 | 0.016 |
| B2 | 0.428 | 0.428 | 0.019 |
| B3 | 0.531 | 0.447 | 0.017 |
| B4 | 0.755 | 0.446 | 0.018 |
| B5 | 0.468 | 0.460 | 0.012 |
| B6 | 0.465 | 0.457 | 0.013 |

All seven MgO landings end within 0.019 Å of the gas B0 shape, with identical
Fe–Fe bond statistics (54 bonds; CN histogram {6: 12, 7: 3, 15: 1}). So the MgO
rows compare one isomer seven times. The README inferred this from the
negative strain energies; it is now shown geometrically. The stored 0.755 Å for B4
mostly reflects atoms trading symmetry-equivalent positions, not a 0.75 Å shape
change.

**Graphene, landed basin B6** (`--substrate 0:160 --site-elements C`). Six Fe bond
to C: 3 bridge, 2 hollow-4, 1 hollow-3 (bonded = C–Fe ≤ 2.736 Å); closest C–Fe
2.214 Å. The lowest Fe is 1.92 Å above the local sheet. The surface spans z −0.31 to
1.08 Å (1.39 Å buckle), so the stored `height` (0.76 Å, measured from the
highest C) is not a cluster–sheet height. Fe–Fe: 58 bonded pairs (gas: 54),
2.146–3.495 Å; shortest 2.146 Å, below the gas minimum 2.378 Å. The shape changed
by 0.522 Å vs its own gas basin (stored trajectory value 0.998 Å) and is closer to
gas B0 (0.287 Å). 6 Fe–Fe bonds formed and 2 broken relative to gas B6.

What this does **not** show: anything about Fe oxidation states, local
moments, the spin state of the supported cluster or binding energetics. Those
need the VASP interaction labels (`vasp-collect`: SCF convergence, total moment)
and, for MgO, the model's missing Fe/oxide chemistry noted in the README.

## Gaussian job structure

This change does not touch Gaussian input generation, routes, checkpoints,
fragment assignments or charge/multiplicity lines. The validated broken-symmetry
tandem (fragment guess, then `--Link1--` with `Stable=Opt` only in the second,
fixed-nuclei stage; IOP(5/13=1) kept in every link) is unchanged.
