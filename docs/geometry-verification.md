# Numerical verification of cluster and adsorption geometry

2026-10-13. Tool: `cluster-mlip geometry-check` (`src/cluster_mlip/geometry_checks.py`,
`pip install -e .[geometry]`: numpy, scipy, ASE). Tests: `tests/test_geometry_checks.py`.
Worked example: `examples/supported_fe16_tem/s12_geometry_check.py` →
`examples/supported_fe16_tem/out/geometry_check.json`.

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
| Supported cluster vs gas | `s2_relax.kabsch_rmsd`, `height`, `n_contact` | Same atom order only; height measured from the highest support atom |
| Adsorption site, adsorbate bonds/orientation, connectivity changes, close contacts | none | missing |

## Conventions and settings (all echoed in the JSON output)

- **"Bond" is operational**: `d <= tol * (r_i + r_j)` with Cordero covalent
  radii (Fe 1.52 high-spin, O 0.66, C 0.76, Mg 1.41 Å), default `tol = 1.2`, the
  same table and tolerance as `stratify`. This defines a connectivity graph; it is
  not a chemical-bonding assignment. Fe–Fe cutoff 3.648 Å, Fe–O 2.616 Å, Fe–C
  2.736 Å, Fe–Mg 3.516 Å.
- **Close contact**: `d < 0.6 (r_i + r_j)` (Fe–Fe 1.82 Å). 0.75 flagged ordinary
  2.15–2.27 Å cluster Fe–Fe. Per-element-pair minima are always reported.
- **Periodicity**: minimum image along periodic axes; groups (cluster, adsorbate)
  are unwrapped through their own connectivity graph; groups wider than half a
  cell are refused.
- **Atom matching (isolated clusters)**: element-preserving Hungarian assignment
  alternated with proper (det = +1) Kabsch rotation. Starts: distance-fingerprint
  assignment, identity order, 4 principal-axis frames and 64 random rotations
  (`scipy.spatial.transform.Rotation.random(64, random_state=0)`), at most 50
  rounds each, 70 starts in total. The lowest RMSD found is returned with these
  settings (`"search"` / `"matching"` in the JSON).
- **Supported comparisons** use the substrate frame: same atom order, minimum
  image, mean substrate drift removed. They report a trajectory RMSD (same atom
  order: how far atoms moved) and a shape RMSD (best mapping as above: how
  different the shapes are).

### Matching limitation

The matcher is a heuristic. It returns the best mapping among the starts above,
not a proven global optimum, and it has no error bound. On near-spherical Fe16,
forward and reverse searches agree within about 1e-3 Å (0.28869 vs 0.28893 Å for
MgO-landed B4 vs gas B1). That agreement is a consistency check, not a bound. A
regression test pins one case where the earlier, fewer-start search stopped at
0.716 Å while 0.289 Å exists. Exact-identity claims (same structure,
symmetry-equivalent mappings) should use the exhaustive
`spin_audit.alignments`.

### Adsorption sites

Site atoms are the surface atoms (substrate atoms with no substrate atom above
them within 1.0 Å laterally) within the bond cutoff of an adsorbing atom. An
ideal label (`top`, `bridge`, `hollow-n`) is given only when the pattern is
ideal:
- the adsorbing atom is within 0.5 Å laterally of the site atom, pair midpoint or polygon centroid;
- for bridge/hollow, no site atom is within 0.5 Å laterally of directly beneath it;
- the site atoms' heights spread by at most 0.3 Å.

Anything else is `irregular-n` with the reasons (`off-centre`, `atom-beneath`,
`buckled`). Irregular, reconstructed or strongly deformed environments are
reported as such and not forced into an ideal label.

- `--site-elements` is **required on ionic supports**, i.e. whenever cations lie
  inside the bond cutoff of the adsorbing atom. On MgO(100) an Fe 2.0 Å above an
  O is 2.9 Å from the four neighbouring Mg, inside the 3.516 Å Fe–Mg cutoff, so
  without `--site-elements O` the pattern is `irregular-5:Mg4O` instead of
  `top:O`.
- **Height** is local: the adsorbing atom's z minus the mean z of surface atoms
  within 3 Å laterally. On a buckled support it differs from the height above the
  highest support atom by up to the buckling amplitude, so both are quoted with
  their definitions.

## Verification

`python -m unittest tests.test_geometry_checks`: 12 tests, OK. Full suite:
`python -m unittest discover -s tests` passes (319 tests, 2 skipped). mypy is clean
on `geometry_checks.py` and `cli.py`.

- A distorted Fe13 + end-on NO, rotated, translated and atom-shuffled, gives
  RMSD < 1e-6 Å, recovers the permutation, and yields identical coordination,
  contact and adsorbate reports. A chiral mirror image is not superposed.
- An Fe4 on MgO(100) straddling the cell edge is unwrapped. After a lateral shift,
  wrap and reordering of all 68 atoms, it keeps the same site (`top:O`), local
  height (2.000 Å) and closest contact (Fe–O 2.000 Å), and the shift gives zero
  displacement. An Fe slid 0.8 Å off the O is `irregular-1:O` (`off-centre`).
- A connectivity-sensitivity test confirms that a pair between the 1.15 and 1.20
  cutoffs changes the formed/broken count exactly there.

## Worked comparison: Fe16 on MgO(100) and graphene

Data: committed MACE-MP-0 outputs in `examples/supported_fe16_tem/out/`
(cluster atoms 192–207 on MgO, 160–175 on graphene). Commands:

```bash
cluster-mlip geometry-check examples/supported_fe16_tem/out/supported_MgO100.extxyz@4 \
  --substrate 0:192 --site-elements O --metal Fe -o mgo_b4.json
cluster-mlip geometry-check examples/supported_fe16_tem/out/supported_graphene.extxyz@6 \
  --substrate 0:160 --site-elements C --metal Fe -o graphene_b6.json
python examples/supported_fe16_tem/s12_geometry_check.py   # writes out/geometry_check.json
```

Stored values are from `results.json` and keep their original definitions:
- stored RMSD: `s2_relax.kabsch_rmsd`, same atom order;
- stored height: lowest Fe z minus the highest z of any support atom.

The new values sit alongside and leave the stored ones unchanged.

### MgO(100)

| landed basin | stored RMSD vs own gas start (same order) | shape RMSD vs own gas start | shape RMSD vs gas B0 | stored height (from highest support atom) | local height | shortest Fe–O / cutoff |
|---|---|---|---|---|---|---|
| B0 | 0.011 | 0.010 | 0.010 | 3.33 | 3.37 | 3.383 / 2.616 |
| B1 | 0.297 | 0.289 | 0.016 | 3.37 | 3.41 | 3.397 / 2.616 |
| B2 | 0.428 | 0.428 | 0.019 | 3.41 | 3.45 | 3.473 / 2.616 |
| B3 | 0.531 | 0.447 | 0.017 | 3.43 | 3.47 | 3.525 / 2.616 |
| B4 | 0.755 | 0.446 | 0.018 | 3.40 | 3.45 | 3.449 / 2.616 |
| B5 | 0.468 | 0.460 | 0.012 | 3.41 | 3.46 | 3.476 / 2.616 |
| B6 | 0.465 | 0.457 | 0.013 | 3.46 | 3.51 | 3.525 / 2.616 |

All values in Å.

- **No Fe–O contacts satisfy the bond cutoff** (2.616 Å) in any MgO structure;
  the shortest Fe–O distance is 3.383–3.525 Å. No Fe–Mg contact satisfies its
  cutoff either (B4 shortest 3.755 vs 3.516 Å). No site is assigned.
- **All seven relaxed structures are geometrically close to gas B0** (shape RMSD
  0.010–0.019 Å), while each is 0.29–0.46 Å from its own gas starting structure.
  They also have the same Fe–Fe connectivity statistics: 54 pairs within the
  cutoff, CN histogram {6: 12, 7: 3, 15: 1}. This is a geometric statement. It
  does not by itself establish that they share an electronic state or an energy
  basin. The README's "relaxed into B0" reading comes from the strain energies;
  the geometry is consistent with it.
- Where the stored same-order RMSD exceeds the shape RMSD (B3 0.531 vs 0.447, B4
  0.755 vs 0.446), the difference reflects atoms ending in each other's
  equivalent positions rather than a larger change of shape.

### Graphene

| landed basin | stored RMSD vs own gas start (same order) | shape RMSD vs own gas start | shape RMSD vs gas B0 | stored height (from highest support atom) | local height | shortest Fe–C / cutoff |
|---|---|---|---|---|---|---|
| B0 | 0.361 | 0.361 | 0.361 | 0.80 | 1.95 | 2.214 / 2.736 |
| B1 | 0.457 | 0.234 | 0.353 | 1.00 | 1.95 | 2.215 / 2.736 |
| B2 | 0.613 | 0.445 | 0.426 | 0.85 | 1.92 | 2.168 / 2.736 |
| B3 | 0.665 | 0.443 | 0.355 | 0.87 | 1.96 | 2.256 / 2.736 |
| B4 | 0.436 | 0.436 | 0.418 | 0.77 | 1.96 | 2.228 / 2.736 |
| B5 | 0.704 | 0.411 | 0.404 | 0.62 | 1.93 | 2.249 / 2.736 |
| B6 | 0.998 | 0.522 | 0.287 | 0.76 | 1.92 | 2.214 / 2.736 |

All values in Å.

- The sheet buckles: surface z spans 1.18–1.50 Å across the seven structures
  (B6: −0.31 to 1.08 Å). The stored height is therefore 0.6–1.0 Å, while the
  local height is 1.92–1.96 Å.
- **Graphene B6.** Six Fe atoms are within the Fe–C cutoff of surface C. Two
  patterns are ideal under the criteria above:
  - a `hollow-3:C3` (offset 0.05 Å);
  - a `bridge:C2` (offset 0.27 Å).
  Four are irregular:
  - `buckled` (site-atom height spread 0.38 Å);
  - `buckled` + `atom-beneath`;
  - `atom-beneath`;
  - `off-centre` + `atom-beneath`.
  Across B0 and B6, only 3 of 11 contact patterns meet the ideal criteria. A
  3-carbon "hollow" is also not a standard graphene site (top / C–C bridge /
  hexagon centre). The graphene contact patterns are therefore reported as
  cutoff patterns, and no adsorption-site preference is claimed.
- **Changes in the cutoff-defined Fe–Fe connectivity graph, graphene B6 vs its
  own gas start** (best mapping):

  | tol | Fe–Fe cutoff (Å) | formed | broken |
  |---|---|---|---|
  | 1.10 | 3.344 | 4 | 2 |
  | 1.15 | 3.496 | 6 | 2 |
  | **1.20** | **3.648** | **6** | **2** |
  | 1.25 | 3.800 | 6 | 1 |
  | 1.30 | 3.952 | 3 | 3 |

  The default count (6 formed, 2 broken) holds for tolerances 1.15–1.20. It
  changes outside that range because several Fe–Fe distances lie within about
  0.15–0.3 Å of the cutoff. Quote it as "about 4–6 formed and 1–3 broken for
  tolerances 1.10–1.30", not as a fixed number. Per-basin sensitivities for all
  14 structures are in `out/geometry_check.json`; for example, MgO B1 goes from
  +2/−0 at 1.10 to +0/−2 at 1.20.
- The shortest Fe–Fe distance on graphene B6 is 2.146 Å (gas start: 2.378 Å).
  It is not a close contact at the 0.6 scale.

What this does **not** show: Fe oxidation states, local moments, the spin
or electronic state of the supported cluster, energy basins or binding
energetics. Those need electronic-structure outputs with convergence checks (the
VASP interaction labels from `vasp-collect`: SCF convergence, total moment) and,
for MgO, the model's missing Fe/oxide chemistry noted in the example README.

## Gaussian job structure

This change does not touch Gaussian input generation, routes, checkpoints,
fragment assignments or charge/multiplicity lines. The validated broken-symmetry
tandem (fragment guess, then `--Link1--` with `Stable=Opt` only in the second,
fixed-nuclei stage; IOP(5/13=1) kept in every link) is unchanged.
