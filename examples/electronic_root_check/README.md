# Package 35 (proposed): electronic-root check on package 26 matches

Owner: ClusterMLIP. Source: `examples/electronic_root_check/` in lgutsev/ClusterMLIP.
Read-only: no Gaussian calculations, no training, nothing is deleted or merged.

## Why
Package 26 found geometry matches (≤ 0.02 Å) between the legacy collection
(`/ddnB/project/ramu/gutsev/glg`) and the warehouses, and inside the legacy collection:

| Overlap | Pairs |
|---|---:|
| FenOm_Warehouse2, same charge/multiplicity and identical route | 219,077 |
| FenOm_Warehouse2, same q/M, different route string | 58,688 |
| Inside the legacy collection | about 7.77 M (most are neighbouring steps of one optimization) |

Same q/M and the same route do not prove the same SCF solution. This is the cheap first check from
`REVIEW_26.md`, item 2: attach each record's own SCF energy to every same-(q, M) match and flag pairs whose
energy difference is too large for the geometric difference.

Those flagged pairs serve two purposes:
- **Duplicate screen.** They show which "duplicates" are not safe to merge.
- **Spin-state screen.** Fe16 (and other Fe_n) pairs with the same geometry and q/M but different energies are
  free candidates for alternative broken-symmetry solutions. This is exactly what the Fe16 broken-symmetry pilot
  tries to create with new jobs, so check these candidates before submitting that pilot.

## Pairs and flags
Pairs examined:
- legacy vs warehouse, same q/M (identical route and different route reported separately);
- legacy vs legacy, same q/M, **different source files**. Steps within one optimization are skipped.

Flags:
- `candidate_distinct_root`: |ΔE| ≥ 0.10 eV at aligned RMSD ≤ 0.005 Å.
- `large_dE_at_match`: |ΔE| ≥ 0.50 eV at any matched RMSD.

`bins.csv` holds the full (RMSD, |ΔE|) distribution per kind, so the thresholds can be judged.

For flagged legacy records, ⟨S²⟩ before and after annihilation is read from the source `.out`. The script finds
the `SCF Done` line carrying that record's energy and takes the next ⟨S²⟩ line. Warehouse-side sources are not
read. ⟨S²⟩ is supporting evidence only.

A flag is a candidate, not a verdict. Other explanations to rule out:
- a different method behind a different route string;
- an unconverged SCF: `IOP(5/13=1)` lets one continue, so look for "Convergence criterion not met" in the
  source;
- a wrong energy–geometry pairing in the legacy parse.

## Run on QB4
Package 26's run must exist: `run_1075007` (override with `INVENTORY_RUN`).
- Partition `single`, 8 cores for memory, 4 h cap, no `--mem`.
- Runtime: `/project/lgutsev/env/cluster_mlip_runtime`. The script uses only the Python standard library.
- `root_check.py --self-test` runs first.
- Outputs go to a new, non-overwriting
  `Gutsev_Legacy_2026-10/root_check_<jobid>/` directory, with compact copies in `outputs/`.

## Review (ClusterMLIP)
`outputs/DONE` and exit code 0 mean the scan completed. Then:
1. Read `summary.json`:
   - counts per kind;
   - `flagged_fe16`;
   - the ⟨S²⟩ status mix.
2. In `pairs_flagged.csv`, sort the Fe16 / Fe_n rows by |ΔE|. Check the route columns and ⟨S²⟩, and open the top
   few source logs for convergence warnings.
3. Feed real same-geometry, same-M alternative solutions to the Fe16 local-spin question, using the roadmap's
   reopening gate.
