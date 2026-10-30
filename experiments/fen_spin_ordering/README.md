# Fe_n spin-state ordering: MagPT on the legacy UBPW91 series (n = 2–25)

**Question:** does the public MagPT checkpoint (`MagPT_399168.pth`; see `../fe16_magpt/README.md` for the model,
environment and conventions) order the spin states of bare Fe_n clusters the way UBPW91/6-311+G* does, across many
sizes rather than only Fe16?

**Answer (2026-10-29): no, not usefully. A larger MagPT pilot is still not justified.** MagPT's apparent ordering
skill comes from structure: with all moments set to zero it orders the states just as well. On same-geometry,
adjacent-multiplicity pairs it does carry a real spin signal for small and medium clusters. That signal fails for
n ≥ 19, does not survive ±10% moment scaling, and is not improved by the per-atom Mulliken pattern.

Diagnostic only: nothing here touches the Gaussian campaign or any training set.

## Data

Source: the Oct9 legacy documents, one Word file per size, unpacked at
`E:\Research_Archive\Research_June18_2025\2019\Oct9`. They are the same bytes as
`E:\Research_Archive\Gennady_Storage\General_Warehouse.zip` (sha256 `bd2d6d66…`, the General Warehouse source).
The documents keep the original Gaussian text for each state: `Charge/Multiplicity`, the geometry, the Mulliken
spin table, frequencies and the archive `HF=…\S2=…`.

`src/cluster_mlip/legacy_doc_states.py` (tests: `tests/test_legacy_doc_states.py`) parses them:

- **Level per block:**
  - The block's own route line wins. It is marked ambiguous if the block's `$$$` title names a different
    functional or basis.
  - Otherwise a functional or basis named in the block title.
  - Otherwise the document title, used only if no differently-leveled route appeared earlier in the file.
  - The documents mix BPW91 with BLYP, PBE, BP86, BPBE, B3LYP, LSDA, M06-L, TPSS and other bases.
- **Old layouts handled:**
  - G98 five-column geometry tables, and tables without an "orientation" title.
  - `Total atomic spin densities:` headers.
  - Fe₂ dimers recorded only as an optimized Z-matrix bond length.
  - Archive energies that Word wrapped mid-number (`HF= -15166.23493 58\S2=`). Digit groups are re-joined only when
    the `\S2` field follows.
- **Skipped and counted:** vertical-ion summaries ("IP = … eV"), counterpoise blocks with ghost atoms, and spin
  tables that do not match the geometry.
- **Flags (recorded, not filtered):**
  - `artifact_sites`: |m| > 5 μB, beyond a physical Fe moment. This is the diffuse-basis Mulliken artifact on
    central atoms seen in Fe16.
  - `iop_5_13`: the route lets an unconverged SCF continue silently.
  - `s2_excess > 1`: expected for broken-symmetry states.
  - `spin_sum_mismatch`: excluded.

`extract_states.py` writes every parsed state (all levels, with flags) to
`D:\MLIP_Work_Folder\Cluster_MLIP\fen_oct9_states\states.extxyz` and `states.csv`, outside Git. **Clean set** =
UBPW91/6-311+G*, spin sum = M − 1, deduplicated, and no disagreement with the warehouse's independent extraction:

| | |
|---|---:|
| Charge/Multiplicity blocks parsed into states | 1,033 (999 unique) |
| Clean states | **695**, in 57 (n, q) groups; 55 groups have ≥ 2 multiplicities |
| Cross-check vs General Warehouse extraction (all unique states) | 871 agree within 5 meV, 7 disagree (excluded), 121 not in the warehouse at the same (q, M) |
| Clean states with a genuinely antiparallel Fe (not an artifact site) | 214 |
| Clean states with an artifact site | 151 |
| Saddles (imaginary modes) | 29 |

Energies are sane: −1263.87 to −1263.60 Ha per Fe atom. Cation and anion offsets match ionization energies and
electron affinities.

## Method

`run_magpt_ordering.py` (resumable; MagPT with no periodic cell, 4 CPU threads) evaluates each clean state at its own
geometry with collinear moments m_i = (0, 0, s_i):

| Condition | Moments |
|---|---|
| `true` | Mulliken |
| `uniform` | (M−1)/n on every atom: same total, no per-atom pattern |
| `scale0.9`, `scale1.1` | Mulliken × 0.9 and × 1.1 |

`run_zero_moment_control.py` adds `zero` (all moments 0: MagPT's structure-only energy). `analyze_ordering.py`
compares predictors with UBPW91 within each (n, q) group, using:

- Spearman rank correlation.
- The fraction of state pairs with |ΔE| ≥ 0.1 eV that are ordered the same way.
- The ground-state multiplicity.
- Same-geometry pairs (sorted-distance RMS < 0.05 Å, different M), where the structure term cancels.

Only relative energies within a group are compared.

## Results

**Whole groups (adiabatic: each state at its own geometry), 55 groups, all clean states:**

| Predictor | Median Spearman | Pairs ordered correctly (\|ΔE\| ≥ 0.1 eV, n = 9,991) | Ground-state M |
|---|---:|---:|---:|
| MagPT, Mulliken moments | +0.43 | 62% | 21/55 |
| MagPT, zero moments (structure only) | +0.46 | 63% | 17/55 |
| MagPT magnetic part (true − zero) | +0.31 | 58% | 22/55 |
| MagPT, uniform moments | +0.60 | 76% | 19/55 |
| "Lower M is lower" | +0.05 | 40% | 4/55 |

Restricting to minima, or dropping artifact states, changes these by a few points only (`report/ordering_summary.json`).

**Same-geometry pairs (the spin-only test):**

- **All ΔM:** 776 pairs, 721 with |ΔE| ≥ 0.05 eV. Sign agreement 55% (chance). MagPT says higher M is lower in 34%
  of pairs; Gaussian says so in 69%.
- **Adjacent M (ΔM = 2), one pair per (n, q, M):** 134 independent pairs. Sign agreement 64–65%, Spearman 0.33.
  By size, sign agreement is:

  | Sizes | Sign agreement (all ΔM = 2 pairs) |
  |---|---:|
  | Fe₂–Fe₆ | 74% |
  | Fe₇–Fe₁₂ | 84% |
  | Fe₁₃–Fe₁₈ | 69% |
  | Fe₁₉–Fe₂₅ | 39% |

  For Fe₁₉–Fe₂₅ MagPT says higher M is lower in only 12% of pairs (Gaussian: 63%), the same low-M bias as Fe16.
- **Artifact sites:** pairs touching one agree 54%; pairs without, 74%.
- **Gap size:** MagPT gaps are ~2.3× the Gaussian ones (median 0.78 vs 0.34 eV).
- **Moment-magnitude stability:** only 52% of the 134 adjacent pairs keep their sign between × 0.9 and × 1.1, and ΔE
  moves by a median 1.5 eV over that range. Agreement is 60% / 64% / 43% at × 0.9 / 1.0 / 1.1. Uniform moments
  agree 68%.

## Interpretation

1. **The ordering of whole groups is structural.** MagPT without moments orders isomers and multiplicities as well as
   MagPT with them. The positive correlations come from geometry, not from magnetism.
2. **There is a real but narrow spin signal.** For adjacent multiplicities at fixed geometry, MagPT gets the sign
   right well above chance for Fe₂–Fe₁₈. The signal is fragile:
   - It sits within ±10% of a magnitude where E(|m|) is steep.
   - It drops below chance for n ≥ 19, where Mulliken moments are large and MagPT strongly prefers lower M.
   - It is hurt by artifact sites.
3. **The per-atom Mulliken pattern does not help.** Spreading the same total moment uniformly does as well or
   better. This again points to a moment-definition mismatch: Mulliken in a diffuse basis vs DeltaSpin's orbital
   projection.

These results are consistent with the Fe16 diagnostic and extend it from one size to 24.

## Files

- **`report/` (small, in Git):**
  - `extraction_summary.json`, `files.csv` (per-source sha256 and skip counts).
  - `magpt_states.csv`, `magpt_zero_moment.csv`, `ordering_summary.json`.
  - `groups.csv` (per-group metrics, minima).
  - `same_geometry_pairs.csv`.
- **`D:\MLIP_Work_Folder\Cluster_MLIP\fen_oct9_states\` (outside Git):** the extracted states.

Reproduce:

```bash
python experiments/fen_spin_ordering/extract_states.py --source E:/Research_Archive/Research_June18_2025/2019/Oct9 \
  --data D:/MLIP_Work_Folder/Cluster_MLIP/fen_oct9_states --report experiments/fen_spin_ordering/report \
  --warehouse D:/MLIP_Work_Folder/Cluster_MLIP/General_Warehouse/extracted/seeds.extxyz
D:/MLIP_Work_Folder/envs/magpt_dp/python.exe experiments/fen_spin_ordering/run_magpt_ordering.py \
  --states D:/MLIP_Work_Folder/Cluster_MLIP/fen_oct9_states/states.extxyz \
  --checkpoint D:/MLIP_Work_Folder/external/magpt/MagPT/model/MagPT_399168.pth --out experiments/fen_spin_ordering/report
D:/MLIP_Work_Folder/envs/magpt_dp/python.exe experiments/fen_spin_ordering/run_zero_moment_control.py  (same arguments)
python experiments/fen_spin_ordering/analyze_ordering.py --states <states.extxyz> --report experiments/fen_spin_ordering/report
```

The MagPT runs take about 40 min each on CPU. The environment sits on the external D: drive. A dropout killed the
first run, so both runners append each state as it finishes and resume on restart.

## Beyond MagPT

These 695 clean states (24 sizes, 3 charges, up to 50 multiplicities per size, 214 with antiparallel sites) are a
spin-label set for ClusterMLIP's own spin-aware MACE: total M plus per-atom moments, with energies. They have no
forces, since all are stationary points, and the Mulliken moments carry the artifact flag. The set is extracted but
not merged into any training data.
