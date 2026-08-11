# `Oct9.zip` extraction audit

Audit date: 2026-08-11

## Archive inventory

- 2,268 files;
- approximately 302 MB uncompressed;
- 303 `.log`, 3 `.out`, approximately 1,950 `.doc`, and 6 `.docx` files;
- approximately 21,300 searchable `HF=` energy records in the Word corpus;
- approximately 14,900 searchable input-orientation tables;
- no detected Gaussian IRC point markers or IRC route stream.

The documents mix methods and property calculations. Frequent content includes
BPW91/6-311+G*, UBPW91/6-311+G*, `GEN` basis calculations, alternative
functionals, MP2 checks, vertical charge states, frequencies, TDDFT, and
polarizabilities. The legacy energies must not be combined as one MLIP label.

## Reproducible Fe/N/O seed extraction

Command:

```bash
cluster-mlip extract Oct9.zip \
  -o extracted_feno \
  --elements Fe,N,O \
  --require-elements Fe \
  --max-atoms 20
```

Result:

| Classification | Unique seeds |
| --- | ---: |
| Minimum (zero retained imaginary modes) | 4,250 |
| Optimized, frequencies unavailable | 394 |
| First-order saddle (one imaginary mode, not explicitly TS-labeled) | 116 |
| Higher-order saddle | 34 |
| Unknown | 31 |
| Explicit transition-state stream | 0 |
| IRC points | 0 |
| Total | 4,825 |

All 2,268 archive members were handled without a document-decoding exception.

These counts describe recoverable seed states, not independent DFT training
labels. Similar geometries may represent different methods or electronic
solutions in the research records. New, consistent energy-and-force
calculations are still required.

## Recommended first campaign

Begin with the smaller Fe/N/O compositions rather than submitting all 4,825
seeds. Stratify by:

- Fe cluster size;
- O/N stoichiometry;
- charge;
- multiplicity;
- configuration classification;
- structural diversity.

Keep first-order and higher-order saddles as useful off-minimum seeds, but do
not call them reaction transition states unless connectivity is subsequently
verified.

## `FenOm_Warehouse.zip` audit

Archive inventory:

- 3,589 total ZIP members;
- 881 real high-precision `.txt` files after excluding AppleDouble `._*`
  resource forks;
- 1,676 lower-precision `.mdt` topology/coordinate companions;
- 50 `.crt` files plus research documents and spreadsheets;
- no Gaussian IRC route, IRC point, `Opt=TS`, QST2, or QST3 markers.

The v0.2 native parser recovered 881 coordinate records and deduplicated them
to 568 unique structures. Applying strict `--elements Fe,O
--require-elements Fe,O` filtering yields 498 unique Fe/O structures. The
remaining parsed structures include pure Fe clusters and Ga/Ti/U/Tc/I/Xe/Tl
substitution examples.

This warehouse should be used for isomer and cluster-size coverage. A separate
checkpoint/output warehouse is still required for the claimed TS and IRC
reaction-path coverage.
