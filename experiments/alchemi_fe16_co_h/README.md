# Optional NVIDIA ALCHEMI batch backend for the Fe16 + CO + 2H pilot

Results and recommendation: [docs/alchemi_cluster_feasibility.md](../../docs/alchemi_cluster_feasibility.md).

Environment: nvalchemi-toolkit 0.3.0 installed over an existing CUDA torch + mace-torch, with
`constraints.txt` so pip does not replace them:

    pip install nvalchemi-toolkit==0.3.0 -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128

On Windows set `TORCHDYNAMO_DISABLE=1` (no Triton). `POOL`, `MODEL`, `OUT` below are paths of your choice.

| Step | Command |
|---|---|
| Frozen pool and fixture | `python build_pool.py RUN_DIR ANALYSIS_DIR --out OUT/pool` |
| Single-point parity | `python parity_single_point.py OUT/pool/fixture12.extxyz MODEL --dtype float64 --out OUT/parity/parity_float64.json` |
| Charge/spin inputs | `python electronic_inputs.py SPIN_MODEL CATALYST.extxyz --out OUT/parity/electronic_inputs.json` |
| Fixed-batch benchmark | `python relax_bench.py POOL MODEL --backend alchemi --batch-size 16 --out OUT/bench/alchemi_b16` (and `--backend native`) |
| Inflight batching | `python inflight_bench.py POOL MODEL --slots 16 --out OUT/inflight/slots16` |
| Compare | `python compare_backends.py OUT/bench --out OUT/bench/compare.json`; `python inflight_compare.py OUT/inflight OUT/bench --out OUT/inflight/compare.json` |
| Reliability | `python reliability_checks.py POOL MODEL --out OUT/reliability`; `python native_repeatability.py POOL MODEL --out OUT/reliability/native_repeatability.json` |

Check each script's `--help` for the exact options. `alchemi_evaluator.AlchemiEvaluator` is the object to pass
to `AdsorbateSearch.run_batched`; `cluster_mlip.adsorbate_search.native_evaluator` is the fallback.

All energies here come from MACE-MPA-0, which failed the pilot's reference gate. They rank candidates for
DFT; they are not results.
