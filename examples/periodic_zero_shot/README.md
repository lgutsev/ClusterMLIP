# Zero-shot check of foundation MLIPs on periodic VASP frames

How far are foundation models from periodic DFT on a supported cluster, before any training?
The input is `cluster-mlip vasp-ingest` output. The models take no charge or spin input.

```bash
cluster-mlip vasp-ingest path/to/outputs -o frames --cluster-elements Fe
# one call per model, each in the environment that has its package
python zero_shot.py predict frames/frames.extxyz --backend mace --model MACE.model --name mp0-medium -o preds
TORCHDYNAMO_DISABLE=1 python zero_shot.py predict frames/frames.extxyz --backend uma \
    --model uma-s-1p1.pt --task omat --name uma-s-1p1-omat -o preds
python zero_shot.py report frames/frames.extxyz preds/*.npz --reactions reactions.json -o report
```

`predict` runs a single point at every DFT geometry; nothing is relaxed. `report` writes
`zero_shot_report.json` and `zero_shot_table.md` with these metrics:

- **Forces on mobile atoms**, overall and per element. Frozen selective-dynamics atoms are
  excluded. The reference is the predict-zero baseline.
- **Forces at the DFT minima.** These are the cleanest test, because DFT forces there are
  close to zero.
- **Relative energies along each job** (E_i − E_last). This removes each model's constant
  offset for a fixed composition.
- **Reaction energies between final frames**, taken from `--reactions`
  (`{"name": {"job": coefficient}}`).
- **Same-geometry final frames.** These are jobs that differ only in their electronic state,
  so a spin-blind model gives them identical predictions by construction.

Notes:

- On Windows, set `TORCHDYNAMO_DISABLE=1` for UMA. Its CPU path otherwise tries to compile
  with MSVC. `predict` also uses UMA's `general` execution mode, because the default GPU
  path needs Triton.
- `FAIRChemCalculator` passes `atoms.info["charge"]` and `["spin"]` to every UMA task head,
  but only `omol` was trained on them; the other heads saw 0. `vasp-ingest` frames carry
  the DFT multiplicity, so `predict` removes both keys for non-`omol` tasks. Left in, they
  shifted N2 adsorption energies by 5–90 eV.
- Compare raw energies only within one composition or through reactions. MACE-MP/MPA and
  UMA-omat are trained on raw VASP PBE(+U) totals, so their absolute energies carry offsets
  of order 10–200 meV/atom. These come from +U on the oxide training data, D3, and
  reference choices. UMA's `oc20` head uses RPBE and its own referencing.
- Results stay with the data, not in this repository.
