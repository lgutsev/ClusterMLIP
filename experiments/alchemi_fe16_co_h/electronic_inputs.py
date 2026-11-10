"""Does the toolkit's MACE wrapper carry charge / multiplicity to a model that needs them?

    python electronic_inputs.py SPIN_MODEL CATALYST.extxyz --out DIR

SPIN_MODEL is the project's Fe16 MACE with a per-graph ``total_spin`` (multiplicity)
embedding. The same bare Fe16 geometry is evaluated at several multiplicities:

  native       ASE MACECalculator, ``atoms.info["spin"]`` = M
  builtin      nvalchemi ``MACEWrapper`` as shipped (v0.3.0)
  project      ``SpinMACEWrapper`` below: the built-in wrapper plus the embedding
               inputs read from per-graph batch fields, in mixed-M batches
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.io import read

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors

from alchemi_backend import load_wrapper


class SpinMACEWrapper(MACEWrapper):
    """Built-in wrapper + the embedding inputs the checkpoint was trained with.

    Every name in the model's ``embedding_specs`` must be present on the batch as a
    per-graph field; a missing field raises instead of falling back to a default.
    """

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])
        inputs = super().adapt_input(data, **kwargs)
        for name in getattr(self.model, "embedding_specs", {}) or {}:
            try:
                value = data[name]
            except (KeyError, AttributeError) as exc:
                raise KeyError(f"model needs per-graph input {name!r}; the batch does not carry it") from exc
            inputs[name] = value.reshape(-1).to(inputs["positions"].device)
        return inputs


def batch_for(atoms, mults, device, dtype) -> Batch:
    items = []
    for m in mults:
        d = AtomicData.from_atoms(atoms, device=device, dtype=dtype)
        d.add_system_property("total_spin", torch.tensor([float(m)], device=device, dtype=dtype))
        d.add_system_property("total_charge", torch.tensor([0.0], device=device, dtype=dtype))
        items.append(d)
    return Batch.from_data_list(items, device=device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("catalyst", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--multiplicities", nargs="+", type=int, default=[49, 47, 51, 53])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dtype = torch.float64
    atoms = read(args.catalyst)
    atoms.calc = None
    from mace.calculators import MACECalculator

    raw = torch.load(args.model, map_location="cpu", weights_only=False)
    specs = getattr(raw, "embedding_specs", None)
    report: dict[str, Any] = {"model": args.model, "embedding_specs": specs, "elements": [int(z) for z in raw.atomic_numbers],
                              "multiplicities": args.multiplicities}
    calc = MACECalculator(model_paths=args.model, device=args.device, default_dtype="float64")
    native = {}
    for m in args.multiplicities:
        a = atoms.copy()
        a.info.update(charge=0, spin=m)
        a.calc = calc
        native[m] = (float(a.get_potential_energy()), np.array(a.get_forces()))
        calc.reset()
    report["native_energy_eV"] = {str(m): native[m][0] for m in args.multiplicities}
    report["native_spread_eV"] = float(max(v[0] for v in native.values()) - min(v[0] for v in native.values()))

    builtin = load_wrapper(args.model, device=args.device, dtype=dtype)
    b = batch_for(atoms, args.multiplicities, args.device, dtype)
    compute_neighbors(b, config=builtin.model_config.neighbor_config)
    try:
        out = builtin(b)
        e = out["energy"].detach().cpu().numpy().reshape(-1)
        report["builtin_wrapper"] = {"status": "ran", "energies_eV": e.tolist(),
                                     "distinguishes_multiplicity": bool(np.ptp(e) > 1e-9),
                                     "max_abs_dE_vs_native_eV": float(max(abs(e[i] - native[m][0])
                                                                          for i, m in enumerate(args.multiplicities)))}
    except Exception as exc:
        report["builtin_wrapper"] = {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    project = SpinMACEWrapper(builtin.model)
    out = project(b)
    e = out["energy"].detach().cpu().numpy().reshape(-1)
    f = out["forces"].detach().cpu().numpy().reshape(len(args.multiplicities), len(atoms), 3)
    report["project_wrapper_mixed_M_batch"] = {
        "energies_eV": e.tolist(),
        "max_abs_dE_vs_native_eV": float(max(abs(e[i] - native[m][0]) for i, m in enumerate(args.multiplicities))),
        "max_force_component_diff_vs_native": float(max(np.abs(f[i] - native[m][1]).max()
                                                        for i, m in enumerate(args.multiplicities)))}
    # reversed batch order: each graph must keep its own multiplicity
    rev = args.multiplicities[::-1]
    b2 = batch_for(atoms, rev, args.device, dtype)
    compute_neighbors(b2, config=builtin.model_config.neighbor_config)
    e2 = project(b2)["energy"].detach().cpu().numpy().reshape(-1)
    report["project_wrapper_reversed_order_max_abs_dE_eV"] = float(max(abs(e2[i] - native[m][0]) for i, m in enumerate(rev)))
    # missing field must fail loudly
    b3 = Batch.from_data_list([AtomicData.from_atoms(atoms, device=args.device, dtype=dtype)], device=args.device)
    compute_neighbors(b3, config=builtin.model_config.neighbor_config)
    try:
        project(b3)
        report["project_wrapper_missing_input"] = "ran silently (BAD)"
    except KeyError as exc:
        report["project_wrapper_missing_input"] = f"raised KeyError: {exc}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "electronic_inputs.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
