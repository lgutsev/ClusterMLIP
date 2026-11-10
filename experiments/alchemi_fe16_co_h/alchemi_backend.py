"""Thin bridge between the project's ASE structures and nvalchemi-toolkit (v0.3.0).

Nothing here changes a model or a force field. The functions convert ASE ``Atoms``
(with the project's per-atom ``role`` / ``mol_id`` arrays) into toolkit batches, run
single points or batched relaxations, and hand back plain numpy results in the
caller's order. Candidate identity is kept *outside* the toolkit tensors: every
structure carries an integer ``slot`` and results are matched back by it.

Import of this module needs the toolkit environment; the rest of the package and
the native ASE path do not.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from ase import Atoms

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors

ROLE_ADSORBATE = 2


def load_wrapper(path: str, *, device: str = "cuda", dtype: torch.dtype = torch.float32,
                 wrapper: type[MACEWrapper] = MACEWrapper) -> MACEWrapper:
    """The same checkpoint file as the native calculator: no cuEq, no compilation.

    Loads the file directly instead of ``MACEWrapper.from_checkpoint``: on Windows
    that helper sends a ``D:/...`` path to the foundation-model downloader, which
    reads the drive letter as a URL scheme.
    """
    model = torch.load(path, map_location=torch.device(device), weights_only=False)
    model = model.to(dtype=dtype).to(torch.device(device))
    return wrapper(model)


def to_data(atoms: Atoms, *, device: str, dtype: torch.dtype, slot: int) -> AtomicData:
    bare = Atoms(atoms.get_chemical_symbols(), positions=atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
    data = AtomicData.from_atoms(bare, device=device, dtype=dtype)
    role = atoms.arrays["role"] if "role" in atoms.arrays else np.ones(len(atoms), dtype=int)
    data.add_node_property("frozen", torch.as_tensor(role != ROLE_ADSORBATE, device=device))
    data.add_system_property("slot", torch.tensor([slot], device=device, dtype=torch.long))
    return data


def to_batch(structures: Sequence[Atoms], *, device: str = "cuda", dtype: torch.dtype = torch.float32,
             slots: Sequence[int] | None = None) -> Batch:
    slots = list(range(len(structures))) if slots is None else list(slots)
    return Batch.from_data_list([to_data(a, device=device, dtype=dtype, slot=s) for a, s in zip(structures, slots)],
                                device=device)


def single_points(model: MACEWrapper, structures: Sequence[Atoms], *, device: str = "cuda",
                  dtype: torch.dtype = torch.float32) -> list[dict[str, Any]]:
    """Energies and forces for one batch, returned per structure in input order."""
    batch = to_batch(structures, device=device, dtype=dtype)
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    out = model(batch)
    energy = out["energy"].detach().double().cpu().numpy().reshape(-1)
    forces = out["forces"].detach().double().cpu().numpy()
    ptr = batch.batch_ptr.cpu().numpy()
    edges = batch.neighbor_list.cpu().numpy()
    graph_of = batch.batch_idx.cpu().numpy()
    cross = int((graph_of[edges[:, 0]] != graph_of[edges[:, 1]]).sum())
    results = []
    for k in range(len(structures)):
        in_graph = graph_of[edges[:, 0]] == k
        results.append({"energy": float(energy[k]), "forces": forces[ptr[k]:ptr[k + 1]].copy(),
                        "edges": int(in_graph.sum()), "cross_system_edges": cross})
    return results
