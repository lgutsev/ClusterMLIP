"""Shared helpers: load a MACE model and build training batches exactly as MACE does."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from ase.io import read
from mace import data as mace_data
from mace.tools import torch_geometric, torch_tools, utils

torch.set_default_dtype(torch.float64)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

KEYSPEC = mace_data.KeySpecification(
    info_keys={"energy": "REF_energy", "total_charge": "charge", "total_spin": "spin"},
    arrays_keys={"forces": "REF_forces"},
)


def load_model(path):
    model = torch.load(path, map_location=DEV, weights_only=False)
    return model.double().to(DEV)


def graphs(model, atoms_list, keyspec=KEYSPEC):
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    heads = getattr(model, "heads", ["Default"])
    out = []
    for atoms in atoms_list:
        cfg = mace_data.config_from_atoms(atoms, key_specification=keyspec, head_name="Default")
        out.append(mace_data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max), heads=heads))
    return out


def batch_of(graph_list):
    loader = torch_geometric.dataloader.DataLoader(graph_list, batch_size=len(graph_list), shuffle=False)
    return next(iter(loader)).to(DEV)


def frames(path, idx=None):
    fr = read(path, ":")
    return fr if idx is None else [fr[i] for i in idx]
