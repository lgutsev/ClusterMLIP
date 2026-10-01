"""Write a MACE checkpoint (EMA weights, as MACE saves them) into a loadable .model.

usage: python ckpt_to_model.py TEMPLATE.model CHECKPOINT.pt OUT.model

TEMPLATE must have the identical architecture (e.g. the same `cluster-mlip train` run
regenerated with --max-num-epochs 1 --extra-arg=--lr=0.0); the state dict is loaded
strictly, so any mismatch fails. Used when a run was interrupted before MACE wrote its
own final model from the best checkpoint.
"""
import sys
import torch

template, ckpt, out = sys.argv[1:4]
model = torch.load(template, map_location="cpu", weights_only=False)
model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=True)
torch.save(model, out)
print(f"wrote {out}")
