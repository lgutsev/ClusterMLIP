"""Staged memorization test driving the real MACE model with its own optimizer groups.

usage: python memorize.py MODEL FRAMES.extxyz IDX(comma|all) MODE(E|F|EF) STEPS [LR] [CLIP] [WE] [WF]

Full-batch Adam(amsgrad) with MACE's parameter groups and weight decays. Logs the
per-atom energy MSE and force MSE, their weighted contributions, gradient norms by
group, and at the end the parameter change by group.
"""
import copy, json, math, os, sys
import torch
from common import *

model_path, frames_path, idx_s, mode, steps = sys.argv[1:6]
steps = int(steps)
lr = float(sys.argv[6]) if len(sys.argv) > 6 else 0.005
clip = None if len(sys.argv) <= 7 or sys.argv[7] == "none" else float(sys.argv[7])
we = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
wf = float(sys.argv[9]) if len(sys.argv) > 9 else 100.0
if mode == "E": wf = 0.0
if mode == "F": we = 0.0

m = load_model(model_path)
fr = frames(frames_path)
idx = list(range(len(fr))) if idx_s == "all" else [int(i) for i in idx_s.split(",")]
# Chunked gradient accumulation: every frame has the same atom count, so weighting
# each chunk's mean losses by its share of frames reproduces the full-batch loss
# and gradient exactly while fitting in GPU memory.
CHUNK = int(os.environ.get("CHUNK", "10"))
all_graphs = graphs(m, [fr[i] for i in idx])
chunks = [batch_of(all_graphs[k:k + CHUNK]) for k in range(0, len(all_graphs), CHUNK)]

groups = {"embedding": list(m.node_embedding.parameters()),
          "interactions": list(m.interactions.parameters()),
          "products": list(m.products.parameters()),
          "readouts": list(m.readouts.parameters())}
if hasattr(m, "joint_embedding"): groups["joint_embedding"] = list(m.joint_embedding.parameters())
if hasattr(m, "embedding_readout"): groups["embedding_readout"] = list(m.embedding_readout.parameters())
grouped = [id(p) for ps in groups.values() for p in ps]
all_params = list(m.named_parameters())
orphans = [n for n, p in all_params if id(p) not in grouped]
dupes = len(grouped) - len(set(grouped))
frozen = [n for n, p in all_params if not p.requires_grad]
decay_names = {id(p) for n, p in m.interactions.named_parameters() if "linear.weight" in n or "skip_tp_full.weight" in n}
opt_groups = []
for name, ps in groups.items():
    if name == "interactions":
        opt_groups.append({"params": [p for p in ps if id(p) in decay_names], "weight_decay": 5e-7})
        opt_groups.append({"params": [p for p in ps if id(p) not in decay_names], "weight_decay": 0.0})
    else:
        opt_groups.append({"params": ps, "weight_decay": 5e-7 if name == "products" else 0.0})
opt = torch.optim.Adam(opt_groups, lr=lr, amsgrad=True)
init = {n: p.detach().clone() for n, p in all_params}

def chunk_losses(b):
    out = m(b.to_dict(), training=True, compute_force=True)
    n_atoms = (b.ptr[1:] - b.ptr[:-1]).double()
    le = torch.mean(torch.square((b["energy"] - out["energy"]) / n_atoms))
    lf = torch.mean(torch.square(b["forces"] - out["forces"]))
    return le, lf


def accumulate(backward):
    """Full-set mean losses; with backward=True also accumulates their gradient."""
    le_tot = lf_tot = 0.0
    for b in chunks:
        w = b.num_graphs / len(all_graphs)
        le, lf = chunk_losses(b)
        if backward:
            (w * (we * le + wf * lf)).backward()
        le_tot += w * float(le); lf_tot += w * float(lf)
    return le_tot, lf_tot


def evaluate():
    le, lf = accumulate(backward=False)
    return None, torch.tensor(le), torch.tensor(lf)

def gnorm(ps):
    g = [p.grad for p in ps if p.grad is not None]
    return math.sqrt(sum(float(x.pow(2).sum()) for x in g)) if g else 0.0

log = []
report_at = sorted({0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, steps - 1})
print(json.dumps({"frames": len(idx), "mode": mode, "lr": lr, "clip": clip, "we": we, "wf": wf,
                  "n_params": sum(p.numel() for _, p in all_params), "orphan_params": orphans,
                  "duplicate_group_params": dupes, "frozen_params": frozen}))
for step in range(steps):
    opt.zero_grad()
    le, lf = accumulate(backward=True)
    le, lf = torch.tensor(le), torch.tensor(lf)
    gn = {k: round(gnorm(v), 6) for k, v in groups.items()}
    total = math.sqrt(sum(v * v for v in gn.values()))
    if clip is not None:
        torch.nn.utils.clip_grad_norm_([p for _, p in all_params], clip)
    opt.step()
    if step in report_at:
        row = {"step": step, "rmse_e_mev_atom": round(1000 * float(le.sqrt()), 3),
               "rmse_f_mev_A": round(1000 * float(lf.sqrt()), 3),
               "loss_e_weighted": float(we * le), "loss_f_weighted": float(wf * lf),
               "grad_total": round(total, 6), "grad": gn}
        print(json.dumps(row)); log.append(row)
change = {}
for name, ps in groups.items():
    ids = {id(p) for p in ps}
    num = sum(float((p.detach() - init[n]).pow(2).sum()) for n, p in all_params if id(p) in ids)
    den = sum(float(init[n].pow(2).sum()) for n, p in all_params if id(p) in ids)
    change[name] = round(math.sqrt(num / den), 5) if den else None
per_frame = []
for b in chunks:
    out = m(b.to_dict(), training=False, compute_force=True)
    dF = (b["forces"] - out["forces"]).detach()
    ref = b["forces"].detach()
    for g in range(b.num_graphs):
        lo, hi = int(b.ptr[g]), int(b.ptr[g + 1])
        per_frame.append((round(1000 * float(dF[lo:hi].pow(2).mean().sqrt()), 2),
                          round(1000 * float(ref[lo:hi].pow(2).mean().sqrt()), 2)))
print(json.dumps({"per_frame_force_rmse_and_label_rms_meV_A": dict(zip(idx, per_frame))}))
_, le, lf = evaluate()
print(json.dumps({"final_rmse_e_mev_atom": round(1000 * float(le.sqrt()), 3),
                  "final_rmse_f_mev_A": round(1000 * float(lf.sqrt()), 3),
                  "relative_param_change": change}))
