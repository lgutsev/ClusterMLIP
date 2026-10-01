"""Score a MACE checkpoint on train/valid/test, per group and multiplicity, against baselines.

usage: python eval_checkpoint.py TEMPLATE.model CHECKPOINT.pt DATASET_DIR [OUT.json]

TEMPLATE.model is any model with the identical architecture (e.g. the lr-0 "init" model);
the checkpoint's state dict is loaded into it strictly, so a mismatch fails loudly.
MACE saves checkpoints inside its EMA context, so these are the EMA weights it validated.

Baselines: forces -> zero force; energies -> the training-set mean energy per atom of the
same multiplicity (what a model that ignored geometry entirely could do).
"""
import collections, json, math, sys
import numpy as np
import torch
from common import DEV, batch_of, graphs, load_model
from ase.io import read

template, ckpt_path, dataset = sys.argv[1:4]
out_path = sys.argv[4] if len(sys.argv) > 4 else None
m = load_model(template)
state = torch.load(ckpt_path, map_location=DEV, weights_only=False)["model"]
m.load_state_dict(state, strict=True)
m.eval()


def predict(frames, chunk=16):
    E, F = [], []
    for k in range(0, len(frames), chunk):
        b = batch_of(graphs(m, frames[k:k + chunk]))
        out = m(b.to_dict(), training=False, compute_force=True)
        E += list(out["energy"].detach().cpu().numpy())
        f = out["forces"].detach().cpu().numpy()
        for g in range(b.num_graphs):
            F.append(f[int(b.ptr[g]):int(b.ptr[g + 1])])
    return np.array(E), F


def group_of(a):
    md = a.info.get("metadata")
    if isinstance(md, str):
        try:
            md = json.loads(md)
        except ValueError:
            md = {}
    return (md or {}).get("split_group") or a.info.get("parent_record_id", "?")


splits = {s: read(f"{dataset}/{s}.extxyz", ":") for s in ("train", "valid", "test")}
train_e_by_m = collections.defaultdict(list)
for a in splits["train"]:
    train_e_by_m[int(a.info["spin"])].append(a.info["REF_energy"] / len(a))
mean_e = {k: float(np.mean(v)) for k, v in train_e_by_m.items()}
all_mean = float(np.mean([e for v in train_e_by_m.values() for e in v]))

report = {"checkpoint": ckpt_path, "splits": {}}
for name, frames in splits.items():
    E, F = predict(frames)
    rows = []
    for a, e, f in zip(frames, E, F):
        n = len(a); M = int(a.info["spin"]); ref_f = a.arrays["REF_forces"]
        rows.append(dict(group=group_of(a), M=M,
                         de=(e - a.info["REF_energy"]) / n,
                         de0=mean_e.get(M, all_mean) - a.info["REF_energy"] / n,
                         df2=float(((f - ref_f) ** 2).mean()), f2=float((ref_f ** 2).mean())))

    def summarize(rs):
        return {"frames": len(rs),
                "rmse_e_meV_atom": round(1000 * math.sqrt(np.mean([r["de"] ** 2 for r in rs])), 2),
                "baseline_e_meV_atom": round(1000 * math.sqrt(np.mean([r["de0"] ** 2 for r in rs])), 2),
                "rmse_f_meV_A": round(1000 * math.sqrt(np.mean([r["df2"] for r in rs])), 2),
                "zero_force_meV_A": round(1000 * math.sqrt(np.mean([r["f2"] for r in rs])), 2)}

    by_group = collections.defaultdict(list); by_m = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r); by_m[r["M"]].append(r)
    report["splits"][name] = {"all": summarize(rows),
                              "by_group": {g: summarize(v) for g, v in sorted(by_group.items())},
                              "by_M": {str(k): summarize(v) for k, v in sorted(by_m.items())}}

for name, s in report["splits"].items():
    a = s["all"]
    print(f"{name:5} {a['frames']:4d} frames | E {a['rmse_e_meV_atom']:7.2f} (baseline {a['baseline_e_meV_atom']:6.2f}) meV/atom"
          f" | F {a['rmse_f_meV_A']:7.2f} (zero-force {a['zero_force_meV_A']:6.2f}) meV/A")
    for g, v in s["by_group"].items():
        print(f"        {g[:30]:30} {v['frames']:4d} | E {v['rmse_e_meV_atom']:7.2f} ({v['baseline_e_meV_atom']:6.2f}) | F {v['rmse_f_meV_A']:7.2f} ({v['zero_force_meV_A']:6.2f})")
if out_path:
    json.dump(report, open(out_path, "w"), indent=2)
