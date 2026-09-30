"""Table of the memorization grid from logs/*.jsonl."""
import glob, json, os
rows = []
for path in sorted(glob.glob(os.path.join(os.path.dirname(__file__) or ".", "logs", "n*.jsonl"))):
    recs = [json.loads(l) for l in open(path) if l.startswith("{")]
    if not recs: continue
    head, steps, final = recs[0], [r for r in recs if "step" in r], recs[-1]
    if "final_rmse_f_mev_A" not in final: continue
    s0, s50 = steps[0], next((r for r in steps if r["step"] == 50), steps[-1])
    rows.append((head["frames"], head["mode"], head["lr"], s0["rmse_e_mev_atom"], s0["rmse_f_mev_A"],
                 s50["rmse_e_mev_atom"], s50["rmse_f_mev_A"], final["final_rmse_e_mev_atom"], final["final_rmse_f_mev_A"],
                 s0["loss_e_weighted"], s0["loss_f_weighted"], s0["grad_total"], steps[-1]["grad_total"],
                 max(v for v in final["relative_param_change"].values() if v is not None)))
print("| frames | loss | lr | E0 | F0 | E@50 | F@50 | E final | F final | wL_E@0 | wL_F@0 | |g|@0 | |g|end | max rel dparam |")
print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in sorted(rows, key=lambda r: (r[0], r[1], -r[2])):
    print("| " + " | ".join(f"{x:.3g}" if isinstance(x, float) else str(x) for x in r) + " |")
