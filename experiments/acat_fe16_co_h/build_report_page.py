"""Build the single-file HTML report page from the analysis outputs.

    python build_report_page.py ANALYSIS_DIR GATE_DIR --out page.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path

REASON = {
    "lowest_of_motif": "Lowest of its motif",
    "near_degenerate": "Near-degenerate competitor",
    "largest_model_disagreement": "Models disagree most",
    "diverse_random_control": "Diverse random control",
    "constructed_control": "Constructed control",
}
SITE = {"top": "top", "bridge": "bridge", "hollow3": "hollow", "hollow4": "4-fold hollow"}


def motif_text(motif: str) -> str:
    parts = []
    for item in motif.split("|"):
        name, where = item.split("@")
        if where == "gas":
            parts.append(f"{name} desorbed")
            continue
        element, kind = where.split("-", 1)
        label = SITE.get(kind, kind)
        if name == "CO" and element == "O":
            label += " (O-bound)"
        parts.append(f"{name} {label}")
    return " · ".join(parts)


def img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def num(value: str, digits: int = 3) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "–"
    if v != v:
        return "–"
    return f"{v:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", type=Path)
    ap.add_argument("gate", type=Path)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    curves = json.loads((args.analysis / "curves.json").read_text(encoding="utf-8"))
    runs = list(csv.DictReader(open(args.analysis / "runs.csv", encoding="utf-8")))
    shortlist = list(csv.DictReader(open(args.analysis / "shortlist.csv", encoding="utf-8")))
    gate_e = json.loads((args.gate / "gate_small_clusters.json").read_text(encoding="utf-8"))
    gate_f = json.loads((args.gate / "gate_forces.json").read_text(encoding="utf-8"))

    run_rows = []
    for r in sorted(runs, key=lambda r: (r["arm"] != "random", int(r["seed"]))):
        run_rows.append(
            f"<tr><td><span class='dot {r['arm']}'></span>{r['arm'].capitalize()}</td><td class='n'>{r['seed']}</td>"
            f"<td class='n'>{r['proposals']}</td><td class='n'>{r['invalid']} / {r['duplicate_start']}</td>"
            f"<td class='n'>{r['relaxations']}</td><td class='n'>{r['converged']}</td>"
            f"<td class='n'>{r['duplicates_after_relax']}</td><td class='n'>{r['unique']}</td><td class='n'>{r['motifs']}</td>"
            f"<td class='n'>{int(r['calc_calls']):,}</td><td class='n strong'>{float(r['best_e_rel_to_pool_min']):.3f}</td></tr>")

    names = {"CFe2O": "Fe2(CO)", "CFe4O": "Fe4(CO)", "H2Fe2": "Fe2H2", "H2Fe4": "Fe4H2", "Fe16": "Fe16 (bare)"}
    gate_rows = []
    for key, label in names.items():
        cells = [f"<td>{label}</td><td class='n'>{gate_e['models']['mpa0']['per_formula'][key]['distinct_geometries']}</td>"]
        for model in ("mpa0", "matpes_r2scan"):
            e = gate_e["models"][model]["per_formula"][key]
            acc = 100 * e["same_M_pair_order_accuracy"]
            cls = "ok" if acc >= 90 and e["same_M_rel_energy_mae_ev"] <= 0.03 else "bad"
            cells.append(f"<td class='n {cls}'>{acc:.0f}%</td><td class='n {cls}'>{e['same_M_rel_energy_mae_ev']:.2f}</td>")
        gate_rows.append("<tr>" + "".join(cells) + "</tr>")
    force_rows = []
    for group in ("Fe16 (bare)", "Fe2H2"):
        for el, m in gate_f["models"]["mpa0"][group].items():
            r2 = gate_f["models"]["matpes_r2scan"][group][el]
            force_rows.append(f"<tr><td>{group}</td><td>{el}</td><td class='n bad'>{m['model_rmse_ev_per_A']:.2f}</td>"
                              f"<td class='n bad'>{r2['model_rmse_ev_per_A']:.2f}</td>"
                              f"<td class='n'>{m['zero_model_rmse_ev_per_A']:.2f}</td></tr>")

    cards = []
    for k, r in enumerate(shortlist):
        reason = r["reason"].split(":")[0]
        tag = REASON.get(reason, "Flagged: " + r["reason"].split(":")[-1].replace("_", " "))
        e1 = num(r["model_e_rel_ev"])
        e2 = num(r["second_model_e_rel_ev"])
        energies = (f"<dl><div><dt>MPA-0</dt><dd>{e1}</dd></div><div><dt>r2SCAN</dt><dd>{e2}</dd></div></dl>"
                    if e1 != "–" else "<p class='note'>Built by hand, not relaxed</p>")
        cards.append(
            f"<figure class='cand'><img alt='Structure {html.escape(r['shortlist_id'])}' src='{img(args.analysis / 'pics' / f'shortlist_{k:03d}.png')}'>"
            f"<figcaption><span class='tag {reason}'>{html.escape(tag)}</span>"
            f"<b>{html.escape(motif_text(r['motif']))}</b>{energies}"
            f"<span class='id'>{html.escape(r['shortlist_id'])} · input {k:03d}</span></figcaption></figure>")

    page = args.template.read_text(encoding="utf-8")
    page = (page.replace("/*CURVES*/", json.dumps(curves))
            .replace("<!--RUN_ROWS-->", "\n".join(run_rows))
            .replace("<!--GATE_ROWS-->", "\n".join(gate_rows))
            .replace("<!--FORCE_ROWS-->", "\n".join(force_rows))
            .replace("<!--CARDS-->", "\n".join(cards))
            .replace("SITES_IMG", img(args.analysis / "pics" / "sites.png")))
    args.out.write_text(page, encoding="utf-8")
    print(f"wrote {args.out} ({len(page) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
