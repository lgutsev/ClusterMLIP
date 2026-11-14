"""Build the Warehouse Atlas page (and optional sub-atlases) from local warehouse mirrors.

Usage:
    python build_atlas.py [--root D:/MLIP_Work_Folder/Cluster_MLIP] [--out out] [--scopes scopes.json]

Aggregates each warehouse's seeds.extxyz plus FenOm_Warehouse2's monitoring and
dataset_spin_v0 files, then inlines the data into atlas_template.html. Without
--scopes it writes one root page. With --scopes, every seed is assigned to the
first scope whose rule matches it, the root page gains an owner layer, and each
scope gets its own page with the same layout (see README.md for the format).
"""
import argparse, json, re, hashlib, math, collections as C
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--root", type=Path, default=Path(r"D:/MLIP_Work_Folder/Cluster_MLIP"), help="folder holding the warehouse mirrors")
ap.add_argument("--out", type=Path, default=HERE / "out", help="where the root atlas_data.json and warehouse_atlas.html go")
ap.add_argument("--scopes", type=Path, help="optional JSON file defining sub-atlases (kept outside the repo)")
args = ap.parse_args()
ROOT = args.root
args.out.mkdir(parents=True, exist_ok=True)
# Add a fourth warehouse here: "KEY": ("FolderName", path to its seeds.extxyz). The page adapts.
WH = {
    "W1": ("FenOm_Warehouse", ROOT / "FenOm_Warehouse/extracted/seeds.extxyz"),
    "W2": ("FenOm_Warehouse2", ROOT / "FenOm_Warehouse2/seeds.extxyz"),
    "G": ("General_Warehouse", ROOT / "General_Warehouse/extracted/seeds.extxyz"),
    "L": ("Gutsev_Legacy_Warehouse", ROOT / "Gutsev_Legacy_Warehouse/extracted/seeds.extxyz"),
}
CAMPAIGN_WH = "W2"   # the warehouse whose monitoring/ and dataset_spin_v0/ feed the campaign and dataset sections
KV = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')


def frames(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.strip():
                continue
            n = int(line)
            info = {k: v.strip('"') for k, v in KV.findall(fh.readline())}
            sym, pos, extra = [], [], []
            for _ in range(n):
                p = fh.readline().split()
                sym.append(p[0]); pos.append(tuple(float(x) for x in p[1:4])); extra.append(p[4:])
            yield info, sym, pos, extra


def parse_formula(f):
    return {e: int(n or 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", f)}


def family(comp):
    others = sorted(e for e in comp if e != "Fe")
    if "Fe" not in comp:
        return "no Fe"
    if not others:
        return "Fe"
    if others == ["O"]:
        return "Fe–O"
    if set(others) <= {"O", "N", "H", "C"}:
        return "Fe–" + "".join(o for o in ["O", "N", "C", "H"] if o in others)
    return "Fe + other metal"


def geom_hash(sym, pos, q, m):
    s = ";".join(f"{a}{x:.4f},{y:.4f},{z:.4f}" for a, (x, y, z) in zip(sym, pos)) + f"|{q}|{m}"
    return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()


# ---- scope rules ----
def rule_matches(rule, seed):
    """A rule is a dict; every key present must hold. {"any": [...]} matches if one sub-rule does."""
    if "any" in rule:
        return any(rule_matches(r, seed) for r in rule["any"])
    comp, src = seed["comp"], seed["src"]
    if rule.get("rest"):
        return True
    if "source_files" in rule and src.split("/")[-1] not in set(rule["source_files"]):
        return False
    if "formulas" in rule and seed["f"] not in set(rule["formulas"]):
        return False
    if "contains" in rule and not all(e in comp for e in rule["contains"]):
        return False
    if "only_elements" in rule and not set(comp) <= set(rule["only_elements"]):
        return False
    if "require" in rule and not all(e in comp for e in rule["require"]):
        return False
    if "warehouses" in rule and seed["wh"] not in rule["warehouses"]:
        return False
    return True


cfg_scopes = json.loads(args.scopes.read_text(encoding="utf-8"))["scopes"] if args.scopes else []


def owner_of(seed):
    for s in cfg_scopes:
        if rule_matches(s["match"], seed):
            return s["id"]
    return None


class Agg:
    """Everything the page needs, accumulated over the seeds that fall in one scope."""
    def __init__(self):
        self.formulas, self.hashes = {}, C.defaultdict(set)
        self.tree, self.spin, self.spinAt, self.cfg, self.chg = ({k: C.Counter() for k in WH} for _ in range(5))
        self.seeds, self.sources = C.Counter(), {k: set() for k in WH}

    def add(self, s, sym, pos, own):
        key, f, q, m, ct, src = s["wh"], s["f"], s["q"], s["m"], s["ct"], s["src"]
        parts = src.split("/")
        folder = parts[0] if len(parts) > 1 else "(root)"
        sub = parts[1] if len(parts) > 2 else "·"
        comp = s["comp"]; nfe = comp.get("Fe", 0)
        self.seeds[key] += 1; self.sources[key].add(src)
        self.tree[key][(folder, sub)] += 1
        self.spin[key][(nfe, m - 1)] += 1
        self.spinAt[key][(len(sym), m - 1)] += 1
        self.cfg[key][ct] += 1; self.chg[key][q] += 1
        self.hashes[s["h"]].add(key)
        a = self.formulas.setdefault(f, {
            "f": f, "comp": comp, "fam": family(comp), "nFe": nfe, "nAt": len(sym),
            "wh": C.Counter(), "cfg": C.Counter(), "mult": C.Counter(), "q": C.Counter(),
            "folders": C.Counter(), "own": C.Counter(), "best": None,
        })
        a["wh"][key] += 1; a["cfg"][ct] += 1; a["mult"][m] += 1; a["q"][q] += 1
        a["folders"][f"{key}:{folder}"] += 1
        if own:
            a["own"][own] += 1
        # Representative geometry: lowest legacy energy, preferring true minima.
        e = s["e"]
        rank = (0 if ct == "minimum" else 1, e if e is not None else 0.0)
        if a["best"] is None or rank < a["best"][0]:
            a["best"] = (rank, {"sym": sym, "pos": [[round(c, 3) for c in p] for p in pos],
                                "src": f"{key} · {src}", "q": q, "m": m, "ct": ct, "e": e})

    def export(self):
        out = []
        for a in self.formulas.values():
            out.append({
                "f": a["f"], "fam": a["fam"], "nFe": a["nFe"], "nAt": a["nAt"],
                "nX": a["nAt"] - a["nFe"], "comp": a["comp"],
                "wh": dict(a["wh"]), "n": sum(a["wh"].values()),
                "cfg": dict(a["cfg"]), "mult": {str(k): v for k, v in sorted(a["mult"].items())},
                "q": {str(k): v for k, v in sorted(a["q"].items())},
                "folders": dict(a["folders"].most_common(6)),
                "own": dict(a["own"]), "geo": a["best"][1],
            })
        out.sort(key=lambda r: -r["n"])
        keys = [k for k in WH if self.seeds[k]]
        pairs = lambda d: {k: [[a, b, c] for (a, b), c in d[k].items()] for k in keys}
        return {
            "warehouses": [{"key": k, "name": WH[k][0]} for k in keys],
            "seeds": {k: self.seeds[k] for k in keys},
            "sources": {k: len(self.sources[k]) for k in keys},
            "cfg": {k: dict(self.cfg[k]) for k in keys},
            "chg": {k: {str(q): n for q, n in sorted(self.chg[k].items())} for k in keys},
            "spin": pairs(self.spin), "spinAt": pairs(self.spinAt), "tree": pairs(self.tree),
            "overlap": dict(C.Counter("+".join(sorted(s)) for s in self.hashes.values())),
            "formulas": out,
        }


root = Agg()
scoped = {s["id"]: Agg() for s in cfg_scopes}
for key, (name, path) in WH.items():
    n = 0
    for info, sym, pos, _ in frames(path):
        n += 1
        f = info.get("formula") or "".join(sorted(set(sym)))
        q, m = int(info.get("charge", 0)), int(info.get("multiplicity", 1))
        e = info.get("legacy_energy_hartree")
        seed = {"wh": key, "f": f, "comp": parse_formula(f), "q": q, "m": m,
                "ct": info.get("config_type", "unknown"), "src": info.get("source", ""),
                "e": float(e) if e not in (None, "", "nan") else None, "h": geom_hash(sym, pos, q, m)}
        own = owner_of(seed)
        root.add(seed, sym, pos, own)
        if own:
            scoped[own].add(seed, sym, pos, own)
    print(key, n, "seeds")

# ---- Warehouse-level file summaries (root only) ----
def report(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None

files = {}
for key, (name, _) in WH.items():
    an = ROOT / name / "analysis"
    if (an / "files.csv").exists():
        d = pd.read_csv(an / "files.csv")
        d["suffix"] = d["suffix"].where(~d["suffix"].str.match(r"^\.o\d+$", na=False), ".o<jobid>")
        r = pd.read_csv(an / "records.csv", usecols=["geometry_state_sha256"])
        files[key] = {
            "status": d["status"].value_counts().to_dict(),
            "suffix": d.groupby("suffix")["size_bytes"].agg(["count", "sum"]).sort_values("count", ascending=False)
                       .head(10).reset_index().values.tolist(),
            "bytes": int(d["size_bytes"].sum()), "records": len(r), "unique": int(r["geometry_state_sha256"].nunique()),
        }
    else:
        m = pd.read_csv(ROOT / name / "extracted/manifest.csv", usecols=["source"])
        files[key] = {"records": None, "unique": len(m), "source_files": int(m["source"].nunique())}

# ---- campaign (one cell per job) ----
W2 = ROOT / WH[CAMPAIGN_WH][0]
ra = pd.read_csv(W2 / "monitoring/route_audit.csv")
jp = pd.read_csv(W2 / "monitoring/job_progress.csv")
def base(inp):
    s = Path(str(inp)).name
    return re.sub(r"__(routefix|restart)\d+.*?(?=\.gjf$)", "", s)
jp["base"] = jp["input"].map(base)
latest = jp.set_index("base")[["state", "observed_multiplicity", "last_energy_hartree", "completed_stages", "planned_stages"]].to_dict("index")
issues = pd.read_csv(W2 / "monitoring/audit_issues.csv")
issue_by = C.defaultdict(list)
for _, r in issues.iterrows():
    issue_by[base(r["input"])].append(r["detail"].split(" (")[0].split(":")[0])
jobs = []
for _, r in ra.iterrows():
    b = base(r["input"])
    name = b.split("__")[0]
    parts = b.split("__")
    st, snap = r["state"], "Sep 10"
    if b in latest:
        st, snap = latest[b]["state"], "Sep 23"
    jobs.append({
        "b": int(r["batch"][-4:]), "n": name, "fx": parts[1] if len(parts) > 1 else "",
        "ct": r["config_type"], "s": st, "snap": snap,
        "lad": parts[-1].replace(".gjf", "").replace("auto-spin-", "") if len(parts) > 5 else "",
        "rl": bool(r["must_relaunch"]), "iss": issue_by.get(b, []),
    })
campaign = {
    "jobs": jobs,
    "restart_plan": len(pd.read_csv(W2 / "restart_plan.csv")),
    "route_fix_plan": len(pd.read_csv(W2 / "route_fix_plan.csv")),
    "spin_plan": pd.read_csv(W2 / "spin_plan.csv")["status"].value_counts().to_dict(),
    "skipped_seeds": pd.read_csv(W2 / "skipped_spin_seeds.csv")["reason"]
        .str.replace(r":multiplicity.*", "", regex=True).value_counts().to_dict(),
    "failed_outputs": sum(1 for _ in open(W2 / "dataset_spin_v0/failed_outputs.tsv", encoding="utf-8")),
    "w2files": files[CAMPAIGN_WH],   # whole-warehouse parse stats for the pipeline strip
}

# ---- dataset_spin_v0 (labeled frames) ----
ds = {"split": {}, "frms": [], "byFormula": C.Counter()}
for split in ["train", "valid", "test"]:
    n = 0
    for info, sym, pos, extra in frames(W2 / f"dataset_spin_v0/{split}.extxyz"):
        n += 1
        fr = [float(x) for row in extra for x in row[:3]]
        rms = math.sqrt(sum(x * x for x in fr) / max(len(fr), 1))
        ds["frms"].append([split[0], round(math.log10(max(rms, 1e-4)), 3), int(info.get("multiplicity", 1)),
                           len(sym), info.get("formula", "")])
        ds["byFormula"][info.get("formula", "")] += 1
    ds["split"][split] = n
    print(split, n)
ds["byFormula"] = dict(ds["byFormula"].most_common())


def scope_has_campaign(agg):
    """Campaign and dataset sections go to a page only if every labeled formula belongs to it."""
    fs = set(agg.formulas)
    return bool(fs) and all(f in fs for f in ds["byFormula"])


def write_page(data, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, separators=(",", ":"))
    tpl = (HERE / "atlas_template.html").read_text(encoding="utf-8")
    path.write_text(tpl.replace("/*TITLE*/", title).replace("/*DATA*/", blob), encoding="utf-8")
    print("page", path, f"{len(blob):,} data bytes")
    return blob


stamp = pd.Timestamp.now().strftime("%Y-%m-%d")
root_data = root.export()
root_scope = {"id": "root", "title": "Warehouse Atlas", "root": True, "links": [], "root_url": None}
for s in cfg_scopes:
    agg = scoped[s["id"]]
    root_scope["links"].append({"id": s["id"], "title": s["title"], "url": s.get("url"), "blurb": s.get("card", ""),
                                "seeds": sum(agg.seeds.values()), "formulas": len(agg.formulas)})
root_page = args.out / "warehouse_atlas.html"
root_url = json.loads(args.scopes.read_text(encoding="utf-8")).get("root_url") if args.scopes else None
blob = write_page({**root_data, "generated": stamp, "files": files, "campaign": campaign, "dataset": ds,
                   "scope": root_scope}, root_page, "Warehouse Atlas")
(args.out / "atlas_data.json").write_text(blob)

for s in cfg_scopes:
    agg = scoped[s["id"]]
    d = agg.export()
    sub_files = {k: {"records": None, "unique": None, "source_files": d["sources"][k]} for k in d["seeds"]}
    has_c = scope_has_campaign(agg)
    scope = {k: s.get(k) for k in ("id", "title", "eyebrow", "lede", "blurbs", "pending", "axes", "spinX", "note", "families")}
    scope["pending"] = [{**p, "found": sum(agg.formulas[p["f"]]["wh"].values()) if p["f"] in agg.formulas else 0}
                        for p in (s.get("pending") or [])]
    scope.update(root=False, root_url=root_url, links=[l for l in root_scope["links"] if l["id"] != s["id"]])
    write_page({**d, "generated": stamp, "files": sub_files, "campaign": campaign if has_c else None,
                "dataset": ds if has_c else None, "scope": scope}, Path(s["out"]), s["title"])
    print(s["id"], sum(agg.seeds.values()), "seeds,", len(agg.formulas), "formulas", "+ campaign" if has_c else "")
print("formulas", len(root_data["formulas"]), "overlap", root_data["overlap"])
