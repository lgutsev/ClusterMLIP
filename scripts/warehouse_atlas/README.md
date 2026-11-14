# Warehouse Atlas

Builds one self-contained, interactive HTML page that summarizes the Gaussian
structure warehouses: a chemical-space map linked to a 3D structure viewer,
element and spin coverage, exact-geometry overlap between warehouses, the
FenOm_Warehouse2 spin-ladder campaign, and the `dataset_spin_v0` labels.

```bash
python scripts/warehouse_atlas/build_atlas.py --root D:/MLIP_Work_Folder/Cluster_MLIP
```

The output goes to `scripts/warehouse_atlas/out/` (gitignored):
`atlas_data.json` and `warehouse_atlas.html`. A full build takes a few
minutes because it reads every `seeds.extxyz`.

## Adding a warehouse

Add one entry to the `WH` dict at the top of `build_atlas.py`:

```python
"W3": ("New_Warehouse", ROOT / "New_Warehouse/extracted/seeds.extxyz"),
```

The page reads the warehouse list from the data, so cards, filters, table
columns, the overlap chart and the spin toggle all pick it up. The layout
reserves room for four warehouse cards. For a short description on its card,
add a line to `blurb` in `atlas_template.html`. Otherwise the card shows a
generic description.

The campaign and dataset sections are specific to FenOm_Warehouse2 (`W2`).

## Sub-atlases

`--scopes scopes.json` splits the seeds into sub-atlases. Each seed goes to
the first scope whose `match` rule holds. The root page gains an owner filter,
an owner column and links to the sub-atlases. Each scope gets its own page with
the same layout. Keep the scopes file outside the repo when it names projects
that must not live here.

```json
{
  "root_url": "https://claude.ai/artifact/<root id>",
  "scopes": [
    {"id": "A", "title": "Example Ion Atlas", "families": "molecule",
     "match": {"any": [{"source_files": ["X.doc"]},
                       {"only_elements": ["C", "H", "N", "O"], "require": ["C", "H", "N", "O"]}]},
     "pending": [{"f": "CH3NO2", "label": "nitromethane"}],
     "axes": ["nAt", "mmax"], "spinX": "at", "out": "path/to/a.html", "url": null},
    {"id": "B", "title": "Example Fe Atlas", "match": {"contains": ["Fe"]}, "out": "path/to/b.html"},
    {"id": "C", "title": "Example Rest Atlas", "families": "metal", "match": {"rest": true}, "out": "path/to/c.html"}
  ]
}
```

- **Rule keys:** `source_files` (basenames), `formulas`, `contains`, `only_elements`,
  `require`, `warehouses`, `rest`, and `any` (a list of sub-rules, any of which may match).
- **`families` picks the colour key:**
  - `fe` (the default) for Fe chemistry;
  - `metal` for non-Fe material, grouped by main metal, with non-metal molecules in grey;
  - `molecule` for small organic molecules and ions.
- **`pending`** lists formulas that are not collected yet. Each shows as found once it appears in a warehouse.
- **`url`** links the pages to each other. Publish once, fill in the URLs, then rebuild.
- **Campaign and dataset sections:** a scope gets them only when every labeled formula belongs to it.
