"""Generate MACE training campaigns from a `collect` dataset.

This replaces the two hand-edited shell scripts (`configs/train_from_scratch.sh`,
`configs/finetune_foundation.sh`) with a config-driven generator, the same idea
as InterfaceForge's `iface train mace`, specialized for isolated-cluster
Gaussian data.

The parts that make a run *Gaussian- and cluster-appropriate* are locked, not
left as flags to get wrong:

- ``--default_dtype=float64`` -- Gaussian energies are effectively exact;
  float32 would inject noise into the labels the model fits;
- ``--stress_weight=0`` and no ``--compute_stress`` -- isolated clusters have
  no cell; every training frame is asserted ``pbc="F F F"``;
- ``--energy_key=REF_energy`` / ``--forces_key=REF_forces`` -- what `collect`
  writes;
- graph-level total-charge and total-spin conditioning is always on (either as
  MACE's native charge/spin embedding when fine-tuning an OMol/POLAR
  foundation, or as explicit ``--embedding_specs`` categorical modules when
  training from scratch);
- a run is refused if the dataset mixes electronic-structure label routes
  (the README's "do not mix energies or forces from different methods in one
  target head"), unless that is made explicit.

Nothing here imports ``mace``/``torch``: it only renders the command line and a
provenance manifest, so it is fully unit-testable with the base install -- the
same split as `evaluate.summarize_evaluation` vs `evaluate.predict_with_mace`.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from .io import parse_extxyz_info_line

SCHEMA_VERSION = 1

DEFAULT_SEED = 20260811
REQUIRED_SPLITS = ("train.extxyz", "valid.extxyz", "test.extxyz")

Mode = Literal["scratch", "finetune"]

# Foundation checkpoints that carry their own graph-level total-charge and
# total-spin conditioning, passed through ``atoms.info["charge"]`` /
# ``atoms.info["spin"]`` -- the same keys `mace_glue.record_to_atoms` already
# sets. Fine-tuning one of these must NOT also attach custom --embedding_specs
# modules: the conditioning already exists in the pretrained weights.
POLAR_FOUNDATIONS = {"polar-1-s", "polar-1-m", "polar-1-l"}
OMOL_FOUNDATIONS = {"mace-omol", "omol", "mace-omol-0"}
# Generic materials foundations (mace-mp-0 family). These were pretrained
# without any charge/spin channel, so a charge/spin cluster run has to add the
# embedding modules on top -- and whether a given mace-torch version tolerates
# that on a foundation checkpoint is the open question already flagged in
# configs/finetune_foundation.sh. Allowed, but with a loud PREFLIGHT note.
GENERIC_FOUNDATIONS = {"small", "medium", "large"}

# Locked flags: these define "isolated-cluster Gaussian MACE" and are not
# accepted as overrides. Recorded verbatim in the manifest.
LOCKED_ARGS: dict[str, str | int] = {
    "default_dtype": "float64",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "stress_weight": 0,
}

# Default categorical embedding sizes for a from-scratch run. Matches
# configs/train_from_scratch.sh. `spin` here is Gaussian multiplicity
# (1=singlet, ...), so class = multiplicity - offset. `charge` is shifted by
# `charge_offset` so negative integer charges become non-negative class ids.
DEFAULT_EMB_DIM = 128
DEFAULT_SPIN_NUM_CLASSES = 101
DEFAULT_SPIN_OFFSET = 0
DEFAULT_CHARGE_NUM_CLASSES = 201
DEFAULT_CHARGE_OFFSET = 100


class SeedRun(TypedDict):
    seed: int
    directory: str
    script: str
    argv: list[str]


class TrainingPlan(TypedDict):
    schema_version: int
    run_name: str
    mode: Mode
    model: str
    foundation_model: str | None
    dataset_dir: str
    dataset_files: dict[str, str]
    charge_range: list[int]
    multiplicity_range: list[int]
    label_routes: list[str]
    e0s: str
    seeds: list[int]
    seed_runs: list[SeedRun]
    locked_args: dict[str, str | int]
    warnings: list[str]


@dataclass
class DatasetFacts:
    files: dict[str, str]  # split filename -> sha256
    charges: set[int]
    multiplicities: set[int]
    label_routes: set[str]
    n_frames: int


@dataclass
class TrainingConfig:
    dataset_dir: Path
    output_dir: Path
    run_name: str = "cluster_charge_spin"
    mode: Mode = "scratch"
    seeds: tuple[int, ...] = (DEFAULT_SEED,)
    foundation_model: str = "polar-1-m"
    # Hyperparameters (defaults reproduce configs/train_from_scratch.sh, or
    # configs/finetune_foundation.sh when mode == "finetune").
    r_max: float = 6.0
    num_interactions: int = 2
    correlation: int = 3
    max_ell: int = 3
    num_radial_basis: int = 8
    hidden_irreps: str = "128x0e + 128x1o + 128x2e"
    mlp_irreps: str = "16x0e"
    energy_weight: float = 1.0
    forces_weight: float = 100.0
    lr: float | None = None  # None -> 0.005 scratch / 0.0001 finetune
    batch_size: int = 8
    valid_batch_size: int = 8
    max_num_epochs: int | None = None  # None -> 500 scratch / 100 finetune
    patience: int | None = None  # None -> 60 scratch / 20 finetune
    ema_decay: float | None = None  # None -> 0.999 scratch / 0.99 finetune
    e0s: str = "average"
    device: str = "cuda"
    multiheads_finetuning: bool = False
    # Categorical embedding sizing (scratch, and generic-foundation finetune).
    emb_dim: int = DEFAULT_EMB_DIM
    spin_num_classes: int = DEFAULT_SPIN_NUM_CLASSES
    spin_offset: int = DEFAULT_SPIN_OFFSET
    charge_num_classes: int = DEFAULT_CHARGE_NUM_CLASSES
    charge_offset: int = DEFAULT_CHARGE_OFFSET
    allow_mixed_method: bool = False
    force: bool = False
    extra_args: tuple[str, ...] = ()  # appended verbatim, for genuine one-offs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(dataset_dir: Path) -> DatasetFacts:
    """Read only the info lines of the split files -- cheap, no geometry parse.

    Collects the charge/multiplicity coverage (to size or validate the
    embedding), the distinct force-label routes (to enforce method
    consistency), and asserts every frame is non-periodic.
    """
    dataset_dir = dataset_dir.resolve()
    missing = [name for name in REQUIRED_SPLITS if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{dataset_dir} is missing {', '.join(missing)}; run `cluster-mlip collect` first"
        )

    files: dict[str, str] = {}
    charges: set[int] = set()
    multiplicities: set[int] = set()
    label_routes: set[str] = set()
    n_frames = 0

    for name in ("all.extxyz", *REQUIRED_SPLITS):
        path = dataset_dir / name
        if not path.is_file():
            continue
        files[name] = _sha256(path)
        if name == "all.extxyz":
            continue  # hashed for provenance, but its frames duplicate the splits
        lines = path.read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            n_atoms = int(lines[i].strip())
            info = parse_extxyz_info_line(lines[i + 1])
            pbc = info.get("pbc", "F F F").replace('"', "").strip().upper()
            if set(pbc.split()) - {"F"}:
                raise ValueError(
                    f"{path.name} frame {n_frames} has pbc={pbc!r}; the cluster MACE "
                    "workflow is for isolated (non-periodic) clusters only"
                )
            n_frames += 1
            charges.add(int(info.get("charge", 0)))
            multiplicities.add(int(info.get("multiplicity", info.get("spin", 1))))
            if "metadata" in info:
                try:
                    meta = json.loads(info["metadata"])
                except (TypeError, ValueError):
                    meta = {}
                route = str(meta.get("link1_route") or meta.get("first_route") or "").strip()
                if route:
                    label_routes.add(route)
            i += n_atoms + 2

    if n_frames == 0:
        raise ValueError(f"{dataset_dir}: split files contain no frames")
    return DatasetFacts(files, charges, multiplicities, label_routes, n_frames)


def _foundation_family(foundation_model: str) -> str:
    key = foundation_model.strip().lower()
    if key in POLAR_FOUNDATIONS:
        return "polar"
    if key in OMOL_FOUNDATIONS:
        return "omol"
    if key in GENERIC_FOUNDATIONS:
        return "generic"
    # A filesystem path or an unrecognized name: treat as generic (safe
    # default -- adds the embedding modules and warns), the user can still
    # pass a native-charge/spin path and drop the embedding via --extra-args.
    return "generic"


def _embedding_specs(config: TrainingConfig) -> str:
    return json.dumps(
        {
            "total_spin": {
                "type": "categorical",
                "per": "graph",
                "in_dim": 1,
                "emb_dim": config.emb_dim,
                "num_classes": config.spin_num_classes,
                "offset": config.spin_offset,
            },
            "total_charge": {
                "type": "categorical",
                "per": "graph",
                "in_dim": 1,
                "emb_dim": config.emb_dim,
                "num_classes": config.charge_num_classes,
                "offset": config.charge_offset,
            },
        },
        separators=(",", ":"),
    )


def _validate_embedding_coverage(config: TrainingConfig, facts: DatasetFacts) -> None:
    """Refuse to silently mis-bin a charge/multiplicity outside the embedding."""
    bad_spin = sorted(
        m for m in facts.multiplicities
        if not 0 <= m - config.spin_offset < config.spin_num_classes
    )
    if bad_spin:
        raise ValueError(
            f"multiplicities {bad_spin} fall outside the spin embedding "
            f"[offset={config.spin_offset}, num_classes={config.spin_num_classes}); "
            "raise --spin-num-classes / adjust --spin-offset"
        )
    bad_charge = sorted(
        c for c in facts.charges
        if not 0 <= c + config.charge_offset < config.charge_num_classes
    )
    if bad_charge:
        raise ValueError(
            f"charges {bad_charge} fall outside the charge embedding "
            f"[offset={config.charge_offset}, num_classes={config.charge_num_classes}); "
            "raise --charge-num-classes / adjust --charge-offset"
        )


def _build_argv(config: TrainingConfig, facts: DatasetFacts, seed: int) -> list[str]:
    lr = config.lr if config.lr is not None else (0.0001 if config.mode == "finetune" else 0.005)
    max_epochs = (
        config.max_num_epochs
        if config.max_num_epochs is not None
        else (100 if config.mode == "finetune" else 500)
    )
    patience = config.patience if config.patience is not None else (20 if config.mode == "finetune" else 60)
    ema_decay = config.ema_decay if config.ema_decay is not None else (0.99 if config.mode == "finetune" else 0.999)

    dataset = config.dataset_dir.resolve()
    family = _foundation_family(config.foundation_model) if config.mode == "finetune" else "scratch"
    native_charge_spin = family in {"polar", "omol"}
    model = "PolarMACE" if family == "polar" else "ScaleShiftMACE"

    argv: list[str] = [
        "mace_run_train",
        f"--name={config.run_name}_seed{seed}",
        f"--model={model}",
        f"--train_file={dataset / 'train.extxyz'}",
        f"--valid_file={dataset / 'valid.extxyz'}",
        f"--test_file={dataset / 'test.extxyz'}",
        f"--energy_key={LOCKED_ARGS['energy_key']}",
        f"--forces_key={LOCKED_ARGS['forces_key']}",
        "--total_charge_key=charge",
        "--total_spin_key=spin",
    ]

    if config.mode == "finetune":
        argv.append(f"--foundation_model={config.foundation_model}")
        argv.append(f"--multiheads_finetuning={config.multiheads_finetuning}")

    if not native_charge_spin:
        _validate_embedding_coverage(config, facts)
        argv.append(f"--embedding_specs={_embedding_specs(config)}")
        argv.append("--use_embedding_readout")

    argv.append(f"--E0s={config.e0s}")

    if config.mode == "scratch":
        argv += [
            "--interaction_first=RealAgnosticResidualNonLinearInteractionBlock",
            "--interaction=RealAgnosticResidualNonLinearInteractionBlock",
            f"--num_interactions={config.num_interactions}",
            f"--correlation={config.correlation}",
            f"--max_ell={config.max_ell}",
            f"--r_max={config.r_max}",
            f"--num_radial_basis={config.num_radial_basis}",
            f"--hidden_irreps={config.hidden_irreps}",
            f"--MLP_irreps={config.mlp_irreps}",
            "--scaling=rms_forces_scaling",
        ]

    argv += [
        "--loss=weighted",
        f"--energy_weight={config.energy_weight}",
        f"--forces_weight={config.forces_weight}",
        f"--stress_weight={LOCKED_ARGS['stress_weight']}",
        f"--lr={lr}",
        f"--batch_size={config.batch_size}",
        f"--valid_batch_size={config.valid_batch_size}",
        f"--max_num_epochs={max_epochs}",
        f"--patience={patience}",
        "--ema",
        f"--ema_decay={ema_decay}",
        f"--default_dtype={LOCKED_ARGS['default_dtype']}",
        f"--device={config.device}",
        f"--seed={seed}",
        "--keep_checkpoints",
    ]
    if config.mode == "finetune":
        argv.append("--amsgrad")
    argv.extend(config.extra_args)
    return argv


# A token (or the value after ``--flag=``) is left unquoted only if it is made
# entirely of shell-safe characters. Anything else -- spaces, and the ``{}"``
# in ``--embedding_specs`` JSON, the ``+`` in irreps, ``()`` in a route -- is
# single-quoted so the generated script is copy-pasteable and stable.
_SHELL_SAFE = re.compile(r"^[A-Za-z0-9_./:=+@%,-]*$")


def _quote_arg(part: str) -> str:
    if "=" in part and part.startswith("--"):
        flag, _, value = part.partition("=")
        if _SHELL_SAFE.fullmatch(value):
            return part
        return f"{flag}={shlex.quote(value)}"
    return part if _SHELL_SAFE.fullmatch(part) else shlex.quote(part)


def _render_script(argv: list[str]) -> str:
    body = " \\\n  ".join(_quote_arg(part) for part in argv)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "# Generated by `cluster-mlip train`. Edit train_manifest.json's inputs\n"
        "# and regenerate rather than hand-editing this file.\n\n"
        f'cd "$(dirname -- "${{BASH_SOURCE[0]}}")"\n\n'
        f"{body}\n"
    )


def _preflight_text(config: TrainingConfig, family: str) -> str:
    lines = [
        f"# Fine-tuning preflight: {config.run_name}",
        "",
        f"Foundation model: `{config.foundation_model}` (family: {family}).",
        "",
        "Before pointing a real allocation at this, run one short job on a",
        "handful of structures and confirm it starts and improves. In",
        "particular:",
        "",
    ]
    if family == "polar":
        lines += [
            "- `PolarMACE` needs the electrostatics module:",
            "  `pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git`",
            "  (see the MACE 'Electrostatic MACE' guide), plus `mace-torch>=0.3.16`.",
            "- Charge and spin are the foundation's native channels; this run does",
            "  NOT attach custom --embedding_specs. Verify the checkpoint loads and",
            "  `atoms.info['charge']`/`atoms.info['spin']` are honored.",
            "- MACE-POLAR-1 was trained at wB97M-V. Your labels are UBPW91/Gen --",
            "  a naive fine-tune shifts the whole model to your method; keep the",
            "  scratch model as the control.",
        ]
    elif family == "omol":
        lines += [
            "- MACE-OMOL carries native total-charge/total-spin embedding; this",
            "  run does NOT attach custom --embedding_specs.",
            "- OMOL is wB97M-V; your labels are UBPW91/Gen. See the note above.",
        ]
    else:
        lines += [
            "- This foundation was pretrained WITHOUT a charge/spin channel, so",
            "  this run attaches --embedding_specs modules on top of it. Whether",
            "  your installed mace-torch tolerates new embedding modules on a",
            "  foundation checkpoint (vs. a state-dict mismatch) is unverified --",
            "  this is exactly the open question in configs/finetune_foundation.sh.",
            "- If it errors, the documented fallback is multihead-replay",
            "  (`--multiheads_finetuning=True --pt_train_file ...`), which also",
            "  needs pretraining-replay data.",
        ]
    lines += [
        "",
        "- Naive fine-tuning can catastrophically forget; the MACE docs recommend",
        "  multihead replay for robustness. `--E0s=average` is used here; confirm",
        "  that is appropriate for your compositions.",
        "",
    ]
    return "\n".join(lines)


def write_training_campaign(config: TrainingConfig) -> TrainingPlan:
    """Scan the dataset, render one run script per seed, write the manifest."""
    facts = scan_dataset(config.dataset_dir)

    if len(facts.label_routes) > 1 and not config.allow_mixed_method:
        preview = "; ".join(sorted(facts.label_routes)[:3])
        raise ValueError(
            f"dataset mixes {len(facts.label_routes)} distinct force-label routes "
            f"({preview}...). Training one MACE head on energies/forces from "
            "different electronic-structure methods is unsound (see README "
            "\"Prepare consistent Gaussian labels\"). Pass --allow-mixed-method "
            "only if you are certain the routes are equivalent."
        )

    output = config.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not config.force:
        raise FileExistsError(
            f"{output} is not empty; use a fresh -o directory per training campaign "
            "or pass --force"
        )
    output.mkdir(parents=True, exist_ok=True)

    seeds = list(dict.fromkeys(config.seeds))  # dedup, keep order
    family = _foundation_family(config.foundation_model) if config.mode == "finetune" else "scratch"

    seed_runs: list[SeedRun] = []
    for seed in seeds:
        argv = _build_argv(config, facts, seed)
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        script = seed_dir / "run.sh"
        script.write_text(_render_script(argv), encoding="utf-8")
        script.chmod(0o755)
        seed_runs.append(
            {
                "seed": seed,
                "directory": seed_dir.name,
                "script": script.relative_to(output).as_posix(),
                "argv": argv,
            }
        )

    warnings: list[str] = []
    if not facts.label_routes:
        warnings.append(
            "no force-label route found in frame metadata; method-consistency "
            "was not checked"
        )
    if config.mode == "finetune" and family in {"polar", "omol"}:
        warnings.append(
            f"{config.foundation_model} was trained at wB97M-V hybrid DFT; the "
            "dataset labels are a different method -- keep the scratch model as "
            "the control"
        )

    plan: TrainingPlan = {
        "schema_version": SCHEMA_VERSION,
        "run_name": config.run_name,
        "mode": config.mode,
        "model": "PolarMACE" if family == "polar" else "ScaleShiftMACE",
        "foundation_model": config.foundation_model if config.mode == "finetune" else None,
        "dataset_dir": str(config.dataset_dir.resolve()),
        "dataset_files": facts.files,
        "charge_range": [min(facts.charges), max(facts.charges)],
        "multiplicity_range": [min(facts.multiplicities), max(facts.multiplicities)],
        "label_routes": sorted(facts.label_routes),
        "e0s": config.e0s,
        "seeds": seeds,
        "seed_runs": seed_runs,
        "locked_args": dict(LOCKED_ARGS),
        "warnings": warnings,
    }
    (output / "train_manifest.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if len(seeds) > 1:
        submit = output / "run_all_seeds.sh"
        submit.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n\n"
            'cd "$(dirname -- "${BASH_SOURCE[0]}")"\n\n'
            + "\n".join(f"bash {run['script']}" for run in seed_runs)
            + "\n",
            encoding="utf-8",
        )
        submit.chmod(0o755)

    if config.mode == "finetune":
        (output / "PREFLIGHT.md").write_text(_preflight_text(config, family), encoding="utf-8")

    return plan
