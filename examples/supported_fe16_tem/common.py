"""Model selection shared by the supported-Fe16 scripts (s2, s3, s11).

Every script takes the same three model options:

  --model          gas-phase cluster term E_gas(A; q, M)       default mace-mp:medium+d3
  --support-model  support term E_support(B)                   default mace-mp:medium+d3
  --interaction    dE_int: 'subtractive' (E(AB) - E(A) - E(B) from --support-model),
                   'subtractive:<spec>' (same, from another model), or the spec/path
                   of a model trained on `cluster-mlip vasp-collect` interaction labels

Specs are those of `cluster_mlip.delta.load_model`: ``mace-mp:<size>[+d3]``,
``mace-polar:<name>``, or a path to a trained ``.model`` (charge/spin-aware unless
suffixed ``:blind``). With the defaults the Delta-model is exactly MACE-MP-0 + D3 on the
whole system -- the baseline in the README -- and the scripts use the single model
directly (identical numbers, three times cheaper).

Outputs of the baseline go to out/; any other combination writes to out_<tag>/ (under
$SUPPORTED_FE16_RUNS when set, else next to the scripts) so the baseline results are
never overwritten. s4-s6 read the directory in $EXAMPLE_OUT.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))  # run from a checkout without installing

from cluster_mlip.delta import DeltaCalculator, LoadedModel, SubtractiveInteraction, load_model  # noqa: E402

BASELINE = "mace-mp:medium+d3"
DEFAULT_MULTIPLICITIES = "49,51,53"


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=BASELINE, help="gas-phase cluster model spec")
    parser.add_argument("--support-model", default=BASELINE, help="support model spec")
    parser.add_argument("--interaction", default="subtractive", help="interaction term (see common.py)")
    parser.add_argument("--multiplicities", default=DEFAULT_MULTIPLICITIES,
                        help="multiplicities scanned when the gas model is spin-aware")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", default=None, help="cuda/cpu (default: cuda when available)")
    parser.add_argument("--out", default=None, help="output directory (default: out/ or out_<tag>/)")


@dataclass
class Models:
    gas: LoadedModel
    support: LoadedModel
    interaction_spec: str
    interaction: object  # the dE_int calculator on its own (s11 compares it with VASP)
    total: object  # an ASE calculator for the whole supported system
    is_delta: bool
    out: Path
    multiplicities: list[int] | None

    def describe(self) -> dict:
        return {"gas": self.gas.spec, "support": self.support.spec,
                "interaction": self.interaction_spec, "delta": self.is_delta,
                "multiplicities": self.multiplicities}


def build_models(args: argparse.Namespace) -> Models:
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache: dict[str, LoadedModel] = {}

    def get(spec: str) -> LoadedModel:
        if spec not in cache:
            cache[spec] = load_model(spec, device=device, dtype=args.dtype)
        return cache[spec]

    gas = get(args.model)
    support = get(args.support_model)
    inter = args.interaction
    if inter == "subtractive" or inter.startswith("subtractive:"):
        source = support if inter == "subtractive" else get(inter.split(":", 1)[1])
        single = source.spec == gas.spec == support.spec
        interaction = SubtractiveInteraction(source.calculator)
        inter_tag = "" if source.spec == support.spec else f"_int-{source.tag}"
    else:
        single = False
        interaction = get(inter).calculator
        inter_tag = f"_int-{get(inter).tag}"
    total = gas.calculator if single else DeltaCalculator(gas.calculator, support.calculator, interaction)
    baseline = single and gas.spec == BASELINE
    if args.out:
        out = Path(args.out)
    elif baseline:
        out = HERE / "out"
    else:
        sup_tag = "" if support.spec == BASELINE else f"_sup-{support.tag}"
        # $SUPPORTED_FE16_RUNS keeps model runs off the repo (e.g. on the D: work drive)
        root = Path(os.environ.get("SUPPORTED_FE16_RUNS") or HERE)
        out = root / f"out_{gas.tag}{sup_tag}{inter_tag}"
    out.mkdir(parents=True, exist_ok=True)
    mults = [int(m) for m in args.multiplicities.split(",")] if gas.spin_aware else None
    print(f"device {device}; gas={gas.spec} support={support.spec} interaction={inter} "
          f"({'single model' if single else 'Delta-model'}); spin scan={mults}; out={out}", flush=True)
    return Models(gas, support, inter, interaction, total, not single, out, mults)


def example_out() -> Path:
    """Directory read by s4-s6: $EXAMPLE_OUT (or the older $TEM_OUT), else out/."""
    return Path(os.environ.get("EXAMPLE_OUT") or os.environ.get("TEM_OUT") or HERE / "out")
