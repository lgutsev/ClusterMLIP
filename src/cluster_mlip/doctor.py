from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from dataclasses import dataclass

# Status severity, from best to worst.
OK = "ok"
MISSING_OPTIONAL = "missing_optional"
MISSING_REQUIRED = "missing_required"


@dataclass
class DoctorCheck:
    name: str
    status: str
    detail: str


def _binary_check(name: str, binaries: list[str], required_for: str) -> DoctorCheck:
    for binary in binaries:
        path = shutil.which(binary)
        if path:
            return DoctorCheck(name, OK, f"'{binary}' found at {path}")
    tried = "/".join(binaries)
    return DoctorCheck(
        name, MISSING_OPTIONAL, f"none of '{tried}' on PATH; needed for {required_for}"
    )


def _module_check(name: str, module: str, required_for: str) -> DoctorCheck:
    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        found = False
    if found:
        return DoctorCheck(name, OK, f"'{module}' is importable")
    return DoctorCheck(name, MISSING_OPTIONAL, f"'{module}' not importable; needed for {required_for}")


def run_checks() -> list[DoctorCheck]:
    """Probe every external binary/package the pipeline can call out to.

    Nothing here is required for `analyze`/`extract`/`spin-extract` on
    already-formatted-checkpoint or plain-text/log input -- these checks
    exist so a multi-thousand-file HPC run doesn't discover a missing
    `formchk` on file #4000. Everything is advisory (missing_optional)
    except the interpreter version, since a command can still legitimately
    run without any single one of these tools depending on what's in the
    warehouse.
    """
    checks: list[DoctorCheck] = []

    version = sys.version_info
    if (version.major, version.minor) >= (3, 10):
        checks.append(DoctorCheck("python", OK, f"Python {platform.python_version()}"))
    else:
        checks.append(
            DoctorCheck(
                "python",
                MISSING_REQUIRED,
                f"Python {platform.python_version()} is below the required 3.10",
            )
        )

    checks.append(_binary_check("strings", ["strings"], "reading legacy binary .doc Word documents"))
    checks.append(_binary_check("formchk", ["formchk"], "reading binary Gaussian .chk checkpoints"))
    checks.append(
        _binary_check(
            "gaussian", ["g16", "g09"], "running generated Gaussian jobs (run_one.sh)"
        )
    )
    checks.append(
        _module_check(
            "mace-torch",
            "mace",
            "training/fine-tuning (configs/*.sh) and 'select-next-batch'",
        )
    )
    checks.append(
        _module_check("torch", "torch", "training/fine-tuning and 'select-next-batch'")
    )
    return checks


def format_report(checks: list[DoctorCheck]) -> tuple[str, str]:
    """Return (report text, worst status seen)."""
    markers = {OK: "OK", MISSING_OPTIONAL: "WARN", MISSING_REQUIRED: "FAIL"}
    order = {OK: 0, MISSING_OPTIONAL: 1, MISSING_REQUIRED: 2}
    lines = ["ClusterMLIP environment check", ""]
    worst = OK
    name_width = max((len(check.name) for check in checks), default=0)
    for check in checks:
        lines.append(f"[{markers[check.status]:4s}] {check.name:<{name_width}s}  {check.detail}")
        if order[check.status] > order[worst]:
            worst = check.status
    lines.append("")
    if worst == OK:
        lines.append("All checks passed.")
    elif worst == MISSING_OPTIONAL:
        lines.append(
            "Optional tools are missing. Commands that don't need them still work; "
            "install what's flagged above before relying on the features it lists."
        )
    else:
        lines.append("A required check failed; fix it before running the pipeline.")
    return "\n".join(lines), worst
