# Audit whether total spin is enough for the Fe16 labels

`audit-spin-labels` is a **read-only dataset diagnostic**. It neither launches
Gaussian nor trains MACE, edits labels, submits missing minima, or generates
rattles. It needs Python >=3.10 and numpy (`pip install -e '.[audit]'` in a
separate environment if numpy is not already available). MACE/CUDA are not needed.
The ordinary extraction/training-command generation tools retain their dependency-free imports.

## LONI launch

Use the updated **repository**, not the old `fe16_loni_kit/src` snapshot. The
launcher puts the updated source first on PYTHONPATH. Submit from the repository
root so Slurm's script spooling cannot change which code is used:

```bash
cd /ddnB/work/lgutsev/ClusterMLIP
git pull --ff-only

CLUSTER_MLIP_PYTHON=/project/lgutsev/env/lgutsev_dev/bin/python \
sbatch scripts/run_spin_audit_slurm.sh \
  /ddnB/work/lgutsev/ClusterMLIP/campaigns/FenOm_Warehouse2/gaussian_spin_qb3_g09_v3/slurm_batches \
  --gaussian --formula Fe16 \
  -o /ddnB/work/lgutsev/ClusterMLIP/fe16_v1/spin_audit_v1
```

This command reads every `.log` and `.out` recursively, including archived
`before-restart` and restarted logs. It selects Fe16 by parsed composition,
not filename. No collection step is required. `input_logs.csv` inventories
selected frames, logs with no selected force frames, missing explicit Q/M,
normal/error terminations, and parse errors. Files without force tables cannot
support an E/F comparison and remain visible in the inventory. All force-bearing
steps are considered by default, including intermediate steps in failed or
unfinished jobs; this is an audit, not acceptance into a training set. Each pair
records whether both sections terminated normally. Read that field before
interpreting a conflict. Frames without explicit Q/M are excluded, never
silently assigned a singlet. No cross-restart trajectory ordering is guessed.
`--file-glob '*__fe16__*.log'` can restrict scanning using your campaign naming
scheme, but the parsed `--formula Fe16` check is still recommended.
`--gaussian-frames last-per-section` reduces to final force frames for a quick
screen; it can miss useful same-geometry pairs and trajectory changes.

Alternatively, omit `--gaussian` and pass collected labeled extxyz files. If the kit named it differently,
replace that filename with its actual `all.extxyz` equivalent. Do not use
unlabeled seeds. If only disjoint `train.extxyz`, `valid.extxyz`, and `test.extxyz`
exist, provide those three paths instead of `all.extxyz`; never both.
This reader deliberately requires the standard ClusterMLIP
`species:S:1:pos:R:3:REF_forces:R:3` schema, explicit charge/multiplicity, eV
energies, eV/Angstrom forces, and nonperiodic clusters. Other schemas fail instead
of guessing which columns contain forces.

In extxyz mode, `--verify-raw` checks labels against their original logs.
`--raw-root` replaces the campaign directory, preserving each frame's
`metadata.gaussian_output` relative path. With multiple campaigns, omit it and
use each frame's recorded `collection_campaign`. Raw verification requires both
that output path and `force_frame_index`; it reports missing provenance instead
of guessing a file by basename. Logs are only read. No SSH/API access is involved.

Default resources: `single`, one CPU, 12 hours, account `loni_perovsk27`.
Override with normal `sbatch` flags. Override `CLUSTER_MLIP_REPO` if submitting
from another directory. `CLUSTER_MLIP_PYTHON` defaults to the existing
`lgutsev_dev` Python; the script checks numpy before doing the audit and does not
install or upgrade anything. Scheduler logs land in the submission directory.

A local/direct invocation uses the same arguments:

```bash
PYTHONPATH=src python -m cluster_mlip.cli audit-spin-labels \
  /path/to/all.extxyz --formula Fe16 --verify-raw -o private_audits/fe16_spin
```

The output directory must be new or empty. Exit 0 means the audit finished,
**not that the dataset passed**. Exit 2 means the pair or mapping budget was
exceeded, or a raw log had a parse error: inspect the partial report, then rerun in a new directory with larger
`--max-pairs`/`--mapping-budget` if necessary. Other errors fail the job.

## What to read

- `report.md`: short findings and interpretation.
- `summary.json`: all thresholds, source SHA256 hashes, coverage, route and raw
  verification counts, search completeness, pair classifications.
- `input_logs.csv`: direct-mode log coverage, including files without usable forces.
- `frames.csv`: stable audit frame indices, input filename/frame offset, record
  identity, E/F summary, local-spin completeness and sum, S2, raw-log verdict.
- `pairs.csv`: atom-matched comparisons; `left` and `right` join to `frames.csv`.
- `trajectory_flags.csv`: adjacent force-index spin changes within one output,
  charge and multiplicity. These are possible root changes, not proof.

Share `report.md` and `summary.json` first; pairs/frames identify the concrete
Gaussian calculations worth investigating. These files belong with the private
LONI dataset, not committed to the repository.

## Hypotheses and limits

1. **Missing local magnetic state:** exact matched geometry/Q/multiplicity but
   inconsistent E/F, with distinguishable site-resolved spins, supports this
   hypothesis. A missing or inconsistent spin sum, or failed requested raw-spin
   verification, prevents attributing a conflict to those spins. SCF warnings
   and raw E/F/geometry mismatches are classified as label-quality problems.
2. **Global multiplicity is enough for the observed pairs:** different-M pairs
   are controls, not contradictory inputs. No same-M conflicts is inconclusive
   if the dataset never sampled competing same-M roots at identical geometry.
3. **Geometry differences alone:** near geometries are reported separately;
   differences in their forces/energies are not a logical contradiction.
4. **Parser/provenance problems:** raw verification reparses the exact Gaussian
   force index and compares its geometry, Q/M, E/F and per-atom populations.
   Only populations between that SCF energy and its force table are accepted.
   Populations elsewhere in the log are not backfilled. Old extxyz files with
   incomplete provenance remain unverified. No data are repaired automatically.

Coordinates are compared by species-preserving distance-graph mappings followed
by a proper Kabsch rotation and a maximum-atom-displacement check. Sorted pair
fingerprints only prune candidates; they never establish structural equivalence.
All mappings found within the budget are considered. Spins remain attached to
sites, and global spin reversal is treated as equivalent at zero field. This is
a collinear audit, not a noncollinear spin-orbit model.
Forces are permuted and rotated with the atoms. Spin RMS and force RMS are
minimized separately over valid mappings to make conflict claims conservative;
`force_at_best_spin_mapping_eV_A` gives the force difference at the spin-optimal
mapping. For collinear/single-atom geometries, force comparisons are withheld
because the free rotation about the axis is not fixed by the nuclei.

Defaults (all configurable): exact max displacement 1e-5 Angstrom, near 0.03
Angstrom, energy difference 0.01 eV **per cluster**, Cartesian-component force
RMS 0.05 eV/Angstrom, local spin RMS 0.3, spin-sum error 0.1 relative to
`multiplicity - 1`. A finite exact tolerance is still a numerical convention.
Mulliken spins are population descriptors, not exact atomic spin quantum numbers;
S2 values are reported without rejecting broken-symmetry determinants as such.

Different recorded label routes trigger review instead of a magnetic conclusion;
identical route strings do not prove identical Gen basis contents. Missing routes
are reported. Verify method/basis, SCF convergence and stability before treating
any conflict as physical. Pair counts are not counts of independent states.
Trajectory comparisons assume preserved Gaussian center numbering; continuous
physical changes can also exceed the chosen spin threshold.

This command does not test model inference sensitivity, learn spins, assess
force/curvature coverage sufficiency, or prove that a model can relax moments.
Those remain later experiments. Existing missing-minima/rattle restrictions
remain in effect.

## Validation status

Synthetic regression tests exercise rigid transformations/permutations, symmetry,
global spin reversal, same-histogram different-site spins, near versus exact
matches, force-only differences, capped searches, raw provenance and parser
metadata isolation. A previously uploaded Fe16N2 Gaussian log also passed a direct-input smoke (55 force frames).
The LONI launch and actual Fe16 dataset have **not been run
by this implementation task**; that cluster smoke is still required.
