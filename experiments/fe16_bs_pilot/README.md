# Fe16 broken-symmetry pilot: tandem fragment kit

Generated 2026-10-08 on branch `fix/bs-fragment-tandem`. **Nothing has been submitted; submitting is
your decision.** This kit replaces `fe16_bs_pilot_kit` (2026-10-06), which was never run. That kit put
`Stable=Opt` in the fragment link, used a default-guess P0, and had a mangled scratch path. Method,
archive audit and deviations: `docs/fragment-tandem-method.md`.

## The question

At a fixed Fe16 geometry with fixed q = 0 and M = 49, does UBPW91/6-311++G* have more than one stable
SCF solution with different site-resolved spins and different forces? Separately: do neutral and
charge-separated fragment guesses change convergence or the final state?

## Every input (three links, one checkpoint)

```
link 0  SP Guess=(Fragment=N)                                 archived syntax
link 1  Stable=Opt Pop=Hirshfeld Geom=Checkpoint Guess=Read
link 2  Force Pop=Hirshfeld Geom=Checkpoint Guess=Read
every link: IOP(5/13=1,5/36=1,8/11=1), as archived; analyze.py flags any unconverged SCF
```

Every file passed `inspect_tandem_input` when it was written.

## Sub-kits, in order (copy the whole folder to `/work/$USER/fe16_bs_tandem_kit` on QB3, submit from qbc)

| Step | Folder | Jobs | Batches |
|---|---|---|---|
| 1 | `validation/` | V1–V5 Fe₂ (2.02 Å, M=1) + H1/H2 replays of two archived AFM jobs | 1 × (7 jobs × 4 cores, 8 GB), 12 h |
| 2 | `smoke/` | Fe16 geometry 1, P1 | 1 × 12 cores, 24 GB, 48 h |
| 3 | `pilot/` | 3 geometries × P0 (all α), P1 (central β), P2/P3 (surface β) | 3 × (4 × 12 cores), 48 h |
| 3 | `charge_init/` | geometry 1: P0Q, P1Q (central Fe(−1), surface Fe(+1)), C0 default guess | 1 × (3 × 12 cores), 48 h |

```bash
cd validation && bash submit_gaussian_batches.sh
```

Gate after step 1: on the laptop, after `pack_results.sh` and copying back, run

```bash
PYTHONPATH=src python experiments/fe16_bs_pilot/make_validation_kit.py --check <kit>/validation
```

Every job should be `ok`. H1/H2 should reproduce the archived fragment-SCF energies, guess ⟨S²⟩
(3.9992 and 5.0082) and first link-1 SCF (−2787.66713025 and −2829.80312953 Ha, within ~10⁻⁵).
V1 and V5 must agree.

Compare V1 with V2 (with and without `,Only`), V1 with V3 (charge-separated), and V1 with V4 (default
guess). Note whether link 0 ran a supermolecule SCF (`s1_supermolecule_scf`) and the cycle cap
actually printed.

Gate after step 2: `ok`, a measured wall time, and a Hirshfeld table. Only then submit step 3.

## Analysis (laptop)

```bash
PYTHONPATH=src python experiments/fe16_bs_pilot/analyze.py <kit>/pilot
PYTHONPATH=src python experiments/fe16_bs_pilot/analyze.py <kit>/charge_init
```

These write `bs_states.csv` (per job: link-0 fragment SCFs, guess ⟨S²⟩ against ideal, the link-1 SCF
history and instabilities, i.e. the state before and after Stable=Opt, link-2 reproduction, geometry
check, energy, ⟨S²⟩, Hirshfeld and Mulliken spins, Hirshfeld and CM5 charges), `bs_pairs.csv` and
`bs_summary.md`.

**Same state** requires energy (1 meV), per-site Hirshfeld spin (0.2), ⟨S²⟩ (0.05) and force RMS
(5 meV/Å) all to agree. **Ambiguous** pairs are listed, not counted. Higher-energy stable states are
kept as metastable states. **Success:** at least 2 geometries with at least 2 distinct stable states.

## Read this before interpreting P1

In all three reference solutions, the central Fe (CN 14–15) already carries a Mulliken spin of −5.0
to −5.9 (and a Mulliken charge of ≈ +12). If Hirshfeld confirms a reversed central moment, P1 is
probably the reference state and P0 (all aligned) is the real alternative. `sites.csv` lists each
site's index, CN, radius, role and reference Mulliken spin.

## Do not

- add any result to the accepted training dataset before its states are shown to be consistent;
- submit step 3 before steps 1–2 pass.
