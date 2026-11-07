"""Step 4: abTEM HRTEM of the MD trajectories (tutorial settings: 200 kV, Cs=-8 um, Scherzer).

Two beam directions per support:
  profile - beam parallel to the surface (the LinkedIn / tutorial side view)
  plan    - beam along the surface normal (looking down on the cluster)
and three exposures: short (last frame + 12 frozen-phonon configs),
10 ps time average (all 200 MD frames), and that average with shot noise.
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

warnings.filterwarnings("ignore")
import abtem  # noqa: E402

abtem.config.set({"device": "cpu", "diagnostics.progress_bar": False})
OUT = Path(os.environ.get("EXAMPLE_OUT") or os.environ.get("TEM_OUT") or Path(__file__).parent / "out")
ENERGY = 200e3
DOSE = 2e4  # e-/A^2, as in the tutorial


Z_PAD = 4.0  # the support sits ~1 A above z=0 and dips below it during MD


def _inplane_wrap(a: Atoms) -> Atoms:
    """wrap laterally only; along the normal just shift by a constant pad, so atoms that
    dip below z=0 are not folded to the top of the cell (same cell for every frame)."""
    a = a.copy()
    a.wrap(pbc=[True, True, False])
    a.positions[:, 2] += Z_PAD
    a.cell[2, 2] += 2 * Z_PAD
    return a


def to_profile(a: Atoms) -> Atoms:
    """beam (new z) along old y; image vertical (new y) = surface normal."""
    a = _inplane_wrap(a)
    lx, ly, lz = a.cell.diagonal()
    p = a.positions
    return Atoms(a.get_chemical_symbols(), positions=np.c_[p[:, 0], p[:, 2], ly - p[:, 1]],
                 cell=[lx, lz, ly], pbc=True)


def to_plan(a: Atoms) -> Atoms:
    a = _inplane_wrap(a)
    return Atoms(a.get_chemical_symbols(), positions=a.positions, cell=np.diag(a.cell.diagonal()),
                 pbc=True)


def ctf_for(wave):
    ctf = abtem.CTF(Cs=-8e-6 * 1e10, energy=wave.energy, defocus="scherzer")
    ctf.focal_spread = 1.0e-3 * 1e10 * 0.3 / wave.energy
    return ctf


def image(frames: list[Atoms], frozen: bool, fixed_mask: np.ndarray):
    if frozen:
        sig = np.where(fixed_mask, 0.0, 0.08)
        src = abtem.FrozenPhonons(frames[-1], num_configs=12, sigmas=sig, seed=13)
    else:
        src = frames
    pot = abtem.Potential(src, sampling=0.05, slice_thickness=1, projection="infinite")
    wave = abtem.PlaneWave(energy=ENERGY)
    exit_wave = wave.multislice(pot)
    img = exit_wave.apply_ctf(ctf_for(wave)).intensity().mean(0).compute()
    return img


SUPPORTS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["graphene", "MgO100"]
npz = OUT / "tem_images.npz"
results = dict(np.load(npz)) if npz.exists() else {}
for sname in SUPPORTS:
    traj = read(OUT / f"md_{sname}.extxyz", ":")
    is_fe = np.array([s == "Fe" for s in traj[0].get_chemical_symbols()])
    z_sup = traj[0].positions[~is_fe, 2]
    fixed = (~is_fe) & (traj[0].positions[:, 2] < z_sup.min() + 0.5) if sname == "MgO100" \
        else np.zeros(len(is_fe), bool)
    for view, tf in (("profile", to_profile), ("plan", to_plan)):
        frames = [tf(a) for a in traj]
        short = image(frames, frozen=True, fixed_mask=fixed)
        avg = image(frames, frozen=False, fixed_mask=fixed)
        noisy = avg.poisson_noise(dose_per_area=DOSE, seed=0)
        key = f"{sname}_{view}"
        ext = short.extent
        results[key + "_short"] = short.array
        results[key + "_avg"] = avg.array
        results[key + "_noisy"] = noisy.array
        results[key + "_extent"] = np.array(ext)
        # Fe positions in image coordinates for annotation
        fe = np.array([f.positions[is_fe] for f in frames])
        results[key + "_fe"] = fe
        results[key + "_sup_top"] = np.array(
            [f.positions[~is_fe][:, 1].max() if view == "profile" else 0.0 for f in frames])
        print(key, short.array.shape, "extent", np.round(ext, 2), flush=True)

np.savez_compressed(npz, **results)
print("saved", OUT / "tem_images.npz")
