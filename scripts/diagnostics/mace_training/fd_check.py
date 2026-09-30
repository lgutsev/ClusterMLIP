"""Predicted forces vs central finite differences of the model's own energy."""
import sys, numpy as np, torch
from common import *
m = load_model(sys.argv[1]); fr = frames(sys.argv[2])
rng = np.random.default_rng(0)
for i in (14, 9, 0):
    a = fr[i]
    out = m(batch_of(graphs(m, [a])).to_dict(), training=False, compute_force=True)
    F = out["forces"].detach().cpu().numpy()
    h = 1e-4; fd = []; pr = []
    for atom, comp in [(int(x), int(y)) for x, y in zip(rng.integers(0, 16, 6), rng.integers(0, 3, 6))]:
        e = []
        for s in (+1, -1):
            b = a.copy(); b.positions[atom, comp] += s * h
            e.append(float(m(batch_of(graphs(m, [b])).to_dict(), training=False, compute_force=False)["energy"]))
        fd.append(-(e[0] - e[1]) / (2 * h)); pr.append(F[atom, comp])
    fd, pr = np.array(fd), np.array(pr)
    print(f"frame {i}: max|F_pred - F_fd| = {np.abs(fd - pr).max():.2e} eV/A  (|F| up to {np.abs(pr).max():.2e})")
