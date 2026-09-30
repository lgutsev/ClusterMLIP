"""Force change per unit displacement: cross-stage near-identical pairs vs consecutive steps.

For a single smooth PES (one electronic state), |dF_i| <~ k |dx| with k a local stiffness,
and consecutive optimisation steps sample that relation. Cross-stage pairs at the same M,
aligned by rotation + atom permutation, should follow the same relation unless the two
labels come from different electronic states at the same geometry.
"""
import collections, itertools, math, sys, numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from cluster_mlip.dataset import read_labeled_extxyz
from cluster_mlip.spin import _pair_fingerprint


def fingerprint_distance(left, right):
    """RMS difference of sorted element-pair distances (rotation/permutation invariant)."""
    if left.keys() != right.keys() or any(len(left[k]) != len(right[k]) for k in left):
        return math.inf
    diffs = [a - b for k in left for a, b in zip(left[k], right[k])]
    return math.sqrt(sum(d * d for d in diffs) / len(diffs)) if diffs else 0.0

fr = read_labeled_extxyz(Path(sys.argv[1]))
X = [np.array([[a.x, a.y, a.z] for a in f.record.atoms]) for f in fr]
F = [np.array(f.forces_ev_ang) for f in fr]
key = [(f.record.metadata["gaussian_output"], f.record.metadata["job_id"]) for f in fr]
M = [f.record.multiplicity for f in fr]
fp = [_pair_fingerprint(f.record.atoms) for f in fr]

def align(a, b, iters=4):
    """Rotation R and permutation p so that (b[p]-cb) @ R.T ~ a - ca."""
    A = a - a.mean(0); B = b - b.mean(0); p = np.arange(len(a)); R = np.eye(3)
    for _ in range(iters):
        R, _ = Rotation.align_vectors(A, B[p]); R = R.as_matrix()
        Bt = B @ R.T
        _, p = linear_sum_assignment(((A[:, None] - Bt[None]) ** 2).sum(-1))
    R, _ = Rotation.align_vectors(A, B[p]); R = R.as_matrix()
    return R, p, np.linalg.norm(A - B[p] @ R.T, axis=1)

def ratio(i, j, aligned=True):
    if aligned:
        R, p, dx = align(X[i], X[j]); dF = np.linalg.norm(F[i] - F[j][p] @ R.T, axis=1)
    else:
        dx = np.linalg.norm(X[i] - X[j], axis=1); dF = np.linalg.norm(F[i] - F[j], axis=1)
    return dx.max(), dF.max(), dF.max() / max(dx.max(), 1e-6)

# reference: consecutive steps inside one stage (same state by construction of an optimisation)
stages = collections.defaultdict(list)
for i, k in enumerate(key): stages[k].append(i)
cons = []
for idx in stages.values():
    idx.sort(key=lambda i: int(fr[i].record.metadata["force_frame_index"]))
    for i, j in zip(idx, idx[1:]):
        dxm, dFm, r = ratio(i, j)
        if dxm < 0.05: cons.append((dxm, dFm, r))
cons = np.array(cons)
# cross-stage pairs at the same M, geometry nearly identical
cross = []
for i, j in itertools.combinations(range(len(fr)), 2):
    if M[i] != M[j] or key[i] == key[j]: continue
    if fingerprint_distance(fp[i], fp[j]) > 0.02: continue
    dxm, dFm, r = ratio(i, j)
    if dxm < 0.05: cross.append((dxm, dFm, r, i, j))
cr = np.array([c[:3] for c in cross])
q = lambda a: "median %.2f  90%% %.2f  max %.2f" % (np.median(a), np.percentile(a, 90), a.max())
print(f"consecutive steps with max|dx| < 0.05 A: {len(cons)}")
print("  max|dF| (eV/A):", q(cons[:, 1]), "| max|dF|/max|dx| (eV/A^2):", q(cons[:, 2]))
print(f"cross-stage same-M pairs with max|dx| < 0.05 A after alignment: {len(cr)}")
if len(cr):
    print("  max|dF| (eV/A):", q(cr[:, 1]), "| max|dF|/max|dx| (eV/A^2):", q(cr[:, 2]))
    thr = np.percentile(cons[:, 2], 99)
    bad = [c for c in cross if c[2] > thr]
    print(f"  cross pairs above the 99th percentile of consecutive-step ratio ({thr:.1f} eV/A^2): {len(bad)} of {len(cr)}")
    for dxm, dFm, r, i, j in sorted(bad, key=lambda c: -c[2])[:6]:
        a, b = fr[i].record.metadata, fr[j].record.metadata
        print(f"    M={M[i]} dx={dxm:.4f} dF={dFm:.3f} ratio={r:.0f} dE={1000*abs(fr[i].energy_ev-fr[j].energy_ev):.1f} meV  "
              f"{a['job_id']}#{a['force_frame_index']} vs {b['job_id']}#{b['force_frame_index']}")
