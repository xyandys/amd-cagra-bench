import numpy as np
import sys

# duplicate-heavy clustered data: each of U unique centres is replicated R times
# with a tiny perturbation.  Real embedding sets are full of near-duplicates and
# that is exactly where an exact kNN graph turns into a pile of parallel edges.
n, nq, d = 250000, 1000, 256
mode = sys.argv[1] if len(sys.argv) > 1 else 'dup'
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rng = np.random.default_rng(seed)

if mode == 'dup':
    U, eps = 25000, 0.01
    uniq = rng.normal(size=(U, d)).astype(np.float32)
    uniq /= np.linalg.norm(uniq, axis=1, keepdims=True)
    rep = rng.integers(0, U, size=n + nq)
    x = uniq[rep] + eps * rng.normal(size=(n + nq, d)).astype(np.float32)
else:
    x = rng.normal(size=(n + nq, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)

x = x.astype(np.float32)


def write_fvecs(path, mat):
    flat = np.empty((mat.shape[0], d + 1), dtype=np.int32)
    flat[:, 0] = d
    flat[:, 1:] = mat.view(np.int32)
    flat.tofile(path)


write_fvecs('/workspace/w/gmix_base.fvecs', x)
write_fvecs('/workspace/w/gmix_query.fvecs', x[n:n + nq])
print('wrote', x.shape, mode)
