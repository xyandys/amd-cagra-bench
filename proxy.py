import numpy as np
import sys

n, nq, d, nc = 250000, 1000, 256, 2000
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

rng = np.random.default_rng(seed)
centers = rng.normal(size=(nc, d)).astype(np.float32)
centers /= np.linalg.norm(centers, axis=1, keepdims=True)

idx = rng.integers(0, nc, size=n + nq)
x = centers[idx] + alpha * rng.normal(size=(n + nq, d)).astype(np.float32)
x /= np.linalg.norm(x, axis=1, keepdims=True)


def write_fvecs(path, mat):
    flat = np.empty((mat.shape[0], d + 1), dtype=np.int32)
    flat[:, 0] = d
    flat[:, 1:] = mat.view(np.int32)
    flat.tofile(path)


write_fvecs('/workspace/w/gmix_base.fvecs', x)
write_fvecs('/workspace/w/gmix_query.fvecs', x[n:n + nq])
print('wrote', x.shape, 'alpha', alpha)
