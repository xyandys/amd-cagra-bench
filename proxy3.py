import numpy as np

# base = clustered embeddings, queries = points that live *between* clusters
# (out-of-distribution queries are where a purely intra-cluster kNN graph has
# the least to follow).
n, nq, d, nc = 250000, 1000, 256, 2000
rng = np.random.default_rng(7)

centers = rng.normal(size=(nc, d)).astype(np.float32)
centers /= np.linalg.norm(centers, axis=1, keepdims=True)
idx = rng.integers(0, nc, size=n)
base = centers[idx] + 0.35 * rng.normal(size=(n, d)).astype(np.float32)
base /= np.linalg.norm(base, axis=1, keepdims=True)

q = rng.normal(size=(nq, d)).astype(np.float32)
q /= np.linalg.norm(q, axis=1, keepdims=True)


def write_fvecs(path, mat):
    flat = np.empty((mat.shape[0], d + 1), dtype=np.int32)
    flat[:, 0] = d
    flat[:, 1:] = mat.view(np.int32)
    flat.tofile(path)


full = np.vstack([base, q]).astype(np.float32)
write_fvecs('/workspace/w/gmix_base.fvecs', full)
write_fvecs('/workspace/w/gmix_query.fvecs', q)
print('wrote ood', full.shape)
