"""Build an index over random vectors and query it.

Runs in about a minute on any CUDA GPU. Recall is measured against exact
search on the same vectors, which is the only ground truth this repository
trusts.

A warning about what this example is. Isotropic Gaussian vectors are the worst
case for any proximity graph: no cluster structure means a query's true
neighbours are scattered, and the defaults -- tuned for embeddings -- leave
recall near 0.74 here. The sweep at the end shows the fix, and it is the point
of the method rather than a workaround: more trajectories per query reach more
of the answer. On real embeddings the same defaults sit above 0.99. Judge the
library on your own vectors, not on these.
"""
import numpy as np
import torch

import swann

n, d, nq, k = 100_000, 768, 1_000, 10
rng = np.random.default_rng(0)

X = rng.standard_normal((n, d), dtype=np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)
Q = rng.standard_normal((nq, d), dtype=np.float32)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)

print(f"building over {n:,} vectors of {d} dimensions")
index = swann.Index.build(X, metric="ip")
print(index)

ids, dist = index.search(Q, k=k)

Xg = torch.from_numpy(X).cuda()
Qg = torch.from_numpy(Q).cuda()
truth = torch.topk(Qg @ Xg.T, k, dim=1).indices.cpu().numpy()
hit = np.mean([len(set(a) & set(b)) / k for a, b in zip(ids, truth)])
print(f"recall@{k}: {hit:.4f}")

# recall is an arrival problem: the branches are what arrive
print("\nrecall against the number of trajectories per query")
for branches in (8, 32, 128):
    ids, _ = index.search(Q, k=k, branches=branches)
    hit = np.mean([len(set(a) & set(b)) / k for a, b in zip(ids, truth)])
    print(f"  branches={branches:4}  recall@{k} {hit:.4f}")
