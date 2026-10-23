"""The public interface, end to end, against exact search.

Small enough to run on any CUDA GPU in under a minute, and strict enough that
a kernel regression fails it: recall against the exact answer, not against a
stored expectation that could drift with the implementation.
"""
import numpy as np
import pytest
import torch

import swann

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")


def corpus(n=8000, d=128, nq=200, seed=0, normalise=True):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)
    Q = rng.standard_normal((nq, d), dtype=np.float32)
    if normalise:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return X, Q


def exact(X, Q, k, metric):
    Xg, Qg = torch.from_numpy(X).cuda(), torch.from_numpy(Q).cuda()
    if metric == "ip":
        return torch.topk(Qg @ Xg.T, k, dim=1).indices.cpu().numpy()
    return torch.topk(torch.cdist(Qg, Xg), k, dim=1, largest=False).indices.cpu().numpy()


def recall(got, truth, k):
    return float(np.mean([len(set(a[:k]) & set(b[:k])) / k for a, b in zip(got, truth)]))


@pytest.mark.parametrize("metric", ["ip", "l2"])
def test_recall_is_high(metric):
    X, Q = corpus(normalise=(metric == "ip"))
    index = swann.Index.build(X, metric=metric, degree=48, candidates=64)
    ids, dist = index.search(Q, k=10)
    assert ids.shape == (Q.shape[0], 10) and dist.shape == ids.shape
    assert recall(ids, exact(X, Q, 10, metric), 10) > 0.90


def test_more_branches_do_not_lose_recall():
    X, Q = corpus()
    index = swann.Index.build(X, degree=48, candidates=64)
    truth = exact(X, Q, 10, "ip")
    few = recall(index.search(Q, k=10, branches=8)[0], truth, 10)
    many = recall(index.search(Q, k=10, branches=64)[0], truth, 10)
    assert many >= few - 1e-9


def test_round_trip(tmp_path):
    X, Q = corpus()
    index = swann.Index.build(X, degree=48, candidates=64)
    before, _ = index.search(Q, k=10)
    index.save(tmp_path / "ix")
    after, _ = swann.Index.load(tmp_path / "ix", X).search(Q, k=10)
    assert np.array_equal(before, after)


def test_batch_limit_is_explained():
    X, _ = corpus(n=2000)
    index = swann.Index.build(X, degree=32, candidates=48)
    with pytest.raises(ValueError, match="65535"):
        index.search(np.zeros((70000, X.shape[1]), dtype=np.float32), k=10)
