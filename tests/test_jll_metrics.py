import numpy as np
from orthogonal_projection.projection import jll_dimension, generate_orthogonal_basis, project_data
from orthogonal_projection.evaluation import compute_distortion
from sklearn.metrics import pairwise_distances


def test_jll_k_and_metrics_shapes():
    rng = np.random.default_rng(0)
    n, d = 60, 200
    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    eps = 0.5
    k = min(jll_dimension(n, eps), d)
    basis = generate_orthogonal_basis(d, k, seed=42)
    Y = project_data(X, basis)

    assert Y.shape == (n, k)

    mean_dist, max_dist, D1_sq, D2_sq = compute_distortion(X, Y)
    # Non-negativity and finite checks
    assert np.isfinite(mean_dist)
    assert np.isfinite(max_dist)
    assert mean_dist >= 0.0
    assert max_dist >= 0.0
    # Distance matrices should be square and match in shape
    assert D1_sq.shape == (n, n)
    assert D2_sq.shape == (n, n)
