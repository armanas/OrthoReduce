import numpy as np

from orthogonal_projection.projection import (
    generate_orthogonal_basis,
    jll_dimension,
    project_data,
)
from orthogonal_projection.evaluation import compute_distortion, nearest_neighbor_overlap


# Helper used by multiple tests

def _evaluate_projection(n, d, epsilon, seed=0):
    """Generate synthetic data, project it and return metrics."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    k = jll_dimension(n, epsilon)
    k = min(k, d)
    basis = generate_orthogonal_basis(d, k, seed=seed)
    Y = project_data(X, basis)

    mean_dist, max_dist, *_ = compute_distortion(X, Y)
    overlap = nearest_neighbor_overlap(X, Y, k=min(5, n - 1))
    return k, mean_dist, max_dist, overlap


def test_jll_dimension_monotonic():
    n = 100
    eps_small = 0.2
    eps_large = 0.6
    k_small = jll_dimension(n, eps_small)
    k_large = jll_dimension(n, eps_large)
    assert k_large < k_small


def test_llm_embedding_projection():
    # Typical dimension for language model embeddings
    k, mean_dist, max_dist, overlap = _evaluate_projection(n=60, d=768, epsilon=0.5)
    assert k <= 768
    assert mean_dist < 1.0  # Adjusted threshold
    assert max_dist < 2.0
    assert overlap > 0.1


def test_mass_spec_projection():
    # High dimensional spectra-like features
    k, mean_dist, max_dist, overlap = _evaluate_projection(n=40, d=2000, epsilon=0.7)
    assert k <= 2000
    assert mean_dist < 1.0  # Adjusted threshold
    assert max_dist < 2.5
    assert overlap > 0.1
