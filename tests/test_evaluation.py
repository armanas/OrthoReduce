import numpy as np
from orthogonal_projection.evaluation import compute_distortion, nearest_neighbor_overlap
from orthogonal_projection.projection import project_data


def test_compute_distortion_identity_projection():
    n, d = 50, 10
    X = np.random.randn(n, d)
    basis = np.eye(d)
    Y = project_data(X, basis)
    mean_distortion, max_distortion, _, _ = compute_distortion(X, Y)
    assert mean_distortion < 1e-6
    assert max_distortion < 1e-6


def test_nearest_neighbor_overlap_identical():
    n, d = 30, 5
    X = np.random.randn(n, d)
    overlap = nearest_neighbor_overlap(X, X, k=5)
    assert overlap == 1.0
