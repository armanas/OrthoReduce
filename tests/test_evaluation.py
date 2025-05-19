import numpy as np
from sklearn.metrics import pairwise_distances

from orthogonal_projection.evaluation import compute_distortion


def test_compute_distortion_ignores_diagonal():
    # Simple dataset of three points
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # Scale the data to induce non-zero distortion
    Y = 2.0 * X

    mean_d, max_d, D_orig_sq, D_red_sq = compute_distortion(X, Y)

    # Manually compute expected distortion ignoring the diagonal
    D_original = pairwise_distances(X) ** 2
    D_reduced = pairwise_distances(Y) ** 2
    distortion = np.abs(D_reduced - D_original) / (D_original + 1e-9)
    mask = ~np.eye(distortion.shape[0], dtype=bool)
    expected_mean = distortion[mask].mean()
    expected_max = distortion[mask].max()

    assert np.isclose(mean_d, expected_mean)
    assert np.isclose(max_d, expected_max)
