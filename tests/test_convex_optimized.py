import numpy as np
from orthogonal_projection.convex_optimized import project_onto_convex_hull_qp


def _regular_polygon(k=6, radius=1.0, seed=0):
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    V = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    return V


def test_convex_projection_qp_constraints_and_accuracy():
    rng = np.random.default_rng(0)
    V = _regular_polygon(k=5, radius=1.0)

    # Sample interior points as convex combinations of vertices
    n = 50
    alphas_true = rng.random((n, V.shape[0]))
    alphas_true /= alphas_true.sum(axis=1, keepdims=True)
    Y = alphas_true @ V  # (n, 2)

    Y_proj, alphas, V_used = project_onto_convex_hull_qp(Y, tol=1e-8, maxiter=200)

    # Projection error should be very small for interior points
    err = np.linalg.norm(Y - Y_proj, axis=1)
    assert np.median(err) < 1e-4
    assert np.max(err) < 1e-2

    # Constraint validation: sum to 1 and non-negative for most points
    sums = alphas.sum(axis=1)
    nonneg = (alphas >= -1e-6).all(axis=1)
    close_sums = np.isclose(sums, 1.0, atol=1e-3)
    ok = nonneg & close_sums
    frac_ok = ok.mean()
    assert frac_ok >= 0.99
