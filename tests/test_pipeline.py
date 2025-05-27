import numpy as np
from orthogonal_projection.pipeline import (
    run_jll,
    run_poincare_pipeline,
    run_spherical_pipeline,
    run_experiment,
)


def test_run_jll_shape():
    n, d = 50, 20
    k = 5
    X = np.random.randn(n, d)
    Y = run_jll(X, k)
    assert Y.shape == (n, k)


def test_geometric_pipelines_shape():
    n, d = 30, 10
    k = 4
    X = np.random.randn(n, d)
    Y_poincare = run_poincare_pipeline(X, k, c=1.0)
    Y_spherical = run_spherical_pipeline(X, k)
    assert Y_poincare.shape == (n, d)
    assert Y_spherical.shape == (n, d)


def test_run_experiment_small():
    results = run_experiment(n=100, d=20, epsilon=0.5, seed=0)
    assert set(results.keys()) == {"PCA", "JLL", "UMAP", "Poincare", "Spherical"}

