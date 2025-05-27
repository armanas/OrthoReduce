"""
Test the enhanced methods: mixture of gaussians generation and convex hull projection.
"""
import numpy as np
import pytest
from orthogonal_projection.dimensionality_reduction import (
    generate_mixture_gaussians,
    project_onto_convex_hull,
    run_convex,
    run_experiment
)


def test_generate_mixture_gaussians():
    """Test the mixture of Gaussians data generation."""
    n, d = 100, 50
    n_clusters = 5
    cluster_std = 0.5
    seed = 42
    
    X = generate_mixture_gaussians(n, d, n_clusters, cluster_std, seed)
    
    # Check shape
    assert X.shape == (n, d)
    
    # Check that it's different from pure random (should have some structure)
    # Generate purely random data for comparison
    np.random.seed(seed)
    X_random = np.random.randn(n, d)
    
    # The mixture should have different statistics than pure random
    assert not np.allclose(X, X_random, rtol=0.1)


def test_project_onto_convex_hull():
    """Test the convex hull projection function."""
    # Create simple test data
    Y = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [0.5, 0.5]  # This point should be projected
    ])
    
    Y_proj = project_onto_convex_hull(Y)
    
    # Check shape is preserved
    assert Y_proj.shape == Y.shape
    
    # Check it doesn't crash with larger data
    Y_large = np.random.randn(30, 10)
    Y_large_proj = project_onto_convex_hull(Y_large)
    assert Y_large_proj.shape == Y_large.shape


def test_run_convex():
    """Test the convex hull + JLL pipeline."""
    X = generate_mixture_gaussians(50, 20, 3, 0.5, 42)
    k = 5
    seed = 42
    
    Y, runtime = run_convex(X, k, seed)
    
    # Check output shape
    assert Y.shape == (50, k)
    
    # Check runtime is reasonable
    assert runtime >= 0
    assert runtime < 10  # Should be reasonably fast


def test_run_experiment_with_convex():
    """Test that run_experiment works with convex hull projection enabled."""
    results = run_experiment(
        n=30, d=10, epsilon=0.5, seed=42, sample_size=1000,
        use_convex=True, n_clusters=3, cluster_std=0.3,
        use_poincare=False, use_spherical=False, use_elliptic=False
    )
    
    # Check that Convex method is included when use_convex=True
    assert 'Convex' in results
    
    # Check that all methods have the expected metrics
    expected_metrics = ['mean_distortion', 'max_distortion', 'rank_correlation', 
                       'kl_divergence', 'l1', 'runtime']
    
    for method, metrics in results.items():
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))


def test_run_experiment_without_convex():
    """Test that run_experiment works without convex hull projection."""
    results = run_experiment(
        n=30, d=10, epsilon=0.5, seed=42, sample_size=1000,
        use_convex=False, n_clusters=3, cluster_std=0.3,
        use_poincare=False, use_spherical=False, use_elliptic=False
    )
    
    # Check that Convex method is NOT included when use_convex=False
    assert 'Convex' not in results
    
    # Check that basic methods are still there
    assert 'PCA' in results
    assert 'JLL' in results