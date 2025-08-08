import numpy as np
import pytest
from orthogonal_projection.dimensionality_reduction import (
    run_experiment, 
    run_pca_simple, 
    run_jll_simple,
    generate_mixture_gaussians,
    evaluate_projection
)

def test_run_experiment_basic():
    """Test the main experiment function with small data."""
    results = run_experiment(n=50, d=20, epsilon=0.5, seed=42, sample_size=100)
    
    # Check that we got results for expected methods
    assert 'PCA' in results
    assert 'JLL' in results
    assert 'GAUSSIAN' in results
    
    # Check that each method has expected metrics
    for method, metrics in results.items():
        assert 'mean_distortion' in metrics
        assert 'max_distortion' in metrics
        assert 'rank_correlation' in metrics
        assert 'runtime' in metrics
        
        # Basic sanity checks
        assert metrics['runtime'] >= 0
        assert metrics['mean_distortion'] >= 0
        assert metrics['max_distortion'] >= 0

def test_simple_interfaces():
    """Test the simple interface functions."""
    n, d, k = 50, 20, 10
    X = np.random.randn(n, d)
    
    # Test simple interfaces
    Y_pca = run_pca_simple(X, k, seed=42)
    Y_jll = run_jll_simple(X, k, seed=42)
    
    assert Y_pca.shape == (n, k)
    assert Y_jll.shape == (n, k)

def test_generate_mixture_gaussians():
    """Test mixture of Gaussians data generation."""
    X = generate_mixture_gaussians(n=100, d=10, n_clusters=5, cluster_std=0.5, seed=42)
    
    assert X.shape == (100, 10)
    assert np.isfinite(X).all()

def test_evaluate_projection():
    """Test the projection evaluation function."""
    n, d, k = 100, 50, 20
    X = np.random.randn(n, d)
    Y = np.random.randn(n, k)
    
    metrics = evaluate_projection(X, Y, sample_size=50)
    
    assert 'mean_distortion' in metrics
    assert 'max_distortion' in metrics  
    assert 'rank_correlation' in metrics
    
    # Check types
    assert isinstance(metrics['mean_distortion'], float)
    assert isinstance(metrics['max_distortion'], float)
    assert isinstance(metrics['rank_correlation'], float)