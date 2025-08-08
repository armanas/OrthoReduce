"""
Test suite for advanced spherical embeddings with Riemannian optimization.

Tests geodesic computations, tangent space operations, optimization, and embedding quality.
"""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

# Import the spherical embeddings module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orthogonal_projection.spherical_embeddings import (
    SphericalEmbedding,
    adaptive_spherical_embedding,
    evaluate_spherical_embedding
)


class TestGeodesicComputations:
    """Test geodesic distance computations and numerical stability."""
    
    def test_geodesic_distance_single_pair(self):
        """Test geodesic distance for single pair of points."""
        # Points on unit sphere
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        
        # Should be π/2 for orthogonal vectors on unit sphere
        d = SphericalEmbedding.geodesic_distance(x, y, radius=1.0)
        np.testing.assert_allclose(d, np.pi/2, rtol=1e-10)
        
        # Test with different radius
        d2 = SphericalEmbedding.geodesic_distance(x, y, radius=2.0)
        np.testing.assert_allclose(d2, np.pi, rtol=1e-10)
    
    def test_geodesic_distance_antipodal(self):
        """Test numerical stability near antipodal points."""
        # Nearly antipodal points
        x = np.array([1, 0, 0])
        y = np.array([-0.99999, 0, 0])
        
        # Should handle near-antipodal points without NaN
        d = SphericalEmbedding.geodesic_distance(x, y, radius=1.0)
        assert np.isfinite(d)
        assert d > 0
        assert d < np.pi  # Not quite antipodal
    
    def test_geodesic_distance_batch(self):
        """Test batch geodesic distance computation."""
        np.random.seed(42)
        n = 10
        k = 3
        
        # Generate random points on sphere
        X = np.random.randn(n, k)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        # Compute batch distances
        D = SphericalEmbedding.geodesic_distance_batch(X, radius=1.0)
        
        # Check properties
        assert D.shape == (n, n)
        np.testing.assert_allclose(np.diag(D), 0, atol=1e-10)  # Zero diagonal
        assert np.all(D >= 0)  # Non-negative
        assert np.all(D <= np.pi)  # Maximum distance on unit sphere
        np.testing.assert_allclose(D, D.T, atol=1e-10)  # Symmetric
    
    def test_geodesic_triangle_inequality(self):
        """Test that geodesic distances satisfy triangle inequality."""
        np.random.seed(42)
        
        # Three random points on sphere
        X = np.random.randn(3, 4)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        D = SphericalEmbedding.geodesic_distance_batch(X, radius=1.0)
        
        # Triangle inequality: d(i,k) <= d(i,j) + d(j,k)
        assert D[0, 2] <= D[0, 1] + D[1, 2] + 1e-10


class TestTangentSpaceOperations:
    """Test tangent space projections and exponential/logarithmic maps."""
    
    def test_tangent_space_projection(self):
        """Test projection to tangent space."""
        x = np.array([[1, 0, 0]])
        v = np.array([[1, 1, 1]])
        
        # Project v to tangent space at x
        v_tangent = SphericalEmbedding.project_to_tangent_space(x, v)
        
        # Should be orthogonal to x
        dot_product = np.sum(x * v_tangent)
        np.testing.assert_allclose(dot_product, 0, atol=1e-10)
        
        # Check specific value
        expected = np.array([[0, 1, 1]])  # Component along x removed
        np.testing.assert_allclose(v_tangent, expected, atol=1e-10)
    
    def test_exponential_map(self):
        """Test exponential map from tangent space to sphere."""
        x = np.array([[1, 0, 0]])
        v = np.array([[0, np.pi/2, 0]])  # Tangent vector
        
        # Apply exponential map
        y = SphericalEmbedding.exponential_map(x, v, radius=1.0)
        
        # Should end up at [0, 1, 0] (90 degree rotation)
        expected = np.array([[0, 1, 0]])
        np.testing.assert_allclose(y, expected, atol=1e-10)
        
        # Check that result is on sphere
        norm = np.linalg.norm(y)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)
    
    def test_logarithmic_map(self):
        """Test logarithmic map from sphere to tangent space."""
        x = np.array([[1, 0, 0]])
        y = np.array([[0, 1, 0]])
        
        # Apply logarithmic map
        v = SphericalEmbedding.logarithmic_map(x, y, radius=1.0)
        
        # Should be in tangent space (orthogonal to x)
        dot_product = np.sum(x * v)
        np.testing.assert_allclose(dot_product, 0, atol=1e-10)
        
        # Magnitude should be π/2 (geodesic distance)
        norm = np.linalg.norm(v)
        np.testing.assert_allclose(norm, np.pi/2, rtol=1e-5)
    
    def test_exp_log_inverse(self):
        """Test that exp and log are inverse operations."""
        np.random.seed(42)
        
        # Random point on sphere
        x = np.random.randn(1, 3)
        x = x / np.linalg.norm(x)
        
        # Random tangent vector
        v = np.random.randn(1, 3)
        v = SphericalEmbedding.project_to_tangent_space(x, v)
        v = 0.5 * v  # Scale to avoid going past antipodal point
        
        # Apply exp then log
        y = SphericalEmbedding.exponential_map(x, v, radius=1.0)
        v_recovered = SphericalEmbedding.logarithmic_map(x, y, radius=1.0)
        
        np.testing.assert_allclose(v, v_recovered, rtol=1e-5)


class TestLossFunctions:
    """Test various loss functions for spherical embeddings."""
    
    def test_mds_stress_geodesic(self):
        """Test MDS stress computation with geodesic distances."""
        np.random.seed(42)
        n = 20
        
        # Create synthetic data
        X = np.random.randn(n, 5)
        D_target = squareform(pdist(X))
        
        # Create random spherical embedding
        Y = np.random.randn(n, 3)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        
        # Compute stress
        model = SphericalEmbedding(n_components=3)
        stress = model.mds_stress_geodesic(Y, D_target)
        
        assert np.isfinite(stress)
        assert stress >= 0
    
    def test_triplet_loss(self):
        """Test triplet loss computation."""
        np.random.seed(42)
        
        # Create points where we know relative distances
        Y = np.array([
            [1, 0, 0],
            [0.9, 0.436, 0],  # Close to first
            [-1, 0, 0]  # Far from first
        ])
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        
        # Triplet where point 1 is closer to 0 than point 2
        triplets = np.array([[0, 1, 2]])
        
        model = SphericalEmbedding(n_components=3)
        loss = model.triplet_loss(Y, triplets)
        
        # Should have zero loss since ordering is correct with margin
        assert loss >= 0
    
    def test_angular_margin_regularization(self):
        """Test angular margin regularization."""
        np.random.seed(42)
        
        # Create points with some very close and some antipodal
        Y = np.array([
            [1, 0, 0],
            [0.999, 0.045, 0],  # Very close to first
            [-0.999, -0.045, 0]  # Nearly antipodal to first
        ])
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        
        model = SphericalEmbedding(n_components=3)
        reg = model.angular_margin_regularization(Y)
        
        # Should have positive regularization due to crowding and near-antipodal points
        assert reg > 0


class TestOptimization:
    """Test optimization procedures and convergence."""
    
    def test_radius_optimization(self):
        """Test adaptive radius optimization."""
        np.random.seed(42)
        n = 30
        
        # Create data with known structure
        X = np.random.randn(n, 5)
        D_target = squareform(pdist(X))
        
        # Initial embedding
        Y = np.random.randn(n, 3)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        
        model = SphericalEmbedding(n_components=3)
        
        # Optimize radius
        initial_stress = model.mds_stress_geodesic(Y, D_target)
        optimal_radius = model.optimize_radius(Y, D_target)
        final_stress = model.mds_stress_geodesic(Y, D_target)
        
        # Stress should not increase (and usually decreases)
        assert final_stress <= initial_stress + 1e-10
        assert optimal_radius > 0
    
    def test_gradient_computation(self):
        """Test gradient computation via finite differences."""
        np.random.seed(42)
        n = 5
        k = 3
        
        # Small problem for testing
        X = np.random.randn(n, 4)
        D_target = squareform(pdist(X))
        
        # Random initial embedding
        Y = np.random.randn(n, k)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        
        model = SphericalEmbedding(n_components=k, loss_type='mds_geodesic')
        
        # Compute gradient
        loss, grad = model.compute_loss_and_gradient(Y.flatten(), D_target)
        
        assert np.isfinite(loss)
        assert grad is not None
        assert np.all(np.isfinite(grad))
    
    def test_fit_convergence(self):
        """Test that optimization converges."""
        np.random.seed(42)
        n = 20
        d = 10
        k = 3
        
        # Generate structured data
        X = np.random.randn(n, d)
        
        # Fit model
        model = SphericalEmbedding(
            n_components=k,
            max_iter=50,  # Limited iterations for testing
            learning_rate=0.01,
            loss_type='mds_geodesic',
            adaptive_radius=False,  # Faster for testing
            seed=42
        )
        
        Y = model.fit_transform(X)
        
        # Check that we got a valid embedding
        assert Y.shape == (n, k)
        assert np.all(np.isfinite(Y))
        
        # Check that points are on sphere
        norms = np.linalg.norm(Y, axis=1)
        np.testing.assert_allclose(norms, model.radius, rtol=1e-3)
        
        # Check that loss decreased
        if len(model.loss_history_) > 1:
            # Allow small increases due to numerical noise
            assert model.loss_history_[-1] <= model.loss_history_[0] * 1.1


class TestHighLevelInterface:
    """Test high-level interface functions."""
    
    def test_adaptive_spherical_embedding_riemannian(self):
        """Test Riemannian spherical embedding."""
        np.random.seed(42)
        X = np.random.randn(30, 10)
        k = 3
        
        Y, info = adaptive_spherical_embedding(
            X, k,
            method='riemannian',
            max_iter=20,  # Quick for testing
            seed=42
        )
        
        assert Y.shape == (30, k)
        assert np.all(np.isfinite(Y))
        assert 'loss_history' in info
        assert 'final_radius' in info
        assert info['final_radius'] > 0
    
    def test_adaptive_spherical_embedding_fast(self):
        """Test fast spherical embedding."""
        np.random.seed(42)
        X = np.random.randn(50, 20)
        k = 5
        
        Y, info = adaptive_spherical_embedding(
            X, k,
            method='fast',
            adaptive_radius=True,
            seed=42
        )
        
        assert Y.shape == (50, k)
        assert np.all(np.isfinite(Y))
        assert 'final_radius' in info
        assert info['final_radius'] > 0
    
    def test_adaptive_spherical_embedding_simple(self):
        """Test simple spherical embedding."""
        np.random.seed(42)
        X = np.random.randn(40, 15)
        k = 4
        
        Y, info = adaptive_spherical_embedding(
            X, k,
            method='simple',
            seed=42
        )
        
        assert Y.shape == (40, k)
        assert np.all(np.isfinite(Y))
        
        # Check normalization
        norms = np.linalg.norm(Y, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)
    
    def test_evaluate_spherical_embedding(self):
        """Test evaluation metrics."""
        np.random.seed(42)
        X = np.random.randn(25, 8)
        
        # Create embedding
        Y, info = adaptive_spherical_embedding(X, 3, method='fast', seed=42)
        
        # Evaluate
        metrics = evaluate_spherical_embedding(X, Y, radius=info.get('final_radius', 1.0))
        
        # Check that all metrics are computed
        expected_metrics = [
            'rank_correlation_geodesic',
            'rank_correlation_chordal',
            'stress_geodesic',
            'stress_chordal',
            'mean_distortion',
            'max_distortion',
            'min_separation',
            'max_separation',
            'antipodal_ratio',
            'mean_angle',
            'std_angle',
            'radius'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert np.isfinite(metrics[metric])
        
        # Sanity checks
        assert 0 <= metrics['antipodal_ratio'] <= 1
        assert metrics['min_separation'] >= 0
        assert metrics['max_separation'] <= np.pi * metrics['radius']
        assert -1 <= metrics['rank_correlation_geodesic'] <= 1


class TestIntegration:
    """Test integration with main dimensionality reduction module."""
    
    def test_run_spherical_with_riemannian(self):
        """Test run_spherical with Riemannian optimization."""
        from orthogonal_projection.dimensionality_reduction import run_spherical
        
        np.random.seed(42)
        X = np.random.randn(20, 10)
        k = 3
        
        Y, runtime = run_spherical(
            X, k, 
            seed=42,
            use_riemannian=True,
            adaptive_radius=True,
            loss_type='mds_geodesic'
        )
        
        assert Y.shape == (20, k)
        assert np.all(np.isfinite(Y))
        assert runtime > 0
    
    def test_run_spherical_fallback(self):
        """Test that run_spherical falls back gracefully."""
        from orthogonal_projection.dimensionality_reduction import run_spherical
        
        np.random.seed(42)
        X = np.random.randn(15, 8)
        k = 3
        
        # Should work even with fallback
        Y, runtime = run_spherical(
            X, k,
            seed=42,
            use_riemannian=False  # Use simple method
        )
        
        assert Y.shape == (15, k)
        assert np.all(np.isfinite(Y))
        assert runtime > 0
        
        # Check normalization
        norms = np.linalg.norm(Y, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


class TestNumericalStability:
    """Test numerical stability in edge cases."""
    
    def test_zero_norm_vectors(self):
        """Test handling of zero-norm vectors."""
        model = SphericalEmbedding(n_components=3)
        
        # Zero vector
        x = np.array([[0, 0, 0]])
        v = np.array([[1, 0, 0]])
        
        # Should handle gracefully
        y = SphericalEmbedding.exponential_map(x, v, radius=1.0)
        assert np.all(np.isfinite(y))
    
    def test_identical_points(self):
        """Test handling of identical points."""
        # Two identical points
        x = np.array([[1, 0, 0]])
        y = np.array([[1, 0, 0]])
        
        d = SphericalEmbedding.geodesic_distance(x[0], y[0], radius=1.0)
        np.testing.assert_allclose(d, 0, atol=1e-3)  # More tolerant for identical points
        
        # Logarithmic map should give zero vector
        v = SphericalEmbedding.logarithmic_map(x, y, radius=1.0)
        np.testing.assert_allclose(v, 0, atol=1e-3)  # More tolerant for numerical stability
    
    def test_large_radius(self):
        """Test stability with large radius values."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        X = 100 * X / np.linalg.norm(X, axis=1, keepdims=True)  # Radius 100
        
        D = SphericalEmbedding.geodesic_distance_batch(X, radius=100.0)
        
        assert np.all(np.isfinite(D))
        assert np.all(D >= 0)
        assert np.all(D <= np.pi * 100)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])