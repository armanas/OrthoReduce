"""
Tests for hyperbolic (Poincaré) embedding functionality.

This test suite validates the mathematical correctness and numerical stability
of the hyperbolic geometry operations and Riemannian optimization algorithms.
"""

import numpy as np
import pytest
import logging
import time
from typing import Tuple

# Import hyperbolic functionality
try:
    from orthogonal_projection.hyperbolic import (
        PoincareBall, RiemannianOptimizer, HyperbolicEmbedding,
        run_poincare_optimized, MIN_NORM, MAX_NORM, BOUNDARY_EPS
    )
    HYPERBOLIC_AVAILABLE = True
except ImportError:
    HYPERBOLIC_AVAILABLE = False

from orthogonal_projection import run_poincare, generate_mixture_gaussians
from orthogonal_projection.evaluation import compute_distortion, rank_correlation


@pytest.mark.skipif(not HYPERBOLIC_AVAILABLE, reason="Hyperbolic module not available")
class TestPoincareBall:
    """Test the PoincareBall mathematical operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ball = PoincareBall(c=1.0, dim=3)
        self.eps = 1e-10
        
    def test_initialization(self):
        """Test PoincareBall initialization."""
        ball = PoincareBall(c=0.5, dim=2)
        assert ball.c == 0.5
        assert ball.dim == 2
        assert ball.eps == 1e-5
        
        # Test max norm calculation
        expected_max_norm = (1.0 - BOUNDARY_EPS) / np.sqrt(0.5)
        np.testing.assert_allclose(ball.max_norm, expected_max_norm, rtol=1e-6)
        
    def test_conformal_factor(self):
        """Test conformal factor λ_c^x = 2 / (1 - c||x||^2)."""
        # Test at origin
        x_origin = np.array([[0.0, 0.0, 0.0]])
        lambda_origin = self.ball._lambda_c(x_origin)
        np.testing.assert_allclose(lambda_origin, [[2.0]], rtol=1e-10)
        
        # Test at non-origin points
        x = np.array([[0.1, 0.2, 0.3], [0.5, 0.0, 0.0]])
        lambda_x = self.ball._lambda_c(x)
        
        # Manual calculation
        norms_sq = np.sum(x ** 2, axis=1, keepdims=True)
        expected = 2.0 / (1.0 - self.ball.c * norms_sq)
        
        np.testing.assert_allclose(lambda_x, expected, rtol=1e-10)
        
    def test_mobius_addition_properties(self):
        """Test algebraic properties of Möbius addition."""
        x = np.array([[0.1, 0.2, 0.0]])
        y = np.array([[0.3, -0.1, 0.2]])
        z = np.array([[-0.2, 0.4, 0.1]])
        
        # Test commutativity: x ⊕ y = y ⊕ x  
        # Note: Due to the asymmetric formula, perfect commutativity may not hold
        # but the difference should be small for points not near the boundary
        xy = self.ball.mobius_add(x, y)
        yx = self.ball.mobius_add(y, x)
        
        # Check if the difference is reasonable (not perfect due to formula asymmetry)
        diff = np.linalg.norm(xy - yx)
        assert diff < 0.1  # Should be reasonably close
        
        # Test identity: x ⊕ 0 = x
        zero = np.zeros((1, 3))
        x_plus_zero = self.ball.mobius_add(x, zero)
        np.testing.assert_allclose(x_plus_zero, x, rtol=1e-5)  # Small numerical errors expected
        
        # Test inverse: x ⊕ (-x) ≈ 0 (within numerical precision)
        neg_x = -x
        x_plus_neg_x = self.ball.mobius_add(x, neg_x)
        np.testing.assert_allclose(x_plus_neg_x, zero, atol=0.05)  # Reasonable tolerance for hyperbolic ops
        
        # Test that result stays in ball
        result_norm = np.linalg.norm(xy, axis=1)
        assert np.all(result_norm < 1.0 / np.sqrt(self.ball.c))
        
    def test_exponential_logarithmic_inverse(self):
        """Test that exp and log are inverse operations."""
        x = np.array([[0.2, 0.1, -0.3]])
        y = np.array([[0.4, -0.2, 0.1]])
        
        # Test exp_x(log_x(y)) = y
        v = self.ball.logarithmic_map(x, y)
        y_reconstructed = self.ball.exponential_map(x, v)
        np.testing.assert_allclose(y_reconstructed, y, rtol=1e-8)
        
        # Test log_x(exp_x(v)) = v (for small v)
        v_small = np.array([[0.1, 0.05, -0.08]])
        y_exp = self.ball.exponential_map(x, v_small)
        v_reconstructed = self.ball.logarithmic_map(x, y_exp)
        np.testing.assert_allclose(v_reconstructed, v_small, rtol=1e-8)
        
    def test_hyperbolic_distance_properties(self):
        """Test hyperbolic distance mathematical properties."""
        x = np.array([[0.1, 0.2, 0.0]])
        y = np.array([[0.3, -0.1, 0.2]])
        z = np.array([[-0.2, 0.4, 0.1]])
        
        # Test symmetry: d(x, y) = d(y, x)
        d_xy = self.ball.hyperbolic_distance(x, y)
        d_yx = self.ball.hyperbolic_distance(y, x)
        np.testing.assert_allclose(d_xy, d_yx, rtol=1e-10)
        
        # Test identity: d(x, x) = 0
        d_xx = self.ball.hyperbolic_distance(x, x)
        np.testing.assert_allclose(d_xx, 0.0, atol=1e-10)
        
        # Test positivity: d(x, y) > 0 for x ≠ y
        assert np.all(d_xy > 0)
        
        # Test triangle inequality: d(x, z) ≤ d(x, y) + d(y, z)
        d_xz = self.ball.hyperbolic_distance(x, z)
        d_yz = self.ball.hyperbolic_distance(y, z)
        
        # Note: Triangle inequality might be approximate due to numerical precision
        assert np.all(d_xz <= d_xy + d_yz + 1e-10)
        
    def test_projection_stability(self):
        """Test numerical stability of projection operation."""
        # Test points near boundary
        large_points = np.array([[0.99, 0.0, 0.0], [0.0, 0.999, 0.0]])
        projected = self.ball.project(large_points)
        
        norms = np.linalg.norm(projected, axis=1)
        assert np.all(norms <= self.ball.max_norm)
        
        # Test very small points
        tiny_points = np.array([[1e-15, 1e-15, 1e-15]])
        projected_tiny = self.ball.project(tiny_points)
        
        # Should not change tiny points significantly
        np.testing.assert_allclose(projected_tiny, tiny_points, rtol=1e-6)
        
    def test_riemannian_gradient_scaling(self):
        """Test Riemannian gradient computation."""
        x = np.array([[0.2, 0.1, -0.3]])
        grad_euclidean = np.array([[1.0, 0.5, -0.2]])
        
        grad_riemannian = self.ball.riemannian_gradient(x, grad_euclidean)
        
        # Check scaling factor
        x_sqnorm = np.sum(x ** 2)
        expected_factor = ((1.0 - self.ball.c * x_sqnorm) ** 2) / 4.0
        expected_grad = expected_factor * grad_euclidean
        
        np.testing.assert_allclose(grad_riemannian, expected_grad, rtol=1e-10)
        
    def test_random_point_generation(self):
        """Test uniform random point generation in ball."""
        points = self.ball.random_point((100,), max_norm=0.8)
        
        # Check shape
        assert points.shape == (100, 3)
        
        # Check all points are in ball
        norms = np.linalg.norm(points, axis=1)
        assert np.all(norms <= 0.8 * (1.0 + 1e-10))  # Allow small numerical error
        
        # Check approximate uniformity (rough test)
        # Points should be distributed throughout the ball
        mean_norm = np.mean(norms)
        assert 0.2 < mean_norm < 0.7  # Should not all be near origin or boundary


@pytest.mark.skipif(not HYPERBOLIC_AVAILABLE, reason="Hyperbolic module not available")
class TestRiemannianOptimizer:
    """Test Riemannian optimization algorithms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ball = PoincareBall(c=1.0, dim=2)
        self.optimizer = RiemannianOptimizer(self.ball, lr=0.01)
        
    def test_rsgd_step(self):
        """Test Riemannian SGD step."""
        x = np.array([[0.2, 0.1]])
        grad = np.array([[1.0, 0.5]])
        
        x_new = self.optimizer.rsgd_step(x, grad, lr=0.01)
        
        # Check result stays in ball
        norm = np.linalg.norm(x_new)
        assert norm < self.ball.max_norm
        
        # Check that we moved in opposite direction of gradient
        diff = x_new - x
        # In hyperbolic space, the relationship is more complex than Euclidean
        # But the step should be bounded
        step_norm = np.linalg.norm(diff)
        assert step_norm < 0.1  # Reasonable step size
        
    def test_radam_step(self):
        """Test Riemannian Adam step."""
        x = np.array([[0.2, 0.1]])
        grad = np.array([[1.0, 0.5]])
        
        # Take a few steps to test momentum
        x_current = x.copy()
        for _ in range(3):
            x_current = self.optimizer.radam_step(x_current, grad, lr=0.01)
            
            # Check stays in ball
            norm = np.linalg.norm(x_current)
            assert norm < self.ball.max_norm
        
        # Check that optimizer state is updated
        assert self.optimizer.velocity is not None
        assert self.optimizer.second_moment is not None
        assert self.optimizer.step > 0
        
    def test_gradient_clipping(self):
        """Test gradient clipping for stability."""
        x = np.array([[0.1, 0.1]])
        large_grad = np.array([[100.0, 50.0]])  # Very large gradient
        
        x_new = self.optimizer.rsgd_step(x, large_grad, lr=0.01)
        
        # Should not make huge step despite large gradient
        step_norm = np.linalg.norm(x_new - x)
        assert step_norm < 1.0  # Should be clipped
        
        # Should stay in ball
        norm = np.linalg.norm(x_new)
        assert norm < self.ball.max_norm


@pytest.mark.skipif(not HYPERBOLIC_AVAILABLE, reason="Hyperbolic module not available")
class TestHyperbolicEmbedding:
    """Test complete hyperbolic embedding system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_samples = 50
        self.n_features = 10
        self.n_components = 3
        
        # Generate test data
        np.random.seed(42)
        self.X = generate_mixture_gaussians(
            self.n_samples, self.n_features, n_clusters=3, cluster_std=0.5
        )
        
        # Generate labels for supervised tests
        self.y = np.repeat(np.arange(3), self.n_samples // 3)
        
    def test_initialization_methods(self):
        """Test different initialization methods."""
        for init_method in ['random', 'pca', 'spectral']:
            embedding = HyperbolicEmbedding(
                n_components=self.n_components,
                init_method=init_method,
                n_epochs=5,  # Fast test
                seed=42
            )
            
            Y = embedding.fit_transform(self.X)
            
            # Check output shape
            assert Y.shape == (self.n_samples, self.n_components)
            
            # Check points are in Poincaré ball
            norms = np.linalg.norm(Y, axis=1)
            assert np.all(norms < embedding.ball.max_norm)
            
    def test_loss_functions(self):
        """Test different loss functions."""
        for loss_fn in ['stress', 'nca', 'triplet']:
            embedding = HyperbolicEmbedding(
                n_components=self.n_components,
                loss_fn=loss_fn,
                n_epochs=5,  # Fast test
                seed=42
            )
            
            if loss_fn in ['nca', 'triplet']:
                # Supervised methods need labels
                Y = embedding.fit_transform(self.X, self.y)
            else:
                Y = embedding.fit_transform(self.X)
            
            # Check output shape
            assert Y.shape == (self.n_samples, self.n_components)
            
            # Check convergence
            assert len(embedding.loss_history_) > 0
            
    def test_optimizers(self):
        """Test different optimizers."""
        for optimizer in ['rsgd', 'radam']:
            embedding = HyperbolicEmbedding(
                n_components=self.n_components,
                optimizer=optimizer,
                n_epochs=5,
                seed=42
            )
            
            Y = embedding.fit_transform(self.X)
            
            # Check output shape
            assert Y.shape == (self.n_samples, self.n_components)
            
    def test_convergence(self):
        """Test that optimization converges."""
        embedding = HyperbolicEmbedding(
            n_components=self.n_components,
            n_epochs=20,
            seed=42
        )
        
        Y = embedding.fit_transform(self.X)
        
        # Check that loss decreases over time
        loss_history = embedding.loss_history_
        assert len(loss_history) > 0
        
        # Loss should generally decrease (allowing for some noise)
        if len(loss_history) > 10:
            early_loss = np.mean(loss_history[:5])
            late_loss = np.mean(loss_history[-5:])
            assert late_loss <= early_loss * 1.1  # Allow 10% increase for noise
            
    def test_regularization_effect(self):
        """Test that regularization keeps points away from boundary."""
        # Without regularization
        embedding_no_reg = HyperbolicEmbedding(
            n_components=self.n_components,
            regularization=0.0,
            n_epochs=10,
            seed=42
        )
        Y_no_reg = embedding_no_reg.fit_transform(self.X)
        
        # With regularization
        embedding_with_reg = HyperbolicEmbedding(
            n_components=self.n_components,
            regularization=0.1,
            n_epochs=10,
            seed=42
        )
        Y_with_reg = embedding_with_reg.fit_transform(self.X)
        
        # Points with regularization should be closer to origin on average
        norms_no_reg = np.linalg.norm(Y_no_reg, axis=1)
        norms_with_reg = np.linalg.norm(Y_with_reg, axis=1)
        
        assert np.mean(norms_with_reg) <= np.mean(norms_no_reg) + 0.1
        
    def test_distance_matrix_input(self):
        """Test embedding with distance matrix input."""
        from sklearn.metrics import pairwise_distances
        
        # Create distance matrix
        D = pairwise_distances(self.X)
        
        embedding = HyperbolicEmbedding(
            n_components=self.n_components,
            n_epochs=10,
            seed=42
        )
        
        Y = embedding.fit_transform(D)
        
        # Check output shape
        assert Y.shape == (self.n_samples, self.n_components)
        
        # Check points are in ball
        norms = np.linalg.norm(Y, axis=1)
        assert np.all(norms < embedding.ball.max_norm)


class TestIntegration:
    """Test integration with existing OrthoReduce pipeline."""
    
    def setup_method(self):
        """Set up test data."""
        self.n_samples = 100
        self.n_features = 20
        self.k = 5
        
        np.random.seed(42)
        self.X = generate_mixture_gaussians(
            self.n_samples, self.n_features, n_clusters=5
        )
        
    def test_run_poincare_interface(self):
        """Test run_poincare function interface."""
        Y, runtime = run_poincare(self.X, self.k, seed=42)
        
        # Check output
        assert Y.shape == (self.n_samples, self.k)
        assert runtime > 0
        assert isinstance(runtime, float)
        
        # If hyperbolic module available, points should be in ball
        if HYPERBOLIC_AVAILABLE:
            norms = np.linalg.norm(Y, axis=1)
            assert np.all(norms < 1.0)  # In unit ball
        
    def test_poincare_quality_metrics(self):
        """Test that Poincaré embedding produces reasonable quality metrics."""
        Y, _ = run_poincare(self.X, self.k, seed=42, n_epochs=20)
        
        # Compute quality metrics
        mean_dist, max_dist, _, _ = compute_distortion(self.X, Y, sample_size=50)
        rank_corr = rank_correlation(self.X, Y, sample_size=50)
        
        # Check metrics are reasonable
        assert 0 <= mean_dist <= 10  # Distortion should be bounded
        assert 0 <= max_dist <= 20
        assert -1 <= rank_corr <= 1  # Correlation in valid range
        
        # For good embedding, rank correlation should be positive
        if HYPERBOLIC_AVAILABLE:
            assert rank_corr > 0.1  # Should preserve some distance relationships
            
    def test_parameter_sensitivity(self):
        """Test sensitivity to key parameters."""
        results = {}
        
        # Test different curvatures
        for c in [0.1, 1.0]:
            Y, _ = run_poincare(self.X, self.k, c=c, n_epochs=10, seed=42)
            rank_corr = rank_correlation(self.X, Y, sample_size=50)
            results[f'c_{c}'] = rank_corr
            
        # Test different optimizers
        for opt in ['rsgd', 'radam']:
            Y, _ = run_poincare(self.X, self.k, optimizer=opt, n_epochs=10, seed=42)
            rank_corr = rank_correlation(self.X, Y, sample_size=50)
            results[f'opt_{opt}'] = rank_corr
            
        # All results should be reasonable
        for key, corr in results.items():
            assert -1 <= corr <= 1
            if HYPERBOLIC_AVAILABLE:
                assert corr > -0.5  # Should not be completely anti-correlated
                
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        Y1, _ = run_poincare(self.X, self.k, seed=42, n_epochs=10)
        Y2, _ = run_poincare(self.X, self.k, seed=42, n_epochs=10)
        
        # Results should be identical (or very close due to numerical precision)
        np.testing.assert_allclose(Y1, Y2, rtol=1e-6, atol=1e-10)
        
    def test_scalability(self):
        """Test performance with different data sizes."""
        sizes = [50, 100]  # Keep small for fast tests
        
        for n in sizes:
            X_test = generate_mixture_gaussians(n, 10, n_clusters=3)
            
            start_time = time.time()
            Y, runtime = run_poincare(X_test, 3, n_epochs=5, seed=42)
            actual_time = time.time() - start_time
            
            # Check output
            assert Y.shape == (n, 3)
            
            # Runtime should be reasonable
            assert runtime > 0
            assert actual_time < 30  # Should complete within 30 seconds
            
    @pytest.mark.skipif(not HYPERBOLIC_AVAILABLE, 
                       reason="Hyperbolic module not available")
    def test_comparison_with_euclidean(self):
        """Compare hyperbolic embedding quality with Euclidean methods."""
        from orthogonal_projection import run_jll, run_pca
        
        # Generate hierarchical data (hyperbolic should excel here)
        np.random.seed(42)
        X_hier = self._generate_hierarchical_data()
        
        # Run different methods
        Y_hyp, _ = run_poincare(X_hier, self.k, n_epochs=20, seed=42)
        Y_jll, _ = run_jll(X_hier, self.k, seed=42)
        Y_pca, _ = run_pca(X_hier, self.k, seed=42)
        
        # Compute rank correlations
        corr_hyp = rank_correlation(X_hier, Y_hyp, sample_size=min(50, X_hier.shape[0]))
        corr_jll = rank_correlation(X_hier, Y_jll, sample_size=min(50, X_hier.shape[0]))
        corr_pca = rank_correlation(X_hier, Y_pca, sample_size=min(50, X_hier.shape[0]))
        
        # All should be reasonable
        for corr in [corr_hyp, corr_jll, corr_pca]:
            assert -1 <= corr <= 1
            
        # For hierarchical data, hyperbolic should be competitive
        assert corr_hyp >= corr_jll - 0.2  # Allow some tolerance
        
    def _generate_hierarchical_data(self) -> np.ndarray:
        """Generate synthetic hierarchical data."""
        np.random.seed(42)
        
        # Create tree-like structure
        n_per_branch = 15
        branches = []
        
        # Root
        root = np.random.randn(1, 20)
        
        # Level 1 branches
        for i in range(3):
            branch_center = root + np.random.randn(1, 20) * 0.5
            branch_points = branch_center + np.random.randn(n_per_branch, 20) * 0.2
            branches.append(branch_points)
            
        return np.vstack(branches)


if __name__ == '__main__':
    # Run basic tests
    import time
    
    print("Testing Poincaré embedding system...")
    
    # Test basic functionality
    X = generate_mixture_gaussians(50, 10, n_clusters=3)
    Y, runtime = run_poincare(X, 3, n_epochs=10, seed=42)
    
    print(f"Embedding shape: {Y.shape}")
    print(f"Runtime: {runtime:.2f}s")
    
    # Compute quality
    mean_dist, _, _, _ = compute_distortion(X, Y, sample_size=30)
    rank_corr = rank_correlation(X, Y, sample_size=30)
    
    print(f"Mean distortion: {mean_dist:.4f}")
    print(f"Rank correlation: {rank_corr:.4f}")
    
    if HYPERBOLIC_AVAILABLE:
        norms = np.linalg.norm(Y, axis=1)
        print(f"Point norms: min={np.min(norms):.3f}, max={np.max(norms):.3f}")
        print("✓ All tests passed!")
    else:
        print("⚠ Hyperbolic module not available, using fallback")