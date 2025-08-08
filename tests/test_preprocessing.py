"""
Tests for preprocessing module

This module tests all preprocessing functionality including standardization,
whitening, denoising, and metric utilities.
"""
import pytest
import numpy as np
from sklearn.datasets import make_blobs
from scipy import sparse

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from orthogonal_projection.preprocessing import (
    standardize_features,
    l2_normalize_rows,
    whiten_data,
    pca_denoise,
    jl_denoise,
    compute_cosine_distances,
    spearman_loss,
    compute_distance_matrix,
    adaptive_preprocessing_pipeline
)


class TestStandardizeFeatures:
    """Test standardization functions."""
    
    def test_zscore_standardization(self):
        """Test z-score standardization."""
        # Create test data with known mean and std
        np.random.seed(42)
        X = np.random.randn(100, 10) * 5 + 10
        
        X_std, metadata = standardize_features(X, method='zscore')
        
        # Check that mean is approximately zero
        assert np.allclose(X_std.mean(axis=0), 0, atol=1e-10)
        
        # Check that std is approximately one
        assert np.allclose(X_std.std(axis=0, ddof=1), 1, atol=1e-10)
        
        # Check metadata
        assert metadata['method'] == 'zscore'
        assert metadata['center'] == True
        assert metadata['scale'] == True
        assert 'means' in metadata
        assert 'stds' in metadata
    
    def test_unit_variance_standardization(self):
        """Test unit variance standardization."""
        np.random.seed(42)
        X = np.random.randn(100, 10) * 3 + 5
        
        X_std, metadata = standardize_features(X, method='unit_variance')
        
        # Check that variance is approximately one
        assert np.allclose(X_std.var(axis=0, ddof=1), 1, atol=1e-10)
        
        # Check metadata
        assert metadata['method'] == 'unit_variance'
        assert 'vars' in metadata
    
    def test_robust_standardization(self):
        """Test robust standardization."""
        np.random.seed(42)
        X = np.random.randn(100, 10) * 2 + 3
        
        # Add outliers
        X[0, :] = 100
        X[1, :] = -100
        
        X_std, metadata = standardize_features(X, method='robust')
        
        # Check metadata
        assert metadata['method'] == 'robust'
        assert 'medians' in metadata
        assert 'mads' in metadata
        
        # Robust method should be less affected by outliers
        assert np.abs(X_std.mean()) < np.abs(X.mean())
    
    def test_center_only(self):
        """Test centering without scaling."""
        np.random.seed(42)
        X = np.random.randn(100, 10) * 3 + 5
        
        X_centered, metadata = standardize_features(X, center=True, scale=False)
        
        # Should be centered but not scaled
        assert np.allclose(X_centered.mean(axis=0), 0, atol=1e-10)
        assert not np.allclose(X_centered.std(axis=0), 1)
        
        assert metadata['center'] == True
        assert metadata['scale'] == False
    
    def test_invalid_method(self):
        """Test invalid standardization method."""
        X = np.random.randn(100, 10)
        
        with pytest.raises(Exception):  # Should raise ValidationError
            standardize_features(X, method='invalid')
    
    def test_small_sample_warning(self):
        """Test behavior with very small samples."""
        X = np.random.randn(1, 10)
        
        X_result, metadata = standardize_features(X)
        
        # Should return original data and metadata
        assert X_result.shape == X.shape
        assert metadata['n_samples'] == 1


class TestL2NormalizeRows:
    """Test L2 row normalization."""
    
    def test_basic_normalization(self):
        """Test basic L2 normalization."""
        np.random.seed(42)
        X = np.random.randn(100, 50) * 5 + 2
        
        X_norm, metadata = l2_normalize_rows(X)
        
        # Check that each row has unit norm
        norms = np.linalg.norm(X_norm, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)
        
        # Check metadata
        assert metadata['method'] == 'l2_normalize_rows'
        assert 'mean_norm' in metadata
        assert 'std_norm' in metadata
        assert 'zero_norm_count' in metadata
    
    def test_zero_norm_handling(self):
        """Test handling of zero-norm rows."""
        X = np.random.randn(10, 5)
        X[0, :] = 0  # Zero row
        
        X_norm, metadata = l2_normalize_rows(X, epsilon=1e-6)
        
        # Zero row should be normalized by epsilon
        assert not np.allclose(X_norm[0, :], 0)
        assert metadata['zero_norm_count'] == 1
    
    def test_copy_parameter(self):
        """Test copy parameter."""
        X = np.random.randn(10, 5)
        X_original = X.copy()
        
        # With copy=True (default)
        X_norm, _ = l2_normalize_rows(X, copy=True)
        assert np.array_equal(X, X_original)  # Original unchanged
        
        # With copy=False
        X_norm, _ = l2_normalize_rows(X, copy=False)
        assert not np.array_equal(X, X_original)  # Original changed


class TestWhitenData:
    """Test data whitening functions."""
    
    def test_pca_whitening(self):
        """Test PCA whitening."""
        np.random.seed(42)
        # Create correlated data
        X = np.random.randn(200, 10)
        X = X @ np.random.randn(10, 10)  # Add correlations
        
        X_white, metadata = whiten_data(X, method='pca')
        
        # Check that covariance is approximately identity
        cov = np.cov(X_white.T)
        assert np.allclose(cov, np.eye(X_white.shape[1]), atol=0.2)
        
        # Check metadata
        assert metadata['method'] == 'pca'
        assert 'explained_variance_ratio' in metadata
        assert 'components' in metadata
    
    def test_zca_whitening(self):
        """Test ZCA whitening."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X = X @ np.random.randn(5, 5)  # Add correlations
        
        X_white, metadata = whiten_data(X, method='zca')
        
        # Check that covariance is approximately identity
        cov = np.cov(X_white.T)
        assert np.allclose(cov, np.eye(X_white.shape[1]), atol=0.2)
        
        # Check metadata
        assert metadata['method'] == 'zca'
        assert 'eigenvals' in metadata
        assert 'whitening_matrix' in metadata
    
    def test_cholesky_whitening(self):
        """Test Cholesky whitening."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X = X @ np.random.randn(5, 5)  # Add correlations
        
        X_white, metadata = whiten_data(X, method='cholesky')
        
        # Check that covariance is approximately identity
        cov = np.cov(X_white.T)
        assert np.allclose(cov, np.eye(X_white.shape[1]), atol=0.2)
        
        # Check metadata
        assert metadata['method'] == 'cholesky'
    
    def test_regularization(self):
        """Test regularization parameter."""
        np.random.seed(42)
        # Create singular matrix
        X = np.random.randn(100, 10)
        X[:, -1] = X[:, 0]  # Make last column identical to first
        
        X_white, metadata = whiten_data(X, method='zca', regularization=1e-3)
        
        # Should not raise error and produce valid result
        assert X_white.shape == X.shape
        assert np.all(np.isfinite(X_white))
    
    def test_n_components(self):
        """Test n_components parameter for PCA whitening."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        
        n_comp = 5
        X_white, metadata = whiten_data(X, method='pca', n_components=n_comp)
        
        # Should reduce dimensionality
        assert X_white.shape[1] == n_comp
        assert metadata['n_components'] == n_comp


class TestDenoising:
    """Test denoising functions."""
    
    def test_pca_denoise(self):
        """Test PCA denoising."""
        np.random.seed(42)
        # Create low-rank data + noise
        U = np.random.randn(100, 5)
        V = np.random.randn(5, 20)
        X = U @ V + 0.1 * np.random.randn(100, 20)  # Low-rank + noise
        
        X_denoised, metadata = pca_denoise(X, n_components=5)
        
        # Should preserve shape
        assert X_denoised.shape == X.shape
        
        # Should reduce noise (lower reconstruction error for clean signal)
        assert metadata['snr_improvement'] > 1.0
        
        # Check metadata
        assert metadata['method'] == 'pca_denoise'
        assert metadata['n_components'] == 5
        assert 'explained_variance_ratio' in metadata
        assert 'compression_ratio' in metadata
    
    def test_pca_denoise_adaptive(self):
        """Test adaptive PCA denoising."""
        np.random.seed(42)
        X = np.random.randn(100, 500)  # High-dimensional
        
        X_denoised, metadata = pca_denoise(X, adaptive_components=True)
        
        # Should adaptively select components
        n_comp = metadata['n_components']
        assert 50 <= n_comp <= 300  # In adaptive range (allow some flexibility)
        
        assert X_denoised.shape == X.shape
    
    def test_jl_denoise(self):
        """Test Johnson-Lindenstrauss denoising."""
        np.random.seed(42)
        X = np.random.randn(100, 200)
        
        X_denoised, metadata = jl_denoise(X, n_components=50)
        
        # Should preserve shape
        assert X_denoised.shape == X.shape
        
        # Check metadata
        assert metadata['method'] == 'jl_denoise'
        assert metadata['n_components'] == 50
        assert 'reconstruction_error' in metadata
        assert 'variance_retained' in metadata
    
    def test_jl_denoise_adaptive(self):
        """Test adaptive JL denoising."""
        np.random.seed(42)
        X = np.random.randn(100, 1000)  # High-dimensional
        
        X_denoised, metadata = jl_denoise(X, adaptive_components=True)
        
        # Should adaptively select components
        n_comp = metadata['n_components']
        assert 64 <= n_comp <= 256  # In adaptive range
        
        assert X_denoised.shape == X.shape
    
    def test_explained_variance_threshold(self):
        """Test explained variance threshold in PCA denoising."""
        np.random.seed(42)
        # Create data where first few components explain most variance
        U = np.random.randn(100, 3)
        V = np.random.randn(3, 20)
        X = U @ V + 0.01 * np.random.randn(100, 20)
        
        X_denoised, metadata = pca_denoise(X, explained_variance_threshold=0.99, 
                                         adaptive_components=False)
        
        # Should select few components that explain most variance
        assert metadata['total_explained_variance'] >= 0.99


class TestMetricUtilities:
    """Test metric computation utilities."""
    
    def test_cosine_distances(self):
        """Test cosine distance computation."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        Y = np.random.randn(30, 10)
        
        # Test pairwise distances
        D = compute_cosine_distances(X, Y)
        
        assert D.shape == (50, 30)
        assert np.all(D >= 0)  # Cosine distances are non-negative
        assert np.all(D <= 2)  # Cosine distances are <= 2
    
    def test_cosine_distances_self(self):
        """Test cosine distances within same dataset."""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        
        D = compute_cosine_distances(X)
        
        assert D.shape == (20, 20)
        assert np.allclose(np.diag(D), 0, atol=1e-10)  # Diagonal should be zero
        assert np.allclose(D, D.T, atol=1e-10)  # Should be symmetric
    
    def test_cosine_distances_batch(self):
        """Test batched cosine distance computation."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        Y = np.random.randn(50, 10)
        
        # Compute with batching
        D_batch = compute_cosine_distances(X, Y, batch_size=10)
        
        # Compute without batching
        D_full = compute_cosine_distances(X, Y)
        
        # Should be approximately equal
        assert np.allclose(D_batch, D_full, atol=1e-10)
    
    def test_spearman_loss(self):
        """Test Spearman rank correlation loss."""
        np.random.seed(42)
        n = 20
        
        # Create two distance matrices
        D1 = np.random.rand(n, n)
        D1 = (D1 + D1.T) / 2  # Make symmetric
        np.fill_diagonal(D1, 0)
        
        # Perfect correlation (same matrix)
        loss_perfect = spearman_loss(D1, D1)
        assert np.isclose(loss_perfect, 0, atol=1e-10)
        
        # Random matrix (should have some correlation)
        D2 = np.random.rand(n, n)
        D2 = (D2 + D2.T) / 2
        np.fill_diagonal(D2, 0)
        
        loss_random = spearman_loss(D1, D2)
        assert 0 <= loss_random <= 2  # Loss should be in valid range
    
    def test_spearman_loss_sampling(self):
        """Test Spearman loss with sampling."""
        np.random.seed(42)
        n = 100
        D1 = np.random.rand(n, n)
        D1 = (D1 + D1.T) / 2
        np.fill_diagonal(D1, 0)
        
        D2 = D1 + 0.1 * np.random.rand(n, n)  # Similar matrix
        
        # With sampling
        loss_sampled = spearman_loss(D1, D2, sample_size=1000)
        
        # Without sampling
        loss_full = spearman_loss(D1, D2)
        
        # Should be approximately similar
        assert abs(loss_sampled - loss_full) < 0.2
    
    def test_distance_matrix_metrics(self):
        """Test different distance metrics."""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        
        # Test different metrics
        D_euclidean = compute_distance_matrix(X, metric='euclidean')
        D_cosine = compute_distance_matrix(X, metric='cosine')
        D_manhattan = compute_distance_matrix(X, metric='manhattan')
        
        # All should be square matrices
        assert D_euclidean.shape == (20, 20)
        assert D_cosine.shape == (20, 20)
        assert D_manhattan.shape == (20, 20)
        
        # All should have zero diagonal
        assert np.allclose(np.diag(D_euclidean), 0, atol=1e-10)
        assert np.allclose(np.diag(D_cosine), 0, atol=1e-10)
        assert np.allclose(np.diag(D_manhattan), 0, atol=1e-10)
        
        # All should be symmetric
        assert np.allclose(D_euclidean, D_euclidean.T, atol=1e-10)
        assert np.allclose(D_cosine, D_cosine.T, atol=1e-10)
        assert np.allclose(D_manhattan, D_manhattan.T, atol=1e-10)


class TestAdaptivePipeline:
    """Test adaptive preprocessing pipeline."""
    
    def test_basic_pipeline(self):
        """Test basic adaptive pipeline."""
        np.random.seed(42)
        # Create data that needs standardization
        X = np.random.randn(100, 50) * 5 + 10
        
        X_processed, metadata = adaptive_preprocessing_pipeline(X)
        
        # Should have processed the data
        assert X_processed.shape[0] == X.shape[0]  # Same number of samples
        
        # Check metadata
        assert metadata['adaptive_pipeline'] == True
        assert metadata['target_method'] == 'jll'
        assert 'pipeline_steps' in metadata
        assert 'data_stats' in metadata
    
    def test_high_dimensional_pipeline(self):
        """Test pipeline with high-dimensional data."""
        np.random.seed(42)
        # High-dimensional noisy data
        X = np.random.randn(100, 2000) * 3 + 5
        
        X_processed, metadata = adaptive_preprocessing_pipeline(
            X, max_components=200
        )
        
        # Should apply denoising for high-dimensional data
        steps = metadata.get('pipeline_steps', [])
        assert any('denoising' in step for step in steps)
        
        # May have reduced dimensionality
        assert X_processed.shape[0] == X.shape[0]
    
    def test_cosine_target_pipeline(self):
        """Test pipeline for cosine-based methods."""
        np.random.seed(42)
        X = np.random.randn(100, 50) * 2 + 3
        
        X_processed, metadata = adaptive_preprocessing_pipeline(
            X, target_method='umap'
        )
        
        # Should apply L2 normalization for cosine-based methods
        steps = metadata.get('pipeline_steps', [])
        assert 'l2_normalization' in steps
        
        # Check that rows are normalized
        norms = np.linalg.norm(X_processed, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)
    
    def test_pipeline_metadata(self):
        """Test comprehensive pipeline metadata."""
        np.random.seed(42)
        X = np.random.randn(100, 100) * 2
        
        X_processed, metadata = adaptive_preprocessing_pipeline(X)
        
        # Check required metadata fields
        required_fields = [
            'adaptive_pipeline', 'target_method', 'quality_threshold',
            'original_shape', 'final_shape', 'data_stats', 
            'pipeline_steps', 'steps'
        ]
        
        for field in required_fields:
            assert field in metadata
        
        # Check data stats
        assert 'mean_norm' in metadata['data_stats']
        assert 'std_norm' in metadata['data_stats']
        assert 'condition_number' in metadata['data_stats']


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_data(self):
        """Test behavior with empty data."""
        X = np.array([]).reshape(0, 5)
        
        # Should handle gracefully (may raise or return empty)
        try:
            X_norm, metadata = l2_normalize_rows(X)
            assert X_norm.shape == X.shape
        except (ValueError, IndexError):
            pass  # Acceptable to raise error for empty data
    
    def test_single_sample(self):
        """Test behavior with single sample."""
        X = np.random.randn(1, 10)
        
        X_norm, metadata = l2_normalize_rows(X)
        assert X_norm.shape == (1, 10)
        
        # Normalization should still work
        norm = np.linalg.norm(X_norm)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_constant_features(self):
        """Test behavior with constant features."""
        X = np.ones((100, 10))  # All constant
        X[:, 0] = np.random.randn(100)  # One variable feature
        
        X_std, metadata = standardize_features(X, method='zscore')
        
        # Should handle constant features gracefully
        assert X_std.shape == X.shape
        assert np.all(np.isfinite(X_std))
    
    def test_very_small_values(self):
        """Test numerical stability with very small values."""
        X = np.random.randn(100, 10) * 1e-10
        
        X_norm, metadata = l2_normalize_rows(X, epsilon=1e-12)
        
        # Should handle gracefully without overflow/underflow
        assert X_norm.shape == X.shape
        assert np.all(np.isfinite(X_norm))


if __name__ == '__main__':
    pytest.main([__file__])