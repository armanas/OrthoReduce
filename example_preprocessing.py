#!/usr/bin/env python3
"""
Example demonstrating the new preprocessing utilities in OrthoReduce.

This script shows how to use the various preprocessing functions for 
preparing data before dimensionality reduction.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Import the new preprocessing functions
from orthogonal_projection.preprocessing import (
    standardize_features,
    l2_normalize_rows,
    whiten_data,
    pca_denoise,
    jl_denoise,
    compute_cosine_distances,
    spearman_loss,
    adaptive_preprocessing_pipeline
)

# Also import dimensionality reduction functions
from orthogonal_projection import (
    run_jll_simple,
    run_pca_simple,
    evaluate_projection
)


def demonstrate_standardization():
    """Demonstrate different standardization methods."""
    print("=== Standardization Examples ===")
    
    # Create test data with different scales
    np.random.seed(42)
    X = np.random.randn(100, 5)
    X[:, 0] *= 10  # Large scale
    X[:, 1] += 50  # Large offset
    X[:, 2] *= 0.1 # Small scale
    
    print(f"Original data stats:")
    print(f"  Mean: {X.mean(axis=0)}")
    print(f"  Std:  {X.std(axis=0)}")
    
    # Z-score standardization
    X_zscore, meta = standardize_features(X, method='zscore')
    print(f"\nAfter Z-score standardization:")
    print(f"  Mean: {X_zscore.mean(axis=0)}")
    print(f"  Std:  {X_zscore.std(axis=0)}")
    
    # Robust standardization
    X_robust, meta = standardize_features(X, method='robust')
    print(f"\nAfter robust standardization:")
    print(f"  Median: {np.median(X_robust, axis=0)}")
    print(f"  MAD:    {np.median(np.abs(X_robust - np.median(X_robust, axis=0)), axis=0)}")


def demonstrate_normalization():
    """Demonstrate L2 row normalization."""
    print("\n=== L2 Row Normalization ===")
    
    # Create data with different row magnitudes
    np.random.seed(42)
    X = np.random.randn(50, 10)
    X[0] *= 10  # Large magnitude row
    X[1] *= 0.1 # Small magnitude row
    X[2] = 0    # Zero row
    
    print(f"Original row norms (first 5): {np.linalg.norm(X[:5], axis=1)}")
    
    # L2 normalize rows
    X_norm, meta = l2_normalize_rows(X)
    normalized_norms = np.linalg.norm(X_norm, axis=1)
    
    print(f"Normalized row norms (first 5): {normalized_norms[:5]}")
    print(f"Zero-norm rows handled: {meta['zero_norm_count']}")


def demonstrate_whitening():
    """Demonstrate data whitening."""
    print("\n=== Data Whitening ===")
    
    # Create correlated data
    np.random.seed(42)
    X = np.random.randn(200, 5)
    # Add correlations
    correlation_matrix = np.array([
        [1.0, 0.8, 0.3, 0.1, 0.0],
        [0.8, 1.0, 0.5, 0.2, 0.1],
        [0.3, 0.5, 1.0, 0.4, 0.2],
        [0.1, 0.2, 0.4, 1.0, 0.3],
        [0.0, 0.1, 0.2, 0.3, 1.0]
    ])
    
    # Generate correlated data
    L = np.linalg.cholesky(correlation_matrix)
    X_corr = X @ L.T
    
    print(f"Original covariance matrix:")
    orig_cov = np.cov(X_corr.T)
    print(f"  Max off-diagonal: {np.max(np.abs(orig_cov - np.diag(np.diag(orig_cov)))):.3f}")
    
    # PCA whitening
    X_white, meta = whiten_data(X_corr, method='pca')
    white_cov = np.cov(X_white.T)
    print(f"\nAfter PCA whitening:")
    print(f"  Max off-diagonal: {np.max(np.abs(white_cov - np.diag(np.diag(white_cov)))):.3f}")
    print(f"  Explained variance: {meta['total_explained_variance']:.3f}")


def demonstrate_denoising():
    """Demonstrate denoising capabilities."""
    print("\n=== Denoising Examples ===")
    
    # Create low-rank data with noise
    np.random.seed(42)
    n_samples, n_features = 100, 50
    true_rank = 5
    
    # Low-rank signal
    U = np.random.randn(n_samples, true_rank)
    V = np.random.randn(true_rank, n_features)
    X_clean = U @ V
    
    # Add noise
    noise_level = 0.5
    X_noisy = X_clean + noise_level * np.random.randn(n_samples, n_features)
    
    print(f"Data shape: {X_noisy.shape}")
    print(f"True rank: {true_rank}")
    print(f"Noise level: {noise_level}")
    
    # PCA denoising
    X_pca_denoised, meta_pca = pca_denoise(X_noisy, n_components=true_rank)
    pca_error = np.mean((X_clean - X_pca_denoised)**2)
    print(f"\nPCA denoising (k={true_rank}):")
    print(f"  Reconstruction error: {pca_error:.4f}")
    print(f"  SNR improvement: {meta_pca['snr_improvement']:.2f}")
    
    # JL denoising
    X_jl_denoised, meta_jl = jl_denoise(X_noisy, n_components=true_rank*2)
    jl_error = np.mean((X_clean - X_jl_denoised)**2)
    print(f"\nJL denoising (k={true_rank*2}):")
    print(f"  Reconstruction error: {jl_error:.4f}")
    if 'variance_retained' in meta_jl:
        print(f"  Variance retained: {meta_jl['variance_retained']:.3f}")
    else:
        print(f"  Fallback used: {meta_jl.get('fallback_used', False)}")
        if 'error' in meta_jl:
            print(f"  Error: {meta_jl['error']}")


def demonstrate_metrics():
    """Demonstrate metric utilities."""
    print("\n=== Metric Utilities ===")
    
    # Create two datasets
    np.random.seed(42)
    X = np.random.randn(20, 10)
    Y = np.random.randn(15, 10)
    
    # Compute cosine distances
    D_cosine = compute_cosine_distances(X, Y)
    print(f"Cosine distances shape: {D_cosine.shape}")
    print(f"Cosine distance range: [{D_cosine.min():.3f}, {D_cosine.max():.3f}]")
    
    # Self-distances
    D_self = compute_cosine_distances(X)
    print(f"Self cosine distances diagonal: {np.diag(D_self)[:5]}")
    
    # Spearman loss between distance matrices
    from sklearn.metrics import pairwise_distances
    D1 = pairwise_distances(X[:10], metric='euclidean')
    D2 = pairwise_distances(X[:10], metric='cosine')
    
    loss = spearman_loss(D1, D2)
    print(f"Spearman loss between Euclidean and cosine distances: {loss:.3f}")


def demonstrate_adaptive_pipeline():
    """Demonstrate adaptive preprocessing pipeline."""
    print("\n=== Adaptive Pipeline ===")
    
    # Create challenging data
    np.random.seed(42)
    n_samples, n_features = 200, 500
    
    # High-dimensional, non-standardized, noisy data
    X = np.random.randn(n_samples, n_features) * 5 + 10
    X += 0.5 * np.random.randn(n_samples, n_features)  # Additional noise
    
    print(f"Original data shape: {X.shape}")
    print(f"Original data mean: {X.mean():.3f}, std: {X.std():.3f}")
    
    # Apply adaptive pipeline for JLL
    X_processed, meta = adaptive_preprocessing_pipeline(X, target_method='jll')
    
    print(f"\nProcessed data shape: {X_processed.shape}")
    print(f"Pipeline steps applied: {meta['pipeline_steps']}")
    print(f"Compression ratio: {meta['compression_ratio']:.2f}")
    
    # Apply adaptive pipeline for UMAP (cosine-based)
    X_processed_umap, meta_umap = adaptive_preprocessing_pipeline(X, target_method='umap')
    
    print(f"\nUMAP preprocessing steps: {meta_umap['pipeline_steps']}")
    # Check if L2 normalized
    if 'l2_normalization' in meta_umap['pipeline_steps']:
        norms = np.linalg.norm(X_processed_umap, axis=1)
        print(f"Row norms after UMAP preprocessing: mean={norms.mean():.6f}, std={norms.std():.6f}")


def demonstrate_integration_with_dr():
    """Demonstrate integration with dimensionality reduction."""
    print("\n=== Integration with Dimensionality Reduction ===")
    
    # Create test data
    np.random.seed(42)
    X, _ = make_blobs(n_samples=200, centers=5, n_features=100, 
                      center_box=(-10.0, 10.0), cluster_std=2.0)
    
    print(f"Original data shape: {X.shape}")
    
    # Without preprocessing
    Y_raw = run_jll_simple(X, k=20)
    metrics_raw = evaluate_projection(X, Y_raw, sample_size=1000)
    
    print(f"\nWithout preprocessing:")
    print(f"  Mean distortion: {metrics_raw['mean_distortion']:.4f}")
    print(f"  Rank correlation: {metrics_raw['rank_correlation']:.4f}")
    
    # With preprocessing
    X_preprocessed, _ = adaptive_preprocessing_pipeline(X, target_method='jll')
    Y_preprocessed = run_jll_simple(X_preprocessed, k=20)
    metrics_preprocessed = evaluate_projection(X_preprocessed, Y_preprocessed, sample_size=1000)
    
    print(f"\nWith preprocessing:")
    print(f"  Mean distortion: {metrics_preprocessed['mean_distortion']:.4f}")
    print(f"  Rank correlation: {metrics_preprocessed['rank_correlation']:.4f}")
    
    improvement_distortion = (metrics_raw['mean_distortion'] - metrics_preprocessed['mean_distortion']) / metrics_raw['mean_distortion']
    improvement_correlation = (metrics_preprocessed['rank_correlation'] - metrics_raw['rank_correlation']) / abs(metrics_raw['rank_correlation'])
    
    print(f"\nImprovement:")
    print(f"  Distortion reduction: {improvement_distortion*100:.1f}%")
    print(f"  Correlation improvement: {improvement_correlation*100:.1f}%")


if __name__ == "__main__":
    print("OrthoReduce Preprocessing Utilities Demo")
    print("=" * 50)
    
    try:
        demonstrate_standardization()
        demonstrate_normalization()
        demonstrate_whitening()
        demonstrate_denoising()
        demonstrate_metrics()
        demonstrate_adaptive_pipeline()
        demonstrate_integration_with_dr()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("The preprocessing utilities are ready for use.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()