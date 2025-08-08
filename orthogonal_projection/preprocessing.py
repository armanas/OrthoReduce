"""
preprocessing.py - Data preprocessing utilities for dimensionality reduction

This module provides efficient preprocessing functions for preparing data before
dimensionality reduction. It includes standardization, whitening, denoising,
and metric-specific utilities.
"""
from __future__ import annotations

import numpy as np
import logging
from typing import Dict, Optional, Tuple, Union, Literal
from numpy.typing import NDArray

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import pairwise_distances
from scipy import sparse
from scipy.stats import spearmanr

try:
    from .projection import jll_dimension, generate_orthogonal_basis, project_data
    from .exceptions import ValidationError, DimensionalityError
except ImportError:
    from projection import jll_dimension, generate_orthogonal_basis, project_data
    from exceptions import ValidationError, DimensionalityError

logger = logging.getLogger(__name__)

# Type aliases for better readability
ArrayLike = Union[NDArray[np.float64], sparse.spmatrix]
PreprocessingResult = Tuple[NDArray[np.float64], Dict[str, Union[float, int, str]]]


def standardize_features(
    X: ArrayLike,
    method: Literal["zscore", "unit_variance", "robust"] = "zscore",
    center: bool = True,
    scale: bool = True,
    copy: bool = True
) -> PreprocessingResult:
    """
    Standardize features to have zero mean and unit variance.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data to standardize
    method : {'zscore', 'unit_variance', 'robust'}
        Standardization method:
        - 'zscore': Standard z-score normalization (mean=0, std=1)
        - 'unit_variance': Scale to unit variance without centering
        - 'robust': Use median and MAD for outlier robustness
    center : bool, default=True
        Whether to center the data (subtract mean)
    scale : bool, default=True
        Whether to scale the data (divide by standard deviation)
    copy : bool, default=True
        Whether to make a copy of the input data
        
    Returns
    -------
    tuple
        (standardized_data, metadata_dict)
        
    Examples
    --------
    >>> X = np.random.randn(100, 50) * 5 + 10
    >>> X_std, metadata = standardize_features(X, method='zscore')
    >>> np.allclose(X_std.mean(axis=0), 0, atol=1e-10)
    True
    >>> np.allclose(X_std.std(axis=0), 1, atol=1e-10)
    True
    """
    X = np.asarray(X, dtype=np.float64)
    if copy:
        X = X.copy()
    
    n_samples, n_features = X.shape
    
    if n_samples < 2:
        logger.warning("Cannot standardize data with fewer than 2 samples")
        return X, {"method": method, "n_samples": n_samples, "n_features": n_features}
    
    metadata = {
        "method": method,
        "center": center,
        "scale": scale,
        "n_samples": n_samples,
        "n_features": n_features
    }
    
    if method == "zscore":
        if center:
            means = np.mean(X, axis=0)
            X -= means
            metadata["means"] = means
        
        if scale:
            stds = np.std(X, axis=0, ddof=1)
            # Avoid division by zero
            stds = np.where(stds < 1e-10, 1.0, stds)
            X /= stds
            metadata["stds"] = stds
            
    elif method == "unit_variance":
        if center:
            means = np.mean(X, axis=0)
            X -= means
            metadata["means"] = means
            
        if scale:
            vars_ = np.var(X, axis=0, ddof=1)
            # Avoid division by zero
            vars_ = np.where(vars_ < 1e-10, 1.0, vars_)
            X /= np.sqrt(vars_)
            metadata["vars"] = vars_
            
    elif method == "robust":
        if center:
            medians = np.median(X, axis=0)
            X -= medians
            metadata["medians"] = medians
            
        if scale:
            # Use Median Absolute Deviation (MAD)
            mads = np.median(np.abs(X - np.median(X, axis=0)), axis=0)
            # MAD can be zero for constant features
            mads = np.where(mads < 1e-10, 1.0, mads)
            X /= mads
            metadata["mads"] = mads
    else:
        raise ValidationError(f"Unknown standardization method: {method}")
    
    return X, metadata


def l2_normalize_rows(
    X: ArrayLike, 
    epsilon: float = 1e-10,
    copy: bool = True
) -> PreprocessingResult:
    """
    L2-normalize each row to unit length for cosine-based operations.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data to normalize
    epsilon : float, default=1e-10
        Small value to prevent division by zero
    copy : bool, default=True
        Whether to make a copy of the input data
        
    Returns
    -------
    tuple
        (normalized_data, metadata_dict)
        
    Examples
    --------
    >>> X = np.random.randn(100, 50) * 5 + 2
    >>> X_norm, metadata = l2_normalize_rows(X)
    >>> norms = np.linalg.norm(X_norm, axis=1)
    >>> np.allclose(norms, 1.0, atol=1e-10)
    True
    """
    X = np.asarray(X, dtype=np.float64)
    if copy:
        X = X.copy()
    
    n_samples, n_features = X.shape
    
    # Compute L2 norms for each row
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    
    # Count zero-norm rows
    zero_norm_mask = (norms.flatten() < epsilon)
    zero_norm_count = np.sum(zero_norm_mask)
    
    # Avoid division by zero - replace small norms with epsilon
    safe_norms = np.where(norms < epsilon, epsilon, norms)
    X /= safe_norms
    
    # For zero-norm rows, set them to small random unit vectors
    if zero_norm_count > 0:
        np.random.seed(42)  # For reproducibility
        X[zero_norm_mask] = np.random.randn(zero_norm_count, n_features) * epsilon
        # Renormalize these rows to unit length
        X[zero_norm_mask] /= np.linalg.norm(X[zero_norm_mask], axis=1, keepdims=True)
    
    metadata = {
        "method": "l2_normalize_rows",
        "epsilon": epsilon,
        "n_samples": n_samples,
        "n_features": n_features,
        "zero_norm_count": int(zero_norm_count),
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms))
    }
    
    return X, metadata


def whiten_data(
    X: ArrayLike,
    method: Literal["zca", "pca", "cholesky"] = "pca",
    regularization: float = 1e-6,
    n_components: Optional[int] = None,
    copy: bool = True
) -> PreprocessingResult:
    """
    Whiten data to have identity covariance matrix.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data to whiten
    method : {'zca', 'pca', 'cholesky'}
        Whitening method:
        - 'pca': PCA whitening (decorrelates and normalizes)
        - 'zca': ZCA whitening (preserves structure better)
        - 'cholesky': Cholesky-based whitening (fastest)
    regularization : float, default=1e-6
        Regularization parameter for numerical stability
    n_components : int, optional
        Number of components to keep (for PCA method only)
    copy : bool, default=True
        Whether to make a copy of the input data
        
    Returns
    -------
    tuple
        (whitened_data, metadata_dict)
    """
    X = np.asarray(X, dtype=np.float64)
    if copy:
        X = X.copy()
    
    n_samples, n_features = X.shape
    
    if n_samples < n_features:
        logger.warning(f"More features ({n_features}) than samples ({n_samples}). "
                      "Consider using regularization or dimensionality reduction first.")
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    metadata = {
        "method": method,
        "regularization": regularization,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components": n_components
    }
    
    if method == "pca":
        # PCA whitening
        if n_components is None:
            n_components = min(n_samples, n_features) - 1
            
        pca = PCA(n_components=n_components, whiten=True)
        X_whitened = pca.fit_transform(X_centered)
        
        metadata.update({
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "total_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
            "singular_values": pca.singular_values_,
            "components": pca.components_,
            "mean": pca.mean_
        })
        
    elif method == "zca":
        # ZCA (Zero-phase Component Analysis) whitening
        # Preserves the structure better than PCA whitening
        cov_matrix = np.cov(X_centered.T)
        
        # Add regularization for numerical stability
        cov_matrix += regularization * np.eye(n_features)
        
        # Eigen decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Avoid division by very small eigenvalues
        eigenvals = np.maximum(eigenvals, regularization)
        
        # ZCA transformation matrix
        sqrt_inv_eigenvals = 1.0 / np.sqrt(eigenvals)
        whitening_matrix = eigenvecs @ np.diag(sqrt_inv_eigenvals) @ eigenvecs.T
        
        X_whitened = X_centered @ whitening_matrix
        
        metadata.update({
            "eigenvals": eigenvals,
            "eigenvecs": eigenvecs,
            "whitening_matrix": whitening_matrix
        })
        
    elif method == "cholesky":
        # Cholesky whitening - fastest method
        cov_matrix = np.cov(X_centered.T)
        
        # Add regularization
        cov_matrix += regularization * np.eye(n_features)
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(cov_matrix)
            # Whitening transformation
            L_inv = np.linalg.inv(L)
            X_whitened = X_centered @ L_inv.T
            
            metadata.update({
                "cholesky_factor": L,
                "cholesky_inv": L_inv
            })
        except np.linalg.LinAlgError:
            # Fallback to eigendecomposition if Cholesky fails
            logger.warning("Cholesky decomposition failed, falling back to eigendecomposition")
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            eigenvals = np.maximum(eigenvals, regularization)
            sqrt_inv_eigenvals = 1.0 / np.sqrt(eigenvals)
            whitening_matrix = eigenvecs @ np.diag(sqrt_inv_eigenvals) @ eigenvecs.T
            X_whitened = X_centered @ whitening_matrix
            
            metadata.update({
                "eigenvals": eigenvals,
                "eigenvecs": eigenvecs,
                "whitening_matrix": whitening_matrix,
                "fallback_used": True
            })
    else:
        raise ValidationError(f"Unknown whitening method: {method}")
    
    return X_whitened, metadata


def pca_denoise(
    X: ArrayLike,
    n_components: Optional[int] = None,
    explained_variance_threshold: float = 0.95,
    adaptive_components: bool = True,
    copy: bool = True
) -> PreprocessingResult:
    """
    Denoise data using PCA by keeping top components.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data to denoise
    n_components : int, optional
        Number of components to keep. If None, uses explained_variance_threshold
    explained_variance_threshold : float, default=0.95
        Keep components that explain this fraction of variance
    adaptive_components : bool, default=True
        Adaptively select components between 100-300 based on data size
    copy : bool, default=True
        Whether to make a copy of the input data
        
    Returns
    -------
    tuple
        (denoised_data, metadata_dict)
    """
    X = np.asarray(X, dtype=np.float64)
    if copy:
        X = X.copy()
    
    n_samples, n_features = X.shape
    
    if n_components is None:
        if adaptive_components:
            # Adaptive selection: 100-300 components based on data characteristics
            n_components = min(
                max(100, n_features // 10),  # At least 100, or 10% of features
                min(300, n_features - 1, n_samples - 1)  # At most 300, or data constraints
            )
        else:
            # Use explained variance threshold
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = int(np.argmax(cumsum_var >= explained_variance_threshold)) + 1
            n_components = min(n_components, n_features - 1, n_samples - 1)
    
    # Ensure n_components is valid
    n_components = max(1, min(n_components, n_features - 1, n_samples - 1))
    
    # Apply PCA denoising
    pca = PCA(n_components=n_components)
    X_denoised = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_denoised)
    
    # Calculate noise reduction metrics
    noise_variance = np.mean((X - X_reconstructed) ** 2)
    signal_variance = np.mean(X_reconstructed ** 2)
    snr_improvement = signal_variance / (noise_variance + 1e-10)
    
    metadata = {
        "method": "pca_denoise",
        "n_components": n_components,
        "n_samples": n_samples,
        "n_features": n_features,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "total_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "noise_variance": float(noise_variance),
        "signal_variance": float(signal_variance),
        "snr_improvement": float(snr_improvement),
        "compression_ratio": float(n_features / n_components)
    }
    
    return X_reconstructed, metadata


def jl_denoise(
    X: ArrayLike,
    n_components: Optional[int] = None,
    epsilon: float = 0.1,
    adaptive_components: bool = True,
    method: str = "auto",
    copy: bool = True
) -> PreprocessingResult:
    """
    Denoise data using Johnson-Lindenstrauss random projection.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data to denoise
    n_components : int, optional
        Target dimension. If None, uses JL bound with epsilon
    epsilon : float, default=0.1
        JL distortion parameter for automatic dimension selection
    adaptive_components : bool, default=True
        Adaptively select components between 64-256 based on data size
    method : str, default="auto"
        JL projection method (passed to generate_orthogonal_basis)
    copy : bool, default=True
        Whether to make a copy of the input data
        
    Returns
    -------
    tuple
        (denoised_data, metadata_dict)
    """
    X = np.asarray(X, dtype=np.float64)
    if copy:
        X = X.copy()
    
    n_samples, n_features = X.shape
    
    if n_components is None:
        if adaptive_components:
            # Adaptive selection: 64-256 components based on data characteristics
            n_components = min(
                max(64, n_features // 20),  # At least 64, or 5% of features
                min(256, n_features - 1, jll_dimension(n_samples, epsilon))
            )
        else:
            # Use JL theoretical bound
            n_components = min(jll_dimension(n_samples, epsilon), n_features - 1)
    
    # Ensure n_components is valid
    n_components = max(1, min(n_components, n_features - 1))
    
    # Apply JL projection and back-projection for denoising
    try:
        # Handle 'auto' method
        if method == 'auto':
            method = 'gaussian'  # Default to gaussian for denoising
        
        basis = generate_orthogonal_basis(n_features, n_components, method=method)
        X_projected = project_data(X, basis)
        
        # Back-project to original dimension (pseudo-inverse for denoising)
        # basis is (n_features, n_components), so basis.T is (n_components, n_features)
        # X_projected is (n_samples, n_components)
        # We want to get back (n_samples, n_features)
        basis_pinv = np.linalg.pinv(basis)  # Shape: (n_components, n_features)
        X_denoised = X_projected @ basis_pinv  # (n_samples, n_components) @ (n_components, n_features) -> (n_samples, n_features)
        
        # Calculate denoising metrics
        reconstruction_error = np.mean((X - X_denoised) ** 2)
        original_variance = np.var(X)
        denoised_variance = np.var(X_denoised)
        
        metadata = {
            "method": "jl_denoise",
            "n_components": n_components,
            "n_samples": n_samples,
            "n_features": n_features,
            "epsilon": epsilon,
            "jl_method": method,
            "reconstruction_error": float(reconstruction_error),
            "original_variance": float(original_variance),
            "denoised_variance": float(denoised_variance),
            "variance_retained": float(denoised_variance / (original_variance + 1e-10)),
            "compression_ratio": float(n_features / n_components)
        }
        
        return X_denoised, metadata
        
    except Exception as e:
        logger.error(f"JL denoising failed: {e}")
        # Fallback to original data
        metadata = {
            "method": "jl_denoise",
            "n_components": n_components,
            "error": str(e),
            "fallback_used": True
        }
        return X, metadata


def compute_cosine_distances(
    X: ArrayLike,
    Y: Optional[ArrayLike] = None,
    batch_size: Optional[int] = None
) -> NDArray[np.float64]:
    """
    Compute cosine distances between rows of X and Y.
    
    Parameters
    ----------
    X : array-like, shape (n_samples_X, n_features)
        First set of samples
    Y : array-like, shape (n_samples_Y, n_features), optional
        Second set of samples. If None, computes distances within X
    batch_size : int, optional
        Process in batches for memory efficiency
        
    Returns
    -------
    ndarray, shape (n_samples_X, n_samples_Y)
        Cosine distance matrix
    """
    X = np.asarray(X, dtype=np.float64)
    
    # L2 normalize X
    X_norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norms = np.where(X_norms < 1e-10, 1e-10, X_norms)
    X_normalized = X / X_norms
    
    if Y is None:
        Y_normalized = X_normalized
    else:
        Y = np.asarray(Y, dtype=np.float64)
        Y_norms = np.linalg.norm(Y, axis=1, keepdims=True)
        Y_norms = np.where(Y_norms < 1e-10, 1e-10, Y_norms)
        Y_normalized = Y / Y_norms
    
    # Cosine similarity = dot product of normalized vectors
    # Cosine distance = 1 - cosine similarity
    if batch_size is None or X.shape[0] * Y_normalized.shape[0] <= batch_size ** 2:
        # Compute all at once
        cosine_sim = X_normalized @ Y_normalized.T
        cosine_dist = 1.0 - cosine_sim
    else:
        # Batch processing for large matrices
        n_batches_x = (X.shape[0] + batch_size - 1) // batch_size
        cosine_dist = np.zeros((X.shape[0], Y_normalized.shape[0]))
        
        for i in range(n_batches_x):
            start_x = i * batch_size
            end_x = min((i + 1) * batch_size, X.shape[0])
            
            batch_sim = X_normalized[start_x:end_x] @ Y_normalized.T
            cosine_dist[start_x:end_x] = 1.0 - batch_sim
    
    # Ensure distances are non-negative (numerical stability)
    cosine_dist = np.maximum(cosine_dist, 0.0)
    
    return cosine_dist


def spearman_loss(
    distances_original: NDArray[np.float64],
    distances_projected: NDArray[np.float64],
    sample_size: Optional[int] = None
) -> float:
    """
    Compute Spearman rank correlation loss between distance matrices.
    
    Parameters
    ----------
    distances_original : array-like, shape (n, n)
        Original distance matrix
    distances_projected : array-like, shape (n, n)
        Projected distance matrix
    sample_size : int, optional
        Sample this many pairs for efficiency
        
    Returns
    -------
    float
        Spearman rank correlation loss (1 - correlation)
    """
    distances_original = np.asarray(distances_original)
    distances_projected = np.asarray(distances_projected)
    
    # Extract upper triangular values (no diagonal)
    triu_idx = np.triu_indices(distances_original.shape[0], k=1)
    orig_vals = distances_original[triu_idx]
    proj_vals = distances_projected[triu_idx]
    
    # Sample for efficiency if needed
    if sample_size is not None and len(orig_vals) > sample_size:
        np.random.seed(42)  # Reproducible sampling
        idx = np.random.choice(len(orig_vals), sample_size, replace=False)
        orig_vals = orig_vals[idx]
        proj_vals = proj_vals[idx]
    
    try:
        # Compute Spearman correlation
        corr, _ = spearmanr(orig_vals, proj_vals)
        
        # Handle NaN/inf values
        if not np.isfinite(corr):
            return 1.0  # Maximum loss
        
        # Return loss (1 - correlation)
        return 1.0 - corr
        
    except Exception as e:
        logger.error(f"Error computing Spearman loss: {e}")
        return 1.0  # Maximum loss on error


def compute_distance_matrix(
    X: ArrayLike,
    metric: Literal["euclidean", "cosine", "manhattan", "chebyshev"] = "euclidean",
    **kwargs
) -> NDArray[np.float64]:
    """
    Compute distance matrix with different metrics.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data
    metric : {'euclidean', 'cosine', 'manhattan', 'chebyshev'}
        Distance metric to use
    **kwargs
        Additional arguments passed to the distance function
        
    Returns
    -------
    ndarray, shape (n_samples, n_samples)
        Distance matrix
    """
    X = np.asarray(X, dtype=np.float64)
    
    if metric == "cosine":
        return compute_cosine_distances(X, **kwargs)
    else:
        return pairwise_distances(X, metric=metric, **kwargs)


def adaptive_preprocessing_pipeline(
    X: ArrayLike,
    target_method: str = "jll",
    quality_threshold: float = 0.9,
    max_components: int = 300,
    copy: bool = True
) -> PreprocessingResult:
    """
    Adaptive preprocessing pipeline that selects optimal preprocessing steps.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data
    target_method : str, default="jll"
        Target dimensionality reduction method ("jll", "pca", "umap")
    quality_threshold : float, default=0.9
        Quality threshold for preprocessing decisions
    max_components : int, default=300
        Maximum number of components to keep in preprocessing
    copy : bool, default=True
        Whether to make a copy of the input data
        
    Returns
    -------
    tuple
        (preprocessed_data, metadata_dict)
    """
    X = np.asarray(X, dtype=np.float64)
    if copy:
        X = X.copy()
    
    n_samples, n_features = X.shape
    
    pipeline_steps = []
    metadata = {
        "adaptive_pipeline": True,
        "target_method": target_method,
        "quality_threshold": quality_threshold,
        "original_shape": (n_samples, n_features),
        "steps": []
    }
    
    # Step 1: Handle data characteristics
    data_stats = {
        "mean_norm": float(np.mean(np.linalg.norm(X, axis=1))),
        "std_norm": float(np.std(np.linalg.norm(X, axis=1))),
        "condition_number": float(np.linalg.cond(X.T @ X + 1e-10 * np.eye(n_features)))
    }
    metadata["data_stats"] = data_stats
    
    # Step 2: Standardization (if data is not already standardized)
    if np.abs(np.mean(X)) > 0.1 or np.abs(np.std(X) - 1.0) > 0.2:
        X, std_meta = standardize_features(X, method="zscore", copy=False)
        pipeline_steps.append("standardization")
        metadata["steps"].append({"standardization": std_meta})
    
    # Step 3: Denoising (if high-dimensional and noisy)
    if n_features > 1000 and data_stats["condition_number"] > 100:
        # Use PCA denoising for very high-dimensional data
        if n_features > 5000:
            X, denoise_meta = pca_denoise(X, n_components=min(max_components, n_features // 10), copy=False)
            pipeline_steps.append("pca_denoising")
        else:
            X, denoise_meta = jl_denoise(X, n_components=min(max_components, n_features // 5), copy=False)
            pipeline_steps.append("jl_denoising")
        
        metadata["steps"].append({pipeline_steps[-1]: denoise_meta})
    
    # Step 4: Normalization (for cosine-based methods)
    if target_method in ["umap"] or "cosine" in target_method.lower():
        X, norm_meta = l2_normalize_rows(X, copy=False)
        pipeline_steps.append("l2_normalization")
        metadata["steps"].append({"l2_normalization": norm_meta})
    
    metadata.update({
        "pipeline_steps": pipeline_steps,
        "final_shape": X.shape,
        "compression_ratio": float(n_features / X.shape[1]) if X.shape[1] != n_features else 1.0
    })
    
    return X, metadata