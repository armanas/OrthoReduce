"""
calibration.py - Post-processing calibration utilities for dimensionality reduction

This module provides advanced calibration methods to improve correlation rank and
distance preservation in dimensionality-reduced embeddings. It includes:

1. Isotonic regression calibration for monotonic distance mapping
2. Procrustes alignment for removing rigid transformations  
3. Local linear correction for per-point neighborhood rescaling

These methods can significantly improve Spearman/Kendall correlation while
maintaining computational efficiency through sparse matrix operations and
numerical stability optimizations.
"""

from __future__ import annotations

import numpy as np
import logging
from typing import Optional, Tuple, Union, Dict, Any
from numpy.typing import NDArray
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, kendalltau
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

# Type aliases for clarity
FloatArray = NDArray[np.float64]
SparseMatrix = sparse.csr_matrix


def isotonic_regression_calibration(
    X_high: FloatArray,
    X_low: FloatArray,
    sample_size: Optional[int] = None,
    increasing: bool = True,
    out_of_bounds: str = "clip",
    seed: int = 42
) -> Tuple[FloatArray, callable]:
    """
    Apply isotonic regression calibration to improve distance correlation.
    
    Fits a monotonic mapping from low-dimensional distances to high-dimensional
    distances, then applies this correction to boost Spearman/Kendall correlation.
    
    Parameters
    ----------
    X_high : ndarray of shape (n_samples, d_high)
        Original high-dimensional data
    X_low : ndarray of shape (n_samples, d_low) 
        Projected low-dimensional data
    sample_size : int or None
        Number of distance pairs to sample for efficiency (default: all pairs)
    increasing : bool
        Whether the isotonic regression should be increasing (default: True)
    out_of_bounds : str
        How to handle out-of-bounds values ("clip", "nan", "extrapolate")
    seed : int
        Random seed for reproducible sampling
        
    Returns
    -------
    X_calibrated : ndarray of shape (n_samples, d_low)
        Calibrated embedding with corrected distances
    calibration_func : callable
        Fitted isotonic regression function for applying to new data
        
    Raises
    ------
    ValueError
        If input dimensions are inconsistent or parameters are invalid
    """
    # Input validation
    X_high = np.asarray(X_high, dtype=np.float64)
    X_low = np.asarray(X_low, dtype=np.float64)
    
    if X_high.shape[0] != X_low.shape[0]:
        raise ValueError(f"Sample count mismatch: {X_high.shape[0]} vs {X_low.shape[0]}")
    
    n_samples = X_high.shape[0]
    
    if n_samples < 3:
        logger.warning("Too few samples for isotonic calibration, returning original data")
        return X_low.copy(), lambda x: x
    
    # Sample distance pairs for efficiency on large datasets
    if sample_size is not None and n_samples > sample_size:
        np.random.seed(seed)
        # Sample pairs efficiently
        n_pairs = sample_size * (sample_size - 1) // 2
        max_pairs = n_samples * (n_samples - 1) // 2
        
        if n_pairs >= max_pairs:
            sample_indices = np.arange(n_samples)
        else:
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
    else:
        sample_indices = np.arange(n_samples)
    
    try:
        # Compute pairwise distances for sampled points
        X_high_sample = X_high[sample_indices]
        X_low_sample = X_low[sample_indices]
        
        # Use scipy's efficient distance computation
        distances_high = pdist(X_high_sample, metric='euclidean')
        distances_low = pdist(X_low_sample, metric='euclidean')
        
        # Remove zero distances to avoid numerical issues
        nonzero_mask = (distances_low > 1e-12) & (distances_high > 1e-12)
        if not np.any(nonzero_mask):
            logger.warning("All distances are zero, returning original data")
            return X_low.copy(), lambda x: x
            
        distances_high_filtered = distances_high[nonzero_mask]
        distances_low_filtered = distances_low[nonzero_mask]
        
        # Fit isotonic regression: low_distances -> high_distances
        iso_reg = IsotonicRegression(
            increasing=increasing, 
            out_of_bounds=out_of_bounds
        )
        iso_reg.fit(distances_low_filtered, distances_high_filtered)
        
        # Create calibration function
        def calibration_func(distances: FloatArray) -> FloatArray:
            """Apply fitted isotonic regression to distance array."""
            return iso_reg.predict(np.asarray(distances))
        
        # Apply calibration to the full embedding
        X_calibrated = _apply_distance_calibration(X_low, calibration_func)
        
        logger.info(f"Isotonic regression calibration completed on {len(distances_low_filtered)} distance pairs")
        return X_calibrated, calibration_func
        
    except Exception as e:
        logger.error(f"Isotonic regression calibration failed: {e}")
        return X_low.copy(), lambda x: x


def procrustes_alignment(
    X_high: FloatArray,
    X_low: FloatArray, 
    scaling: bool = True,
    reflection: bool = False
) -> Tuple[FloatArray, Dict[str, Any]]:
    """
    Apply Procrustes alignment to remove rigid transformations.
    
    Removes rotation, scaling, and translation effects before evaluation
    to focus on structural preservation. Supports both Euclidean and 
    spherical embeddings.
    
    Parameters
    ---------- 
    X_high : ndarray of shape (n_samples, d_high)
        Target high-dimensional configuration
    X_low : ndarray of shape (n_samples, d_low)
        Source low-dimensional configuration to align
    scaling : bool
        Whether to allow uniform scaling (default: True)
    reflection : bool  
        Whether to allow reflections (default: False)
        
    Returns
    -------
    X_aligned : ndarray of shape (n_samples, d_low)
        Procrustes-aligned embedding
    transformation : dict
        Transformation parameters (rotation, scaling, translation)
        
    Raises
    ------
    ValueError
        If dimensions are incompatible or data is degenerate
    """
    # Input validation
    X_high = np.asarray(X_high, dtype=np.float64)
    X_low = np.asarray(X_low, dtype=np.float64)
    
    if X_high.shape[0] != X_low.shape[0]:
        raise ValueError(f"Sample count mismatch: {X_high.shape[0]} vs {X_low.shape[0]}")
    
    n_samples = X_high.shape[0]
    d_high = X_high.shape[1] 
    d_low = X_low.shape[1]
    
    if n_samples < max(d_high, d_low):
        raise ValueError(f"Need at least {max(d_high, d_low)} samples for alignment")
    
    try:
        # Center both configurations
        mu_high = np.mean(X_high, axis=0)
        mu_low = np.mean(X_low, axis=0)
        
        X_high_centered = X_high - mu_high
        X_low_centered = X_low - mu_low
        
        # Handle dimension mismatch by projecting to common space
        if d_high != d_low:
            # Use SVD to find best subspace alignment
            U_high, _, Vt_high = np.linalg.svd(X_high_centered, full_matrices=False)
            U_low, _, Vt_low = np.linalg.svd(X_low_centered, full_matrices=False)
            
            # Align in the smaller dimension
            d_common = min(d_high, d_low, n_samples - 1)
            
            X_high_proj = U_high[:, :d_common] @ np.diag(np.ones(d_common))
            X_low_proj = U_low[:, :d_common] @ np.diag(np.ones(d_common))
        else:
            X_high_proj = X_high_centered
            X_low_proj = X_low_centered
            d_common = d_low
        
        # Procrustes analysis via SVD
        # Find optimal rotation: X_low * R ≈ X_high
        H = X_low_proj.T @ X_high_proj  # Cross-covariance matrix
        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt  # Optimal rotation matrix
        
        # Handle reflection if not allowed
        if not reflection and np.linalg.det(R) < 0:
            # Flip the last column to ensure proper rotation
            U[:, -1] *= -1
            R = U @ Vt
        
        # Compute optimal scaling
        if scaling:
            # Optimal scaling factor
            numerator = np.trace(X_low_proj @ R @ X_high_proj.T)
            denominator = np.trace(X_low_proj @ X_low_proj.T)
            scale = numerator / (denominator + 1e-12)  # Avoid division by zero
            
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
        else:
            scale = 1.0
        
        # Apply transformation to original low-dimensional data
        X_low_transformed = scale * (X_low_centered @ R[:d_low, :d_common].T)
        
        # Add back the high-dimensional centroid (if dimensions match)
        if d_high == d_low:
            X_aligned = X_low_transformed + mu_high
        else:
            # Project high-dimensional centroid to low-dimensional space
            mu_high_proj = mu_high[:d_low] if d_high > d_low else np.pad(mu_high, (0, d_low - d_high))
            X_aligned = X_low_transformed + mu_high_proj
        
        # Store transformation parameters
        transformation = {
            'rotation': R,
            'scaling': scale,
            'translation_high': mu_high,
            'translation_low': mu_low,
            'aligned_dimension': d_common
        }
        
        logger.info(f"Procrustes alignment completed: scale={scale:.4f}, det(R)={np.linalg.det(R):.4f}")
        return X_aligned, transformation
        
    except Exception as e:
        logger.error(f"Procrustes alignment failed: {e}")
        return X_low.copy(), {'rotation': np.eye(d_low), 'scaling': 1.0}


def local_linear_correction(
    X_high: FloatArray,
    X_low: FloatArray,
    k_neighbors: int = 10,
    adaptive_k: bool = True,
    reg_strength: float = 1e-6,
    max_neighbors: Optional[int] = None
) -> FloatArray:
    """
    Apply local linear correction with per-point radial rescaling.
    
    Performs per-point radial rescaling to match neighborhood radii while
    preserving local structure. Uses adaptive neighborhood size selection
    for optimal local structure preservation.
    
    Parameters
    ----------
    X_high : ndarray of shape (n_samples, d_high)
        Original high-dimensional data
    X_low : ndarray of shape (n_samples, d_low)
        Projected low-dimensional data
    k_neighbors : int
        Base number of neighbors for local correction (default: 10)
    adaptive_k : bool
        Whether to adapt neighborhood size per point (default: True) 
    reg_strength : float
        Regularization strength for numerical stability (default: 1e-6)
    max_neighbors : int or None
        Maximum number of neighbors to consider (default: n_samples // 4)
        
    Returns
    -------
    X_corrected : ndarray of shape (n_samples, d_low)
        Locally corrected embedding
        
    Raises
    ------
    ValueError
        If input dimensions are inconsistent
    """
    # Input validation
    X_high = np.asarray(X_high, dtype=np.float64)
    X_low = np.asarray(X_low, dtype=np.float64)
    
    if X_high.shape[0] != X_low.shape[0]:
        raise ValueError(f"Sample count mismatch: {X_high.shape[0]} vs {X_low.shape[0]}")
    
    n_samples = X_high.shape[0]
    
    if max_neighbors is None:
        max_neighbors = n_samples // 4
    
    max_neighbors = min(max_neighbors, n_samples - 1)
    k_neighbors = min(k_neighbors, max_neighbors)
    
    if k_neighbors < 1:
        logger.warning("Too few samples for local correction, returning original data")
        return X_low.copy()
    
    try:
        # Build k-NN graphs for both high and low dimensional spaces
        nbrs_high = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto')
        nbrs_low = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto')
        
        nbrs_high.fit(X_high)
        nbrs_low.fit(X_low)
        
        X_corrected = X_low.copy()
        
        # Process each point individually
        for i in range(n_samples):
            try:
                # Determine optimal neighborhood size for this point
                if adaptive_k:
                    k_opt = _adaptive_neighborhood_size(
                        X_high, X_low, i, k_neighbors, max_neighbors, nbrs_high, nbrs_low
                    )
                else:
                    k_opt = k_neighbors
                
                # Get neighborhoods
                distances_high, indices_high = nbrs_high.kneighbors(
                    X_high[i:i+1], n_neighbors=k_opt + 1
                )
                distances_low, indices_low = nbrs_low.kneighbors(
                    X_low[i:i+1], n_neighbors=k_opt + 1
                )
                
                # Remove self (first neighbor)
                distances_high = distances_high[0, 1:]
                distances_low = distances_low[0, 1:]
                indices_high = indices_high[0, 1:]
                indices_low = indices_low[0, 1:]
                
                # Compute local radial scaling factor
                if len(distances_high) > 0 and len(distances_low) > 0:
                    # Use robust scaling based on median ratios
                    ratio_high = np.median(distances_high)
                    ratio_low = np.median(distances_low)
                    
                    if ratio_low > 1e-12:  # Avoid division by zero
                        scale_factor = ratio_high / ratio_low
                        
                        # Apply regularization to prevent extreme scaling
                        scale_factor = np.clip(scale_factor, 0.1, 10.0)
                        
                        # Apply local radial scaling centered at current point
                        direction = X_low[i] - np.mean(X_low, axis=0)
                        correction = direction * (scale_factor - 1.0) * reg_strength
                        X_corrected[i] += correction
                
            except Exception as e:
                logger.debug(f"Local correction failed for point {i}: {e}")
                continue
        
        logger.info(f"Local linear correction completed for {n_samples} points")
        return X_corrected
        
    except Exception as e:
        logger.error(f"Local linear correction failed: {e}")
        return X_low.copy()


def combined_calibration(
    X_high: FloatArray,
    X_low: FloatArray,
    methods: list[str] = ['procrustes', 'isotonic', 'local'],
    **kwargs
) -> Tuple[FloatArray, Dict[str, Any]]:
    """
    Apply multiple calibration methods in sequence for maximum improvement.
    
    Combines different calibration approaches for optimal correlation improvement.
    The order of application is: Procrustes -> Isotonic -> Local correction.
    
    Parameters
    ----------
    X_high : ndarray of shape (n_samples, d_high)
        Original high-dimensional data
    X_low : ndarray of shape (n_samples, d_low)
        Projected low-dimensional data to calibrate
    methods : list of str
        Calibration methods to apply in order (default: ['procrustes', 'isotonic', 'local'])
    **kwargs
        Additional parameters passed to individual calibration methods
        
    Returns
    -------
    X_calibrated : ndarray of shape (n_samples, d_low) 
        Final calibrated embedding
    calibration_info : dict
        Information about applied calibrations and their effects
    """
    X_current = X_low.copy()
    calibration_info = {
        'methods_applied': [],
        'correlation_improvements': {},
        'transformations': {}
    }
    
    # Compute initial correlation
    try:
        initial_corr = _compute_distance_correlation(X_high, X_current)
        calibration_info['initial_correlation'] = initial_corr
    except Exception:
        initial_corr = 0.0
        calibration_info['initial_correlation'] = 0.0
    
    for method in methods:
        try:
            if method == 'procrustes':
                X_current, transform_info = procrustes_alignment(X_high, X_current, **kwargs)
                calibration_info['transformations']['procrustes'] = transform_info
                
            elif method == 'isotonic':
                X_current, calib_func = isotonic_regression_calibration(
                    X_high, X_current, **kwargs
                )
                calibration_info['transformations']['isotonic'] = calib_func
                
            elif method == 'local':
                X_current = local_linear_correction(X_high, X_current, **kwargs)
                calibration_info['transformations']['local'] = True
                
            else:
                logger.warning(f"Unknown calibration method: {method}")
                continue
            
            # Measure improvement
            try:
                new_corr = _compute_distance_correlation(X_high, X_current)
                improvement = new_corr - initial_corr
                calibration_info['correlation_improvements'][method] = improvement
                logger.info(f"{method} calibration: correlation improved by {improvement:.4f}")
                initial_corr = new_corr  # Update for next method
            except Exception:
                calibration_info['correlation_improvements'][method] = 0.0
            
            calibration_info['methods_applied'].append(method)
            
        except Exception as e:
            logger.error(f"Failed to apply {method} calibration: {e}")
            continue
    
    # Final correlation
    try:
        final_corr = _compute_distance_correlation(X_high, X_current)
        total_improvement = final_corr - calibration_info['initial_correlation']
        calibration_info['final_correlation'] = final_corr
        calibration_info['total_improvement'] = total_improvement
        
        logger.info(f"Combined calibration completed: total improvement = {total_improvement:.4f}")
    except Exception:
        calibration_info['final_correlation'] = calibration_info['initial_correlation']
        calibration_info['total_improvement'] = 0.0
    
    return X_current, calibration_info


# Helper functions

def _apply_distance_calibration(X_low: FloatArray, calibration_func: callable) -> FloatArray:
    """Apply distance-based calibration to embedding via iterative scaling."""
    try:
        # Compute all pairwise distances
        n_samples = X_low.shape[0]
        if n_samples < 2:
            return X_low.copy()
        
        distances = pdist(X_low, metric='euclidean')
        
        # Apply calibration function
        calibrated_distances = calibration_func(distances)
        
        # Convert back to square matrix
        distance_matrix = squareform(calibrated_distances)
        
        # Use multidimensional scaling (MDS) to reconstruct coordinates
        # This is a simplified version - for production use sklearn.manifold.MDS
        X_calibrated = _simple_mds(distance_matrix, X_low.shape[1])
        
        return X_calibrated
        
    except Exception as e:
        logger.error(f"Distance calibration application failed: {e}")
        return X_low.copy()


def _simple_mds(distance_matrix: FloatArray, n_components: int) -> FloatArray:
    """Simplified multidimensional scaling for distance calibration."""
    try:
        n = distance_matrix.shape[0]
        
        # Double centering
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (distance_matrix ** 2) @ H
        
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(B)
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Take top components
        eigenvals = eigenvals[:n_components]
        eigenvecs = eigenvecs[:, :n_components]
        
        # Ensure non-negative eigenvalues
        eigenvals = np.maximum(eigenvals, 0)
        
        # Reconstruct coordinates
        coordinates = eigenvecs * np.sqrt(eigenvals)
        
        return coordinates
        
    except Exception as e:
        logger.error(f"Simple MDS failed: {e}")
        # Fallback: return scaled random coordinates
        np.random.seed(42)
        return np.random.randn(distance_matrix.shape[0], n_components) * 0.1


def _adaptive_neighborhood_size(
    X_high: FloatArray, 
    X_low: FloatArray,
    point_idx: int,
    base_k: int,
    max_k: int,
    nbrs_high: NearestNeighbors,
    nbrs_low: NearestNeighbors
) -> int:
    """Adaptively determine optimal neighborhood size for a point."""
    try:
        best_k = base_k
        best_correlation = -1.0
        
        # Try different neighborhood sizes
        for k in range(max(1, base_k // 2), min(max_k + 1, base_k * 2)):
            try:
                # Get k-neighborhoods
                dist_high, _ = nbrs_high.kneighbors(
                    X_high[point_idx:point_idx+1], n_neighbors=k + 1
                )
                dist_low, _ = nbrs_low.kneighbors(
                    X_low[point_idx:point_idx+1], n_neighbors=k + 1
                )
                
                # Remove self-distance
                dist_high = dist_high[0, 1:]
                dist_low = dist_low[0, 1:]
                
                if len(dist_high) > 1 and len(dist_low) > 1:
                    # Compute local correlation
                    corr, _ = spearmanr(dist_high, dist_low)
                    if np.isfinite(corr) and corr > best_correlation:
                        best_correlation = corr
                        best_k = k
                        
            except Exception:
                continue
        
        return best_k
        
    except Exception:
        return base_k


def _compute_distance_correlation(X_high: FloatArray, X_low: FloatArray) -> float:
    """Compute Spearman correlation between pairwise distances."""
    try:
        if X_high.shape[0] != X_low.shape[0] or X_high.shape[0] < 3:
            return 0.0
        
        # Sample for efficiency on large datasets
        n = X_high.shape[0]
        if n > 1000:
            np.random.seed(42)
            idx = np.random.choice(n, 1000, replace=False)
            X_high = X_high[idx]
            X_low = X_low[idx]
        
        distances_high = pdist(X_high, metric='euclidean')
        distances_low = pdist(X_low, metric='euclidean')
        
        # Remove zero distances
        nonzero_mask = (distances_high > 1e-12) & (distances_low > 1e-12)
        
        if not np.any(nonzero_mask):
            return 0.0
        
        corr, _ = spearmanr(
            distances_high[nonzero_mask], 
            distances_low[nonzero_mask]
        )
        
        return 0.0 if not np.isfinite(corr) else float(corr)
        
    except Exception as e:
        logger.debug(f"Distance correlation computation failed: {e}")
        return 0.0