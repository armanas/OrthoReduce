"""
Fast Johnson-Lindenstrauss Transform (FJLT) Implementation

This module implements the Fast Johnson-Lindenstrauss Transform which achieves
O(d log k) time complexity instead of O(dk) for standard random projections.

Based on:
- Ailon & Chazelle (2009): "The Fast Johnson-Lindenstrauss Transform"  
- Liberty et al. (2007): "An improved algorithm for the fast Johnson-Lindenstrauss transform"
"""
import numpy as np
from typing import Optional, Tuple

def _next_power_of_2(n: int) -> int:
    """Find the next power of 2 >= n."""
    return 1 << (n - 1).bit_length()

def _walsh_hadamard_transform(x: np.ndarray) -> np.ndarray:
    """
    Compute Walsh-Hadamard Transform using Fast Walsh-Hadamard Transform (FWHT).
    
    Time complexity: O(d log d) where d is the length of x.
    
    Parameters:
    - x: 1D array of length d (must be power of 2)
    
    Returns:
    - Transformed array of same length
    """
    x = x.copy()
    n = len(x)
    
    # FWHT using in-place butterfly operations
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                u, v = x[j], x[j + step]
                x[j], x[j + step] = u + v, u - v
        step *= 2
    
    return x / np.sqrt(n)  # Normalized Walsh-Hadamard

def _pad_to_power_of_2(X: np.ndarray, axis: int = 1) -> Tuple[np.ndarray, int]:
    """
    Pad array to next power of 2 along specified axis.
    
    Returns:
    - Padded array and original dimension
    """
    original_dim = X.shape[axis]
    padded_dim = _next_power_of_2(original_dim)
    
    if padded_dim == original_dim:
        return X, original_dim
    
    # Create padding specification
    pad_width = [(0, 0)] * X.ndim
    pad_width[axis] = (0, padded_dim - original_dim)
    
    X_padded = np.pad(X, pad_width, mode='constant', constant_values=0)
    return X_padded, original_dim

def generate_fjlt_matrix(d: int, k: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a Fast Johnson-Lindenstrauss Transform matrix.
    
    The FJLT uses a structured approach: P * H * D where:
    - D is a diagonal matrix with random ±1 entries (Rademacher)
    - H is a Walsh-Hadamard matrix (or DHT/DCT)
    - P is a sparse random sampling matrix
    
    This achieves O(d log k) matrix-vector multiplication time.
    
    Parameters:
    - d: Original dimension
    - k: Target dimension  
    - seed: Random seed
    
    Returns:
    - FJLT matrix of shape (d, k) for compatibility with existing API
    
    Note: For true O(d log k) performance, use fjlt_transform() directly
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Pad d to next power of 2 for optimal Walsh-Hadamard performance
    d_padded = _next_power_of_2(d)
    
    # Generate diagonal matrix D with random ±1 entries
    D = np.random.choice([-1, 1], size=d_padded)
    
    # Create sampling indices for P (uniform random sampling)
    # Sample k indices from d_padded dimensions
    sample_indices = np.random.choice(d_padded, size=k, replace=False)
    
    # Create the FJLT matrix P * H * D
    # Since we need to return a matrix, we'll create the full transform
    # (in practice, you'd apply D, H, P sequentially for O(d log k) speed)
    
    # Create sampling matrix P
    P = np.zeros((k, d_padded))
    P[np.arange(k), sample_indices] = np.sqrt(d_padded / k)
    
    # For the matrix representation, we need to multiply P * H
    # H is Walsh-Hadamard matrix (very memory intensive to store explicitly)
    # Instead, we'll create a structured representation
    
    # Create identity to transform through H
    H_cols = np.eye(d_padded)
    for i in range(d_padded):
        H_cols[:, i] = _walsh_hadamard_transform(H_cols[:, i])
    
    # Apply diagonal matrix D
    H_cols = H_cols * D[None, :]
    
    # Sample columns according to P
    fjlt_matrix = H_cols[:, sample_indices].T * np.sqrt(d_padded / k)
    
    # Truncate to original dimension d
    if d < d_padded:
        fjlt_matrix = fjlt_matrix[:, :d]
    
    return fjlt_matrix.T  # Return as (d, k) for compatibility

def fjlt_transform(X: np.ndarray, k: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Apply Fast Johnson-Lindenstrauss Transform directly to data.
    
    This is the optimized O(n d log k) implementation that doesn't 
    materialize the full transformation matrix.
    
    Parameters:
    - X: Input data of shape (n, d)
    - k: Target dimension
    - seed: Random seed
    
    Returns:
    - Transformed data of shape (n, k)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n, d = X.shape
    
    # Pad to power of 2 if needed
    X_padded, original_d = _pad_to_power_of_2(X, axis=1)
    d_padded = X_padded.shape[1]
    
    # Generate random diagonal matrix D
    D = np.random.choice([-1, 1], size=d_padded)
    
    # Generate sampling indices
    sample_indices = np.random.choice(d_padded, size=k, replace=False)
    
    # Apply FJLT: for each sample, compute D * x, then H * (D * x), then sample
    result = np.zeros((n, k))
    
    for i in range(n):
        # Step 1: Apply diagonal matrix D
        y = X_padded[i] * D
        
        # Step 2: Apply Walsh-Hadamard transform H  
        y = _walsh_hadamard_transform(y)
        
        # Step 3: Apply sampling matrix P
        result[i] = y[sample_indices] * np.sqrt(d_padded / k)
    
    return result

def benchmark_projection_methods(d: int, k: int, n_trials: int = 10) -> dict:
    """
    Benchmark different projection methods for speed comparison.
    
    Returns timing results for QR, Gaussian, Sparse, and FJLT methods.
    """
    import time
    from . import generate_orthogonal_basis
    
    methods = ['gaussian', 'sparse', 'rademacher']
    if _next_power_of_2(d) == d:  # Only test FJLT if d is power of 2
        methods.append('fjlt')
    
    results = {}
    
    for method in methods:
        times = []
        for trial in range(n_trials):
            start = time.perf_counter()
            _ = generate_orthogonal_basis(d, k, method=method, seed=trial)
            end = time.perf_counter()
            times.append(end - start)
        
        results[method] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times)
        }
    
    return results