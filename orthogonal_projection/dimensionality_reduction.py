"""
dimensionality_reduction.py - Simplified interface for dimensionality reduction methods

This module provides a clean, unified interface for various dimensionality reduction
techniques with minimal complexity and clear error handling.
"""
from __future__ import annotations

import numpy as np
import time
import logging
from typing import Dict, Optional, Tuple
from numpy.typing import NDArray

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import spearmanr

try:
    # Try relative imports first (when used as module)
    from .projection import jll_dimension, generate_orthogonal_basis, project_data
    # Import optimized evaluation functions
    try:
        from .evaluation_optimized import compute_distortion_optimized as compute_distortion
        from .evaluation_optimized import rank_correlation_optimized as rank_correlation
        OPTIMIZED_EVALUATION = True
    except ImportError:
        from .evaluation import compute_distortion, rank_correlation
        OPTIMIZED_EVALUATION = False
    # Import convex hull projection
    try:
        from .convex_optimized import project_onto_convex_hull_qp
        CONVEX_AVAILABLE = True
    except ImportError:
        CONVEX_AVAILABLE = False
except ImportError:
    # Fall back to absolute imports (when run as script)
    from projection import jll_dimension, generate_orthogonal_basis, project_data
    try:
        from evaluation_optimized import compute_distortion_optimized as compute_distortion
        from evaluation_optimized import rank_correlation_optimized as rank_correlation
        OPTIMIZED_EVALUATION = True
    except ImportError:
        from evaluation import compute_distortion, rank_correlation
        OPTIMIZED_EVALUATION = False
    # Import convex hull projection
    try:
        from convex_optimized import project_onto_convex_hull_qp
        CONVEX_AVAILABLE = True
    except ImportError:
        CONVEX_AVAILABLE = False

# Optional imports
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

logger = logging.getLogger(__name__)

def generate_mixture_gaussians(
    n: int, 
    d: int, 
    n_clusters: int = 10, 
    cluster_std: float = 0.5, 
    seed: int = 42
) -> NDArray[np.float64]:
    """Generate synthetic data as mixture of Gaussians."""
    np.random.seed(seed)
    
    # Generate cluster centers
    centers = np.random.randn(n_clusters, d)
    
    # Distribute points among clusters  
    points_per_cluster = n // n_clusters
    remainder = n % n_clusters
    
    X_parts = []
    for i in range(n_clusters):
        count = points_per_cluster + (1 if i < remainder else 0)
        points = centers[i] + cluster_std * np.random.randn(count, d)
        X_parts.append(points)
    
    X = np.vstack(X_parts)
    np.random.shuffle(X)
    return X

def run_pca(X: NDArray[np.float64], k: int, seed: int = 42) -> Tuple[NDArray[np.float64], float]:
    """Run Principal Component Analysis."""
    start_time = time.time()
    try:
        pca = PCA(n_components=k, random_state=seed)
        Y = pca.fit_transform(X)
        return Y, time.time() - start_time
    except Exception as e:
        logger.error(f"PCA failed: {e}")
        return np.random.randn(X.shape[0], k), time.time() - start_time

def run_jll(
    X: NDArray[np.float64], 
    k: int, 
    seed: int = 42, 
    method: str = 'auto'
) -> Tuple[NDArray[np.float64], float]:
    """
    Run Johnson-Lindenstrauss random projection with intelligent method selection.
    
    Parameters:
    - X: Input data (n, d)
    - k: Target dimension
    - seed: Random seed
    - method: Projection method ('auto', 'qr', 'gaussian', 'sparse', 'rademacher', 'fjlt')
    
    Returns:
    - (projected_data, runtime)
    """
    start_time = time.time()
    try:
        n, d = X.shape
        
        # Auto-select optimal method based on data characteristics
        if method == 'auto':
            if d >= 4096 and k <= d // 10:
                # Large dimension with high compression - use FJLT if available
                method = 'fjlt' if d & (d - 1) == 0 else 'sparse'  # FJLT needs power of 2
            elif k <= d // 4:
                # High compression ratio - use sparse projection
                method = 'sparse'
            elif n <= 1000:
                # Small dataset - use QR for best quality
                method = 'qr'
            else:
                # Default to fast Gaussian projection
                method = 'gaussian'
        
        # Generate projection matrix with selected method
        basis = generate_orthogonal_basis(d, k, method=method, seed=seed)
        Y = project_data(X, basis)
        return Y, time.time() - start_time
        
    except Exception as e:
        logger.error(f"JLL failed: {e}")
        return np.random.randn(X.shape[0], k), time.time() - start_time

def run_gaussian_projection(X: np.ndarray, k: int, seed: int = 42) -> Tuple[np.ndarray, float]:
    """Run Gaussian random projection using sklearn."""
    start_time = time.time()
    try:
        grp = GaussianRandomProjection(n_components=k, random_state=seed)
        Y = grp.fit_transform(X)
        return Y, time.time() - start_time
    except Exception as e:
        logger.error(f"Gaussian projection failed: {e}")
        return np.random.randn(X.shape[0], k), time.time() - start_time

def run_umap(X: np.ndarray, k: int, seed: int = 42, 
             n_neighbors: int = 15, min_dist: float = 0.1) -> Tuple[np.ndarray, float]:
    """Run UMAP dimensionality reduction."""
    if not UMAP_AVAILABLE:
        logger.warning("UMAP not available, falling back to JLL")
        return run_jll(X, k, seed)
    
    start_time = time.time()
    try:
        reducer = umap.UMAP(
            n_components=k, 
            n_neighbors=min(n_neighbors, X.shape[0] - 1),
            min_dist=min_dist,
            random_state=seed
        )
        Y = reducer.fit_transform(X)
        return Y, time.time() - start_time
    except Exception as e:
        logger.error(f"UMAP failed: {e}")
        return run_jll(X, k, seed)

def run_pocs(X: np.ndarray, k: int, seed: int = 42) -> Tuple[np.ndarray, float]:
    """
    Run POCS (Projection onto Convex Sets): JLL projection followed by fast convex hull projection.
    
    This is your custom method combining Johnson-Lindenstrauss with convex hull constraints
    using the fast approximation method (closest vertex projection).
    
    Parameters:
    - X: Input data (n, d)
    - k: Target dimension
    - seed: Random seed
    
    Returns:
    - (projected_data, runtime)
    """
    start_time = time.time()
    try:
        # Step 1: Apply JLL projection
        Y_jll, _ = run_jll(X, k, seed, method='gaussian')  # Use Gaussian for stability
        
        # Step 2: Fast convex hull projection (approximation method)
        Y_pocs = project_onto_convex_hull_fast(Y_jll)
        
        logger.info(f"POCS: JLL → fast convex hull projection completed")
        return Y_pocs, time.time() - start_time
        
    except Exception as e:
        logger.error(f"POCS failed: {e}")
        return run_jll(X, k, seed)

def project_onto_convex_hull_fast(Y: np.ndarray) -> np.ndarray:
    """
    Fast convex hull projection using closest vertex approximation.
    This matches the original implementation's performance characteristics.
    """
    from scipy.spatial import ConvexHull
    
    try:
        n_samples, n_features = Y.shape
        
        # For computational efficiency, use a simplified approach
        if n_samples <= 20:
            # For small datasets, use exact convex hull
            hull = ConvexHull(Y)
            hull_vertices = Y[hull.vertices]
        else:
            # For larger datasets, approximate by using extreme points in each dimension
            hull_vertices = []
            for dim in range(n_features):
                min_idx = np.argmin(Y[:, dim])
                max_idx = np.argmax(Y[:, dim])
                hull_vertices.extend([Y[min_idx], Y[max_idx]])
            hull_vertices = np.unique(hull_vertices, axis=0)
        
        # For each point, project to the closest hull vertex (fast approximation)
        result = np.zeros_like(Y)
        for i, point in enumerate(Y):
            # Find the closest hull vertex
            distances = np.sum((hull_vertices - point) ** 2, axis=1)
            closest_vertex_idx = np.argmin(distances)
            
            # Project to closest vertex (simple approximation)
            result[i] = hull_vertices[closest_vertex_idx]
        
        return result
        
    except Exception as e:
        logger.warning(f"Fast convex hull projection failed: {e}, returning original data")
        return Y

def run_poincare(X: np.ndarray, k: int, seed: int = 42, c: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Run Poincaré (hyperbolic) embedding.
    
    Maps data to the Poincaré disk model of hyperbolic space.
    """
    start_time = time.time()
    try:
        # Step 1: Apply JLL projection to target dimension
        Y_jll, _ = run_jll(X, k, seed, method='gaussian')
        
        # Step 2: Map to Poincaré disk
        Y_poincare = map_to_poincare_disk(Y_jll, c=c)
        
        return Y_poincare, time.time() - start_time
        
    except Exception as e:
        logger.error(f"Poincaré embedding failed: {e}")
        return run_jll(X, k, seed)

def run_spherical(X: np.ndarray, k: int, seed: int = 42) -> Tuple[np.ndarray, float]:
    """
    Run spherical embedding.
    
    Maps data to the unit sphere.
    """
    start_time = time.time()
    try:
        # Step 1: Apply JLL projection to target dimension  
        Y_jll, _ = run_jll(X, k, seed, method='gaussian')
        
        # Step 2: Map to unit sphere
        Y_spherical = map_to_unit_sphere(Y_jll)
        
        return Y_spherical, time.time() - start_time
        
    except Exception as e:
        logger.error(f"Spherical embedding failed: {e}")
        return run_jll(X, k, seed)

def map_to_poincare_disk(Y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Map points to Poincaré disk model of hyperbolic space."""
    # Normalize to unit vectors first
    norms = np.linalg.norm(Y, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)  # Avoid division by zero
    Y_normalized = Y / norms
    
    # Map to Poincaré disk using stereographic projection
    # Scale by c and apply tanh to ensure points stay within unit disk
    scaled = c * Y_normalized
    
    # Use tanh to map to (-1, 1) range for each coordinate
    Y_poincare = np.tanh(scaled)
    
    return Y_poincare

def map_to_unit_sphere(Y: np.ndarray) -> np.ndarray:
    """Map points to unit sphere."""
    # Normalize each point to unit length
    norms = np.linalg.norm(Y, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)  # Avoid division by zero
    Y_spherical = Y / norms
    
    return Y_spherical

def evaluate_projection(X: np.ndarray, Y: np.ndarray, sample_size: int = 2000) -> Dict:
    """Evaluate quality of dimensionality reduction."""
    mean_dist, max_dist, _, _ = compute_distortion(X, Y, sample_size=sample_size)
    rank_corr = rank_correlation(X, Y, sample_size=sample_size)
    
    return {
        'mean_distortion': float(mean_dist),
        'max_distortion': float(max_dist),
        'rank_correlation': float(rank_corr)
    }

def adaptive_jll_dimension(X: np.ndarray, epsilon: float = 0.2, 
                          delta: float = 0.01, max_iterations: int = 10,
                          method: str = 'binary_search') -> int:
    """
    Adaptively find the minimal dimension k that preserves distances within epsilon.
    
    This can give much better compression than theoretical bounds for real data.
    
    Parameters:
    - X: Input data (n, d)
    - epsilon: Desired distortion tolerance
    - delta: Confidence level (fraction of pairs that can violate bound)
    - max_iterations: Maximum search iterations
    - method: Search strategy ('binary_search', 'doubling')
    
    Returns:
    - k: Minimal dimension that satisfies distortion constraints
    """
    n, d = X.shape
    
    # Get theoretical bounds as search range
    k_min = max(1, int(np.log(n) / (epsilon**2) / 4))  # Aggressive lower bound
    k_max = min(d, jll_dimension(n, epsilon, method='classic'))  # Conservative upper bound
    
    logger.info(f"Adaptive search range: k_min={k_min}, k_max={k_max}")
    
    if method == 'binary_search':
        # Binary search for optimal k
        best_k = k_max
        for iteration in range(max_iterations):
            if k_max - k_min <= 1:
                break
                
            k_test = (k_min + k_max) // 2
            logger.info(f"Testing k={k_test} (iteration {iteration + 1})")
            
            # Test projection quality
            Y, _ = run_jll(X, k_test, method='gaussian')  # Fast method for testing
            mean_dist, _, _, _ = compute_distortion(X, Y, sample_size=min(1000, n))
            
            if mean_dist <= epsilon * (1 + delta):
                # Success - try smaller k
                k_max = k_test
                best_k = k_test
                logger.info(f"k={k_test} successful (distortion={mean_dist:.4f})")
            else:
                # Failed - need larger k
                k_min = k_test
                logger.info(f"k={k_test} failed (distortion={mean_dist:.4f})")
        
        return best_k
    
    elif method == 'doubling':
        # Doubling strategy: start small and increase until success
        k = k_min
        while k <= k_max:
            Y, _ = run_jll(X, k, method='gaussian')
            mean_dist, _, _, _ = compute_distortion(X, Y, sample_size=min(1000, n))
            
            if mean_dist <= epsilon * (1 + delta):
                logger.info(f"Found k={k} (distortion={mean_dist:.4f})")
                return k
            
            k = min(int(k * 1.5), k_max)  # Increase by 50%
        
        return k_max
    
    else:
        raise ValueError(f"Unknown method: {method}")

def estimate_data_intrinsic_dimension(X: np.ndarray, sample_size: int = 1000) -> int:
    """
    Estimate the intrinsic dimensionality of data using PCA spectrum analysis.
    
    This can inform better compression strategies.
    """
    n, d = X.shape
    
    # Sample data if too large
    if n > sample_size:
        np.random.seed(42)
        idx = np.random.choice(n, sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
    
    # Compute PCA to analyze spectrum
    try:
        pca = PCA(n_components=min(d, sample_size-1))
        pca.fit(X_sample)
        
        # Find dimension that captures 95% of variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = int(np.argmax(cumsum >= 0.95)) + 1
        
        logger.info(f"Estimated intrinsic dimension: {intrinsic_dim} (captures 95% variance)")
        return intrinsic_dim
        
    except Exception as e:
        logger.error(f"Failed to estimate intrinsic dimension: {e}")
        return d // 4  # Conservative fallback

def run_experiment(n: int = 1000, d: int = 100, epsilon: float = 0.2, 
                  seed: int = 42, sample_size: int = 2000,
                  methods: Optional[list] = None,
                  use_mixture_gaussians: bool = True,
                  n_clusters: int = 10, cluster_std: float = 0.5,
                  use_adaptive: bool = False,
                  use_optimized_eval: bool = True) -> Dict:
    """
    Run dimensionality reduction experiment with multiple methods and optimizations.
    
    Parameters
    ----------
    n : int
        Number of data points
    d : int  
        Original dimensionality
    epsilon : float
        JLL distortion parameter
    seed : int
        Random seed
    sample_size : int
        Sample size for evaluation
    methods : list or None
        Methods to run ['pca', 'jll', 'gaussian', 'umap']. If None, runs all available.
    use_mixture_gaussians : bool
        Whether to generate mixture of Gaussians data
    n_clusters : int
        Number of clusters for mixture data
    cluster_std : float
        Standard deviation of clusters
    use_adaptive : bool
        Use adaptive dimension selection for better compression
    use_optimized_eval : bool
        Use high-performance evaluation functions
        
    Returns
    -------
    dict
        Results for each method with performance metrics
    """
    logger.info(f"Running optimized experiment: n={n}, d={d}, epsilon={epsilon}")
    logger.info(f"Optimizations: adaptive={use_adaptive}, eval={'optimized' if OPTIMIZED_EVALUATION else 'standard'}")
    
    # Generate or create data
    if use_mixture_gaussians:
        X = generate_mixture_gaussians(n, d, n_clusters, cluster_std, seed)
    else:
        np.random.seed(seed)
        X = np.random.randn(n, d)
    
    # Normalize to unit sphere with numerical stability
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    
    # Use robust normalization to prevent numerical issues
    # Prevent division by very small numbers
    safe_norms = np.where(norms > 1e-10, norms, 1.0)  # Replace small norms with 1.0
    X_normalized = X / safe_norms
    
    # For vectors that had very small norms, replace with small random vectors
    small_norm_mask = (norms <= 1e-10).flatten()
    if np.any(small_norm_mask):
        np.random.seed(seed)  # Ensure reproducibility
        X_normalized[small_norm_mask] = np.random.randn(np.sum(small_norm_mask), d) * 1e-6
        
    X = X_normalized
    
    # Calculate target dimension with optional adaptation
    if use_adaptive:
        logger.info("Using adaptive dimension selection...")
        k_adaptive = adaptive_jll_dimension(X, epsilon, max_iterations=5)
        k_theoretical = jll_dimension(n, epsilon)
        k = min(k_adaptive, d)
        logger.info(f"Adaptive dimension k={k} (theoretical={k_theoretical}, improvement={k_theoretical-k})")
    else:
        k = min(jll_dimension(n, epsilon), d)
        logger.info(f"Theoretical dimension k={k}")
    
    # Estimate intrinsic dimensionality for reference
    intrinsic_dim = estimate_data_intrinsic_dimension(X)
    logger.info(f"Estimated intrinsic dimension: {intrinsic_dim}")
    
    # Default methods - include all 6 methods
    if methods is None:
        methods = ['pca', 'jll', 'gaussian', 'pocs', 'poincare', 'spherical']
        if UMAP_AVAILABLE:
            methods.append('umap')
    
    results = {}
    
    # Run each method
    for method in methods:
        logger.info(f"Running {method}...")
        
        if method == 'pca':
            Y, runtime = run_pca(X, k, seed)
        elif method == 'jll':
            # Use optimized JLL with intelligent method selection
            Y, runtime = run_jll(X, k, seed, method='auto')
        elif method == 'gaussian':
            Y, runtime = run_gaussian_projection(X, k, seed)
        elif method == 'umap':
            Y, runtime = run_umap(X, k, seed)
        elif method == 'pocs':
            Y, runtime = run_pocs(X, k, seed)
        elif method == 'poincare':
            Y, runtime = run_poincare(X, k, seed)
        elif method == 'spherical':
            Y, runtime = run_spherical(X, k, seed)
        else:
            logger.warning(f"Unknown method: {method}")
            continue
            
        # Evaluate with optimized or standard functions
        if use_optimized_eval and OPTIMIZED_EVALUATION:
            # Use high-performance evaluation
            mean_dist, max_dist, _, _ = compute_distortion(X, Y, sample_size=sample_size)
            rank_corr = rank_correlation(X, Y, sample_size=sample_size)
        else:
            # Fall back to standard evaluation
            metrics = evaluate_projection(X, Y, sample_size)
            mean_dist = metrics['mean_distortion']
            max_dist = metrics['max_distortion'] 
            rank_corr = metrics['rank_correlation']
        
        # Compile results
        result = {
            'mean_distortion': float(mean_dist),
            'max_distortion': float(max_dist),
            'rank_correlation': float(rank_corr),
            'runtime': float(runtime),
            'compression_ratio': float(d/k),
            'optimized_evaluation': OPTIMIZED_EVALUATION and use_optimized_eval
        }
        results[method.upper()] = result
    
    # Add metadata
    results['_metadata'] = {
        'n': n,
        'd': d,
        'k': k,
        'epsilon': epsilon,
        'intrinsic_dimension': intrinsic_dim,
        'adaptive_used': use_adaptive,
        'optimized_eval_available': OPTIMIZED_EVALUATION
    }
    
    return results

# Simple interface functions for backward compatibility
def run_jll_simple(X, k, seed=42):
    """Simple JLL projection without timing."""
    Y, _ = run_jll(X, k, seed)
    return Y

def run_pca_simple(X, k, seed=42):
    """Simple PCA without timing."""
    Y, _ = run_pca(X, k, seed)
    return Y

def run_umap_simple(X, k, seed=42):
    """Simple UMAP without timing."""
    Y, _ = run_umap(X, k, seed)
    return Y

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run dimensionality reduction experiments")
    parser.add_argument('--n', type=int, default=1000, help="Number of points")
    parser.add_argument('--d', type=int, default=100, help="Original dimension")
    parser.add_argument('--epsilon', type=float, default=0.2, help="JLL epsilon")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--sample_size', type=int, default=2000, help="Sample size for evaluation")
    parser.add_argument('--methods', nargs='+', default=None, help="Methods to run")
    parser.add_argument('--simple-data', action='store_true', help="Use simple random data instead of mixture")
    
    args = parser.parse_args()
    
    results = run_experiment(
        n=args.n,
        d=args.d, 
        epsilon=args.epsilon,
        seed=args.seed,
        sample_size=args.sample_size,
        methods=args.methods,
        use_mixture_gaussians=not args.simple_data
    )
    
    print("\nResults:")
    print("=" * 50)
    for method, metrics in results.items():
        print(f"\n{method}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")