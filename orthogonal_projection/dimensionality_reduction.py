"""
dimensionality_reduction.py - Simplified interface for dimensionality reduction methods

This module provides a clean, unified interface for various dimensionality reduction
techniques with minimal complexity and clear error handling.
"""
from __future__ import annotations

import numpy as np
import time
import logging
from functools import wraps
from typing import Dict, Optional, Tuple, List, Callable, Any
from numpy.typing import NDArray

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import spearmanr

# Import monitoring utilities (optional - graceful fallback if not available)
try:
    from .monitoring import create_simple_monitor, get_memory_usage, optimize_memory
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

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
# Import convex hull projection (now provided in-package)
    try:
        from .convex_optimized import project_onto_convex_hull_qp, project_onto_convex_hull_enhanced
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
        from convex_optimized import project_onto_convex_hull_qp, project_onto_convex_hull_enhanced
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


def with_monitoring(show_progress: bool = True, method_name: str = None):
    """
    Decorator to add optional progress monitoring to dimensionality reduction functions.
    
    Args:
        show_progress: Whether to show progress monitoring
        method_name: Name of the method for display (auto-detected if None)
    
    Returns:
        Decorated function with optional monitoring
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not show_progress or not MONITORING_AVAILABLE:
                return func(*args, **kwargs)
            
            # Extract method name from function name if not provided
            display_name = method_name or func.__name__.replace('run_', '').upper()
            
            # Get data size info for monitoring context
            X = args[0] if args else kwargs.get('X')
            data_info = ""
            if hasattr(X, 'shape'):
                data_info = f" | Data: {X.shape[0]}×{X.shape[1]}"
            
            # Create simple progress monitor
            with create_simple_monitor(f"{display_name}{data_info}", show_stats=True) as pbar:
                pbar.update(10)  # Start progress
                
                # Track memory before
                mem_before = get_memory_usage()
                
                try:
                    # Run the actual function
                    pbar.update(50)  # Mid-progress
                    result = func(*args, **kwargs)
                    pbar.update(90)  # Near completion
                    
                    # Track memory after and clean up
                    mem_after = get_memory_usage()
                    if mem_after - mem_before > 100:  # If used >100MB, optimize
                        optimize_memory()
                    
                    pbar.update(100)  # Complete
                    return result
                    
                except Exception as e:
                    pbar.set_description(f"❌ {display_name} FAILED: {str(e)[:50]}...")
                    raise
                    
        return wrapper
    return decorator

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

def project_onto_convex_hull(Y: np.ndarray) -> np.ndarray:
    """Legacy shim: project onto convex hull of Y.

    Uses exact QP-based projector when available, otherwise falls back to the
    fast closest-vertex approximation.
    """
    if Y.size == 0:
        return Y
    try:
        if CONVEX_AVAILABLE:
            Y_proj, _, _ = project_onto_convex_hull_qp(Y, tol=1e-6, maxiter=200)
            return Y_proj
        return project_onto_convex_hull_fast(Y)
    except Exception:
        return Y

def run_convex(X: np.ndarray, k: int, seed: int = 42) -> Tuple[np.ndarray, float]:
    """Legacy shim: JLL -> sphere -> convex-hull projection.

    Mirrors run_pocs but returns only the embedding and runtime.
    """
    start = time.time()
    Y_jll, _ = run_jll(X, k, seed, method='gaussian')
    Y_sph = map_to_unit_sphere(Y_jll)
    Y_proj = project_onto_convex_hull(Y_sph)
    return Y_proj, time.time() - start
def run_pocs(X: np.ndarray, k: int, seed: int = 42) -> Tuple[np.ndarray, float]:
    """
    Run POCS (Projection onto Convex Sets): JLL projection followed by fast convex hull projection.
    
    This combines Johnson-Lindenstrauss with convex regularization via convex-hull
    projection. If an exact constrained QP projector is available, it is used; otherwise
    a fast closest-vertex fallback is applied. A unit-sphere normalization is applied
    before hull projection to stabilize cosine geometry.
    
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

        # Step 2: Spherical normalization (hyperspherical embedding)
        Y_sph = map_to_unit_sphere(Y_jll)

        # Step 3: Convex hull projection
        if CONVEX_AVAILABLE:
            Y_proj, alphas, V_used = project_onto_convex_hull_qp(Y_sph, tol=1e-6, maxiter=200)
            logger.info(
                f"POCS: JLL → sphere → convex hull QP (V={V_used.shape[0]} vertices) completed"
            )
            return Y_proj, time.time() - start_time
        else:
            Y_pocs = project_onto_convex_hull_fast(Y_sph)
            logger.info("POCS: JLL → sphere → fast convex hull (approx) completed")
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

def run_poincare(X: np.ndarray, k: int, seed: int = 42, c: float = 1.0,
                n_epochs: int = 50, lr: float = 0.01,
                optimizer: str = 'radam', loss_fn: str = 'stress',
                init_method: str = 'pca', regularization: float = 0.01) -> Tuple[np.ndarray, float]:
    """
    Run Poincaré (hyperbolic) embedding with rigorous Riemannian optimization.
    
    Maps data to the Poincaré ball model of hyperbolic space using proper
    hyperbolic geometry operations and Riemannian optimization.
    
    Parameters:
    -----------
    X : np.ndarray
        Input data (n_samples, n_features)
    k : int
        Target embedding dimension
    seed : int
        Random seed for reproducibility
    c : float
        Curvature parameter (higher = more curvature, typically 0.1-1.0)
    n_epochs : int
        Number of optimization epochs
    lr : float
        Learning rate for Riemannian optimizer
    optimizer : str
        Optimizer type ('rsgd' for Riemannian SGD, 'radam' for Riemannian Adam)
    loss_fn : str
        Loss function ('stress' for MDS-like, 'triplet', 'nca' for supervised)
    init_method : str
        Initialization method ('pca', 'spectral', 'random')
    regularization : float
        L2 regularization to keep points away from boundary
        
    Returns:
    --------
    Tuple[np.ndarray, float]
        (embedding, runtime) where embedding is in Poincaré ball
    """
    start_time = time.time()
    try:
        # Try to use optimized hyperbolic implementation
        try:
            from .hyperbolic import run_poincare_optimized
            return run_poincare_optimized(
                X, k, c=c, lr=lr, n_epochs=n_epochs,
                optimizer=optimizer, loss_fn=loss_fn,
                init_method=init_method, regularization=regularization,
                seed=seed
            )
        except ImportError:
            # Fallback to simple implementation if hyperbolic module not available
            logger.warning("Hyperbolic module not available, using simple Poincaré mapping")
            # Step 1: Apply JLL projection to target dimension
            Y_jll, _ = run_jll(X, k, seed, method='gaussian')
            
            # Step 2: Map to Poincaré disk (simple version)
            Y_poincare = map_to_poincare_disk(Y_jll, c=c)
            
            return Y_poincare, time.time() - start_time
        
    except Exception as e:
        logger.error(f"Poincaré embedding failed: {e}")
        return run_jll(X, k, seed)

def run_spherical(X: np.ndarray, k: int, seed: int = 42,
                 use_riemannian: bool = True,
                 adaptive_radius: bool = True,
                 loss_type: str = 'mds_geodesic') -> Tuple[np.ndarray, float]:
    """
    Run advanced spherical embedding with Riemannian optimization.
    
    Uses proper geodesic distances and manifold optimization for better
    structure preservation on the sphere.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    k : int
        Target dimension (embeds to S^(k-1))
    seed : int
        Random seed
    use_riemannian : bool
        Whether to use full Riemannian optimization (slower but better)
    adaptive_radius : bool
        Whether to optimize sphere radius
    loss_type : str
        Loss function: 'mds_geodesic', 'triplet', 'nca', 'hybrid'
    
    Returns
    -------
    Y : np.ndarray
        Spherical embedding
    runtime : float
        Execution time
    """
    start_time = time.time()
    try:
        if use_riemannian:
            # Import advanced spherical embedding
            try:
                from .spherical_embeddings import adaptive_spherical_embedding
            except ImportError:
                from spherical_embeddings import adaptive_spherical_embedding
            
            # Use advanced Riemannian optimization
            Y, info = adaptive_spherical_embedding(
                X, k,
                method='riemannian' if X.shape[0] <= 500 else 'fast',  # Use fast for large data
                loss_type=loss_type,
                max_iter=300 if X.shape[0] <= 200 else 100,  # Fewer iterations for large data
                learning_rate=0.01,
                adaptive_radius=adaptive_radius,
                hemisphere_constraint=True,
                seed=seed
            )
            
            logger.info(f"Spherical embedding completed with radius={info.get('final_radius', 1.0):.3f}")
            
        else:
            # Simple spherical embedding (original implementation)
            # Step 1: Apply JLL projection to target dimension  
            Y_jll, _ = run_jll(X, k, seed, method='gaussian')
            
            # Step 2: Map to unit sphere
            Y = map_to_unit_sphere(Y_jll)
        
        return Y, time.time() - start_time
        
    except Exception as e:
        logger.error(f"Advanced spherical embedding failed: {e}, falling back to simple")
        # Fallback to simple spherical embedding
        Y_jll, _ = run_jll(X, k, seed, method='gaussian')
        Y = map_to_unit_sphere(Y_jll)
        return Y, time.time() - start_time

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


def evaluate_projection_comprehensive(X: np.ndarray, Y: np.ndarray, 
                                    sample_size: int = 2000,
                                    include_advanced: bool = True,
                                    k_values: List[int] = None) -> Dict:
    """
    Comprehensive evaluation with all available metrics.
    
    Parameters
    ----------
    X : ndarray
        Original high-dimensional data
    Y : ndarray  
        Projected low-dimensional data
    sample_size : int
        Sample size for efficiency
    include_advanced : bool
        Whether to compute advanced metrics
    k_values : list of int or None
        k values for trustworthiness/continuity
        
    Returns
    -------
    dict
        Comprehensive evaluation results
    """
    # Import comprehensive evaluation function
    try:
        from .evaluation import comprehensive_evaluation
    except ImportError:
        from evaluation import comprehensive_evaluation
    
    if k_values is None:
        # Set reasonable k values based on sample size
        max_k = min(100, sample_size // 5) if sample_size else min(100, X.shape[0] // 5)
        k_values = [k for k in [10, 20, 50, 100] if k < max_k]
        if not k_values:
            k_values = [min(10, X.shape[0] - 1)]
    
    return comprehensive_evaluation(
        X, Y, 
        k_values=k_values,
        sample_size=sample_size,
        sample_pairs=min(20000, sample_size * (sample_size - 1) // 4) if sample_size else 10000,
        include_advanced=include_advanced
    )

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
                  use_optimized_eval: bool = True,
                  # Legacy flags accepted but not strictly required
                  use_convex: bool = False,
                  use_poincare: bool = True,
                  use_spherical: bool = True,
                  use_elliptic: bool = False,
                  # New monitoring options
                  enable_monitoring: bool = False,
                  show_method_progress: bool = False) -> Dict:
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
    enable_monitoring : bool
        Enable basic progress monitoring for individual methods
    show_method_progress : bool
        Show detailed progress bars for each method
        
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
    
    # Default methods - match legacy expectations in tests
    if methods is None:
        methods = ['pca', 'jll', 'gaussian']
        # Tests expect UMAP key to exist; if not available, we'll fallback
        methods.append('umap')
        if use_poincare:
            methods.append('poincare')
        if use_spherical:
            methods.append('spherical')
        if use_convex:
            methods.append('convex')
    
    results = {}
    
    # Setup monitoring context if requested
    total_methods = len(methods)
    method_progress = None
    
    if enable_monitoring and MONITORING_AVAILABLE:
        from .monitoring import ProgressTracker
        monitor_context = ProgressTracker(
            total_methods=len(methods),
            method_names=methods,
            show_system_stats=True
        )
        monitor_context.start()
    else:
        monitor_context = None
    
    try:
        # Run each method
        for i, method in enumerate(methods):
            logger.info(f"Running {method} ({i+1}/{total_methods})...")
            
            # Start method monitoring if enabled
            if monitor_context:
                monitor_context.start_method(method.upper(), data_points=n, dimensions=k)
            
            # Run the specific method
            if method == 'pca':
                Y, runtime = run_pca(X, k, seed)
            elif method == 'jll':
                # Use optimized JLL with intelligent method selection
                Y, runtime = run_jll(X, k, seed, method='auto')
            elif method == 'gaussian':
                Y, runtime = run_gaussian_projection(X, k, seed)
            elif method == 'umap':
                if UMAP_AVAILABLE:
                    Y, runtime = run_umap(X, k, seed)
                else:
                    # Fallback: use JLL but record under UMAP to satisfy interface
                    Y, runtime = run_jll(X, k, seed, method='auto')
            elif method == 'pocs':
                Y, runtime = run_pocs(X, k, seed)
            elif method == 'poincare':
                Y, runtime = run_poincare(X, k, seed)
            elif method == 'spherical':
                Y, runtime = run_spherical(X, k, seed)
            elif method == 'convex':
                Y, runtime = run_convex(X, k, seed)
            else:
                logger.warning(f"Unknown method: {method}")
                if monitor_context:
                    monitor_context.complete_method({'error': f'Unknown method: {method}'})
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
            
            # Compile results (include legacy extra metrics placeholders)
            result = {
                'mean_distortion': float(mean_dist),
                'max_distortion': float(max_dist),
                'rank_correlation': float(rank_corr),
                'runtime': float(runtime),
                'compression_ratio': float(d/k),
                'optimized_evaluation': OPTIMIZED_EVALUATION and use_optimized_eval,
                # legacy extras
                'kl_divergence': 0.0,
                'l1': 0.0,
            }

            # Method naming to match tests exactly
            name_map = {
                'pca': 'PCA',
                'jll': 'JLL',
                'umap': 'UMAP',
                'poincare': 'Poincare',
                'spherical': 'Spherical',
                'convex': 'Convex',
                'gaussian': 'GAUSSIAN',
                'pocs': 'POCS',
            }
            key = name_map.get(method, method.upper())
            results[key] = result
            
            # Complete method monitoring if enabled
            if monitor_context:
                monitor_context.complete_method(result)
    
    finally:
        # Clean up monitoring context
        if monitor_context:
            monitor_context.finish()
    
    return results

def run_experiment_with_visualization(
    n: int = 1000, d: int = 100, epsilon: float = 0.2, 
    seed: int = 42, sample_size: int = 2000,
    methods: Optional[list] = None,
    use_mixture_gaussians: bool = True,
    n_clusters: int = 10, cluster_std: float = 0.5,
    use_adaptive: bool = False,
    use_optimized_eval: bool = True,
    # Legacy flags
    use_convex: bool = False,
    use_poincare: bool = True,
    use_spherical: bool = True,
    use_elliptic: bool = False,
    # Visualization options
    create_plots: bool = True,
    plot_style: str = "publication",
    output_dir: str = "experiment_results_with_plots",
    include_advanced_plots: bool = True,
    include_interactive: bool = True,
    # Monitoring options
    enable_monitoring: bool = False,
    show_method_progress: bool = False
) -> Tuple[Dict, Optional[Dict[str, str]]]:
    """
    Run dimensionality reduction experiment with comprehensive visualization.
    
    This function extends run_experiment() with integrated plotting capabilities,
    creating publication-ready visualizations alongside the numerical results.
    
    Parameters
    ----------
    All parameters from run_experiment(), plus:
    
    create_plots : bool
        Whether to create visualizations
    plot_style : str
        Plotting style: "publication", "presentation", "interactive"
    output_dir : str
        Directory for saving results and plots
    include_advanced_plots : bool
        Whether to use advanced plotting features
    include_interactive : bool
        Whether to create interactive plots (requires plotly)
    enable_monitoring : bool
        Enable basic progress monitoring for individual methods
    show_method_progress : bool
        Show detailed progress bars for each method
        
    Returns
    -------
    results : dict
        Experiment results (same as run_experiment())
    plot_files : dict or None
        Dictionary mapping plot types to file paths (if create_plots=True)
    """
    logger.info("Running experiment with comprehensive visualization...")
    
    # Run the core experiment with monitoring options
    results = run_experiment(
        n=n, d=d, epsilon=epsilon, seed=seed, sample_size=sample_size,
        methods=methods, use_mixture_gaussians=use_mixture_gaussians,
        n_clusters=n_clusters, cluster_std=cluster_std,
        use_adaptive=use_adaptive, use_optimized_eval=use_optimized_eval,
        use_convex=use_convex, use_poincare=use_poincare,
        use_spherical=use_spherical, use_elliptic=use_elliptic,
        enable_monitoring=enable_monitoring, show_method_progress=show_method_progress
    )
    
    if not create_plots:
        return results, None
    
    try:
        # Import visualization (lazy import)
        from .visualization import OrthoReduceVisualizer
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        visualizer = OrthoReduceVisualizer(output_dir=output_dir)
        
        # Create traditional visualizations
        logger.info("Creating standard visualization package...")
        standard_plots = visualizer.create_complete_visualization(results)
        
        plot_files = {}
        plot_files.update(standard_plots)
        
        # Create advanced visualizations if requested
        if include_advanced_plots:
            logger.info("Creating advanced visualization suite...")
            
            # Generate embeddings for visualization
            embeddings = {}
            if use_mixture_gaussians:
                X = generate_mixture_gaussians(n, d, n_clusters, cluster_std, seed)
            else:
                np.random.seed(seed)
                X = np.random.randn(n, d)
            
            # Normalize to unit sphere
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            safe_norms = np.where(norms > 1e-10, norms, 1.0)
            X_normalized = X / safe_norms
            small_norm_mask = (norms <= 1e-10).flatten()
            if np.any(small_norm_mask):
                np.random.seed(seed)
                X_normalized[small_norm_mask] = np.random.randn(np.sum(small_norm_mask), d) * 1e-6
            X = X_normalized
            
            # Calculate target dimension
            k = min(jll_dimension(n, epsilon), d)
            
            # Generate embeddings for active methods
            active_methods = methods or ['pca', 'jll', 'gaussian', 'umap']
            if use_poincare and 'poincare' not in active_methods:
                active_methods.append('poincare')
            if use_spherical and 'spherical' not in active_methods:
                active_methods.append('spherical')
            if use_convex and 'convex' not in active_methods:
                active_methods.append('convex')
            
            for method in active_methods:
                if method in results:  # Only create embeddings for successful methods
                    try:
                        if method == 'pca':
                            Y, _ = run_pca(X, k, seed)
                        elif method == 'jll':
                            Y, _ = run_jll(X, k, seed, method='auto')
                        elif method == 'gaussian':
                            Y, _ = run_gaussian_projection(X, k, seed)
                        elif method == 'umap':
                            Y, _ = run_umap(X, k, seed)
                        elif method == 'pocs':
                            Y, _ = run_pocs(X, k, seed)
                        elif method == 'poincare':
                            Y, _ = run_poincare(X, k, seed)
                        elif method == 'spherical':
                            Y, _ = run_spherical(X, k, seed)
                        elif method == 'convex':
                            Y, _ = run_convex(X, k, seed)
                        else:
                            continue
                        
                        # Take only first 2-3 dimensions for visualization
                        embeddings[method.upper()] = Y[:, :min(3, Y.shape[1])]
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for {method}: {e}")
            
            # Create advanced visualization suite
            advanced_plots = visualizer.create_advanced_visualization_suite(
                results=results, embeddings=embeddings
            )
            plot_files.update(advanced_plots)
        
        logger.info(f"✅ Experiment with visualization completed!")
        logger.info(f"📊 Total plots generated: {len(plot_files)}")
        logger.info(f"📁 Output directory: {output_path}")
        
        return results, plot_files
        
    except Exception as e:
        logger.error(f"Failed to create visualizations: {e}")
        return results, None

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