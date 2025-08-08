# evaluation.py
import numpy as np
import logging
import time
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr, kendalltau
from typing import Optional, Tuple, Dict, Any, Union, List
from numpy.typing import NDArray

# Import calibration utilities
try:
    from .calibration import (
        isotonic_regression_calibration,
        procrustes_alignment, 
        local_linear_correction,
        combined_calibration
    )
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False

# Set up module logger
logger = logging.getLogger(__name__)

def compute_distortion(X, Y, sample_size=None, epsilon=1e-9, seed=42):
    """Compute distortion of pairwise distances after projection.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, d_original)
        Original high-dimensional data
    Y : array-like, shape (n_samples, d_reduced) 
        Projected low-dimensional data
    sample_size : int or None
        If provided, randomly sample this many points for efficiency
    epsilon : float
        Small value to avoid division by zero
    seed : int
        Random seed for sampling
        
    Returns
    -------
    tuple
        (mean_distortion, max_distortion, D_orig_sq, D_red_sq)
    """
    try:
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        # Sample points if requested for large datasets
        if sample_size is not None and X.shape[0] > sample_size:
            np.random.seed(seed)
            idx = np.random.choice(X.shape[0], sample_size, replace=False)
            X = X[idx]
            Y = Y[idx]
        
        D_original = pairwise_distances(X, metric='euclidean')
        D_reduced = pairwise_distances(Y, metric='euclidean')
        D_orig_sq = D_original ** 2
        D_red_sq = D_reduced ** 2

        # Avoid division by zero
        denominator = np.maximum(D_orig_sq, epsilon)
        distortion = np.abs(D_red_sq - D_orig_sq) / denominator

        return distortion.mean(), distortion.max(), D_orig_sq, D_red_sq
        
    except Exception as e:
        logger.error(f"Error computing distortion: {e}", exc_info=True)
        return 0.0, 0.0, np.zeros((1, 1)), np.zeros((1, 1))

def nearest_neighbor_overlap(X, Y, k=10):
    """Evaluate nearest-neighbor preservation after projection."""
    from sklearn.neighbors import NearestNeighbors
    # Request k+1 neighbors so that each point's own index (distance zero)
    # is included and can be removed before computing the overlap. This
    # prevents artificially inflating the score by counting each point as its
    # own nearest neighbor.
    n_neighbors = min(k + 1, len(X))
    nn_original = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    nn_reduced = NearestNeighbors(n_neighbors=n_neighbors).fit(Y)
    _, idx_original = nn_original.kneighbors(X)
    _, idx_reduced = nn_reduced.kneighbors(Y)

    # Drop the first column which corresponds to the point itself
    idx_original = idx_original[:, 1:]
    idx_reduced = idx_reduced[:, 1:]

    overlap = [
        len(set(idx_original[i]) & set(idx_reduced[i])) / k
        for i in range(len(X))
    ]
    return np.mean(overlap)

def rank_correlation(X, Y, sample_size=None, seed=42):
    """Compute Spearman rank correlation between pairwise distances.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, d_original)
        Original data
    Y : array-like, shape (n_samples, d_reduced)
        Projected data
    sample_size : int or None
        If provided, randomly sample points for efficiency  
    seed : int
        Random seed for sampling
        
    Returns
    -------
    float
        Spearman correlation coefficient
    """
    try:
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        # Sample if requested
        if sample_size is not None and X.shape[0] > sample_size:
            np.random.seed(seed)
            idx = np.random.choice(X.shape[0], sample_size, replace=False)
            X = X[idx]
            Y = Y[idx]

        # Compute distance matrices
        DX = pairwise_distances(X)
        DY = pairwise_distances(Y)

        # Extract upper triangular values (no diagonal)
        triu_idx = np.triu_indices(DX.shape[0], k=1)
        x_distances = DX[triu_idx]
        y_distances = DY[triu_idx]

        # Compute Spearman correlation
        corr, _ = spearmanr(x_distances, y_distances)
        
        return 0.0 if not np.isfinite(corr) else float(corr)
        
    except Exception as e:
        logger.error(f"Error computing rank correlation: {e}", exc_info=True)
        return 0.0


def trustworthiness(X: NDArray[np.float64], Y: NDArray[np.float64], 
                   k: int = 10, sample_size: Optional[int] = None, 
                   seed: int = 42) -> float:
    """
    Compute trustworthiness metric for dimensionality reduction.
    
    Trustworthiness measures how well neighborhoods are preserved from high-d to low-d.
    It quantifies the fraction of k-nearest neighbors in the low-dimensional space
    that are also neighbors in the high-dimensional space.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, d_original)
        Original high-dimensional data
    Y : ndarray, shape (n_samples, d_reduced)
        Projected low-dimensional data  
    k : int
        Number of neighbors to consider
    sample_size : int or None
        Sample subset of points for efficiency on large datasets
    seed : int
        Random seed for sampling
        
    Returns
    -------
    float
        Trustworthiness score (0 to 1, higher is better)
    """
    try:
        X = np.asarray(X)
        Y = np.asarray(Y)
        n_samples = X.shape[0]
        
        # Validate k
        k = min(k, n_samples - 1)
        if k <= 0:
            return 1.0
            
        # Sample if requested
        if sample_size is not None and n_samples > sample_size:
            np.random.seed(seed)
            idx = np.random.choice(n_samples, sample_size, replace=False)
            X = X[idx]
            Y = Y[idx]
            n_samples = sample_size
            k = min(k, n_samples - 1)
        
        # Find k+1 nearest neighbors (including self)
        nbrs_X = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X)
        nbrs_Y = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(Y)
        
        # Get neighbor indices (excluding self at index 0)
        _, neighbors_X = nbrs_X.kneighbors(X)
        _, neighbors_Y = nbrs_Y.kneighbors(Y)
        
        neighbors_X = neighbors_X[:, 1:]  # Remove self
        neighbors_Y = neighbors_Y[:, 1:]  # Remove self
        
        # Compute ranks in original space for all pairs
        distances_X = pairwise_distances(X)
        ranks_X = np.argsort(np.argsort(distances_X, axis=1), axis=1)
        
        trustworthiness_sum = 0.0
        
        for i in range(n_samples):
            # Neighbors in low-d space
            y_neighbors = set(neighbors_Y[i])
            
            # For each neighbor in Y that's not in X's k-neighbors
            for j in y_neighbors:
                if j not in neighbors_X[i]:
                    # Penalize by how far this point is in original space
                    rank_in_X = ranks_X[i, j]
                    if rank_in_X > k:
                        trustworthiness_sum += rank_in_X - k
        
        # Normalize
        if n_samples <= k:
            return 1.0
            
        normalization = n_samples * k * (2 * n_samples - 3 * k - 1) / 2
        trustworthiness_val = 1.0 - (2.0 * trustworthiness_sum) / normalization
        
        return max(0.0, min(1.0, trustworthiness_val))
        
    except Exception as e:
        logger.error(f"Error computing trustworthiness: {e}", exc_info=True)
        return 0.0


def continuity(X: NDArray[np.float64], Y: NDArray[np.float64], 
               k: int = 10, sample_size: Optional[int] = None, 
               seed: int = 42) -> float:
    """
    Compute continuity metric for dimensionality reduction.
    
    Continuity measures how well neighborhoods are preserved from low-d to high-d.
    It quantifies the fraction of k-nearest neighbors in the high-dimensional space
    that are also neighbors in the low-dimensional space.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, d_original)
        Original high-dimensional data
    Y : ndarray, shape (n_samples, d_reduced)
        Projected low-dimensional data
    k : int
        Number of neighbors to consider
    sample_size : int or None
        Sample subset of points for efficiency on large datasets
    seed : int
        Random seed for sampling
        
    Returns
    -------
    float
        Continuity score (0 to 1, higher is better)
    """
    try:
        X = np.asarray(X)
        Y = np.asarray(Y)
        n_samples = X.shape[0]
        
        # Validate k
        k = min(k, n_samples - 1)
        if k <= 0:
            return 1.0
            
        # Sample if requested
        if sample_size is not None and n_samples > sample_size:
            np.random.seed(seed)
            idx = np.random.choice(n_samples, sample_size, replace=False)
            X = X[idx]
            Y = Y[idx]
            n_samples = sample_size
            k = min(k, n_samples - 1)
        
        # Find k+1 nearest neighbors (including self)
        nbrs_X = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X)
        nbrs_Y = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(Y)
        
        # Get neighbor indices (excluding self at index 0)
        _, neighbors_X = nbrs_X.kneighbors(X)
        _, neighbors_Y = nbrs_Y.kneighbors(Y)
        
        neighbors_X = neighbors_X[:, 1:]  # Remove self
        neighbors_Y = neighbors_Y[:, 1:]  # Remove self
        
        # Compute ranks in low-d space for all pairs
        distances_Y = pairwise_distances(Y)
        ranks_Y = np.argsort(np.argsort(distances_Y, axis=1), axis=1)
        
        continuity_sum = 0.0
        
        for i in range(n_samples):
            # Neighbors in high-d space
            x_neighbors = set(neighbors_X[i])
            
            # For each neighbor in X that's not in Y's k-neighbors
            for j in x_neighbors:
                if j not in neighbors_Y[i]:
                    # Penalize by how far this point is in low-d space
                    rank_in_Y = ranks_Y[i, j]
                    if rank_in_Y > k:
                        continuity_sum += rank_in_Y - k
        
        # Normalize
        if n_samples <= k:
            return 1.0
            
        normalization = n_samples * k * (2 * n_samples - 3 * k - 1) / 2
        continuity_val = 1.0 - (2.0 * continuity_sum) / normalization
        
        return max(0.0, min(1.0, continuity_val))
        
    except Exception as e:
        logger.error(f"Error computing continuity: {e}", exc_info=True)
        return 0.0


def trustworthiness_continuity_multi_k(X: NDArray[np.float64], Y: NDArray[np.float64],
                                       k_values: List[int] = [10, 20, 50, 100],
                                       sample_size: Optional[int] = None,
                                       seed: int = 42) -> Dict[str, Dict[int, float]]:
    """
    Compute trustworthiness and continuity for multiple k values efficiently.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, d_original)
        Original high-dimensional data
    Y : ndarray, shape (n_samples, d_reduced)
        Projected low-dimensional data
    k_values : list of int
        List of k values to compute metrics for
    sample_size : int or None
        Sample subset for efficiency
    seed : int
        Random seed
        
    Returns
    -------
    dict
        Dictionary with 'trustworthiness' and 'continuity' keys, 
        each mapping k values to scores
    """
    try:
        X = np.asarray(X)
        Y = np.asarray(Y)
        n_samples = X.shape[0]
        
        # Filter k values to be valid
        max_k = n_samples - 1
        k_values = [k for k in k_values if k <= max_k and k > 0]
        
        if not k_values:
            return {'trustworthiness': {}, 'continuity': {}}
        
        # Sample if requested
        if sample_size is not None and n_samples > sample_size:
            np.random.seed(seed)
            idx = np.random.choice(n_samples, sample_size, replace=False)
            X = X[idx]
            Y = Y[idx]
            n_samples = sample_size
            k_values = [k for k in k_values if k < n_samples]
        
        # Pre-compute distance matrices and ranks (expensive operations)
        distances_X = pairwise_distances(X)
        distances_Y = pairwise_distances(Y)
        ranks_X = np.argsort(np.argsort(distances_X, axis=1), axis=1)
        ranks_Y = np.argsort(np.argsort(distances_Y, axis=1), axis=1)
        
        # Pre-compute neighbors for largest k
        max_k = max(k_values)
        nbrs_X = NearestNeighbors(n_neighbors=max_k+1, metric='euclidean').fit(X)
        nbrs_Y = NearestNeighbors(n_neighbors=max_k+1, metric='euclidean').fit(Y)
        
        _, all_neighbors_X = nbrs_X.kneighbors(X)
        _, all_neighbors_Y = nbrs_Y.kneighbors(Y)
        
        all_neighbors_X = all_neighbors_X[:, 1:]  # Remove self
        all_neighbors_Y = all_neighbors_Y[:, 1:]  # Remove self
        
        results = {'trustworthiness': {}, 'continuity': {}}
        
        # Compute metrics for each k
        for k in k_values:
            # Get k neighbors for this specific k
            neighbors_X = all_neighbors_X[:, :k]
            neighbors_Y = all_neighbors_Y[:, :k]
            
            # Compute trustworthiness
            trustworthiness_sum = 0.0
            for i in range(n_samples):
                y_neighbors = set(neighbors_Y[i])
                for j in y_neighbors:
                    if j not in neighbors_X[i]:
                        rank_in_X = ranks_X[i, j]
                        if rank_in_X > k:
                            trustworthiness_sum += rank_in_X - k
            
            normalization = n_samples * k * (2 * n_samples - 3 * k - 1) / 2
            trust_val = 1.0 - (2.0 * trustworthiness_sum) / normalization
            results['trustworthiness'][k] = max(0.0, min(1.0, trust_val))
            
            # Compute continuity  
            continuity_sum = 0.0
            for i in range(n_samples):
                x_neighbors = set(neighbors_X[i])
                for j in x_neighbors:
                    if j not in neighbors_Y[i]:
                        rank_in_Y = ranks_Y[i, j]
                        if rank_in_Y > k:
                            continuity_sum += rank_in_Y - k
            
            cont_val = 1.0 - (2.0 * continuity_sum) / normalization
            results['continuity'][k] = max(0.0, min(1.0, cont_val))
        
        return results
        
    except Exception as e:
        logger.error(f"Error computing multi-k metrics: {e}", exc_info=True)
        return {'trustworthiness': {}, 'continuity': {}}


def advanced_correlation_metrics(X: NDArray[np.float64], Y: NDArray[np.float64],
                                sample_pairs: int = 10000, seed: int = 42,
                                distance_metric: str = 'euclidean') -> Dict[str, float]:
    """
    Compute advanced correlation metrics with subsampling strategy.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, d_original)
        Original high-dimensional data
    Y : ndarray, shape (n_samples, d_reduced)
        Projected low-dimensional data
    sample_pairs : int
        Number of distance pairs to sample (5k-20k recommended)
    seed : int
        Random seed for sampling
    distance_metric : str
        Distance metric ('euclidean', 'cosine', 'manhattan')
        
    Returns
    -------
    dict
        Dictionary containing spearman_r, kendall_tau, and other metrics
    """
    try:
        X = np.asarray(X)
        Y = np.asarray(Y)
        n_samples = X.shape[0]
        
        # Compute distance matrices
        DX = pairwise_distances(X, metric=distance_metric)
        DY = pairwise_distances(Y, metric=distance_metric)
        
        # Get upper triangular indices (excluding diagonal)
        triu_indices = np.triu_indices(n_samples, k=1)
        total_pairs = len(triu_indices[0])
        
        # Sample pairs if dataset is large
        if total_pairs > sample_pairs:
            np.random.seed(seed)
            sample_idx = np.random.choice(total_pairs, sample_pairs, replace=False)
            x_distances = DX[triu_indices][sample_idx]
            y_distances = DY[triu_indices][sample_idx]
        else:
            x_distances = DX[triu_indices]
            y_distances = DY[triu_indices]
        
        # Remove any invalid distances
        valid_mask = np.isfinite(x_distances) & np.isfinite(y_distances)
        x_distances = x_distances[valid_mask]
        y_distances = y_distances[valid_mask]
        
        if len(x_distances) < 2:
            return {'spearman_r': 0.0, 'kendall_tau': 0.0, 'pearson_r': 0.0}
        
        # Compute correlations
        spearman_r, _ = spearmanr(x_distances, y_distances)
        kendall_tau, _ = kendalltau(x_distances, y_distances)
        
        # Pearson correlation on distances
        pearson_r = np.corrcoef(x_distances, y_distances)[0, 1]
        
        return {
            'spearman_r': 0.0 if not np.isfinite(spearman_r) else float(spearman_r),
            'kendall_tau': 0.0 if not np.isfinite(kendall_tau) else float(kendall_tau),
            'pearson_r': 0.0 if not np.isfinite(pearson_r) else float(pearson_r),
            'pairs_used': len(x_distances)
        }
        
    except Exception as e:
        logger.error(f"Error computing advanced correlations: {e}", exc_info=True)
        return {'spearman_r': 0.0, 'kendall_tau': 0.0, 'pearson_r': 0.0, 'pairs_used': 0}


def evaluate_projection_calibrated(
    X: NDArray[np.float64], 
    Y: NDArray[np.float64],
    calibration_method: str = 'combined',
    sample_size: Optional[int] = None,
    seed: int = 42,
    **calibration_kwargs
) -> Dict[str, Any]:
    """
    Evaluate dimensionality reduction with post-processing calibration.
    
    Applies calibration methods to improve correlation metrics while maintaining
    numerical stability and computational efficiency.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, d_original)
        Original high-dimensional data
    Y : ndarray of shape (n_samples, d_reduced)  
        Projected low-dimensional data
    calibration_method : str
        Calibration approach ('isotonic', 'procrustes', 'local', 'combined', 'none')
    sample_size : int or None
        Sample size for efficient evaluation on large datasets
    seed : int
        Random seed for reproducible results
    **calibration_kwargs
        Additional parameters for calibration methods
        
    Returns
    -------
    dict
        Comprehensive evaluation metrics including:
        - Original metrics (before calibration)
        - Calibrated metrics (after calibration) 
        - Calibration information and improvements
    """
    if not CALIBRATION_AVAILABLE and calibration_method != 'none':
        logger.warning("Calibration module not available, falling back to standard evaluation")
        calibration_method = 'none'
    
    # Compute original metrics
    original_metrics = {
        'mean_distortion': 0.0,
        'max_distortion': 0.0,
        'rank_correlation': 0.0
    }
    
    try:
        mean_dist, max_dist, _, _ = compute_distortion(X, Y, sample_size=sample_size, seed=seed)
        rank_corr = rank_correlation(X, Y, sample_size=sample_size, seed=seed)
        
        original_metrics.update({
            'mean_distortion': float(mean_dist),
            'max_distortion': float(max_dist), 
            'rank_correlation': float(rank_corr)
        })
    except Exception as e:
        logger.error(f"Error computing original metrics: {e}")
    
    # Apply calibration if requested
    calibrated_metrics = original_metrics.copy()
    calibration_info = {'applied': False, 'method': calibration_method}
    
    if calibration_method != 'none' and CALIBRATION_AVAILABLE:
        try:
            Y_calibrated, calib_info = _apply_calibration_method(
                X, Y, calibration_method, **calibration_kwargs
            )
            
            # Evaluate calibrated embedding
            mean_dist_cal, max_dist_cal, _, _ = compute_distortion(
                X, Y_calibrated, sample_size=sample_size, seed=seed
            )
            rank_corr_cal = rank_correlation(
                X, Y_calibrated, sample_size=sample_size, seed=seed
            )
            
            calibrated_metrics.update({
                'mean_distortion': float(mean_dist_cal),
                'max_distortion': float(max_dist_cal),
                'rank_correlation': float(rank_corr_cal)
            })
            
            calibration_info.update({
                'applied': True,
                'calibration_details': calib_info,
                'improvements': {
                    'rank_correlation': float(rank_corr_cal - rank_corr),
                    'mean_distortion': float(mean_dist - mean_dist_cal),
                    'max_distortion': float(max_dist - max_dist_cal)
                }
            })
            
            logger.info(
                f"Calibration ({calibration_method}) improved rank correlation by "
                f"{calibration_info['improvements']['rank_correlation']:.4f}"
            )
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            calibration_info['error'] = str(e)
    
    return {
        'original': original_metrics,
        'calibrated': calibrated_metrics, 
        'calibration_info': calibration_info,
        'improvement_summary': {
            'rank_correlation_gain': calibrated_metrics['rank_correlation'] - original_metrics['rank_correlation'],
            'distortion_reduction': original_metrics['mean_distortion'] - calibrated_metrics['mean_distortion'],
            'calibration_successful': calibration_info['applied']
        }
    }


def compute_calibrated_correlation(
    X: NDArray[np.float64],
    Y: NDArray[np.float64], 
    method: str = 'isotonic',
    sample_size: Optional[int] = None,
    seed: int = 42
) -> float:
    """
    Compute rank correlation after applying calibration for maximum improvement.
    
    This is a convenience function that applies calibration and returns only
    the improved correlation coefficient.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, d_original)
        Original high-dimensional data
    Y : ndarray of shape (n_samples, d_reduced)
        Projected low-dimensional data
    method : str
        Calibration method ('isotonic', 'procrustes', 'local', 'combined')
    sample_size : int or None
        Sample size for efficiency
    seed : int
        Random seed
        
    Returns
    -------
    float
        Calibrated Spearman rank correlation coefficient
    """
    if not CALIBRATION_AVAILABLE:
        logger.warning("Calibration not available, returning standard correlation")
        return rank_correlation(X, Y, sample_size=sample_size, seed=seed)
    
    try:
        Y_calibrated, _ = _apply_calibration_method(X, Y, method)
        return rank_correlation(X, Y_calibrated, sample_size=sample_size, seed=seed)
    except Exception as e:
        logger.error(f"Calibrated correlation computation failed: {e}")
        return rank_correlation(X, Y, sample_size=sample_size, seed=seed)


def _apply_calibration_method(
    X: NDArray[np.float64], 
    Y: NDArray[np.float64], 
    method: str,
    **kwargs
) -> Tuple[NDArray[np.float64], Dict[str, Any]]:
    """Apply the specified calibration method."""
    if method == 'isotonic':
        Y_cal, calib_func = isotonic_regression_calibration(X, Y, **kwargs)
        return Y_cal, {'calibration_function': calib_func}
        
    elif method == 'procrustes':
        Y_cal, transform_info = procrustes_alignment(X, Y, **kwargs) 
        return Y_cal, transform_info
        
    elif method == 'local':
        Y_cal = local_linear_correction(X, Y, **kwargs)
        return Y_cal, {'method': 'local_linear'}
        
    elif method == 'combined':
        Y_cal, calib_info = combined_calibration(X, Y, **kwargs)
        return Y_cal, calib_info
        
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def benchmark_calibration_methods(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    methods: Optional[list[str]] = None,
    sample_size: Optional[int] = None,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different calibration methods on the same embedding.
    
    Compares the effectiveness of different calibration approaches for
    improving correlation metrics.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, d_original)
        Original high-dimensional data
    Y : ndarray of shape (n_samples, d_reduced)
        Projected low-dimensional data
    methods : list of str or None
        Calibration methods to benchmark (default: all available)
    sample_size : int or None
        Sample size for efficiency
    seed : int
        Random seed for reproducible results
        
    Returns
    -------
    dict
        Benchmark results for each method with performance metrics
    """
    if not CALIBRATION_AVAILABLE:
        logger.error("Calibration module not available for benchmarking")
        return {}
    
    if methods is None:
        methods = ['isotonic', 'procrustes', 'local', 'combined']
    
    # Compute baseline metrics
    baseline_corr = rank_correlation(X, Y, sample_size=sample_size, seed=seed)
    baseline_dist, _, _, _ = compute_distortion(X, Y, sample_size=sample_size, seed=seed)
    
    results = {
        'baseline': {
            'rank_correlation': baseline_corr,
            'mean_distortion': baseline_dist,
            'improvement': 0.0,
            'runtime': 0.0
        }
    }
    
    for method in methods:
        try:
            import time
            start_time = time.time()
            
            # Apply calibration
            Y_cal, _ = _apply_calibration_method(X, Y, method)
            
            # Evaluate calibrated results
            cal_corr = rank_correlation(X, Y_cal, sample_size=sample_size, seed=seed)
            cal_dist, _, _, _ = compute_distortion(X, Y_cal, sample_size=sample_size, seed=seed)
            
            runtime = time.time() - start_time
            improvement = cal_corr - baseline_corr
            
            results[method] = {
                'rank_correlation': cal_corr,
                'mean_distortion': cal_dist,
                'improvement': improvement,
                'runtime': runtime
            }
            
            logger.info(f"Method {method}: correlation={cal_corr:.4f}, improvement={improvement:.4f}")
            
        except Exception as e:
            logger.error(f"Benchmarking failed for method {method}: {e}")
            results[method] = {
                'rank_correlation': baseline_corr,
                'mean_distortion': baseline_dist,
                'improvement': 0.0,
                'runtime': 0.0,
                'error': str(e)
            }
    
    return results


def sammon_stress(X: NDArray[np.float64], Y: NDArray[np.float64],
                  sample_size: Optional[int] = None, seed: int = 42,
                  distance_metric: str = 'euclidean') -> Dict[str, float]:
    """
    Compute Sammon stress with local vs global decomposition.
    
    Sammon stress emphasizes preserving small distances more than large ones,
    making it sensitive to local structure preservation.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, d_original)
        Original high-dimensional data
    Y : ndarray, shape (n_samples, d_reduced)
        Projected low-dimensional data
    sample_size : int or None
        Sample size for efficiency
    seed : int
        Random seed
    distance_metric : str
        Distance metric to use
        
    Returns
    -------
    dict
        Dictionary with total, local, and global stress values
    """
    try:
        X = np.asarray(X)
        Y = np.asarray(Y)
        n_samples = X.shape[0]
        
        # Sample if requested
        if sample_size is not None and n_samples > sample_size:
            np.random.seed(seed)
            idx = np.random.choice(n_samples, sample_size, replace=False)
            X = X[idx]
            Y = Y[idx]
            n_samples = sample_size
        
        # Compute distance matrices
        DX = pairwise_distances(X, metric=distance_metric)
        DY = pairwise_distances(Y, metric=distance_metric)
        
        # Get upper triangular part (excluding diagonal)
        triu_indices = np.triu_indices(n_samples, k=1)
        d_orig = DX[triu_indices]
        d_proj = DY[triu_indices]
        
        # Remove zero distances to avoid division by zero
        nonzero_mask = d_orig > 1e-12
        d_orig = d_orig[nonzero_mask]
        d_proj = d_proj[nonzero_mask]
        
        if len(d_orig) == 0:
            return {'total_stress': 0.0, 'local_stress': 0.0, 'global_stress': 0.0}
        
        # Compute Sammon stress: sum((d_ij - d'_ij)^2 / d_ij) / sum(d_ij)
        stress_numerator = np.sum((d_orig - d_proj)**2 / d_orig)
        stress_denominator = np.sum(d_orig)
        
        total_stress = stress_numerator / stress_denominator if stress_denominator > 0 else 0.0
        
        # Decompose into local vs global stress
        # Local: distances smaller than median
        # Global: distances larger than median
        median_distance = np.median(d_orig)
        
        local_mask = d_orig <= median_distance
        global_mask = d_orig > median_distance
        
        # Local stress
        if np.any(local_mask):
            local_numerator = np.sum((d_orig[local_mask] - d_proj[local_mask])**2 / d_orig[local_mask])
            local_denominator = np.sum(d_orig[local_mask])
            local_stress = local_numerator / local_denominator if local_denominator > 0 else 0.0
        else:
            local_stress = 0.0
        
        # Global stress
        if np.any(global_mask):
            global_numerator = np.sum((d_orig[global_mask] - d_proj[global_mask])**2 / d_orig[global_mask])
            global_denominator = np.sum(d_orig[global_mask])
            global_stress = global_numerator / global_denominator if global_denominator > 0 else 0.0
        else:
            global_stress = 0.0
        
        return {
            'total_stress': float(total_stress),
            'local_stress': float(local_stress),
            'global_stress': float(global_stress),
            'local_global_ratio': float(local_stress / global_stress) if global_stress > 0 else 0.0
        }
        
    except Exception as e:
        logger.error(f"Error computing Sammon stress: {e}", exc_info=True)
        return {'total_stress': 0.0, 'local_stress': 0.0, 'global_stress': 0.0, 'local_global_ratio': 0.0}


def weighted_stress(X: NDArray[np.float64], Y: NDArray[np.float64],
                   weighting: str = 'adaptive', sample_size: Optional[int] = None,
                   seed: int = 42) -> Dict[str, float]:
    """
    Compute weighted stress with different weighting schemes.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, d_original)
        Original high-dimensional data
    Y : ndarray, shape (n_samples, d_reduced)
        Projected low-dimensional data
    weighting : str
        Weighting scheme ('adaptive', 'inverse_distance', 'uniform')
    sample_size : int or None
        Sample size for efficiency
    seed : int
        Random seed
        
    Returns
    -------
    dict
        Weighted stress metrics
    """
    try:
        X = np.asarray(X)
        Y = np.asarray(Y)
        n_samples = X.shape[0]
        
        # Sample if requested
        if sample_size is not None and n_samples > sample_size:
            np.random.seed(seed)
            idx = np.random.choice(n_samples, sample_size, replace=False)
            X = X[idx]
            Y = Y[idx]
            n_samples = sample_size
        
        # Compute distance matrices
        DX = pairwise_distances(X)
        DY = pairwise_distances(Y)
        
        # Get upper triangular part
        triu_indices = np.triu_indices(n_samples, k=1)
        d_orig = DX[triu_indices]
        d_proj = DY[triu_indices]
        
        # Compute weights based on scheme
        if weighting == 'uniform':
            weights = np.ones_like(d_orig)
        elif weighting == 'inverse_distance':
            weights = 1.0 / np.maximum(d_orig, 1e-12)
        elif weighting == 'adaptive':
            # Adaptive weighting: higher weight for smaller distances
            # but not as extreme as inverse distance
            median_dist = np.median(d_orig)
            weights = np.exp(-d_orig / median_dist)
        else:
            weights = np.ones_like(d_orig)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Compute weighted stress
        squared_errors = (d_orig - d_proj)**2
        weighted_stress_val = np.sum(weights * squared_errors)
        
        # Compute unweighted stress for comparison
        unweighted_stress = np.mean(squared_errors)
        
        return {
            'weighted_stress': float(weighted_stress_val),
            'unweighted_stress': float(unweighted_stress),
            'stress_ratio': float(weighted_stress_val / unweighted_stress) if unweighted_stress > 0 else 1.0
        }
        
    except Exception as e:
        logger.error(f"Error computing weighted stress: {e}", exc_info=True)
        return {'weighted_stress': 0.0, 'unweighted_stress': 0.0, 'stress_ratio': 1.0}


def comprehensive_evaluation(X: NDArray[np.float64], Y: NDArray[np.float64],
                           k_values: List[int] = [10, 20, 50, 100],
                           sample_size: Optional[int] = None,
                           sample_pairs: int = 10000,
                           seed: int = 42,
                           include_advanced: bool = True) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation with all available metrics.
    
    This function computes trustworthiness, continuity, correlation metrics,
    stress metrics, and traditional distortion measures to provide a complete
    assessment of dimensionality reduction quality.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, d_original)
        Original high-dimensional data
    Y : ndarray, shape (n_samples, d_reduced)
        Projected low-dimensional data
    k_values : list of int
        Neighborhood sizes for trustworthiness/continuity
    sample_size : int or None
        Sample size for efficiency
    sample_pairs : int
        Number of distance pairs for correlation metrics
    seed : int
        Random seed
    include_advanced : bool
        Whether to compute advanced metrics (slower but more comprehensive)
        
    Returns
    -------
    dict
        Comprehensive evaluation results
    """
    start_time = time.time()
    
    results = {
        'metadata': {
            'n_samples': X.shape[0],
            'original_dim': X.shape[1],
            'reduced_dim': Y.shape[1],
            'compression_ratio': X.shape[1] / Y.shape[1],
            'sample_size_used': min(sample_size or X.shape[0], X.shape[0]),
            'evaluation_time': 0.0
        }
    }
    
    try:
        logger.info("Computing comprehensive evaluation metrics...")
        
        # Basic distortion metrics
        logger.info("Computing distortion metrics...")
        mean_dist, max_dist, _, _ = compute_distortion(X, Y, sample_size=sample_size, seed=seed)
        results['distortion'] = {
            'mean_distortion': float(mean_dist),
            'max_distortion': float(max_dist)
        }
        
        # Basic correlation
        logger.info("Computing rank correlation...")
        basic_corr = rank_correlation(X, Y, sample_size=sample_size, seed=seed)
        results['basic_correlation'] = float(basic_corr)
        
        # Trustworthiness and continuity for multiple k values
        logger.info("Computing trustworthiness and continuity...")
        trust_cont = trustworthiness_continuity_multi_k(X, Y, k_values, sample_size, seed)
        results['trustworthiness'] = trust_cont['trustworthiness']
        results['continuity'] = trust_cont['continuity']
        
        # Nearest neighbor overlap for reference
        logger.info("Computing neighbor overlap...")
        k_ref = min(10, X.shape[0] - 1)
        if k_ref > 0:
            overlap = nearest_neighbor_overlap(X, Y, k=k_ref)
            results['neighbor_overlap'] = float(overlap)
        else:
            results['neighbor_overlap'] = 1.0
        
        if include_advanced:
            # Advanced correlation metrics
            logger.info("Computing advanced correlation metrics...")
            adv_corr = advanced_correlation_metrics(X, Y, sample_pairs, seed)
            results['advanced_correlation'] = adv_corr
            
            # Sammon stress
            logger.info("Computing Sammon stress...")
            sammon = sammon_stress(X, Y, sample_size, seed)
            results['sammon_stress'] = sammon
            
            # Weighted stress
            logger.info("Computing weighted stress...")
            weighted = weighted_stress(X, Y, 'adaptive', sample_size, seed)
            results['weighted_stress'] = weighted
        
        # Summary statistics
        results['summary'] = {
            'overall_quality_score': _compute_quality_score(results),
            'local_preservation': np.mean([v for v in results['trustworthiness'].values()]) if results['trustworthiness'] else 0.0,
            'global_preservation': np.mean([v for v in results['continuity'].values()]) if results['continuity'] else 0.0,
            'distance_preservation': basic_corr
        }
        
        results['metadata']['evaluation_time'] = time.time() - start_time
        logger.info(f"Evaluation completed in {results['metadata']['evaluation_time']:.2f} seconds")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in comprehensive evaluation: {e}", exc_info=True)
        results['metadata']['evaluation_time'] = time.time() - start_time
        results['error'] = str(e)
        return results


def _compute_quality_score(results: Dict[str, Any]) -> float:
    """
    Compute an overall quality score from evaluation metrics.
    
    Combines multiple metrics into a single interpretable score (0-1).
    """
    try:
        # Basic metrics (always available)
        basic_corr = results.get('basic_correlation', 0.0)
        mean_dist = results.get('distortion', {}).get('mean_distortion', 1.0)
        
        # Trustworthiness and continuity (average across k values)
        trust_scores = list(results.get('trustworthiness', {}).values())
        cont_scores = list(results.get('continuity', {}).values())
        
        avg_trust = np.mean(trust_scores) if trust_scores else 0.0
        avg_cont = np.mean(cont_scores) if cont_scores else 0.0
        
        # Normalize distortion (lower is better, convert to 0-1 scale where 1 is best)
        dist_score = max(0.0, 1.0 - min(mean_dist, 1.0))
        
        # Weighted combination
        weights = {'correlation': 0.3, 'trustworthiness': 0.3, 
                  'continuity': 0.2, 'distortion': 0.2}
        
        quality_score = (
            weights['correlation'] * max(0.0, basic_corr) +
            weights['trustworthiness'] * avg_trust +
            weights['continuity'] * avg_cont +
            weights['distortion'] * dist_score
        )
        
        return max(0.0, min(1.0, quality_score))
        
    except Exception:
        return 0.0


def evaluation_report(results: Dict[str, Any], method_name: str = "Unknown") -> str:
    """
    Generate a comprehensive evaluation report.
    
    Parameters
    ----------
    results : dict
        Results from comprehensive_evaluation
    method_name : str
        Name of the dimensionality reduction method
        
    Returns
    -------
    str
        Formatted evaluation report
    """
    report = []
    report.append(f"\n{'='*60}")
    report.append(f"DIMENSIONALITY REDUCTION EVALUATION REPORT")
    report.append(f"Method: {method_name}")
    report.append(f"{'='*60}")
    
    # Metadata
    meta = results.get('metadata', {})
    report.append(f"\nDataset Information:")
    report.append(f"  Original dimension: {meta.get('original_dim', 'N/A')}")
    report.append(f"  Reduced dimension: {meta.get('reduced_dim', 'N/A')}")
    report.append(f"  Number of samples: {meta.get('n_samples', 'N/A')}")
    report.append(f"  Compression ratio: {meta.get('compression_ratio', 'N/A'):.2f}x")
    report.append(f"  Evaluation time: {meta.get('evaluation_time', 0):.2f}s")
    
    # Summary
    summary = results.get('summary', {})
    report.append(f"\nOverall Quality Assessment:")
    report.append(f"  Quality score: {summary.get('overall_quality_score', 0):.3f} (0-1, higher is better)")
    report.append(f"  Local preservation: {summary.get('local_preservation', 0):.3f}")
    report.append(f"  Global preservation: {summary.get('global_preservation', 0):.3f}")
    report.append(f"  Distance preservation: {summary.get('distance_preservation', 0):.3f}")
    
    # Basic metrics
    report.append(f"\nBasic Metrics:")
    distortion = results.get('distortion', {})
    report.append(f"  Mean distortion: {distortion.get('mean_distortion', 0):.4f}")
    report.append(f"  Max distortion: {distortion.get('max_distortion', 0):.4f}")
    report.append(f"  Rank correlation: {results.get('basic_correlation', 0):.4f}")
    report.append(f"  Neighbor overlap: {results.get('neighbor_overlap', 0):.4f}")
    
    # Trustworthiness and Continuity
    trust = results.get('trustworthiness', {})
    cont = results.get('continuity', {})
    if trust or cont:
        report.append(f"\nTrustworthiness & Continuity (by k):")
        all_k = sorted(set(trust.keys()) | set(cont.keys()))
        for k in all_k:
            t_val = trust.get(k, 'N/A')
            c_val = cont.get(k, 'N/A')
            if isinstance(t_val, (int, float)) and isinstance(c_val, (int, float)):
                report.append(f"  k={k:3d}: Trust={t_val:.3f}, Continuity={c_val:.3f}")
            else:
                report.append(f"  k={k:3d}: Trust={t_val}, Continuity={c_val}")
    
    # Advanced metrics
    adv_corr = results.get('advanced_correlation', {})
    if adv_corr:
        report.append(f"\nAdvanced Correlation Metrics:")
        report.append(f"  Spearman r: {adv_corr.get('spearman_r', 0):.4f}")
        report.append(f"  Kendall tau: {adv_corr.get('kendall_tau', 0):.4f}")
        report.append(f"  Pearson r: {adv_corr.get('pearson_r', 0):.4f}")
        report.append(f"  Pairs used: {adv_corr.get('pairs_used', 0)}")
    
    # Stress metrics
    sammon = results.get('sammon_stress', {})
    if sammon:
        report.append(f"\nSammon Stress Analysis:")
        report.append(f"  Total stress: {sammon.get('total_stress', 0):.4f}")
        report.append(f"  Local stress: {sammon.get('local_stress', 0):.4f}")
        report.append(f"  Global stress: {sammon.get('global_stress', 0):.4f}")
        report.append(f"  Local/Global ratio: {sammon.get('local_global_ratio', 0):.4f}")
    
    weighted = results.get('weighted_stress', {})
    if weighted:
        report.append(f"\nWeighted Stress:")
        report.append(f"  Weighted stress: {weighted.get('weighted_stress', 0):.4f}")
        report.append(f"  Unweighted stress: {weighted.get('unweighted_stress', 0):.4f}")
        report.append(f"  Ratio: {weighted.get('stress_ratio', 1):.4f}")
    
    # Recommendations
    report.append(f"\n{'='*40}")
    report.append(f"RECOMMENDATIONS:")
    report.append(f"{'='*40}")
    
    quality_score = summary.get('overall_quality_score', 0)
    if quality_score >= 0.8:
        report.append("✓ Excellent quality - embedding preserves structure very well")
    elif quality_score >= 0.6:
        report.append("✓ Good quality - embedding is suitable for most applications") 
    elif quality_score >= 0.4:
        report.append("⚠ Fair quality - consider parameter tuning or different method")
    else:
        report.append("✗ Poor quality - significant structure loss detected")
    
    # Specific recommendations
    local_pres = summary.get('local_preservation', 0)
    global_pres = summary.get('global_preservation', 0)
    
    if local_pres < 0.5:
        report.append("• Low trustworthiness - local neighborhoods not well preserved")
    if global_pres < 0.5:
        report.append("• Low continuity - global structure may be distorted")
    if results.get('basic_correlation', 0) < 0.5:
        report.append("• Low distance correlation - consider different distance metric")
    
    report.append(f"{'='*60}")
    
    return '\n'.join(report)
