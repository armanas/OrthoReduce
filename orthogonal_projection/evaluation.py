# evaluation.py
import numpy as np
import logging
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

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
