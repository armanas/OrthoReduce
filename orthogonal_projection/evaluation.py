# evaluation.py
import numpy as np
from sklearn.metrics import pairwise_distances

def compute_distortion(X, Y, epsilon=1e-9):
    """Compute distortion of pairwise distances after projection."""
    D_original = pairwise_distances(X)
    D_reduced = pairwise_distances(Y)
    D_orig_sq = D_original ** 2
    D_red_sq = D_reduced ** 2

    # Avoid division by zero or near-zero distances
    distortion = np.abs(D_red_sq - D_orig_sq) / (D_orig_sq + epsilon)

    # Return the distance matrices for debugging if needed
    return distortion.mean(), distortion.max(), D_orig_sq, D_red_sq

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
