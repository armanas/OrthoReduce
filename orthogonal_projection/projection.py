import numpy as np

def jll_dimension(n, epsilon):
    """
    Compute the embedding dimension k using Johnson-Lindenstrauss lemma.
    
    Parameters:
    - n: int, number of points
    - epsilon: float, desired maximum distortion
    
    Returns:
    - k: int, the required dimension to preserve distances within 1 ± epsilon
    """
    return int(np.ceil((4 * np.log(n)) / (epsilon ** 2)))


def generate_orthogonal_basis(d, k, seed=None):
    """Generate a nearly orthogonal basis of dimension k in R^d."""
    if seed is not None:
        np.random.seed(seed)
    random_matrix = np.random.randn(d, k)
    Q, _ = np.linalg.qr(random_matrix)  # Q is orthonormal
    return Q

def project_data(X, basis):
    """Project data X onto the lower-dimensional basis."""
    return X @ basis
