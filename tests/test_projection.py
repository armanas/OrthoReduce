import numpy as np
from orthogonal_projection.projection import generate_orthogonal_basis, project_data

def test_generate_orthogonal_basis():
    d, k = 1000, 100
    basis = generate_orthogonal_basis(d, k)
    assert basis.shape == (d, k)
    assert np.allclose(basis.T @ basis, np.eye(k))  # Verify orthonormality

def test_project_data():
    d, k, n = 1000, 100, 100
    X = np.random.randn(n, d)
    basis = generate_orthogonal_basis(d, k)
    Y = project_data(X, basis)
    assert Y.shape == (n, k)
