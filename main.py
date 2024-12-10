import numpy as np
from orthogonal_projection.projection import generate_orthogonal_basis, project_data, jll_dimension
from orthogonal_projection.evaluation import compute_distortion, nearest_neighbor_overlap

# Parameters for data generation
d = 100   # Original dimension
n = 1000  # Number of points

# Generate synthetic data
X = np.random.randn(n, d)
X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize to the unit sphere

# Choose a desired epsilon for distortion
epsilon = 0.4  # Larger epsilon = less strict accuracy = smaller k
k = jll_dimension(n, epsilon)
print(f"Calculated k using JLL (epsilon={epsilon}, n={n}): {k}")

if k > d:
    print("Calculated k exceeds original dimension d. Adjusting k = d.")
    k = d

basis = generate_orthogonal_basis(d, k, seed=42)

# Check orthogonality
orthogonality_check = np.allclose(np.dot(basis.T, basis), np.eye(k), atol=1e-7)
if orthogonality_check:
    print("Orthogonality check passed. Basis is approximately orthonormal.")
else:
    print("Orthogonality check FAILED. Basis is not orthonormal.")

# Project the data onto k dimensions
Y = project_data(X, basis)

# Evaluate Distortion
mean_distortion, max_distortion, D_orig_sq, D_red_sq = compute_distortion(X, Y, epsilon=1e-9)

print("=== Debugging Distances ===")
print("D_orig_sq min:", np.min(D_orig_sq))
print("D_orig_sq max:", np.max(D_orig_sq))
print("D_red_sq min:", np.min(D_red_sq))
print("D_red_sq max:", np.max(D_red_sq))
print("===========================")

print(f"Mean distortion: {mean_distortion}")
print(f"Max distortion: {max_distortion}")

nn_overlap = nearest_neighbor_overlap(X, Y, k=10)
print(f"Nearest neighbor overlap: {nn_overlap}")
