#!/usr/bin/env python3
"""
Demo script for the enhanced convex hull projection features.

This demonstrates the new capabilities added to convex_optimized.py:
- Ridge regularization for numerical stability
- Robust objective functions (Huber, epsilon-insensitive)
- High-precision float64 support
- Enhanced solver tolerance modes
- Vertex normalization options
"""

import numpy as np
import matplotlib.pyplot as plt
from orthogonal_projection.convex_optimized import (
    project_onto_convex_hull_qp,  # Original function
    project_onto_convex_hull_enhanced  # New enhanced function
)

def demo_basic_comparison():
    """Compare original vs enhanced function on well-behaved data."""
    print("=== Demo 1: Basic Comparison ===")
    
    # Generate test data
    np.random.seed(42)
    n, d = 200, 8
    angles = np.linspace(0, 2*np.pi, n)
    Y = np.column_stack([
        np.cos(angles) + np.random.normal(0, 0.05, n),
        np.sin(angles) + np.random.normal(0, 0.05, n)
    ] + [np.random.normal(0, 0.1, n) for _ in range(d-2)])
    
    print(f"Input data shape: {Y.shape}")
    
    # Original function
    Y_proj_orig, alphas_orig, V_orig = project_onto_convex_hull_qp(
        Y, tol=1e-6, ridge_lambda=0.0  # Old behavior
    )
    
    # Enhanced function
    Y_proj_enh, alphas_enh, V_enh = project_onto_convex_hull_enhanced(
        Y, ridge_lambda=1e-5, solver_mode='balanced'
    )
    
    # Compare constraint satisfaction
    orig_violation = np.abs(alphas_orig.sum(axis=1) - 1.0).max()
    enh_violation = np.abs(alphas_enh.sum(axis=1) - 1.0).max()
    
    print(f"Original: max constraint violation = {orig_violation:.2e}")
    print(f"Enhanced: max constraint violation = {enh_violation:.2e}")
    print(f"Original: {V_orig.shape[0]} vertices, Enhanced: {V_enh.shape[0]} vertices")
    print()


def demo_ridge_regularization():
    """Demonstrate ridge regularization on ill-conditioned data."""
    print("=== Demo 2: Ridge Regularization on Ill-Conditioned Data ===")
    
    # Create nearly collinear data (ill-conditioned)
    np.random.seed(123)
    n = 100
    t = np.linspace(-1, 1, n)
    Y = np.column_stack([
        t,
        t * 0.1 + np.random.normal(0, 0.01, n),  # Nearly parallel
        np.random.normal(0, 0.02, n),
        np.random.normal(0, 0.02, n),
        np.random.normal(0, 0.02, n)
    ])
    
    print(f"Ill-conditioned data shape: {Y.shape}")
    
    # Without ridge regularization
    Y_proj_no_ridge, alphas_no_ridge, _ = project_onto_convex_hull_enhanced(
        Y, ridge_lambda=0.0, solver_mode='balanced'
    )
    
    # With small ridge regularization
    Y_proj_small_ridge, alphas_small_ridge, _ = project_onto_convex_hull_enhanced(
        Y, ridge_lambda=1e-6, solver_mode='balanced'
    )
    
    # With larger ridge regularization
    Y_proj_large_ridge, alphas_large_ridge, _ = project_onto_convex_hull_enhanced(
        Y, ridge_lambda=1e-4, solver_mode='balanced'
    )
    
    violations = []
    for alphas, name in [
        (alphas_no_ridge, "No ridge"),
        (alphas_small_ridge, "Ridge λ=1e-6"),
        (alphas_large_ridge, "Ridge λ=1e-4")
    ]:
        violation = np.abs(alphas.sum(axis=1) - 1.0).max()
        violations.append(violation)
        print(f"{name:15}: max violation = {violation:.2e}")
    
    print()


def demo_robust_objectives():
    """Demonstrate robust objective functions."""
    print("=== Demo 3: Robust Objective Functions ===")
    
    # Create data with some outliers
    np.random.seed(456)
    n = 150
    
    # Generate base data
    angles = np.linspace(0, 2*np.pi, n)
    Y_base = np.column_stack([
        np.cos(angles),
        np.sin(angles),
        np.random.normal(0, 0.1, n),
        np.random.normal(0, 0.1, n)
    ])
    
    # Add some outliers
    n_outliers = 10
    outlier_indices = np.random.choice(n, n_outliers, replace=False)
    Y = Y_base.copy()
    Y[outlier_indices] += np.random.normal(0, 2, (n_outliers, Y.shape[1]))
    
    print(f"Data with outliers shape: {Y.shape}, {n_outliers} outliers added")
    
    objectives = [
        ('quadratic', {}),
        ('huber', {'huber_delta': 1.0}),
        ('huber', {'huber_delta': 0.5}),
        ('epsilon_insensitive', {'epsilon': 0.01})
    ]
    
    for obj_type, params in objectives:
        Y_proj, alphas, V = project_onto_convex_hull_enhanced(
            Y, objective_type=obj_type, ridge_lambda=1e-5, **params
        )
        
        violation = np.abs(alphas.sum(axis=1) - 1.0).max()
        sparsity = (np.abs(alphas) < 1e-6).mean()  # Fraction of near-zero weights
        
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        if param_str:
            param_str = f" ({param_str})"
        
        print(f"{obj_type:20}{param_str:15}: violation={violation:.2e}, "
              f"sparsity={sparsity:.3f}, vertices={V.shape[0]}")
    
    print()


def demo_precision_modes():
    """Demonstrate different precision modes."""
    print("=== Demo 4: Precision Modes ===")
    
    # Generate test data
    np.random.seed(789)
    Y = np.random.randn(80, 6)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)  # Normalize
    
    print(f"Test data shape: {Y.shape}")
    
    precision_configs = [
        ("Loose mode", {'solver_mode': 'loose', 'use_float64': False}),
        ("Balanced mode", {'solver_mode': 'balanced', 'use_float64': False}),
        ("Strict mode", {'solver_mode': 'strict', 'use_float64': False}),
        ("Strict + float64", {'solver_mode': 'strict', 'use_float64': True})
    ]
    
    for name, config in precision_configs:
        Y_proj, alphas, V = project_onto_convex_hull_enhanced(
            Y, ridge_lambda=1e-6, **config
        )
        
        violation = np.abs(alphas.sum(axis=1) - 1.0).max()
        print(f"{name:18}: max violation = {violation:.2e}")
    
    print()


def demo_normalization_modes():
    """Demonstrate vertex normalization options."""
    print("=== Demo 5: Vertex Normalization ===")
    
    # Generate test data with varying scales
    np.random.seed(101112)
    Y = np.random.randn(60, 4)
    # Scale dimensions differently
    Y[:, 0] *= 10  # Large scale
    Y[:, 1] *= 0.1  # Small scale
    
    print(f"Multi-scale data shape: {Y.shape}")
    print(f"Column scales: {np.std(Y, axis=0)}")
    
    normalizations = ['none', 'l2', 'unit_sphere']
    
    for norm_mode in normalizations:
        Y_proj, alphas, V = project_onto_convex_hull_enhanced(
            Y, candidate_normalization=norm_mode, ridge_lambda=1e-5
        )
        
        violation = np.abs(alphas.sum(axis=1) - 1.0).max()
        print(f"Normalization {norm_mode:12}: violation={violation:.2e}, "
              f"vertices={V.shape[0]}")
    
    print()


if __name__ == "__main__":
    print("Enhanced Convex Hull Projection Demo")
    print("=" * 50)
    print()
    
    # Suppress numerical warnings for cleaner output
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    demo_basic_comparison()
    demo_ridge_regularization()
    demo_robust_objectives()
    demo_precision_modes()
    demo_normalization_modes()
    
    print("Demo completed! All enhanced features demonstrated.")
    print()
    print("Key enhancements:")
    print("- Ridge regularization (λ) improves numerical stability")
    print("- Robust objectives (Huber, epsilon-insensitive) handle outliers")
    print("- High precision mode (float64) for demanding applications")
    print("- Flexible solver tolerance modes (loose/balanced/strict)")
    print("- Vertex normalization for multi-scale data")