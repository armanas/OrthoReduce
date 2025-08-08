#!/usr/bin/env python3
"""
main.py - Simple example of dimensionality reduction with OrthoReduce

This demonstrates the basic usage of the library for dimensionality reduction.
"""
import numpy as np
from orthogonal_projection.projection import jll_dimension
from orthogonal_projection.dimensionality_reduction import run_jll_simple, evaluate_projection

def main():
    print("OrthoReduce - Simple Dimensionality Reduction Example")
    print("=" * 55)
    
    # Parameters
    n = 1000  # Number of points  
    d = 100   # Original dimension
    epsilon = 0.3  # JLL distortion parameter
    
    print(f"Generating {n} points in {d} dimensions")
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(n, d)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize to unit sphere
    
    # Calculate target dimension
    k = min(jll_dimension(n, epsilon), d)
    print(f"Target dimension k = {k} (epsilon = {epsilon})")
    
    # Apply Johnson-Lindenstrauss projection
    print("Running JLL projection...")
    Y = run_jll_simple(X, k, seed=42)
    print(f"Projected to {Y.shape[1]} dimensions")
    
    # Evaluate quality
    print("Evaluating projection quality...")
    metrics = evaluate_projection(X, Y, sample_size=500)
    
    print(f"\nResults:")
    print(f"Mean distortion: {metrics['mean_distortion']:.4f}")
    print(f"Max distortion: {metrics['max_distortion']:.4f}")  
    print(f"Rank correlation: {metrics['rank_correlation']:.4f}")
    
    print("\nDone! For more advanced usage, see:")
    print("- python orthogonal_projection/dimensionality_reduction.py --help")
    print("- python ms_pipeline.py --help  (for mass spectrometry data)")

if __name__ == "__main__":
    main()
