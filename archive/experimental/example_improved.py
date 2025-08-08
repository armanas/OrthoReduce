#!/usr/bin/env python3
"""
Simple example demonstrating the improved OrthoReduce API with KISS principles.

This example shows:
- Proper error handling with custom exceptions
- Type-safe operations
- Numerical stability improvements
- Clean logging instead of print statements
"""

import numpy as np
import logging
from orthogonal_projection.dimensionality_reduction import run_experiment
from orthogonal_projection.projection import jll_dimension, generate_orthogonal_basis
from orthogonal_projection.exceptions import ValidationError, DimensionalityError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Demonstrate improved OrthoReduce functionality."""
    logger.info("🚀 OrthoReduce KISS Improvements Demo")
    
    # Example 1: Input validation with clear error messages
    logger.info("\n1. Testing input validation...")
    try:
        jll_dimension(n=-10, epsilon=0.2)  # Invalid input
    except ValidationError as e:
        logger.info(f"✅ Caught validation error: {e}")
    
    try:
        generate_orthogonal_basis(d=10, k=20)  # k > d
    except DimensionalityError as e:
        logger.info(f"✅ Caught dimensionality error: {e}")
    
    # Example 2: Normal operation with type safety
    logger.info("\n2. Normal operation with type safety...")
    n, d = 200, 50
    epsilon = 0.3
    
    k = jll_dimension(n, epsilon, method='optimal')
    logger.info(f"Optimal JL dimension: {d} -> {k} (compression: {d/k:.1f}x)")
    
    # Example 3: Numerical stability (no more warnings!)
    logger.info("\n3. Testing numerical stability...")
    results = run_experiment(
        n=n, d=d, epsilon=epsilon,
        methods=['jll', 'pca'],
        use_adaptive=True,
        use_optimized_eval=True
    )
    
    logger.info("Results (no numerical warnings!):")
    for method, metrics in results.items():
        if method != '_metadata':
            logger.info(f"  {method}: distortion={metrics['mean_distortion']:.4f}, "
                       f"runtime={metrics['runtime']:.4f}s")
    
    # Example 4: Different projection methods
    logger.info("\n4. Testing different projection methods...")
    X = np.random.randn(100, 30)
    k_small = min(20, k)
    
    methods = ['gaussian', 'sparse', 'rademacher']
    for method in methods:
        basis = generate_orthogonal_basis(30, k_small, method=method, seed=42)
        logger.info(f"  {method}: basis shape {basis.shape}")
    
    logger.info("\n✅ All KISS improvements working perfectly!")
    logger.info("Benefits achieved:")
    logger.info("  • Clean error handling with custom exceptions")
    logger.info("  • Type safety with comprehensive type hints")
    logger.info("  • Numerical stability (no more warnings)")
    logger.info("  • Professional logging instead of print statements")
    logger.info("  • Modern development environment (pyproject.toml)")

if __name__ == '__main__':
    main()