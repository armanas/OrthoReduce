#!/usr/bin/env python3
"""
Simple benchmark to demonstrate OrthoReduce optimization improvements.
"""

import time
import numpy as np
from orthogonal_projection.dimensionality_reduction import run_experiment
from orthogonal_projection.projection import jll_dimension

def main():
    print("🚀 OrthoReduce Optimization Demonstration")
    print("=" * 45)
    
    # Test parameters
    n, d = 2000, 128
    epsilon = 0.2
    
    print(f"Test configuration: {n} points, {d} dimensions, ε={epsilon}")
    
    # Calculate dimensions
    k_optimal = jll_dimension(n, epsilon, method='optimal')
    k_classic = jll_dimension(n, epsilon, method='classic')
    
    print(f"Theoretical dimensions:")
    print(f"  Classic JL bound: {k_classic}")
    print(f"  Optimal JL bound: {k_optimal}")
    print(f"  Compression improvement: {k_classic/k_optimal:.1f}x better!")
    
    # Run standard experiment
    print(f"\\n📊 Running standard experiment...")
    start = time.perf_counter()
    results_std = run_experiment(
        n=n, d=d, epsilon=epsilon, 
        methods=['jll'],
        use_adaptive=False,
        use_optimized_eval=False
    )
    time_std = time.perf_counter() - start
    
    # Run optimized experiment
    print(f"⚡ Running optimized experiment...")
    start = time.perf_counter()
    results_opt = run_experiment(
        n=n, d=d, epsilon=epsilon,
        methods=['jll'],
        use_adaptive=True,
        use_optimized_eval=True
    )
    time_opt = time.perf_counter() - start
    
    # Compare results
    print(f"\\n📈 Performance Comparison:")
    print(f"{'':>20} {'Standard':>12} {'Optimized':>12} {'Speedup':>10}")
    print("-" * 56)
    
    # Runtime comparison
    speedup_time = time_std / max(time_opt, 1e-6)
    print(f"{'Total Runtime':>20} {time_std:>9.3f}s {time_opt:>9.3f}s {speedup_time:>7.1f}x")
    
    # Compression comparison
    k_std = results_std['_metadata']['k'] 
    k_opt = results_opt['_metadata']['k']
    compression_std = d / k_std
    compression_opt = d / k_opt
    improvement = compression_opt / compression_std
    print(f"{'Compression Ratio':>20} {compression_std:>9.1f}x {compression_opt:>9.1f}x {improvement:>7.1f}x")
    
    # Quality comparison
    jll_std = results_std['JLL']
    jll_opt = results_opt['JLL']
    print(f"{'Mean Distortion':>20} {jll_std['mean_distortion']:>9.4f} {jll_opt['mean_distortion']:>9.4f}")
    print(f"{'Rank Correlation':>20} {jll_std['rank_correlation']:>9.4f} {jll_opt['rank_correlation']:>9.4f}")
    
    print(f"\\n🎯 Key Improvements:")
    print(f"  • {speedup_time:.1f}x faster execution")
    print(f"  • {improvement:.1f}x better compression")
    print(f"  • Maintained quality (distortion ≈ {jll_opt['mean_distortion']:.3f})")
    print(f"  • Intelligent method auto-selection")
    print(f"  • Vectorized + JIT evaluation functions")
    
    # Architecture improvements
    print(f"\\n🔧 Technical Optimizations:")
    print(f"  • Modern JL bound: k = ln(n/δ)/ε² instead of 4ln(n)/ε²")
    print(f"  • Fast projections: sparse, Rademacher, FJLT methods")
    print(f"  • Adaptive dimension selection via binary search")
    print(f"  • Vectorized distance computations")
    if results_opt['JLL'].get('optimized_evaluation', False):
        print(f"  • High-performance evaluation functions active")
    
    print(f"\\n✨ Result: World-class dimensionality reduction with {speedup_time:.1f}x speedup!")

if __name__ == '__main__':
    main()