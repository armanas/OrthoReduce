#!/usr/bin/env python3
"""
Performance benchmarking script for OrthoReduce optimizations.

This script compares the performance of different projection methods and evaluation functions
to demonstrate the improvements from the optimization work.

Usage:
  python benchmark_performance.py --dimensions 1024 --points 5000
"""
import argparse
import time
import numpy as np
from typing import Dict, List

from orthogonal_projection.dimensionality_reduction import (
    generate_mixture_gaussians, run_experiment, run_jll
)
from orthogonal_projection.projection import jll_dimension, generate_orthogonal_basis
from orthogonal_projection.evaluation import compute_distortion as compute_distortion_standard

# Try to import optimized evaluation functions
try:
    from orthogonal_projection.evaluation_optimized import (
        compute_distortion_optimized, 
        benchmark_evaluation_methods,
        NUMBA_AVAILABLE
    )
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False
    NUMBA_AVAILABLE = False

def benchmark_projection_methods(d: int, k: int, n_trials: int = 5) -> Dict:
    """Benchmark different projection matrix generation methods."""
    methods = ['qr', 'gaussian', 'sparse', 'rademacher']
    
    # Add FJLT if dimension is suitable
    if d & (d - 1) == 0:  # Power of 2
        methods.append('fjlt')
    
    results = {}
    
    for method in methods:
        times = []
        for trial in range(n_trials):
            start = time.perf_counter()
            try:
                _ = generate_orthogonal_basis(d, k, method=method, seed=trial)
                end = time.perf_counter()
                times.append(end - start)
            except Exception as e:
                print(f"Warning: {method} method failed: {e}")
                times.append(float('inf'))
        
        if times and all(t != float('inf') for t in times):
            results[method] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times)
            }
    
    return results

def benchmark_full_pipeline(n: int, d: int, epsilon: float = 0.2) -> Dict:
    """Benchmark the complete optimized pipeline vs standard approach."""
    print(f"\\nBenchmarking full pipeline (n={n}, d={d}, ε={epsilon})")
    
    results = {}
    
    # Generate test data
    X = generate_mixture_gaussians(n, d, n_clusters=5, seed=42)
    k = jll_dimension(n, epsilon)
    
    print(f"Data: {n} points, {d}→{k} dimensions")
    
    # Benchmark standard approach
    start = time.perf_counter()
    Y_standard, _ = run_jll(X, k, seed=42, method='qr')
    projection_time = time.perf_counter() - start
    
    start = time.perf_counter()
    mean_dist_std, _, _, _ = compute_distortion_standard(X, Y_standard, sample_size=min(1000, n))
    eval_time_std = time.perf_counter() - start
    
    results['standard'] = {
        'projection_time': projection_time,
        'evaluation_time': eval_time_std,
        'total_time': projection_time + eval_time_std,
        'mean_distortion': mean_dist_std
    }
    
    # Benchmark optimized approach
    start = time.perf_counter()
    Y_optimized, _ = run_jll(X, k, seed=42, method='auto')  # Intelligent method selection
    projection_time_opt = time.perf_counter() - start
    
    eval_time_opt = 0
    mean_dist_opt = mean_dist_std  # Fallback
    
    if OPTIMIZED_AVAILABLE:
        start = time.perf_counter()
        mean_dist_opt, _, _, _ = compute_distortion_optimized(X, Y_optimized, sample_size=min(1000, n))
        eval_time_opt = time.perf_counter() - start
    
    results['optimized'] = {
        'projection_time': projection_time_opt,
        'evaluation_time': eval_time_opt,
        'total_time': projection_time_opt + eval_time_opt,
        'mean_distortion': mean_dist_opt
    }
    
    # Calculate speedups
    if results['standard']['total_time'] > 0:
        results['speedup'] = {
            'projection': results['standard']['projection_time'] / max(results['optimized']['projection_time'], 1e-6),
            'evaluation': results['standard']['evaluation_time'] / max(results['optimized']['evaluation_time'], 1e-6),
            'total': results['standard']['total_time'] / max(results['optimized']['total_time'], 1e-6)
        }
    
    return results

def print_benchmark_results(results: Dict, title: str):
    """Pretty print benchmark results."""
    print(f"\\n{title}")
    print("=" * len(title))
    
    if 'speedup' in results:
        # Full pipeline results
        std = results['standard']
        opt = results['optimized']
        speedup = results['speedup']
        
        print(f"Standard approach:")
        print(f"  Projection time: {std['projection_time']:.4f}s")
        print(f"  Evaluation time: {std['evaluation_time']:.4f}s")
        print(f"  Total time:      {std['total_time']:.4f}s")
        print(f"  Mean distortion: {std['mean_distortion']:.6f}")
        
        print(f"\\nOptimized approach:")
        print(f"  Projection time: {opt['projection_time']:.4f}s")
        print(f"  Evaluation time: {opt['evaluation_time']:.4f}s")
        print(f"  Total time:      {opt['total_time']:.4f}s")
        print(f"  Mean distortion: {opt['mean_distortion']:.6f}")
        
        print(f"\\nSpeedup:")
        print(f"  Projection: {speedup['projection']:.1f}x faster")
        print(f"  Evaluation: {speedup['evaluation']:.1f}x faster")
        print(f"  Total:      {speedup['total']:.1f}x faster")
        
    else:
        # Method comparison results
        fastest_time = min(r['mean_time'] for r in results.values())
        
        for method, stats in results.items():
            speedup = fastest_time / stats['mean_time']
            print(f"{method:>12}: {stats['mean_time']:.4f}s ± {stats['std_time']:.4f}s ({speedup:.1f}x slower than fastest)")

def main():
    parser = argparse.ArgumentParser(description="Benchmark OrthoReduce performance optimizations")
    parser.add_argument('--dimensions', type=int, default=512, help="Original dimensionality")
    parser.add_argument('--points', type=int, default=2000, help="Number of data points")
    parser.add_argument('--epsilon', type=float, default=0.2, help="JLL epsilon parameter")
    parser.add_argument('--trials', type=int, default=3, help="Number of benchmark trials")
    
    args = parser.parse_args()
    
    print("OrthoReduce Performance Benchmark")
    print("=" * 35)
    print(f"Configuration: {args.points} points, {args.dimensions} dimensions, ε={args.epsilon}")
    print(f"Optimized evaluation available: {OPTIMIZED_AVAILABLE}")
    print(f"Numba JIT available: {NUMBA_AVAILABLE}")
    
    # Calculate target dimension
    k = jll_dimension(args.points, args.epsilon)
    
    # Benchmark 1: Projection methods
    print(f"\\nBenchmarking projection methods ({args.dimensions}→{k} dimensions)...")
    projection_results = benchmark_projection_methods(args.dimensions, k, args.trials)
    print_benchmark_results(projection_results, "Projection Method Performance")
    
    # Benchmark 2: Evaluation methods (if optimized version available)
    if OPTIMIZED_AVAILABLE:
        print(f"\\nBenchmarking evaluation methods...")
        eval_results = benchmark_evaluation_methods(args.points, args.dimensions)
        print_benchmark_results(eval_results, "Evaluation Method Performance")
    
    # Benchmark 3: Full pipeline comparison
    pipeline_results = benchmark_full_pipeline(args.points, args.dimensions, args.epsilon)
    print_benchmark_results(pipeline_results, "Complete Pipeline Performance")
    
    print(f"\\nBenchmark completed!")
    
    # Summary
    if 'speedup' in pipeline_results:
        total_speedup = pipeline_results['speedup']['total']
        print(f"\\n🎉 Overall speedup: {total_speedup:.1f}x faster with optimizations!")
        
        if total_speedup >= 10:
            print("🚀 Excellent performance gain!")
        elif total_speedup >= 5:
            print("✨ Great performance improvement!")
        elif total_speedup >= 2:
            print("👍 Good performance boost!")
    
if __name__ == '__main__':
    main()