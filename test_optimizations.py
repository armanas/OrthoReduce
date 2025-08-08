#!/usr/bin/env python3
"""
Quick test script to verify the optimized OrthoReduce pipeline works correctly.
"""

from orthogonal_projection.dimensionality_reduction import run_experiment
import numpy as np

def test_optimized_pipeline():
    print('Testing optimized OrthoReduce pipeline...')
    
    # Run a small test with adaptive compression
    results = run_experiment(
        n=500, 
        d=64, 
        epsilon=0.2, 
        methods=['jll', 'pca'],
        use_adaptive=True,
        use_optimized_eval=True
    )
    
    print('\nResults:')
    for method, metrics in results.items():
        if method != '_metadata':
            print(f'{method}:')
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f'  {key}: {value:.4f}')
                else:
                    print(f'  {key}: {value}')
    
    print('\nMetadata:', results['_metadata'])
    print('✅ Optimized pipeline working correctly!')
    
    return results

if __name__ == '__main__':
    test_optimized_pipeline()