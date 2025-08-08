#!/usr/bin/env python3

from orthogonal_projection.dimensionality_reduction_old import run_experiment as run_old

# Test the original implementation with all available methods
results = run_old(
    n=500, 
    d=50, 
    epsilon=0.2, 
    seed=42, 
    sample_size=1000,
    use_convex=True,           # Enable convex method
    use_poincare=True,         # Enable Poincaré 
    use_spherical=True,        # Enable spherical
    use_elliptic=False         # Skip elliptic for speed
)

print("Original implementation results (all methods):")
print("=" * 60)
for method, metrics in results.items():
    if method != '_metadata':
        if 'mean_distortion' in metrics and 'runtime' in metrics:
            correlation = metrics.get('rank_correlation', 'N/A')
            print(f'{method:>12}: distortion={metrics["mean_distortion"]:.4f}, runtime={metrics["runtime"]:.6f}s, correlation={correlation}')