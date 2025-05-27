#!/usr/bin/env python3
import argparse
import logging
from orthogonal_projection.dimensionality_reduction import run_experiment

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dimensionality reduction experiments')
    parser.add_argument('--n', type=int, default=15000, help='Number of data points')
    parser.add_argument('--d', type=int, default=1200, help='Original dimensionality')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Desired maximum distortion')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sample_size', type=int, default=5000, help='Sample size for distortion computation')
    parser.add_argument('--use_poincare', action='store_true', help='Use Poincare embedding')
    parser.add_argument('--use_spherical', action='store_true', help='Use Spherical embedding')
    parser.add_argument('--use_elliptic', action='store_true', help='Use Elliptic embedding')
    args = parser.parse_args()

    logger.info("Starting dimensionality reduction experiment...")
    print(f"Starting experiment with parameters: n={args.n}, d={args.d}, epsilon={args.epsilon}, sample_size={args.sample_size}")

    res = run_experiment(
        n=args.n,
        d=args.d,
        epsilon=args.epsilon,
        seed=args.seed,
        sample_size=args.sample_size,
        use_poincare=args.use_poincare,
        use_spherical=args.use_spherical,
        use_elliptic=args.use_elliptic
    )

    print("\nExperiment results:")
    for name, metrics in res.items():
        print(f"=== {name} ===")
        for k, v in metrics.items():
            if isinstance(v, float) or hasattr(v, 'item'):  # Handle numpy types
                print(f"{k}: {float(v):.4f}")
            else:
                print(f"{k}: {v}")

    print("\nExperiment completed successfully.")
