#!/usr/bin/env python3
"""
ms_pipeline_simple.py - Simplified MS data processing pipeline

A cleaner, more maintainable version of the MS pipeline focusing on core functionality:
1. Parse mzML files  
2. Apply dimensionality reduction
3. Evaluate results
4. Save metrics and basic plots

Usage:
  python ms_pipeline_simple.py --mzml data/sample.mzML --epsilon 0.3 --max_spectra 1000
"""
import os
import argparse
import json
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA

from orthogonal_projection.ms_data import parse_mzml_build_matrix
from orthogonal_projection.projection import jll_dimension
# Import optimized evaluation functions if available
try:
    from orthogonal_projection.evaluation_optimized import compute_distortion_optimized as compute_distortion
    from orthogonal_projection.evaluation_optimized import rank_correlation_optimized as rank_correlation
    OPTIMIZED_EVALUATION = True
except ImportError:
    from orthogonal_projection.evaluation import compute_distortion, rank_correlation
    OPTIMIZED_EVALUATION = False
from orthogonal_projection.convex_optimized import project_onto_convex_hull_qp, project_onto_convex_hull_enhanced
from orthogonal_projection.fingerprinting import match_frames_to_peptides

def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def load_peptide_labels(labels_csv: str, n_frames: int) -> Optional[Dict]:
    """Load peptide labels from CSV file."""
    if not labels_csv or not os.path.exists(labels_csv):
        return None
    
    try:
        df = pd.read_csv(labels_csv)
        required_columns = {"frame_index", "peptide_id", "peptide_rt"}
        
        if not required_columns.issubset(df.columns):
            print(f"Warning: CSV missing required columns {required_columns}")
            return None
        
        # Create peptide mapping
        peptide_ids = [f"UNKNOWN_{i}" for i in range(n_frames)]
        for _, row in df.iterrows():
            idx = int(row["frame_index"])
            if 0 <= idx < n_frames:
                peptide_ids[idx] = row["peptide_id"]
        
        rt_by_peptide = df.groupby("peptide_id")["peptide_rt"].median().to_dict()
        
        return {
            "peptide_ids": peptide_ids,
            "rt_by_peptide": rt_by_peptide,
        }
    except Exception as e:
        print(f"Error loading labels: {e}")
        return None

def create_plots(results_dir: str, X: np.ndarray, Y: np.ndarray, rts: np.ndarray, 
                peptide_data: Optional[Dict] = None):
    """Create basic visualization plots."""
    plots_dir = os.path.join(results_dir, "plots")
    ensure_dir(plots_dir)
    
    try:
        # 2D scatter plot of projections
        if Y.shape[1] >= 2:
            Z = Y[:, :2]  # Use first 2 dimensions
        else:
            # Use PCA to get 2D visualization
            pca = PCA(n_components=2)
            Z = pca.fit_transform(Y)
        
        plt.figure(figsize=(8, 6))
        if peptide_data:
            # Color by retention time if available
            plt.scatter(Z[:, 0], Z[:, 1], c=rts, cmap='viridis', s=20, alpha=0.7)
            plt.colorbar(label='Retention Time (min)')
        else:
            plt.scatter(Z[:, 0], Z[:, 1], s=20, alpha=0.7)
        
        plt.title("MS Data Projection (2D)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "projection_scatter.png"), dpi=150)
        plt.close()
        
        print(f"Plots saved to {plots_dir}")
        
    except Exception as e:
        print(f"Warning: Plot generation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Simplified MS data pipeline")
    parser.add_argument("--mzml", required=True, help="Path to mzML file")
    parser.add_argument("--epsilon", type=float, default=0.3, help="JLL epsilon parameter")
    parser.add_argument("--max_spectra", type=int, default=1000, help="Max spectra to process")
    parser.add_argument("--bin_size", type=float, default=0.1, help="m/z bin size")
    parser.add_argument("--results", default="results", help="Output directory")
    parser.add_argument("--labels_csv", help="CSV file with peptide labels")
    parser.add_argument("--use_convex", action='store_true', help="Apply convex hull projection")
    
    # Enhanced convex hull projection parameters
    parser.add_argument("--convex_ridge_lambda", type=float, default=1e-6, 
                       help="Ridge regularization for convex hull (1e-6 to 1e-3)")
    parser.add_argument("--convex_solver_tol", type=float, default=1e-6,
                       help="Solver tolerance for convex optimization")
    parser.add_argument("--convex_objective", choices=['quadratic', 'huber', 'epsilon_insensitive'], 
                       default='quadratic', help="Objective function type")
    parser.add_argument("--convex_use_float64", action='store_true', 
                       help="Use float64 for high-precision convex optimization")
    parser.add_argument("--convex_normalization", choices=['none', 'l2', 'unit_sphere'], 
                       default='none', help="Vertex normalization method")
    parser.add_argument("--convex_solver_mode", choices=['strict', 'balanced', 'loose'], 
                       default='balanced', help="Solver tolerance mode")
    
    args = parser.parse_args()
    
    print("MS Pipeline - Simplified Version")
    print("=" * 40)
    
    # Ensure output directory exists
    ensure_dir(args.results)
    
    # Phase 1: Parse mzML data
    print(f"Loading mzML file: {args.mzml}")
    try:
        X, rts, mz_bins, meta = parse_mzml_build_matrix(
            args.mzml, 
            bin_size=args.bin_size, 
            max_spectra=args.max_spectra
        )
        print(f"Loaded {X.shape[0]} spectra with {X.shape[1]} m/z bins")
    except Exception as e:
        print(f"Error loading mzML: {e}")
        return 1
    
    # Phase 2: Apply dimensionality reduction
    n, d = X.shape
    k = min(jll_dimension(n, args.epsilon), d)
    print(f"Applying dimensionality reduction: {d} -> {k} dimensions")
    
    try:
        # Apply Gaussian random projection
        grp = GaussianRandomProjection(n_components=k, random_state=42)
        Y = grp.fit_transform(X)
        
        # Optional convex hull projection
        if args.use_convex:
            print(f"Applying enhanced convex hull projection (mode: {args.convex_solver_mode}, "
                  f"objective: {args.convex_objective}, ridge_λ: {args.convex_ridge_lambda:.1e})...")
            Y, alphas, vertices = project_onto_convex_hull_enhanced(
                Y,
                ridge_lambda=args.convex_ridge_lambda,
                solver_tol=args.convex_solver_tol,
                objective_type=args.convex_objective,
                use_float64=args.convex_use_float64,
                candidate_normalization=args.convex_normalization,
                solver_mode=args.convex_solver_mode
            )
        
    except Exception as e:
        print(f"Error in dimensionality reduction: {e}")
        return 1
    
    # Phase 3: Evaluate results with optimized functions
    eval_method = "optimized" if OPTIMIZED_EVALUATION else "standard"
    print(f"Evaluating projection quality ({eval_method} evaluation)...")
    mean_dist, max_dist, _, _ = compute_distortion(X, Y, sample_size=min(1000, n))
    rank_corr = rank_correlation(X, Y, sample_size=min(1000, n))
    
    # Phase 4: Load peptide labels and compute accuracy if available
    peptide_data = load_peptide_labels(args.labels_csv, n)
    accuracy = None
    time_accuracy = None
    
    if peptide_data:
        print("Computing peptide matching accuracy...")
        try:
            match_results = match_frames_to_peptides(
                Y, 
                peptide_data["peptide_ids"],
                rts,
                peptide_data["rt_by_peptide"],
                rt_tolerance=0.5
            )
            accuracy = match_results["accuracy"]
            time_accuracy = match_results["time_only_accuracy"]
        except Exception as e:
            print(f"Error computing accuracy: {e}")
    
    # Phase 5: Save results
    metrics = {
        "n_spectra": int(n),
        "n_bins": int(d), 
        "target_dimension": int(k),
        "epsilon": float(args.epsilon),
        "mean_distortion": float(mean_dist),
        "max_distortion": float(max_dist),
        "rank_correlation": float(rank_corr),
        "accuracy": float(accuracy) if accuracy is not None else None,
        "time_accuracy": float(time_accuracy) if time_accuracy is not None else None,
        "used_convex_hull": args.use_convex
    }
    
    # Save metrics
    metrics_file = os.path.join(args.results, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save as CSV too
    pd.DataFrame([metrics]).to_csv(
        os.path.join(args.results, "metrics.csv"), 
        index=False
    )
    
    # Create plots
    create_plots(args.results, X, Y, rts, peptide_data)
    
    # Print summary
    print(f"\nResults Summary:")
    print(f"Mean distortion: {mean_dist:.4f}")
    print(f"Max distortion: {max_dist:.4f}")
    print(f"Rank correlation: {rank_corr:.4f}")
    if accuracy is not None:
        print(f"Peptide accuracy: {accuracy:.4f}")
        print(f"Time-based accuracy: {time_accuracy:.4f}")
    
    print(f"\nResults saved to: {args.results}/")
    print("Done!")
    
    return 0

if __name__ == "__main__":
    exit(main())