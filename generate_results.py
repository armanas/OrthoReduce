#!/usr/bin/env python3
"""
Generate and save OrthoReduce experiment results with professional visualizations.

This script runs comprehensive experiments and automatically creates publication-quality
graphs, charts, and PDF reports for OrthoReduce dimensionality reduction analysis.
"""

import json
import pandas as pd
from orthogonal_projection.dimensionality_reduction import run_experiment
from orthogonal_projection.visualization import OrthoReduceVisualizer

def main():
    """Generate experiment results and save to multiple formats."""
    print("🔬 Generating OrthoReduce Experiment Results...")
    
    # Run comprehensive experiment with ALL 6 methods - fixed implementations
    results = run_experiment(
        n=500, 
        d=50, 
        epsilon=0.2, 
        methods=['jll', 'pca', 'gaussian', 'pocs', 'poincare', 'spherical'],
        use_adaptive=True,
        use_optimized_eval=True
    )
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs('experiment_results', exist_ok=True)
    
    # Save results as JSON
    json_file = 'experiment_results/results.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ JSON results saved to: {json_file}")
    
    # Save results as CSV
    csv_file = 'experiment_results/results.csv' 
    # Convert to DataFrame (excluding metadata)
    df_data = []
    for method, metrics in results.items():
        if method != '_metadata':
            row = {'method': method}
            row.update(metrics)
            df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_file, index=False)
    print(f"✅ CSV results saved to: {csv_file}")
    
    # Save metadata separately
    metadata_file = 'experiment_results/metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(results.get('_metadata', {}), f, indent=2)
    print(f"✅ Metadata saved to: {metadata_file}")
    
    # Create a summary report
    summary_file = 'experiment_results/summary.txt'
    with open(summary_file, 'w') as f:
        f.write("OrthoReduce Experiment Summary\n")
        f.write("=" * 40 + "\n\n")
        
        # Write metadata
        if '_metadata' in results:
            meta = results['_metadata']
            f.write(f"Dataset: {meta['n']} points, {meta['d']} dimensions\n")
            f.write(f"Target dimension: {meta['k']}\n") 
            f.write(f"Compression ratio: {meta['d']/meta['k']:.2f}x\n")
            f.write(f"Epsilon: {meta['epsilon']}\n")
            f.write(f"Intrinsic dimension: {meta['intrinsic_dimension']}\n\n")
        
        # Write results
        f.write("Method Results:\n")
        f.write("-" * 20 + "\n")
        for method, metrics in results.items():
            if method != '_metadata':
                f.write(f"\n{method}:\n")
                f.write(f"  Mean distortion: {metrics['mean_distortion']:.4f}\n")
                f.write(f"  Max distortion: {metrics['max_distortion']:.4f}\n")
                f.write(f"  Rank correlation: {metrics['rank_correlation']:.4f}\n")
                f.write(f"  Runtime: {metrics['runtime']:.4f}s\n")
                f.write(f"  Compression ratio: {metrics['compression_ratio']:.2f}x\n")
    
    print(f"✅ Summary report saved to: {summary_file}")
    
    # Create professional visualizations
    print(f"\n📊 Generating Professional Visualizations...")
    visualizer = OrthoReduceVisualizer('experiment_results')
    
    try:
        # Generate complete visualization package
        viz_files = visualizer.create_complete_visualization(results)
        
        print(f"✅ Visualizations created successfully!")
        print(f"📄 Comprehensive PDF report: {viz_files['comprehensive_report']}")
        print(f"📊 Main dashboard: {viz_files['main_dashboard']}")
        print(f"🖼️ Individual plots: {len(viz_files['individual_plots'])} files")
        
    except Exception as e:
        print(f"⚠️ Visualization generation failed: {e}")
        print("Results data files were still saved successfully.")
    
    # Print quick preview
    print(f"\n📊 Quick Results Preview:")
    print("=" * 50)
    for method, metrics in results.items():
        if method != '_metadata':
            print(f"{method:>10}: distortion={metrics['mean_distortion']:.4f}, "
                  f"runtime={metrics['runtime']:.4f}s, "
                  f"compression={metrics['compression_ratio']:.2f}x")
    
    print(f"\n📁 All results saved in: experiment_results/")
    print("Files created:")
    print("  • results.json       - Full results in JSON format")  
    print("  • results.csv        - Results in CSV format for Excel")
    print("  • metadata.json      - Experiment configuration")
    print("  • summary.txt        - Human-readable summary")
    print("  • plots/             - Professional visualization directory")
    print("    ├── comprehensive_report.pdf - Multi-page publication report")
    print("    ├── main_dashboard.png       - Performance overview")
    print("    └── individual/              - Separate high-res plots")

if __name__ == '__main__':
    main()