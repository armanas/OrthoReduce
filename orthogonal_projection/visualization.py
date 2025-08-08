"""
Professional visualization module for OrthoReduce results.

This module provides publication-quality plots and reports for dimensionality reduction
analysis, optimized for macOS Retina displays and scientific publications.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

# Set up module logger
logger = logging.getLogger(__name__)

# Publication-quality styling configuration
STYLE_CONFIG = {
    'figure.dpi': 300,
    'figure.figsize': (12, 8),
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
}

# Scientific color palette (colorblind-safe)
METHOD_COLORS = {
    'JLL': '#66c2a5',      # Teal
    'PCA': '#fc8d62',      # Orange  
    'GAUSSIAN': '#8da0cb', # Purple
    'UMAP': '#e78ac3',     # Pink
    'POCS': '#a6d854',     # Green (JLL + Convex Hull)
    'POINCARE': '#ffd92f', # Yellow (Hyperbolic)
    'SPHERICAL': '#e5c494', # Tan (Unit Sphere)
    'SPARSE': '#b3de69',   # Light Green
    'RADEMACHER': '#fdb462' # Light Orange
}

# Quality indicators
QUALITY_COLORS = {
    'excellent': '#2ecc71',  # Green
    'good': '#f39c12',       # Orange
    'fair': '#e74c3c',       # Red
    'poor': '#95a5a6'        # Gray
}

def setup_plotting_style():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(STYLE_CONFIG)
    sns.set_palette("Set2")

def get_quality_grade(distortion: float, correlation: float) -> str:
    """Assign quality grade based on metrics."""
    if distortion < 0.1 and correlation > 0.95:
        return 'excellent'
    elif distortion < 0.2 and correlation > 0.9:
        return 'good'
    elif distortion < 0.4 and correlation > 0.8:
        return 'fair'
    else:
        return 'poor'

class OrthoReduceVisualizer:
    """Professional visualization system for OrthoReduce results."""
    
    def __init__(self, output_dir: str = "experiment_results"):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.individual_dir = self.plots_dir / "individual"
        
        # Create directories
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.individual_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup plotting
        setup_plotting_style()
        
        logger.info(f"Visualizer initialized with output dir: {self.output_dir}")
    
    def create_performance_dashboard(self, results: Dict[str, Any]) -> plt.Figure:
        """Create comprehensive performance dashboard."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("OrthoReduce Performance Dashboard", fontsize=16, fontweight='bold')
        
        # Extract method data
        methods = [k for k in results.keys() if k != '_metadata']
        runtimes = [results[m]['runtime'] for m in methods]
        compressions = [results[m]['compression_ratio'] for m in methods]
        distortions = [results[m]['mean_distortion'] for m in methods]
        correlations = [results[m]['rank_correlation'] for m in methods]
        
        colors = [METHOD_COLORS.get(m, '#gray') for m in methods]
        
        # 1. Runtime Performance (Log Scale)
        bars1 = ax1.bar(methods, runtimes, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_yscale('log')
        ax1.set_ylabel('Runtime (seconds)', fontweight='bold')
        ax1.set_title('Runtime Performance (Log Scale)', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add runtime values on bars
        for bar, runtime in zip(bars1, runtimes):
            height = bar.get_height()
            ax1.annotate(f'{runtime:.4f}s', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
        
        # 2. Compression Effectiveness
        bars2 = ax2.bar(methods, compressions, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Compression Ratio (x)', fontweight='bold')
        ax2.set_title('Compression Effectiveness', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add compression values
        for bar, comp in zip(bars2, compressions):
            height = bar.get_height()
            ax2.annotate(f'{comp:.1f}x', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
        
        # 3. Quality vs Speed Trade-off
        scatter = ax3.scatter(runtimes, distortions, c=correlations, s=150, 
                            alpha=0.8, cmap='viridis_r', edgecolors='black', linewidth=1)
        ax3.set_xscale('log')
        ax3.set_xlabel('Runtime (seconds)', fontweight='bold')
        ax3.set_ylabel('Mean Distortion', fontweight='bold') 
        ax3.set_title('Quality vs Speed Trade-off', fontweight='bold')
        
        # Add method labels
        for i, method in enumerate(methods):
            ax3.annotate(method, (runtimes[i], distortions[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontweight='bold', fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Rank Correlation', fontweight='bold')
        
        # 4. Overall Quality Assessment
        grades = [get_quality_grade(d, c) for d, c in zip(distortions, correlations)]
        grade_colors = [QUALITY_COLORS[g] for g in grades]
        
        # Create quality score (lower distortion + higher correlation = better)
        quality_scores = [c / (d + 0.01) for d, c in zip(distortions, correlations)]
        bars4 = ax4.bar(methods, quality_scores, color=grade_colors, alpha=0.8, 
                       edgecolor='black', linewidth=0.5)
        ax4.set_ylabel('Quality Score (Correlation/Distortion)', fontweight='bold')
        ax4.set_title('Overall Quality Assessment', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add quality grades
        for bar, grade in zip(bars4, grades):
            height = bar.get_height()
            ax4.annotate(grade.upper(), xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='white')
        
        plt.tight_layout()
        return fig
    
    def create_method_comparison(self, results: Dict[str, Any]) -> plt.Figure:
        """Create detailed method comparison analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("Method Comparison Analysis", fontsize=16, fontweight='bold')
        
        # Extract data
        methods = [k for k in results.keys() if k != '_metadata']
        metrics = ['mean_distortion', 'max_distortion', 'rank_correlation', 'runtime', 'compression_ratio']
        
        # Create comparison dataframe
        data = []
        for method in methods:
            row = {'Method': method}
            for metric in metrics:
                if metric in results[method]:
                    row[metric] = results[method][metric]
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 1. Distortion vs Correlation Analysis
        scatter = ax1.scatter(df['mean_distortion'], df['rank_correlation'], 
                            s=200, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Color points by compression ratio
        for i, (method, row) in enumerate(df.iterrows()):
            color = METHOD_COLORS.get(row['Method'], '#gray')
            ax1.scatter(row['mean_distortion'], row['rank_correlation'], 
                       s=200, c=[color], alpha=0.8, edgecolors='black', linewidth=1)
            ax1.annotate(row['Method'], (row['mean_distortion'], row['rank_correlation']),
                        xytext=(5, 5), textcoords='offset points', fontweight='bold', fontsize=10)
        
        ax1.set_xlabel('Mean Distortion (lower is better)', fontweight='bold')
        ax1.set_ylabel('Rank Correlation (higher is better)', fontweight='bold')
        ax1.set_title('Quality Analysis: Distortion vs Correlation', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add quality regions
        ax1.axhspan(0.95, 1.0, alpha=0.1, color='green', label='Excellent (>0.95)')
        ax1.axhspan(0.9, 0.95, alpha=0.1, color='orange', label='Good (0.9-0.95)')
        ax1.axhspan(0.8, 0.9, alpha=0.1, color='red', label='Fair (0.8-0.9)')
        ax1.legend(loc='lower left')
        
        # 2. Performance Radar Chart
        # Normalize metrics for radar chart (0-1 scale)
        normalized_data = df.copy()
        normalized_data['mean_distortion'] = 1 - (df['mean_distortion'] / df['mean_distortion'].max())  # Invert
        normalized_data['rank_correlation'] = df['rank_correlation'] / df['rank_correlation'].max()
        normalized_data['runtime'] = 1 - (df['runtime'] / df['runtime'].max())  # Invert (faster is better)
        normalized_data['compression_ratio'] = df['compression_ratio'] / df['compression_ratio'].max()
        
        # Create radar chart
        categories = ['Low Distortion', 'High Correlation', 'Fast Runtime', 'High Compression']
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Convert to polar plot
        ax2.remove()
        ax2 = fig.add_subplot(122, projection='polar')
        
        # Configure polar plot
        ax2.set_theta_offset(np.pi / 2)
        ax2.set_theta_direction(-1)
        ax2.set_thetagrids(np.degrees(angles[:-1]), categories)
        
        for i, (_, row) in enumerate(normalized_data.iterrows()):
            values = [row['mean_distortion'], row['rank_correlation'], 
                     row['runtime'], row['compression_ratio']]
            values += values[:1]  # Complete the circle
            
            color = METHOD_COLORS.get(row['Method'], '#gray')
            ax2.plot(angles, values, 'o-', linewidth=2, label=row['Method'], 
                    color=color, alpha=0.8)
            ax2.fill(angles, values, alpha=0.2, color=color)
        
        ax2.set_ylim(0, 1)
        ax2.set_title('Multi-Metric Performance Radar', fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def create_results_table(self, results: Dict[str, Any]) -> plt.Figure:
        """Create professional results summary table."""
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle("Detailed Results Summary", fontsize=16, fontweight='bold')
        
        # Extract data for table
        methods = [k for k in results.keys() if k != '_metadata']
        table_data = []
        
        for method in methods:
            metrics = results[method]
            grade = get_quality_grade(metrics['mean_distortion'], metrics['rank_correlation'])
            
            row = [
                method,
                f"{metrics['runtime']:.4f}s",
                f"{metrics['compression_ratio']:.2f}x",
                f"{metrics['mean_distortion']:.4f}",
                f"{metrics['max_distortion']:.4f}",
                f"{metrics['rank_correlation']:.4f}",
                grade.upper()
            ]
            table_data.append(row)
        
        # Sort by quality score
        table_data.sort(key=lambda x: float(x[5]), reverse=True)
        
        columns = ['Method', 'Runtime', 'Compression', 'Mean Distortion', 
                  'Max Distortion', 'Rank Correlation', 'Quality Grade']
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=columns, 
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Header styling
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.15)
        
        # Row styling with alternating colors
        for i in range(1, len(table_data) + 1):
            row_color = '#f8f9fa' if i % 2 == 0 else 'white'
            for j in range(len(columns)):
                table[(i, j)].set_facecolor(row_color)
                table[(i, j)].set_height(0.12)
                
                # Color-code quality grades
                if j == 6:  # Quality grade column
                    grade = table_data[i-1][6]
                    if grade == 'EXCELLENT':
                        table[(i, j)].set_facecolor('#2ecc71')
                        table[(i, j)].set_text_props(weight='bold', color='white')
                    elif grade == 'GOOD':
                        table[(i, j)].set_facecolor('#f39c12')  
                        table[(i, j)].set_text_props(weight='bold', color='white')
                    elif grade == 'FAIR':
                        table[(i, j)].set_facecolor('#e74c3c')
                        table[(i, j)].set_text_props(weight='bold', color='white')
        
        ax.axis('off')
        
        # Add metadata information
        if '_metadata' in results:
            meta = results['_metadata']
            info_text = f"""Experiment Configuration:
Dataset: {meta['n']} points, {meta['d']} dimensions → {meta['k']} dimensions
Compression: {meta['d']/meta['k']:.2f}x reduction
Epsilon: {meta['epsilon']}
Intrinsic Dimension: {meta['intrinsic_dimension']}
Optimization: {'Adaptive' if meta.get('adaptive_used', False) else 'Theoretical'}"""
            
            ax.text(0, -0.15, info_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive PDF report with all visualizations."""
        report_path = self.plots_dir / "comprehensive_report.pdf"
        
        logger.info(f"Generating comprehensive report: {report_path}")
        
        with PdfPages(report_path) as pdf:
            # Page 1: Performance Dashboard
            fig1 = self.create_performance_dashboard(results)
            pdf.savefig(fig1, bbox_inches='tight', dpi=300)
            plt.close(fig1)
            
            # Page 2: Method Comparison
            fig2 = self.create_method_comparison(results)
            pdf.savefig(fig2, bbox_inches='tight', dpi=300)
            plt.close(fig2)
            
            # Page 3: Results Table
            fig3 = self.create_results_table(results)
            pdf.savefig(fig3, bbox_inches='tight', dpi=300)
            plt.close(fig3)
        
        logger.info(f"✅ Comprehensive report saved: {report_path}")
        return str(report_path)
    
    def save_individual_plots(self, results: Dict[str, Any]) -> List[str]:
        """Save individual high-quality plots."""
        saved_files = []
        
        # Performance Dashboard
        fig1 = self.create_performance_dashboard(results)
        dashboard_path = self.individual_dir / "performance_dashboard.png"
        fig1.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig1)
        saved_files.append(str(dashboard_path))
        
        # Method Comparison
        fig2 = self.create_method_comparison(results)
        comparison_path = self.individual_dir / "method_comparison.png"
        fig2.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        saved_files.append(str(comparison_path))
        
        # Results Table
        fig3 = self.create_results_table(results)
        table_path = self.individual_dir / "results_table.png"
        fig3.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig3)
        saved_files.append(str(table_path))
        
        logger.info(f"✅ Individual plots saved: {len(saved_files)} files")
        return saved_files
    
    def create_complete_visualization(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Create complete visualization package."""
        logger.info("Creating complete visualization package...")
        
        output_files = {}
        
        # Generate comprehensive PDF report
        pdf_path = self.generate_comprehensive_report(results)
        output_files['comprehensive_report'] = pdf_path
        
        # Save individual plots
        individual_plots = self.save_individual_plots(results)
        output_files['individual_plots'] = individual_plots
        
        # Create main dashboard
        main_dashboard = self.create_performance_dashboard(results)
        main_path = self.plots_dir / "main_dashboard.png"
        main_dashboard.savefig(main_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(main_dashboard)
        output_files['main_dashboard'] = str(main_path)
        
        logger.info("✅ Complete visualization package created!")
        logger.info(f"📁 Output directory: {self.plots_dir}")
        logger.info(f"📊 Main dashboard: {main_path}")
        logger.info(f"📄 PDF report: {pdf_path}")
        logger.info(f"🖼️ Individual plots: {len(individual_plots)} files")
        
        return output_files
    
    def create_advanced_visualization_suite(self, results: Dict[str, Any], 
                                           embeddings: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Create comprehensive visualization suite using advanced plotting capabilities.
        
        This method integrates the new AdvancedPlotter with the existing visualization system
        to provide enhanced plots with better styling, specialized embeddings, and interactive features.
        
        Parameters
        ----------
        results : dict
            Experiment results from run_experiment()
        embeddings : dict or None
            Optional embeddings for enhanced visualization
            
        Returns
        -------
        dict
            Dictionary mapping visualization types to file paths
        """
        try:
            # Import advanced plotting (lazy import to avoid circular dependencies)
            from .advanced_plotting import (
                AdvancedPlotter, InteractivePlotter, create_evaluation_report
            )
            
            logger.info("Creating advanced visualization suite...")
            
            # Create advanced evaluation report
            advanced_output_dir = self.plots_dir / "advanced"
            generated_files = create_evaluation_report(
                results=results,
                embeddings=embeddings,
                output_dir=str(advanced_output_dir),
                include_interactive=True
            )
            
            # Update paths relative to main plots directory
            for key, path in generated_files.items():
                # Make paths relative to main output directory for consistency
                rel_path = Path(path).relative_to(self.output_dir)
                generated_files[key] = str(self.output_dir / rel_path)
            
            logger.info(f"✅ Advanced visualization suite created with {len(generated_files)} components")
            
            return generated_files
            
        except ImportError as e:
            logger.warning(f"Advanced plotting not available: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to create advanced visualization suite: {e}")
            return {}