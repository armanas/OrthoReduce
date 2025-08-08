"""
Enhanced OrthoReduce Dashboard with Advanced Features

This module extends the basic dashboard with advanced analysis capabilities,
parameter sensitivity analysis, and comprehensive export functionality.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import pickle
import io
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import zipfile
import tempfile

try:
    from .dashboard_utils import (
        ExperimentDatabase, DataProcessor, EmbeddingAnalyzer, 
        CacheManager, load_and_process_experiments
    )
    from .embedding_viewer import InteractiveEmbeddingExplorer
    from .visualization import METHOD_COLORS, QUALITY_COLORS
    from .results_aggregator import ResultsAggregator
except ImportError:
    from dashboard_utils import (
        ExperimentDatabase, DataProcessor, EmbeddingAnalyzer,
        CacheManager, load_and_process_experiments
    )
    from embedding_viewer import InteractiveEmbeddingExplorer
    from visualization import METHOD_COLORS, QUALITY_COLORS
    from results_aggregator import ResultsAggregator

logger = logging.getLogger(__name__)


class AdvancedAnalytics:
    """Advanced analytics and insights generation."""
    
    @staticmethod
    def generate_insights(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate automated insights from experiment results."""
        insights = []
        methods = [k for k in results.keys() if k != '_metadata']
        
        if not methods:
            return insights
        
        # Best overall method
        quality_scores = DataProcessor.compute_quality_scores(results)
        best_method = max(quality_scores.items(), key=lambda x: x[1])
        insights.append({
            'type': 'success',
            'title': 'Best Overall Method',
            'content': f'{best_method[0]} achieved the highest quality score ({best_method[1]:.3f})',
            'priority': 'high'
        })
        
        # Runtime analysis
        runtimes = {m: results[m]['runtime'] for m in methods}
        fastest_method = min(runtimes.items(), key=lambda x: x[1])
        slowest_method = max(runtimes.items(), key=lambda x: x[1])
        
        if slowest_method[1] / fastest_method[1] > 100:
            insights.append({
                'type': 'warning',
                'title': 'Runtime Disparity',
                'content': f'{slowest_method[0]} is {slowest_method[1]/fastest_method[1]:.1f}x slower than {fastest_method[0]}',
                'priority': 'medium'
            })
        
        # Quality vs speed trade-off
        correlations = {m: results[m]['rank_correlation'] for m in methods}
        best_correlation = max(correlations.items(), key=lambda x: x[1])
        
        if best_correlation[0] != fastest_method[0]:
            insights.append({
                'type': 'info',
                'title': 'Quality-Speed Trade-off',
                'content': f'{best_correlation[0]} has best correlation ({best_correlation[1]:.3f}) but {fastest_method[0]} is fastest ({fastest_method[1]:.4f}s)',
                'priority': 'medium'
            })
        
        # Pareto optimal methods
        pareto_methods = DataProcessor.get_pareto_frontier(results)
        if len(pareto_methods) > 1:
            insights.append({
                'type': 'info',
                'title': 'Pareto Optimal Methods',
                'content': f'{len(pareto_methods)} methods are Pareto optimal: {", ".join(pareto_methods)}',
                'priority': 'low'
            })
        
        # Compression efficiency
        compressions = {m: results[m].get('compression_ratio', 1) for m in methods}
        best_compression = max(compressions.items(), key=lambda x: x[1])
        
        if best_compression[1] > 2.0:
            insights.append({
                'type': 'success',
                'title': 'High Compression',
                'content': f'{best_compression[0]} achieves {best_compression[1]:.1f}x compression while maintaining quality',
                'priority': 'medium'
            })
        
        return sorted(insights, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)
    
    @staticmethod
    def analyze_convergence_patterns(aggregated_results: Any) -> Dict[str, Any]:
        """Analyze convergence patterns across experiments."""
        convergence_analysis = {}
        
        if hasattr(aggregated_results, 'performance_trends'):
            convergence_data = aggregated_results.performance_trends.get('convergence_analysis', {})
            
            for method, data in convergence_data.items():
                if 'loss_history' in method:
                    convergence_analysis[method] = {
                        'converges': data.get('mean_convergence_rate', 0) > 0.01,
                        'convergence_rate': data.get('mean_convergence_rate', 0),
                        'final_loss': data.get('mean_final_loss', float('inf')),
                        'stability': 1.0 / (1.0 + data.get('std_final_loss', 0))
                    }
        
        return convergence_analysis
    
    @staticmethod
    def generate_recommendations(results: Dict[str, Any], 
                               convergence_analysis: Dict[str, Any]) -> List[str]:
        """Generate method recommendations based on analysis."""
        recommendations = []
        
        # Get method performance
        quality_scores = DataProcessor.compute_quality_scores(results)
        methods = list(quality_scores.keys())
        
        if not methods:
            return ["No methods to analyze"]
        
        # Sort methods by quality
        sorted_methods = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Primary recommendation
        best_method = sorted_methods[0][0]
        recommendations.append(
            f"🎯 **Primary Recommendation**: Use {best_method} for best overall quality "
            f"(score: {sorted_methods[0][1]:.3f})"
        )
        
        # Speed consideration
        runtimes = {m: results[m]['runtime'] for m in methods}
        fastest_method = min(runtimes.items(), key=lambda x: x[1])[0]
        
        if fastest_method != best_method:
            recommendations.append(
                f"⚡ **Speed Alternative**: Consider {fastest_method} for time-critical applications "
                f"(runtime: {runtimes[fastest_method]:.4f}s)"
            )
        
        # Balanced recommendation
        normalized = DataProcessor.normalize_metrics(results)
        balanced_scores = {}
        
        for method in methods:
            norm_metrics = normalized[method]
            balanced_score = (
                0.4 * norm_metrics.get('rank_correlation', 0) +
                0.3 * norm_metrics.get('mean_distortion', 0) +
                0.2 * norm_metrics.get('runtime', 0) +
                0.1 * norm_metrics.get('compression_ratio', 0)
            )
            balanced_scores[method] = balanced_score
        
        balanced_method = max(balanced_scores.items(), key=lambda x: x[1])[0]
        if balanced_method not in [best_method, fastest_method]:
            recommendations.append(
                f"⚖️ **Balanced Choice**: {balanced_method} offers good trade-offs across all metrics"
            )
        
        # Parameter tuning suggestions
        if convergence_analysis:
            poorly_converged = [
                method for method, analysis in convergence_analysis.items()
                if not analysis.get('converges', True)
            ]
            
            if poorly_converged:
                recommendations.append(
                    f"🔧 **Parameter Tuning**: Consider adjusting hyperparameters for: "
                    f"{', '.join(poorly_converged)}"
                )
        
        return recommendations


class ExportManager:
    """Handles all export functionality for the dashboard."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "orthoreduce_exports"
        self.temp_dir.mkdir(exist_ok=True)
    
    def export_results_data(self, results: Dict[str, Any], format: str = 'json') -> bytes:
        """Export results data in specified format."""
        if format.lower() == 'json':
            return json.dumps(results, indent=2, default=str).encode()
        
        elif format.lower() == 'csv':
            # Convert to DataFrame for CSV export
            methods = [k for k in results.keys() if k != '_metadata']
            data_rows = []
            
            for method in methods:
                metrics = results[method]
                row = {'Method': method}
                row.update(metrics)
                data_rows.append(row)
            
            df = pd.DataFrame(data_rows)
            return df.to_csv(index=False).encode()
        
        elif format.lower() == 'excel':
            # Create Excel file with multiple sheets
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Summary sheet
                methods = [k for k in results.keys() if k != '_metadata']
                summary_data = []
                
                for method in methods:
                    metrics = results[method]
                    summary_data.append({
                        'Method': method,
                        'Rank Correlation': metrics.get('rank_correlation', 0),
                        'Mean Distortion': metrics.get('mean_distortion', 0),
                        'Runtime (s)': metrics.get('runtime', 0),
                        'Memory (GB)': metrics.get('memory_usage', 0),
                        'Compression Ratio': metrics.get('compression_ratio', 1)
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Detailed metrics sheet
                detailed_data = []
                for method in methods:
                    metrics = results[method]
                    for metric_name, value in metrics.items():
                        detailed_data.append({
                            'Method': method,
                            'Metric': metric_name,
                            'Value': value
                        })
                
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_excel(writer, sheet_name='Detailed Metrics', index=False)
            
            return buffer.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def create_comprehensive_report(self, results: Dict[str, Any], 
                                  insights: List[Dict[str, Any]],
                                  recommendations: List[str]) -> str:
        """Create comprehensive analysis report."""
        report_lines = []
        
        # Header
        report_lines.extend([
            "# OrthoReduce Experiment Analysis Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            ""
        ])
        
        # Key insights
        for insight in insights[:3]:  # Top 3 insights
            icon = {"success": "✅", "warning": "⚠️", "info": "ℹ️"}.get(insight['type'], "•")
            report_lines.append(f"{icon} **{insight['title']}**: {insight['content']}")
        
        report_lines.extend(["", "## Method Comparison", ""])
        
        # Method performance table
        methods = [k for k in results.keys() if k != '_metadata']
        quality_scores = DataProcessor.compute_quality_scores(results)
        
        # Create markdown table
        report_lines.append("| Method | Quality Score | Rank Correlation | Mean Distortion | Runtime (s) |")
        report_lines.append("|--------|---------------|------------------|-----------------|-------------|")
        
        for method in sorted(methods, key=lambda x: quality_scores.get(x, 0), reverse=True):
            metrics = results[method]
            report_lines.append(
                f"| {method} | {quality_scores.get(method, 0):.3f} | "
                f"{metrics.get('rank_correlation', 0):.4f} | "
                f"{metrics.get('mean_distortion', 0):.4f} | "
                f"{metrics.get('runtime', 0):.4f} |"
            )
        
        # Recommendations
        report_lines.extend(["", "## Recommendations", ""])
        for rec in recommendations:
            report_lines.append(f"- {rec}")
        
        # Detailed analysis
        report_lines.extend(["", "## Detailed Analysis", ""])
        
        pareto_methods = DataProcessor.get_pareto_frontier(results)
        report_lines.append(f"**Pareto Optimal Methods**: {', '.join(pareto_methods)}")
        
        # Performance characteristics
        runtimes = [results[m]['runtime'] for m in methods]
        correlations = [results[m]['rank_correlation'] for m in methods]
        
        report_lines.extend([
            f"**Runtime Range**: {min(runtimes):.4f}s - {max(runtimes):.4f}s",
            f"**Correlation Range**: {min(correlations):.4f} - {max(correlations):.4f}",
            f"**Speed-up Factor**: {max(runtimes)/min(runtimes):.1f}x",
            ""
        ])
        
        return "\n".join(report_lines)
    
    def create_export_package(self, results: Dict[str, Any], 
                            insights: List[Dict[str, Any]],
                            recommendations: List[str],
                            figures: List[go.Figure] = None) -> bytes:
        """Create comprehensive export package as ZIP."""
        buffer = io.BytesIO()
        
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Export data in multiple formats
            zf.writestr("results.json", self.export_results_data(results, 'json'))
            zf.writestr("results.csv", self.export_results_data(results, 'csv'))
            zf.writestr("results.xlsx", self.export_results_data(results, 'excel'))
            
            # Analysis report
            report = self.create_comprehensive_report(results, insights, recommendations)
            zf.writestr("analysis_report.md", report)
            
            # Export figures if provided
            if figures:
                for i, fig in enumerate(figures):
                    # Save as HTML
                    html_content = fig.to_html(include_plotlyjs='inline')
                    zf.writestr(f"figure_{i+1}.html", html_content)
                    
                    # Save as PNG (if possible)
                    try:
                        img_bytes = fig.to_image(format='png', width=1200, height=800, scale=2)
                        zf.writestr(f"figure_{i+1}.png", img_bytes)
                    except Exception:
                        pass  # Skip PNG export if not available
            
            # Metadata
            metadata = {
                "export_timestamp": datetime.now().isoformat(),
                "methods_analyzed": list(results.keys()),
                "export_format": "comprehensive_package",
                "insights_count": len(insights),
                "recommendations_count": len(recommendations)
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
        
        return buffer.getvalue()


def show_enhanced_overview(results: Dict[str, Any]):
    """Show enhanced overview page with insights and recommendations."""
    st.title("📊 Enhanced Experiment Overview")
    
    # Generate insights
    with st.spinner("Generating insights..."):
        insights = AdvancedAnalytics.generate_insights(results)
        convergence_analysis = AdvancedAnalytics.analyze_convergence_patterns(None)
        recommendations = AdvancedAnalytics.generate_recommendations(results, convergence_analysis)
    
    # Insights panel
    if insights:
        st.subheader("🔍 Key Insights")
        
        for insight in insights:
            if insight['type'] == 'success':
                st.success(f"**{insight['title']}**: {insight['content']}")
            elif insight['type'] == 'warning':
                st.warning(f"**{insight['title']}**: {insight['content']}")
            else:
                st.info(f"**{insight['title']}**: {insight['content']}")
    
    # Recommendations panel
    if recommendations:
        st.subheader("💡 Recommendations")
        for rec in recommendations:
            st.markdown(rec)
    
    # Quick stats
    methods = [k for k in results.keys() if k != '_metadata']
    quality_scores = DataProcessor.compute_quality_scores(results)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Methods Analyzed", len(methods))
    
    with col2:
        best_method = max(quality_scores.items(), key=lambda x: x[1])
        st.metric("Best Method", best_method[0], f"Score: {best_method[1]:.3f}")
    
    with col3:
        runtimes = [results[m]['runtime'] for m in methods]
        speedup = max(runtimes) / min(runtimes)
        st.metric("Max Speed-up", f"{speedup:.1f}x")
    
    with col4:
        correlations = [results[m]['rank_correlation'] for m in methods]
        max_correlation = max(correlations)
        st.metric("Best Correlation", f"{max_correlation:.4f}")


def show_parameter_sensitivity_analysis(results: Dict[str, Any]):
    """Show parameter sensitivity analysis page."""
    st.title("⚙️ Parameter Sensitivity Analysis")
    
    # Mock parameter sensitivity data for demonstration
    # In a real implementation, this would come from hyperparameter search results
    sensitivity_data = {
        "Convex Optimization": {
            "k_candidates": {
                "variance_explained": 0.35,
                "optimal_range": "[64, 128]",
                "sensitivity": "High"
            },
            "ridge_lambda": {
                "variance_explained": 0.22,
                "optimal_range": "[1e-6, 1e-4]", 
                "sensitivity": "Medium"
            },
            "tolerance_mode": {
                "variance_explained": 0.15,
                "optimal_range": "balanced",
                "sensitivity": "Low"
            }
        },
        "Geometric Embeddings": {
            "learning_rate": {
                "variance_explained": 0.18,
                "optimal_range": "[0.01, 0.05]",
                "sensitivity": "Medium"
            },
            "curvature": {
                "variance_explained": 0.12,
                "optimal_range": "[1.0, 2.0]",
                "sensitivity": "Medium"
            },
            "max_iterations": {
                "variance_explained": 0.08,
                "optimal_range": "[500, 1000]",
                "sensitivity": "Low"
            }
        }
    }
    
    # Parameter importance visualization
    st.subheader("Parameter Importance by Stage")
    
    for stage, params in sensitivity_data.items():
        st.write(f"**{stage}**")
        
        # Create horizontal bar chart
        param_names = list(params.keys())
        variance_explained = [params[p]["variance_explained"] for p in param_names]
        
        fig = go.Figure(go.Bar(
            x=variance_explained,
            y=param_names,
            orientation='h',
            marker_color=['#FF6B6B' if v > 0.2 else '#4ECDC4' if v > 0.1 else '#95E1D3' for v in variance_explained],
            text=[f'{v:.2%}' for v in variance_explained],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"{stage} Parameter Sensitivity",
            xaxis_title="Variance Explained",
            height=200 + len(param_names) * 30
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Parameter recommendations table
        recommendations_df = pd.DataFrame([
            {
                'Parameter': param,
                'Sensitivity': data['sensitivity'],
                'Optimal Range': data['optimal_range'],
                'Variance Explained': f"{data['variance_explained']:.1%}"
            }
            for param, data in params.items()
        ])
        
        st.dataframe(recommendations_df, use_container_width=True)
        st.markdown("---")


def show_advanced_exports(results: Dict[str, Any]):
    """Show advanced export functionality."""
    st.title("💾 Advanced Export Options")
    
    export_manager = ExportManager()
    
    # Generate insights and recommendations for export
    insights = AdvancedAnalytics.generate_insights(results)
    recommendations = AdvancedAnalytics.generate_recommendations(results, {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Export")
        
        # Single format exports
        export_format = st.selectbox(
            "Select Format:",
            ["JSON", "CSV", "Excel"]
        )
        
        if st.button("Export Data"):
            try:
                data_bytes = export_manager.export_results_data(results, export_format.lower())
                
                # Create download
                st.download_button(
                    label=f"⬇️ Download {export_format}",
                    data=data_bytes,
                    file_name=f"orthoreduce_results.{export_format.lower()}",
                    mime="application/octet-stream"
                )
                
                st.success(f"✅ {export_format} export ready!")
                
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
        
        # Report generation
        st.subheader("📋 Analysis Report")
        
        include_sections = st.multiselect(
            "Include Sections:",
            [
                "Executive Summary",
                "Method Comparison",
                "Key Insights", 
                "Recommendations",
                "Detailed Metrics"
            ],
            default=[
                "Executive Summary",
                "Method Comparison", 
                "Recommendations"
            ]
        )
        
        if st.button("Generate Report"):
            try:
                report_content = export_manager.create_comprehensive_report(
                    results, insights, recommendations
                )
                
                st.download_button(
                    label="⬇️ Download Report",
                    data=report_content,
                    file_name=f"orthoreduce_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
                st.success("✅ Analysis report generated!")
                
                # Preview report
                with st.expander("📖 Report Preview"):
                    st.markdown(report_content)
                
            except Exception as e:
                st.error(f"❌ Report generation failed: {e}")
    
    with col2:
        st.subheader("📦 Comprehensive Package")
        
        st.write("""
        Create a complete export package including:
        - Results data in multiple formats
        - Analysis report with insights
        - Visualizations (if available)
        - Metadata and configuration
        """)
        
        package_name = st.text_input(
            "Package Name:",
            f"orthoreduce_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        if st.button("Create Export Package"):
            try:
                with st.spinner("Creating comprehensive export package..."):
                    # You would pass actual figures here in a real implementation
                    package_bytes = export_manager.create_export_package(
                        results, insights, recommendations
                    )
                
                st.download_button(
                    label="⬇️ Download Package",
                    data=package_bytes,
                    file_name=f"{package_name}.zip",
                    mime="application/zip"
                )
                
                st.success("✅ Export package created!")
                
                # Package contents preview
                with st.expander("📋 Package Contents"):
                    st.write("""
                    - `results.json` - Complete results data
                    - `results.csv` - Tabular results summary  
                    - `results.xlsx` - Excel workbook with multiple sheets
                    - `analysis_report.md` - Comprehensive analysis report
                    - `figure_*.html` - Interactive visualizations
                    - `figure_*.png` - Static visualization images
                    - `metadata.json` - Export metadata and configuration
                    """)
                
            except Exception as e:
                st.error(f"❌ Package creation failed: {e}")
        
        # Custom export options
        st.subheader("🔧 Custom Export")
        
        with st.expander("Advanced Options"):
            include_raw_data = st.checkbox("Include raw embeddings", value=False)
            include_config = st.checkbox("Include experiment configuration", value=True) 
            high_res_figures = st.checkbox("High-resolution figures", value=True)
            
            custom_metadata = st.text_area(
                "Additional Metadata (JSON):",
                placeholder='{"project": "my_project", "version": "1.0"}'
            )
            
            if st.button("Create Custom Package"):
                st.info("Custom export functionality would be implemented here")


def main():
    """Main entry point for enhanced dashboard."""
    st.set_page_config(
        page_title="OrthoReduce Enhanced Dashboard",
        page_icon="🚀", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better appearance
    st.markdown("""
        <style>
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .insight-success {
            border-left: 4px solid #28a745;
            padding-left: 1rem;
            background-color: #d4edda;
        }
        .insight-warning {
            border-left: 4px solid #ffc107;
            padding-left: 1rem;
            background-color: #fff3cd;
        }
        .insight-info {
            border-left: 4px solid #17a2b8;
            padding-left: 1rem;
            background-color: #d1ecf1;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load sample results for demonstration
    sample_results = {
        "PCA": {
            "rank_correlation": 0.998,
            "mean_distortion": 0.030,
            "runtime": 0.0009,
            "memory_usage": 0.1,
            "compression_ratio": 1.19
        },
        "JLL": {
            "rank_correlation": 0.925,
            "mean_distortion": 0.143,
            "runtime": 0.0002,
            "memory_usage": 0.05,
            "compression_ratio": 1.19
        },
        "UMAP": {
            "rank_correlation": 0.891,
            "mean_distortion": 0.287,
            "runtime": 2.45,
            "memory_usage": 0.3,
            "compression_ratio": 1.19
        },
        "Poincare": {
            "rank_correlation": 0.831,
            "mean_distortion": 0.139,
            "runtime": 0.003,
            "memory_usage": 0.08,
            "compression_ratio": 1.19
        }
    }
    
    # Sidebar navigation
    st.sidebar.title("🚀 Enhanced Dashboard")
    page = st.sidebar.selectbox(
        "Navigation",
        [
            "Enhanced Overview",
            "Method Comparison", 
            "Parameter Analysis",
            "Embedding Explorer",
            "Advanced Exports"
        ]
    )
    
    # Main content
    if page == "Enhanced Overview":
        show_enhanced_overview(sample_results)
    elif page == "Method Comparison":
        from .dashboard import show_method_comparison_page, InteractivePlotGenerator
        plot_generator = InteractivePlotGenerator()
        show_method_comparison_page(sample_results, plot_generator)
    elif page == "Parameter Analysis":
        show_parameter_sensitivity_analysis(sample_results)
    elif page == "Embedding Explorer":
        st.title("🔍 Interactive Embedding Explorer")
        st.info("Load experiment with embedding data to use the interactive explorer")
        # Would integrate with actual embedding data
    elif page == "Advanced Exports":
        show_advanced_exports(sample_results)


if __name__ == "__main__":
    main()