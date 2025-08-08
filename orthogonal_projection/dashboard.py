"""
Interactive Dashboard System for OrthoReduce Experiment Results

This module provides a comprehensive web-based dashboard for visualizing and exploring
dimensionality reduction experiment results. Built with Streamlit for an intuitive
user experience with real-time interactivity.

Features:
- Real-time experiment monitoring
- Interactive embedding visualizations with pan/zoom
- Method comparison tools with dynamic filtering
- Parameter sensitivity analysis
- Export functionality for findings
- Responsive design suitable for research presentations
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging
from dataclasses import asdict

# Import OrthoReduce components
try:
    from .results_aggregator import ResultsAggregator, AggregatedResults, ExperimentResult
    from .experiment_orchestration import StagedOptimizationExperiment
    from .experiment_config import ExperimentConfig
    from .visualization import METHOD_COLORS, QUALITY_COLORS
except ImportError:
    from results_aggregator import ResultsAggregator, AggregatedResults, ExperimentResult
    from experiment_orchestration import StagedOptimizationExperiment
    from experiment_config import ExperimentConfig
    from visualization import METHOD_COLORS, QUALITY_COLORS

logger = logging.getLogger(__name__)


class DashboardDataManager:
    """Manages data loading and caching for the dashboard."""
    
    def __init__(self, results_dir: str = "experiment_results"):
        self.results_dir = Path(results_dir)
        self.cached_results = {}
        self.last_refresh = {}
        self.cache_duration = 300  # 5 minutes
    
    @st.cache_data(ttl=300)
    def load_experiment_results(_self, experiment_path: str) -> Optional[Dict[str, Any]]:
        """Load experiment results with caching."""
        try:
            path = Path(experiment_path)
            
            if path.suffix == '.pkl':
                with open(path, 'rb') as f:
                    return pickle.load(f)
            elif path.suffix == '.json':
                with open(path, 'r') as f:
                    return json.load(f)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to load experiment results: {e}")
            return None
    
    @st.cache_data(ttl=60)
    def get_available_experiments(_self) -> List[Dict[str, Any]]:
        """Get list of available experiments with metadata."""
        experiments = []
        
        if not _self.results_dir.exists():
            return experiments
            
        for path in _self.results_dir.rglob("*.json"):
            if "results" in path.name or "summary" in path.name:
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract metadata
                    metadata = {
                        "name": path.stem,
                        "path": str(path),
                        "modified": datetime.fromtimestamp(path.stat().st_mtime),
                        "size_mb": path.stat().st_size / (1024 * 1024),
                        "type": "json"
                    }
                    
                    # Add experiment-specific info if available
                    if isinstance(data, dict):
                        metadata.update({
                            "methods": list(data.keys()) if data else [],
                            "n_methods": len(data) if data else 0
                        })
                    
                    experiments.append(metadata)
                    
                except Exception:
                    continue
        
        # Also look for pickle files
        for path in _self.results_dir.rglob("*.pkl"):
            if "results" in path.name or "aggregated" in path.name:
                try:
                    metadata = {
                        "name": path.stem,
                        "path": str(path),
                        "modified": datetime.fromtimestamp(path.stat().st_mtime),
                        "size_mb": path.stat().st_size / (1024 * 1024),
                        "type": "pkl"
                    }
                    experiments.append(metadata)
                except Exception:
                    continue
        
        return sorted(experiments, key=lambda x: x["modified"], reverse=True)
    
    def get_real_time_status(self, experiment_dir: str) -> Dict[str, Any]:
        """Get real-time experiment status for monitoring."""
        status = {
            "running": False,
            "current_stage": None,
            "progress": 0.0,
            "eta": None,
            "last_update": None
        }
        
        try:
            exp_path = Path(experiment_dir)
            
            # Look for active experiment indicators
            lock_file = exp_path / "experiment.lock"
            log_file = exp_path / "experiment.log"
            
            if lock_file.exists():
                status["running"] = True
                status["last_update"] = datetime.fromtimestamp(lock_file.stat().st_mtime)
                
            if log_file.exists():
                # Parse log for current stage and progress
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-50:]  # Last 50 lines
                    
                for line in reversed(lines):
                    if "Starting stage:" in line:
                        status["current_stage"] = line.split("Starting stage:")[-1].strip()
                        break
                    elif "Progress:" in line:
                        # Extract progress percentage
                        try:
                            progress_str = line.split("Progress:")[-1].strip()
                            status["progress"] = float(progress_str.replace("%", ""))
                        except:
                            pass
                        
        except Exception as e:
            logger.error(f"Failed to get real-time status: {e}")
        
        return status


class InteractivePlotGenerator:
    """Generates interactive plots using Plotly for dashboard."""
    
    def __init__(self):
        self.method_colors = METHOD_COLORS
        self.quality_colors = QUALITY_COLORS
    
    def create_method_comparison_scatter(self, results: Dict[str, Any], 
                                       x_metric: str = "mean_distortion",
                                       y_metric: str = "rank_correlation",
                                       size_metric: str = "runtime") -> go.Figure:
        """Create interactive scatter plot for method comparison."""
        methods = [k for k in results.keys() if k != '_metadata']
        
        # Extract data
        plot_data = []
        for method in methods:
            if method in results:
                metrics = results[method]
                plot_data.append({
                    'method': method,
                    'x': metrics.get(x_metric, 0),
                    'y': metrics.get(y_metric, 0),
                    'size': metrics.get(size_metric, 1),
                    'runtime': metrics.get('runtime', 0),
                    'compression': metrics.get('compression_ratio', 1),
                    'distortion': metrics.get('mean_distortion', 0),
                    'correlation': metrics.get('rank_correlation', 0)
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create scatter plot
        fig = px.scatter(
            df, 
            x='x', 
            y='y',
            size='size',
            color='method',
            color_discrete_map=self.method_colors,
            hover_data=['runtime', 'compression', 'distortion', 'correlation'],
            labels={
                'x': x_metric.replace('_', ' ').title(),
                'y': y_metric.replace('_', ' ').title(),
                'size': size_metric.replace('_', ' ').title()
            }
        )
        
        # Customize layout
        fig.update_layout(
            title=f"Method Comparison: {y_metric.title()} vs {x_metric.title()}",
            showlegend=True,
            height=600,
            hovermode='closest'
        )
        
        # Add method labels
        for _, row in df.iterrows():
            fig.add_annotation(
                x=row['x'],
                y=row['y'],
                text=row['method'],
                showarrow=False,
                yshift=10,
                font=dict(size=10)
            )
        
        return fig
    
    def create_performance_dashboard(self, results: Dict[str, Any]) -> go.Figure:
        """Create comprehensive performance dashboard."""
        methods = [k for k in results.keys() if k != '_metadata']
        
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Runtime Performance (Log Scale)',
                'Quality Assessment',
                'Method Rankings',
                'Resource Usage'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": True}]
            ]
        )
        
        # Extract metrics
        runtimes = [results[m]['runtime'] for m in methods]
        correlations = [results[m]['rank_correlation'] for m in methods]
        distortions = [results[m]['mean_distortion'] for m in methods]
        memory_usage = [results[m].get('memory_usage', 0) for m in methods]
        
        # 1. Runtime Performance
        fig.add_trace(
            go.Bar(
                x=methods,
                y=runtimes,
                name="Runtime (s)",
                marker_color=[self.method_colors.get(m, '#gray') for m in methods],
                text=[f'{r:.4f}s' for r in runtimes],
                textposition='auto'
            ),
            row=1, col=1
        )
        fig.update_yaxes(type="log", row=1, col=1)
        
        # 2. Quality Assessment (Correlation vs Distortion)
        fig.add_trace(
            go.Scatter(
                x=distortions,
                y=correlations,
                mode='markers+text',
                text=methods,
                textposition='top center',
                marker=dict(
                    size=15,
                    color=[self.method_colors.get(m, '#gray') for m in methods]
                ),
                name="Quality"
            ),
            row=1, col=2
        )
        
        # 3. Method Rankings (Composite Score)
        quality_scores = [c / (d + 0.01) for c, d in zip(correlations, distortions)]
        fig.add_trace(
            go.Bar(
                x=methods,
                y=quality_scores,
                name="Quality Score",
                marker_color=[self.method_colors.get(m, '#gray') for m in methods],
                text=[f'{q:.2f}' for q in quality_scores],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # 4. Resource Usage (Runtime vs Memory)
        fig.add_trace(
            go.Scatter(
                x=runtimes,
                y=memory_usage,
                mode='markers+text',
                text=methods,
                textposition='top center',
                marker=dict(
                    size=12,
                    color=[self.method_colors.get(m, '#gray') for m in methods]
                ),
                name="Resource Usage"
            ),
            row=2, col=2
        )
        fig.update_xaxes(type="log", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title_text="OrthoReduce Performance Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def create_parameter_sensitivity_plot(self, sensitivity_data: Dict[str, Any]) -> go.Figure:
        """Create parameter sensitivity analysis plot."""
        if not sensitivity_data:
            # Return empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="No parameter sensitivity data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return fig
        
        # Extract parameter importance
        params = list(sensitivity_data.keys())
        variance_explained = [sensitivity_data[p].get('variance_explained', 0) for p in params]
        
        # Create bar plot
        fig = go.Figure(data=[
            go.Bar(
                x=variance_explained,
                y=params,
                orientation='h',
                marker_color='lightblue',
                text=[f'{v:.3f}' for v in variance_explained],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Parameter Sensitivity Analysis",
            xaxis_title="Variance Explained",
            yaxis_title="Parameters",
            height=max(400, len(params) * 30)
        )
        
        return fig
    
    def create_convergence_plot(self, convergence_data: Dict[str, Any]) -> go.Figure:
        """Create convergence analysis plot."""
        fig = go.Figure()
        
        for method, data in convergence_data.items():
            if 'loss_history' in method:
                # Plot loss convergence
                histories = data.get('loss_histories', [])
                if histories:
                    # Plot mean and std
                    max_len = max(len(h) for h in histories)
                    iterations = list(range(max_len))
                    
                    # Pad histories to same length
                    padded_histories = []
                    for h in histories:
                        padded = h + [h[-1]] * (max_len - len(h))
                        padded_histories.append(padded)
                    
                    mean_loss = np.mean(padded_histories, axis=0)
                    std_loss = np.std(padded_histories, axis=0)
                    
                    fig.add_trace(go.Scatter(
                        x=iterations,
                        y=mean_loss,
                        mode='lines',
                        name=method.replace('_loss_history', '').title(),
                        line=dict(color=self.method_colors.get(method, '#gray'))
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=iterations + iterations[::-1],
                        y=(mean_loss + std_loss).tolist() + (mean_loss - std_loss)[::-1].tolist(),
                        fill='tonexty',
                        fillcolor=self.method_colors.get(method, '#gray'),
                        opacity=0.2,
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False
                    ))
        
        fig.update_layout(
            title="Convergence Analysis",
            xaxis_title="Iteration",
            yaxis_title="Loss",
            height=500
        )
        
        return fig


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="OrthoReduce Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize components
    data_manager = DashboardDataManager()
    plot_generator = InteractivePlotGenerator()
    
    # Sidebar for navigation and controls
    st.sidebar.title("🔬 OrthoReduce Dashboard")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Select View",
        ["Overview", "Method Comparison", "Parameter Analysis", "Real-time Monitor", "Export"]
    )
    
    # Experiment selection
    st.sidebar.subheader("Experiment Selection")
    experiments = data_manager.get_available_experiments()
    
    if experiments:
        experiment_options = [
            f"{exp['name']} ({exp['modified'].strftime('%Y-%m-%d %H:%M')})"
            for exp in experiments
        ]
        selected_exp_idx = st.sidebar.selectbox(
            "Select Experiment",
            range(len(experiment_options)),
            format_func=lambda x: experiment_options[x]
        )
        
        selected_experiment = experiments[selected_exp_idx]
        
        # Load selected experiment
        with st.spinner("Loading experiment data..."):
            results = data_manager.load_experiment_results(selected_experiment['path'])
        
        if results:
            # Main content based on selected page
            if page == "Overview":
                show_overview_page(results, plot_generator)
            elif page == "Method Comparison":
                show_method_comparison_page(results, plot_generator)
            elif page == "Parameter Analysis":
                show_parameter_analysis_page(results, plot_generator)
            elif page == "Real-time Monitor":
                show_monitoring_page(data_manager)
            elif page == "Export":
                show_export_page(results)
        else:
            st.error("Failed to load experiment results")
    else:
        st.warning("No experiment results found in the results directory")
        st.info("Run some experiments first using the OrthoReduce CLI or Python interface")


def show_overview_page(results: Dict[str, Any], plot_generator: InteractivePlotGenerator):
    """Display experiment overview page."""
    st.title("📊 Experiment Overview")
    
    # Key metrics summary
    methods = [k for k in results.keys() if k != '_metadata']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Methods Compared", len(methods))
    
    with col2:
        best_correlation = max([results[m]['rank_correlation'] for m in methods])
        best_method = [m for m in methods if results[m]['rank_correlation'] == best_correlation][0]
        st.metric("Best Correlation", f"{best_correlation:.4f}", f"({best_method})")
    
    with col3:
        min_distortion = min([results[m]['mean_distortion'] for m in methods])
        best_distortion_method = [m for m in methods if results[m]['mean_distortion'] == min_distortion][0]
        st.metric("Lowest Distortion", f"{min_distortion:.4f}", f"({best_distortion_method})")
    
    with col4:
        fastest_runtime = min([results[m]['runtime'] for m in methods])
        fastest_method = [m for m in methods if results[m]['runtime'] == fastest_runtime][0]
        st.metric("Fastest Runtime", f"{fastest_runtime:.4f}s", f"({fastest_method})")
    
    # Performance dashboard
    st.subheader("Performance Dashboard")
    dashboard_fig = plot_generator.create_performance_dashboard(results)
    st.plotly_chart(dashboard_fig, use_container_width=True)
    
    # Method summary table
    st.subheader("Method Summary")
    
    summary_data = []
    for method in methods:
        metrics = results[method]
        summary_data.append({
            'Method': method,
            'Runtime (s)': f"{metrics['runtime']:.4f}",
            'Rank Correlation': f"{metrics['rank_correlation']:.4f}",
            'Mean Distortion': f"{metrics['mean_distortion']:.4f}",
            'Compression Ratio': f"{metrics['compression_ratio']:.2f}x"
        })
    
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True)


def show_method_comparison_page(results: Dict[str, Any], plot_generator: InteractivePlotGenerator):
    """Display method comparison page."""
    st.title("🔄 Method Comparison")
    
    # Interactive controls
    col1, col2, col3 = st.columns(3)
    
    available_metrics = ['mean_distortion', 'max_distortion', 'rank_correlation', 'runtime', 'compression_ratio']
    
    with col1:
        x_metric = st.selectbox("X-axis Metric", available_metrics, index=0)
    
    with col2:
        y_metric = st.selectbox("Y-axis Metric", available_metrics, index=2)
    
    with col3:
        size_metric = st.selectbox("Point Size Metric", available_metrics, index=3)
    
    # Create interactive scatter plot
    scatter_fig = plot_generator.create_method_comparison_scatter(
        results, x_metric, y_metric, size_metric
    )
    st.plotly_chart(scatter_fig, use_container_width=True)
    
    # Method filtering and selection
    st.subheader("Method Details")
    methods = [k for k in results.keys() if k != '_metadata']
    selected_methods = st.multiselect("Select methods to compare:", methods, default=methods)
    
    if selected_methods:
        # Create comparison chart
        metrics_to_compare = ['rank_correlation', 'mean_distortion', 'runtime']
        
        comparison_data = []
        for method in selected_methods:
            for metric in metrics_to_compare:
                comparison_data.append({
                    'Method': method,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': results[method][metric]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create radar chart for selected methods
        fig = go.Figure()
        
        for method in selected_methods:
            method_data = comparison_df[comparison_df['Method'] == method]
            
            # Normalize values for radar chart
            normalized_values = []
            for metric in metrics_to_compare:
                value = results[method][metric]
                if metric == 'runtime' or metric == 'mean_distortion':
                    # For these metrics, lower is better, so invert
                    all_values = [results[m][metric] for m in methods]
                    normalized = 1 - (value - min(all_values)) / (max(all_values) - min(all_values) + 1e-8)
                else:
                    # For correlation, higher is better
                    all_values = [results[m][metric] for m in methods]
                    normalized = (value - min(all_values)) / (max(all_values) - min(all_values) + 1e-8)
                
                normalized_values.append(normalized)
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=[m.replace('_', ' ').title() for m in metrics_to_compare],
                fill='toself',
                name=method,
                line_color=METHOD_COLORS.get(method, '#gray')
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Method Performance Comparison (Normalized)"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_parameter_analysis_page(results: Dict[str, Any], plot_generator: InteractivePlotGenerator):
    """Display parameter sensitivity analysis page."""
    st.title("⚙️ Parameter Analysis")
    
    # Mock sensitivity data for now (would come from actual analysis)
    sensitivity_data = {
        "convex_k_candidates": {"variance_explained": 0.35, "n_experiments": 10},
        "convex_ridge_lambda": {"variance_explained": 0.22, "n_experiments": 10},
        "spherical_learning_rate": {"variance_explained": 0.18, "n_experiments": 8},
        "poincare_curvature": {"variance_explained": 0.15, "n_experiments": 8},
        "preprocessing_components": {"variance_explained": 0.10, "n_experiments": 10}
    }
    
    # Parameter sensitivity plot
    st.subheader("Parameter Sensitivity Analysis")
    sensitivity_fig = plot_generator.create_parameter_sensitivity_plot(sensitivity_data)
    st.plotly_chart(sensitivity_fig, use_container_width=True)
    
    # Parameter recommendations
    st.subheader("Parameter Recommendations")
    
    recommendations = [
        "🎯 **k_candidates**: Most influential parameter - consider grid search over [32, 64, 128]",
        "⚡ **ridge_lambda**: Moderate impact - start with 1e-4 for most datasets",
        "📈 **learning_rate**: Fine-tune for spherical embeddings - try [0.01, 0.02, 0.05]",
        "🌐 **curvature**: Important for hyperbolic embeddings - dataset-dependent",
        "🔧 **preprocessing**: Less critical but can help - use adaptive pipeline"
    ]
    
    for rec in recommendations:
        st.markdown(rec)


def show_monitoring_page(data_manager: DashboardDataManager):
    """Display real-time experiment monitoring page."""
    st.title("📡 Real-time Experiment Monitor")
    
    # Auto-refresh controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Active Experiments")
    
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
    
    if auto_refresh:
        # Auto-refresh every 30 seconds
        time.sleep(30)
        st.experimental_rerun()
    
    # Mock monitoring data (would be real-time in actual implementation)
    active_experiments = [
        {
            "name": "comprehensive_experiment_001",
            "status": "running",
            "current_stage": "convex_optimization",
            "progress": 65.0,
            "eta": "12 minutes",
            "started": "2024-01-15 14:30:00"
        }
    ]
    
    if active_experiments:
        for exp in active_experiments:
            with st.expander(f"🟢 {exp['name']} - {exp['status'].upper()}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Stage", exp['current_stage'])
                
                with col2:
                    st.metric("Progress", f"{exp['progress']}%")
                
                with col3:
                    st.metric("ETA", exp['eta'])
                
                # Progress bar
                progress_bar = st.progress(exp['progress'] / 100.0)
                
                # Stage timeline
                stages = ["preprocessing", "convex_optimization", "geometric_embeddings", "calibration"]
                current_stage_idx = stages.index(exp['current_stage']) if exp['current_stage'] in stages else 0
                
                stage_status = []
                for i, stage in enumerate(stages):
                    if i < current_stage_idx:
                        stage_status.append("✅ " + stage.title())
                    elif i == current_stage_idx:
                        stage_status.append("🔄 " + stage.title())
                    else:
                        stage_status.append("⏳ " + stage.title())
                
                st.write("**Pipeline Progress:**")
                st.write(" → ".join(stage_status))
    else:
        st.info("No active experiments found")
        
        # Button to refresh
        if st.button("🔄 Refresh"):
            st.experimental_rerun()


def show_export_page(results: Dict[str, Any]):
    """Display export functionality page."""
    st.title("💾 Export Results")
    
    st.subheader("Available Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Visualizations")
        
        export_formats = st.multiselect(
            "Select formats:",
            ["PNG", "SVG", "PDF", "HTML"],
            default=["PNG"]
        )
        
        if st.button("Export Visualizations"):
            # Mock export process
            with st.spinner("Exporting visualizations..."):
                time.sleep(2)
            st.success(f"✅ Visualizations exported in {', '.join(export_formats)} format(s)")
    
    with col2:
        st.subheader("📄 Data")
        
        data_formats = st.multiselect(
            "Select data formats:",
            ["JSON", "CSV", "Excel", "Pickle"],
            default=["JSON", "CSV"]
        )
        
        if st.button("Export Data"):
            # Mock export process  
            with st.spinner("Exporting data..."):
                time.sleep(1)
            st.success(f"✅ Data exported in {', '.join(data_formats)} format(s)")
    
    # Report generation
    st.subheader("📋 Generate Report")
    
    report_sections = st.multiselect(
        "Include sections:",
        [
            "Executive Summary",
            "Method Comparison",
            "Statistical Analysis", 
            "Parameter Sensitivity",
            "Recommendations"
        ],
        default=[
            "Executive Summary",
            "Method Comparison",
            "Recommendations"
        ]
    )
    
    report_format = st.radio("Report format:", ["PDF", "HTML", "Word"])
    
    if st.button("📋 Generate Report"):
        with st.spinner("Generating comprehensive report..."):
            time.sleep(3)
        
        st.success(f"✅ Comprehensive report generated in {report_format} format")
        
        # Show download button (mock)
        st.download_button(
            label="⬇️ Download Report",
            data="Mock report data",
            file_name=f"orthoreduce_report.{report_format.lower()}",
            mime="application/octet-stream"
        )


if __name__ == "__main__":
    main()