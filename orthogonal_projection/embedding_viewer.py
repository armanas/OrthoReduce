"""
Interactive Embedding Visualization Component

Provides advanced visualization capabilities for exploring dimensionality reduction results
with interactive pan/zoom, point selection, and embedding quality analysis.
"""

from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import colorcet as cc

try:
    from .dashboard_utils import EmbeddingAnalyzer, DataProcessor
    from .visualization import METHOD_COLORS
except ImportError:
    from dashboard_utils import EmbeddingAnalyzer, DataProcessor
    from visualization import METHOD_COLORS

logger = logging.getLogger(__name__)


class EmbeddingVisualizer:
    """Advanced embedding visualization with interactive features."""
    
    def __init__(self):
        self.method_colors = METHOD_COLORS
        self.current_selection = None
        self.point_annotations = {}
    
    def create_embedding_scatter(self, 
                               embedding: np.ndarray,
                               method_name: str,
                               labels: Optional[np.ndarray] = None,
                               highlight_indices: Optional[List[int]] = None,
                               show_3d: bool = False) -> go.Figure:
        """Create interactive scatter plot of embedding."""
        
        if embedding.shape[1] < 2:
            st.error(f"Embedding has only {embedding.shape[1]} dimensions. Need at least 2 for visualization.")
            return go.Figure()
        
        # Prepare data
        n_points = embedding.shape[0]
        point_indices = np.arange(n_points)
        
        # Use labels for coloring if provided, otherwise use indices
        if labels is not None:
            color_data = labels
            colorscale = px.colors.qualitative.Set1
        else:
            color_data = point_indices
            colorscale = 'viridis'
        
        # Create 2D or 3D scatter plot
        if show_3d and embedding.shape[1] >= 3:
            fig = go.Figure(data=go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=color_data,
                    colorscale=colorscale,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=[f'Point {i}' for i in point_indices],
                hovertemplate='<b>Point %{text}</b><br>' +
                             'X: %{x:.3f}<br>' +
                             'Y: %{y:.3f}<br>' +
                             'Z: %{z:.3f}<extra></extra>',
                name=method_name
            ))
            
            fig.update_layout(
                title=f'{method_name} Embedding (3D)',
                scene=dict(
                    xaxis_title='Component 1',
                    yaxis_title='Component 2',
                    zaxis_title='Component 3',
                    aspectmode='cube'
                ),
                height=700
            )
        else:
            # 2D scatter plot
            fig = go.Figure(data=go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=color_data,
                    colorscale=colorscale,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=[f'Point {i}' for i in point_indices],
                hovertemplate='<b>Point %{text}</b><br>' +
                             'X: %{x:.3f}<br>' +
                             'Y: %{y:.3f}<extra></extra>',
                name=method_name
            ))
            
            fig.update_layout(
                title=f'{method_name} Embedding (2D)',
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                height=600,
                hovermode='closest'
            )
        
        # Highlight specific points if requested
        if highlight_indices:
            highlight_points = embedding[highlight_indices]
            
            if show_3d and embedding.shape[1] >= 3:
                fig.add_trace(go.Scatter3d(
                    x=highlight_points[:, 0],
                    y=highlight_points[:, 1],
                    z=highlight_points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='circle-open',
                        line=dict(width=3, color='red')
                    ),
                    name='Selected Points',
                    hovertemplate='<b>Selected Point</b><br>' +
                                 'X: %{x:.3f}<br>' +
                                 'Y: %{y:.3f}<br>' +
                                 'Z: %{z:.3f}<extra></extra>'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=highlight_points[:, 0],
                    y=highlight_points[:, 1],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='circle-open',
                        line=dict(width=3, color='red')
                    ),
                    name='Selected Points',
                    hovertemplate='<b>Selected Point</b><br>' +
                                 'X: %{x:.3f}<br>' +
                                 'Y: %{y:.3f}<extra></extra>'
                ))
        
        return fig
    
    def create_multi_embedding_comparison(self, 
                                        embeddings: Dict[str, np.ndarray],
                                        original_data: Optional[np.ndarray] = None,
                                        labels: Optional[np.ndarray] = None) -> go.Figure:
        """Create side-by-side comparison of multiple embeddings."""
        
        n_methods = len(embeddings)
        if n_methods == 0:
            return go.Figure()
        
        # Create subplot layout
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols
        
        subplot_titles = list(embeddings.keys())
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"type": "scatter"}] * cols for _ in range(rows)]
        )
        
        # Add each embedding
        for idx, (method_name, embedding) in enumerate(embeddings.items()):
            row = idx // cols + 1
            col = idx % cols + 1
            
            if embedding.shape[1] < 2:
                continue
            
            # Use labels for coloring if provided
            if labels is not None:
                color_data = labels
                colorscale = px.colors.qualitative.Set1
            else:
                color_data = np.arange(embedding.shape[0])
                colorscale = 'viridis'
            
            fig.add_trace(
                go.Scatter(
                    x=embedding[:, 0],
                    y=embedding[:, 1],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=color_data,
                        colorscale=colorscale,
                        opacity=0.7,
                        line=dict(width=0.5, color='white')
                    ),
                    name=method_name,
                    showlegend=False,
                    hovertemplate=f'<b>{method_name}</b><br>' +
                                 'X: %{x:.3f}<br>' +
                                 'Y: %{y:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Embedding Comparison",
            height=300 * rows,
            hovermode='closest'
        )
        
        return fig
    
    def create_embedding_quality_analysis(self, 
                                        original_data: np.ndarray,
                                        embeddings: Dict[str, np.ndarray]) -> go.Figure:
        """Create visualization of embedding quality metrics."""
        
        # Compute quality metrics for each embedding
        quality_data = []
        
        for method_name, embedding in embeddings.items():
            try:
                metrics = EmbeddingAnalyzer.compute_embedding_metrics(original_data, embedding)
                quality_data.append({
                    'Method': method_name,
                    'Distance Correlation': metrics.get('distance_correlation', 0),
                    'Neighborhood Preservation': metrics.get('neighborhood_preservation', 0),
                    'Stress': 1 - min(1, metrics.get('stress', 1)),  # Invert for better visualization
                    'Rank Correlation': metrics.get('rank_correlation', 0)
                })
            except Exception as e:
                logger.error(f"Failed to compute metrics for {method_name}: {e}")
                continue
        
        if not quality_data:
            return go.Figure()
        
        df = pd.DataFrame(quality_data)
        
        # Create radar chart
        metrics_to_plot = ['Distance Correlation', 'Neighborhood Preservation', 'Stress', 'Rank Correlation']
        
        fig = go.Figure()
        
        for _, row in df.iterrows():
            method = row['Method']
            values = [row[metric] for metric in metrics_to_plot]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_to_plot,
                fill='toself',
                name=method,
                line_color=self.method_colors.get(method, '#gray')
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Embedding Quality Analysis",
            height=500
        )
        
        return fig
    
    def create_distance_preservation_plot(self, 
                                        original_data: np.ndarray,
                                        embedding: np.ndarray,
                                        method_name: str,
                                        n_samples: int = 1000) -> go.Figure:
        """Create distance preservation analysis plot."""
        
        # Sample points for efficiency
        n_points = min(n_samples, original_data.shape[0])
        indices = np.random.choice(original_data.shape[0], n_points, replace=False)
        
        X_sample = original_data[indices]
        Y_sample = embedding[indices]
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        
        orig_distances = pdist(X_sample)
        embed_distances = pdist(Y_sample)
        
        # Create scatter plot of distances
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=orig_distances,
            y=embed_distances,
            mode='markers',
            marker=dict(
                size=3,
                color='blue',
                opacity=0.6
            ),
            name=f'{method_name} Distances',
            hovertemplate='Original: %{x:.3f}<br>' +
                         'Embedded: %{y:.3f}<extra></extra>'
        ))
        
        # Add perfect preservation line
        max_dist = max(np.max(orig_distances), np.max(embed_distances))
        fig.add_trace(go.Scatter(
            x=[0, max_dist],
            y=[0, max_dist],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Preservation'
        ))
        
        # Compute and display correlation
        correlation = np.corrcoef(orig_distances, embed_distances)[0, 1]
        
        fig.update_layout(
            title=f'Distance Preservation: {method_name}<br>Correlation: {correlation:.4f}',
            xaxis_title='Original Distances',
            yaxis_title='Embedded Distances',
            height=500
        )
        
        return fig
    
    def create_neighborhood_analysis(self, 
                                   original_data: np.ndarray,
                                   embedding: np.ndarray,
                                   method_name: str,
                                   k_neighbors: int = 10) -> go.Figure:
        """Analyze neighborhood preservation."""
        
        from sklearn.neighbors import NearestNeighbors
        
        n_samples = min(500, original_data.shape[0])
        indices = np.random.choice(original_data.shape[0], n_samples, replace=False)
        
        X_sample = original_data[indices]
        Y_sample = embedding[indices]
        
        # Find k-nearest neighbors in both spaces
        nbrs_orig = NearestNeighbors(n_neighbors=k_neighbors+1).fit(X_sample)
        nbrs_embed = NearestNeighbors(n_neighbors=k_neighbors+1).fit(Y_sample)
        
        _, indices_orig = nbrs_orig.kneighbors(X_sample)
        _, indices_embed = nbrs_embed.kneighbors(Y_sample)
        
        # Compute neighborhood overlaps
        overlaps = []
        for i in range(n_samples):
            neighbors_orig = set(indices_orig[i][1:])  # Exclude self
            neighbors_embed = set(indices_embed[i][1:])
            
            overlap = len(neighbors_orig & neighbors_embed) / k_neighbors
            overlaps.append(overlap)
        
        # Create histogram of overlaps
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=overlaps,
            nbinsx=20,
            name=f'{method_name} Overlaps',
            marker_color='skyblue',
            opacity=0.7
        ))
        
        mean_overlap = np.mean(overlaps)
        fig.add_vline(
            x=mean_overlap,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_overlap:.3f}"
        )
        
        fig.update_layout(
            title=f'Neighborhood Preservation: {method_name}',
            xaxis_title='Neighborhood Overlap',
            yaxis_title='Count',
            height=400
        )
        
        return fig


class InteractiveEmbeddingExplorer:
    """Main interface for interactive embedding exploration."""
    
    def __init__(self):
        self.visualizer = EmbeddingVisualizer()
        self.selected_points = []
        self.current_embeddings = {}
        self.original_data = None
    
    def show_embedding_explorer(self, 
                               embeddings: Dict[str, np.ndarray],
                               original_data: Optional[np.ndarray] = None,
                               labels: Optional[np.ndarray] = None):
        """Show the main embedding exploration interface."""
        
        self.current_embeddings = embeddings
        self.original_data = original_data
        
        st.subheader("🔍 Interactive Embedding Explorer")
        
        if not embeddings:
            st.warning("No embeddings available for visualization")
            return
        
        # Sidebar controls
        with st.sidebar:
            st.subheader("Visualization Controls")
            
            # Method selection
            selected_method = st.selectbox(
                "Select Method",
                list(embeddings.keys()),
                key="embedding_method_selector"
            )
            
            # Visualization options
            show_3d = st.checkbox("3D Visualization", value=False)
            
            # Point selection tools
            st.subheader("Point Selection")
            selection_mode = st.radio(
                "Selection Mode",
                ["Manual", "Random Sample", "Outliers"],
                key="selection_mode"
            )
            
            if selection_mode == "Random Sample":
                n_sample = st.slider("Sample Size", 10, 1000, 100)
                if st.button("Select Random Points"):
                    self.selected_points = np.random.choice(
                        embeddings[selected_method].shape[0], 
                        min(n_sample, embeddings[selected_method].shape[0]), 
                        replace=False
                    ).tolist()
            
            elif selection_mode == "Outliers":
                if st.button("Select Outliers"):
                    self.selected_points = self._find_outliers(embeddings[selected_method])
            
            # Analysis options
            st.subheader("Analysis Options")
            show_quality_analysis = st.checkbox("Quality Analysis", value=True)
            show_distance_preservation = st.checkbox("Distance Preservation", value=False)
            show_neighborhood_analysis = st.checkbox("Neighborhood Analysis", value=False)
        
        # Main visualization area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Primary embedding visualization
            embedding = embeddings[selected_method]
            
            fig = self.visualizer.create_embedding_scatter(
                embedding=embedding,
                method_name=selected_method,
                labels=labels,
                highlight_indices=self.selected_points,
                show_3d=show_3d
            )
            
            # Handle point selection from plot
            selected_data = st.plotly_chart(fig, use_container_width=True, key="main_embedding_plot")
            
            # Multi-embedding comparison
            if len(embeddings) > 1:
                st.subheader("Method Comparison")
                comparison_fig = self.visualizer.create_multi_embedding_comparison(
                    embeddings, original_data, labels
                )
                st.plotly_chart(comparison_fig, use_container_width=True)
        
        with col2:
            # Point information panel
            st.subheader("Point Information")
            
            if self.selected_points:
                st.write(f"Selected {len(self.selected_points)} points")
                
                # Show statistics for selected points
                selected_embedding = embedding[self.selected_points]
                
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean X', 'Mean Y', 'Std X', 'Std Y'],
                    'Value': [
                        np.mean(selected_embedding[:, 0]),
                        np.mean(selected_embedding[:, 1]),
                        np.std(selected_embedding[:, 0]),
                        np.std(selected_embedding[:, 1])
                    ]
                })
                
                st.dataframe(stats_df, use_container_width=True)
                
                if st.button("Clear Selection"):
                    self.selected_points = []
                    st.experimental_rerun()
            else:
                st.info("No points selected")
        
        # Analysis panels
        if show_quality_analysis and original_data is not None:
            st.subheader("📊 Quality Analysis")
            quality_fig = self.visualizer.create_embedding_quality_analysis(
                original_data, embeddings
            )
            st.plotly_chart(quality_fig, use_container_width=True)
        
        if show_distance_preservation and original_data is not None:
            st.subheader("📏 Distance Preservation Analysis")
            distance_fig = self.visualizer.create_distance_preservation_plot(
                original_data, embedding, selected_method
            )
            st.plotly_chart(distance_fig, use_container_width=True)
        
        if show_neighborhood_analysis and original_data is not None:
            st.subheader("🏘️ Neighborhood Analysis")
            k_neighbors = st.slider("Number of Neighbors", 5, 50, 10)
            neighborhood_fig = self.visualizer.create_neighborhood_analysis(
                original_data, embedding, selected_method, k_neighbors
            )
            st.plotly_chart(neighborhood_fig, use_container_width=True)
    
    def _find_outliers(self, embedding: np.ndarray, method: str = 'isolation_forest') -> List[int]:
        """Find outlier points in the embedding."""
        try:
            if method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                clf = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = clf.fit_predict(embedding)
                outlier_indices = np.where(outlier_labels == -1)[0]
            else:
                # Use distance-based outliers
                from sklearn.neighbors import LocalOutlierFactor
                clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
                outlier_labels = clf.fit_predict(embedding)
                outlier_indices = np.where(outlier_labels == -1)[0]
            
            return outlier_indices.tolist()
        
        except Exception as e:
            logger.error(f"Outlier detection failed: {e}")
            return []
    
    def export_visualization(self, fig: go.Figure, filename: str, format: str = 'png'):
        """Export visualization to file."""
        try:
            if format.lower() == 'png':
                fig.write_image(filename, format='png', width=1200, height=800, scale=2)
            elif format.lower() == 'svg':
                fig.write_image(filename, format='svg')
            elif format.lower() == 'html':
                fig.write_html(filename)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False