import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from orthogonal_projection.dimensionality_reduction import run_experiment
import matplotlib.pyplot as plt
import io

st.set_page_config(
    page_title="OrthoReduce Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("OrthoReduce: Dimensionality Reduction Visualization")
st.markdown("""
This application visualizes and compares different dimensionality reduction methods
from the OrthoReduce library. Configure parameters and see how different methods 
perform in terms of distortion, correlation, and other metrics.
""")

# Sidebar for parameters
with st.sidebar:
    st.header("Experiment Parameters")
    
    n = st.slider("Number of data points (n)", min_value=100, max_value=20000, value=5000, step=100)
    d = st.slider("Original dimensionality (d)", min_value=50, max_value=5000, value=1200, step=50)
    k = st.slider("Target dimensionality (k)", min_value=2, max_value=100, value=10, step=1)
    epsilon = st.slider("Desired maximum distortion (epsilon)", min_value=0.05, max_value=0.9, value=0.2, step=0.05)
    seed = st.number_input("Random seed", min_value=0, max_value=1000, value=42, step=1)
    sample_size = st.slider("Sample size for distortion computation", min_value=100, max_value=10000, value=2000, step=100)
    
    st.header("Method Selection")
    use_pca = st.checkbox("PCA", value=True)
    use_jll = st.checkbox("JLL (Random Projection)", value=True)
    use_convex = st.checkbox("Convex Hull Projection", value=False)
    use_umap = st.checkbox("UMAP", value=True)
    use_poincare = st.checkbox("Poincaré (Hyperbolic)", value=False)
    use_spherical = st.checkbox("Spherical", value=False)
    
    st.header("Clustering Parameters")
    n_clusters = st.slider("Number of Gaussian clusters", min_value=2, max_value=50, value=10, step=1)
    cluster_std = st.slider("Cluster standard deviation", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

    run_button = st.button("Run Experiment")

# Function to create visualization plots
def create_metric_comparison(results, metric_name, metric_label):
    methods = list(results.keys())
    values = [results[method][metric_name] for method in methods]
    
    df = pd.DataFrame({
        'Method': methods,
        metric_label: values
    })
    
    fig = px.bar(df, x='Method', y=metric_label, 
                 title=f"Comparison of {metric_label} across methods", 
                 color='Method', height=400)
    
    return fig

def create_runtime_comparison(results):
    methods = list(results.keys())
    runtimes = [results[method]['runtime'] for method in methods]
    
    df = pd.DataFrame({
        'Method': methods,
        'Runtime (seconds)': runtimes
    })
    
    fig = px.bar(df, x='Method', y='Runtime (seconds)', 
                 title="Runtime Comparison", 
                 color='Method', height=400)
    
    return fig

def create_radar_chart(results):
    methods = list(results.keys())
    metrics = ['mean_distortion', 'max_distortion', 'rank_correlation', 'kl_divergence', 'l1']
    metric_labels = ['Mean Distortion', 'Max Distortion', 'Rank Correlation', 'KL Divergence', 'L1 Distance']
    
    # Normalize metrics for radar chart
    normalized_results = {}
    for metric in metrics:
        values = [results[method][metric] for method in methods]
        
        # For distortion and divergence metrics, lower is better
        if metric in ['mean_distortion', 'max_distortion', 'kl_divergence', 'l1']:
            min_val = min(values)
            max_val = max(values)
            if max_val > min_val:
                normalized_results[metric] = [1 - (val - min_val) / (max_val - min_val) for val in values]
            else:
                normalized_results[metric] = [0.5 for _ in values]
        else:  # For correlation, higher is better
            min_val = min(values)
            max_val = max(values)
            if max_val > min_val:
                normalized_results[metric] = [(val - min_val) / (max_val - min_val) for val in values]
            else:
                normalized_results[metric] = [0.5 for _ in values]
    
    fig = go.Figure()
    
    for i, method in enumerate(methods):
        fig.add_trace(go.Scatterpolar(
            r=[normalized_results[metric][i] for metric in metrics],
            theta=metric_labels,
            fill='toself',
            name=method
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Method Performance Comparison (normalized, higher is better)"
    )
    
    return fig

# Main content
if run_button:
    with st.spinner("Running experiment, this may take a while..."):
        try:
            # Only run selected methods
            results = run_experiment(
                n=n, 
                d=d, 
                epsilon=epsilon, 
                seed=seed, 
                sample_size=sample_size,
                use_convex=use_convex,
                n_clusters=n_clusters,
                cluster_std=cluster_std,
                use_poincare=use_poincare,
                use_spherical=use_spherical,
            )
            
            # Filter results based on user selection
            filtered_results = {}
            if use_pca and 'PCA' in results:
                filtered_results['PCA'] = results['PCA']
            if use_jll and 'JLL' in results:
                filtered_results['JLL'] = results['JLL']
            if use_convex and 'Convex' in results:
                filtered_results['Convex'] = results['Convex']
            if use_umap and 'UMAP' in results:
                filtered_results['UMAP'] = results['UMAP']
            if use_poincare and 'Poincare' in results:
                filtered_results['Poincare'] = results['Poincare']
            if use_spherical and 'Spherical' in results:
                filtered_results['Spherical'] = results['Spherical']
            
            st.success("Experiment completed successfully!")
            
            # Display results as a table
            st.header("Experiment Results")
            
            # Convert results to a DataFrame for easier display
            results_df = pd.DataFrame()
            for method, metrics in filtered_results.items():
                method_df = pd.DataFrame(metrics, index=[method])
                results_df = pd.concat([results_df, method_df])
            
            st.dataframe(results_df.style.highlight_min(axis=0, subset=['mean_distortion', 'max_distortion', 'kl_divergence', 'l1'])
                          .highlight_max(axis=0, subset=['rank_correlation']))
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Distortion Metrics", "Correlation & Distribution", "Runtime", "Overall Comparison"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_metric_comparison(filtered_results, 'mean_distortion', 'Mean Distortion'), use_container_width=True)
                with col2:
                    st.plotly_chart(create_metric_comparison(filtered_results, 'max_distortion', 'Max Distortion'), use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_metric_comparison(filtered_results, 'rank_correlation', 'Rank Correlation'), use_container_width=True)
                with col2:
                    st.plotly_chart(create_metric_comparison(filtered_results, 'kl_divergence', 'KL Divergence'), use_container_width=True)
            
            with tab3:
                st.plotly_chart(create_runtime_comparison(filtered_results), use_container_width=True)
            
            with tab4:
                st.plotly_chart(create_radar_chart(filtered_results), use_container_width=True)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("Configure parameters in the sidebar and click 'Run Experiment' to start.")
    
    # Show some information about the methods
    st.header("About the Methods")
    
    methods_info = {
        "PCA": "Principal Component Analysis finds the directions of maximum variance in the data and projects onto those directions.",
        "JLL": "Johnson-Lindenstrauss random projections use random orthogonal matrices to project high-dimensional data while approximately preserving pairwise distances.",
        "Convex": "Enhanced Projection onto Convex Sets (POCS) combines JLL projection with convex hull projection.",
        "UMAP": "Uniform Manifold Approximation and Projection preserves both local and global structure.",
        "Poincaré": "Poincaré embeddings map data to the Poincaré disk (hyperbolic space), which can better preserve hierarchical structures.",
        "Spherical": "Spherical embeddings map data to the unit sphere, useful for directional data or when angular distances are important."
    }
    
    for method, description in methods_info.items():
        with st.expander(method):
            st.write(description)
    
    # Show information about metrics
    st.header("About the Metrics")
    
    metrics_info = {
        "Mean Distortion": "Average distortion of pairwise distances between original and reduced spaces. Lower is better.",
        "Max Distortion": "Maximum distortion of pairwise distances. Lower is better.",
        "Rank Correlation": "Spearman correlation between original and reduced pairwise distances. Higher is better.",
        "KL Divergence": "Kullback-Leibler divergence between original and reduced distributions. Lower is better.",
        "L1 Distance": "L1 distance between distributions. Lower is better.",
        "Runtime": "Execution time in seconds. Lower is better."
    }
    
    for metric, description in metrics_info.items():
        with st.expander(metric):
            st.write(description)