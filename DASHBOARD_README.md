# OrthoReduce Interactive Dashboard System

A comprehensive web-based dashboard for visualizing and exploring dimensionality reduction experiment results. Built with Streamlit and Plotly for an intuitive, interactive research experience.

## 🌟 Features

### Core Capabilities
- **Real-time Experiment Monitoring**: Track running experiments with live progress updates
- **Interactive Embedding Visualization**: Pan, zoom, and select points in 2D/3D embeddings
- **Method Comparison Tools**: Dynamic filtering and comparative analysis across methods
- **Parameter Sensitivity Analysis**: Understand hyperparameter impact on performance
- **Advanced Analytics**: Automated insights generation and recommendations
- **Comprehensive Export**: Multiple formats including packages with visualizations

### Dashboard Views
1. **Overview Page**: Key metrics, insights, and method rankings
2. **Method Comparison**: Interactive scatter plots and radar charts
3. **Parameter Analysis**: Sensitivity analysis and optimization recommendations  
4. **Embedding Explorer**: Interactive 2D/3D visualization with point selection
5. **Real-time Monitor**: Live experiment tracking and progress
6. **Export Center**: Data export in multiple formats with analysis reports

## 🚀 Quick Start

### Installation

```bash
# Install dashboard dependencies
pip install streamlit plotly dash colorcet

# Or install all dependencies
pip install -r requirements.txt
```

### Launch Dashboard

```bash
# Basic launch
python launch_dashboard.py

# Custom configuration
python launch_dashboard.py --port 8502 --host 0.0.0.0 --results-dir ./my_results

# Enable monitoring mode
python launch_dashboard.py --monitor --auto-refresh

# Create sample configuration
python launch_dashboard.py --create-config
```

### Access Dashboard
Navigate to `http://localhost:8501` in your web browser.

## 📊 Usage Guide

### Loading Experiments

1. **Automatic Discovery**: Dashboard automatically finds experiment results in the configured directory
2. **File Formats**: Supports JSON and PKL result files
3. **Real-time Updates**: Auto-refreshes to show new experiments

### Interactive Features

#### Embedding Visualization
- **Pan & Zoom**: Mouse interactions for exploration
- **Point Selection**: Click/drag to select data points
- **3D Mode**: Toggle 3D visualization for high-dimensional embeddings
- **Outlier Detection**: Automatic identification of unusual points
- **Quality Metrics**: Real-time distance preservation analysis

#### Method Comparison
- **Dynamic Filtering**: Select methods and metrics to compare
- **Pareto Analysis**: Identify optimal trade-offs between metrics
- **Radar Charts**: Multi-dimensional method comparison
- **Statistical Tests**: Significance analysis between methods

#### Parameter Analysis
- **Sensitivity Plots**: Visualize parameter impact on performance
- **Optimization Guidance**: Recommendations for hyperparameter tuning
- **Convergence Analysis**: Track optimization progress over iterations

### Export Functionality

#### Data Export Options
- **JSON**: Complete structured results
- **CSV**: Tabular format for spreadsheet analysis
- **Excel**: Multi-sheet workbooks with summary and detailed views

#### Visualization Export
- **PNG**: High-resolution static images
- **SVG**: Vector graphics for publications
- **HTML**: Interactive plots for presentations
- **PDF**: Publication-ready figures

#### Comprehensive Packages
- **ZIP Archives**: Complete export with data, reports, and visualizations
- **Analysis Reports**: Markdown reports with insights and recommendations
- **Metadata**: Experiment configuration and export information

## ⚙️ Configuration

### Dashboard Configuration File

Create `dashboard_config.yaml`:

```yaml
# Server settings
host: localhost
port: 8501
results_dir: experiment_results
cache_dir: dashboard_cache

# Dashboard features
auto_refresh: true
refresh_interval: 30
debug: false
theme: light
max_upload_size: 200

# Advanced features
enable_monitoring: true
enable_real_time: true

# Dashboard settings
dashboard_settings:
  default_view: overview
  show_advanced_metrics: true
  enable_exports: true
  cache_embeddings: true
```

### Environment Variables

```bash
export ORTHOREDUCE_RESULTS_DIR="./experiment_results"
export ORTHOREDUCE_CACHE_DIR="./dashboard_cache"
export ORTHOREDUCE_DEBUG="false"
```

## 🏗️ Architecture

### Components

```
dashboard/
├── dashboard.py              # Main Streamlit application
├── dashboard_utils.py        # Data management and processing
├── embedding_viewer.py       # Interactive embedding visualization
├── enhanced_dashboard.py     # Advanced features and analytics
└── launch_dashboard.py       # CLI launcher
```

### Data Flow

1. **Results Loading**: Automatic discovery and loading of experiment files
2. **Data Processing**: Metric computation and normalization
3. **Caching**: Intelligent caching for performance
4. **Visualization**: Real-time plot generation
5. **Export**: On-demand data and visualization export

### Key Classes

#### `DashboardDataManager`
- Handles data loading and caching
- Manages experiment discovery
- Provides real-time status monitoring

#### `InteractivePlotGenerator`
- Creates Plotly-based interactive visualizations
- Method comparison plots
- Performance dashboards

#### `EmbeddingVisualizer`
- Advanced embedding visualization
- 2D/3D scatter plots with selection
- Quality analysis plots

#### `ExportManager`
- Comprehensive export functionality
- Multiple format support
- Report generation

## 📈 Dashboard Views

### 1. Overview Page
- **Key Metrics**: Best methods, runtime comparisons, quality scores
- **Automated Insights**: AI-generated analysis of results
- **Recommendations**: Method selection guidance
- **Performance Summary**: Visual dashboard with multiple metrics

### 2. Method Comparison
- **Interactive Scatter**: Customizable axis selection and filtering
- **Radar Charts**: Multi-dimensional method comparison
- **Statistical Analysis**: Significance testing and effect sizes
- **Pareto Frontier**: Optimal trade-off identification

### 3. Parameter Analysis
- **Sensitivity Analysis**: Hyperparameter impact visualization
- **Optimization Guidance**: Parameter tuning recommendations
- **Convergence Tracking**: Loss function progression
- **Grid Search Results**: Hyperparameter exploration results

### 4. Embedding Explorer
- **Interactive Visualization**: Pan, zoom, select in 2D/3D
- **Quality Analysis**: Distance preservation metrics
- **Neighborhood Analysis**: Local structure preservation
- **Outlier Detection**: Automatic unusual point identification
- **Point Information**: Detailed statistics for selections

### 5. Real-time Monitor
- **Live Tracking**: Running experiment progress
- **Stage Progression**: Pipeline stage completion status
- **Resource Usage**: Memory and CPU monitoring
- **ETA Estimation**: Completion time predictions

### 6. Export Center
- **Data Export**: Multiple formats (JSON, CSV, Excel)
- **Visualization Export**: Various image and interactive formats
- **Report Generation**: Comprehensive analysis reports
- **Package Creation**: Complete export bundles

## 🎨 Customization

### Themes and Styling
The dashboard supports custom themes and styling through Streamlit configuration:

```toml
[theme]
base = "light"  # or "dark"
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### Method Colors
Customize method colors in `visualization.py`:

```python
METHOD_COLORS = {
    'JLL': '#66c2a5',      # Teal
    'PCA': '#fc8d62',      # Orange
    'GAUSSIAN': '#8da0cb', # Purple
    'UMAP': '#e78ac3',     # Pink
    # Add custom colors...
}
```

### Custom Metrics
Add custom evaluation metrics by extending the `EmbeddingAnalyzer` class:

```python
class CustomEmbeddingAnalyzer(EmbeddingAnalyzer):
    @staticmethod
    def compute_custom_metric(X_orig, X_embed):
        # Your custom metric implementation
        return metric_value
```

## 🔧 Advanced Features

### Real-time Monitoring
- **Experiment Discovery**: Automatic detection of new experiments
- **Progress Tracking**: Live updates from running experiments
- **Resource Monitoring**: Memory and CPU usage tracking
- **Alert System**: Notifications for completed or failed experiments

### Interactive Analytics
- **Automated Insights**: AI-generated analysis and recommendations  
- **Statistical Testing**: Significance analysis between methods
- **Pareto Analysis**: Multi-objective optimization insights
- **Convergence Analysis**: Optimization trajectory visualization

### Export System
- **Multi-format Support**: JSON, CSV, Excel, PNG, SVG, PDF, HTML
- **Comprehensive Packages**: ZIP archives with complete analysis
- **Custom Reports**: Markdown reports with insights and recommendations
- **Publication Ready**: High-resolution figures suitable for papers

## 🐛 Troubleshooting

### Common Issues

#### Dashboard Won't Start
```bash
# Check dependencies
python launch_dashboard.py --check-deps

# Install missing dependencies
python launch_dashboard.py --install-deps

# Try different port
python launch_dashboard.py --port 8502
```

#### No Experiments Found
- Verify `results_dir` contains experiment files
- Check file formats (JSON/PKL supported)
- Ensure proper file naming conventions
- Check file permissions

#### Slow Performance
- Enable caching: Set `cache_embeddings: true` in config
- Reduce data size: Use sampling for large datasets
- Clear cache: Delete `dashboard_cache` directory
- Check available memory

#### Visualization Issues
- Update browser: Ensure modern browser with JavaScript enabled
- Clear browser cache: Remove stored dashboard data
- Check network: Ensure stable connection to dashboard
- Try different visualization backend

### Debug Mode
Enable debug mode for detailed error information:

```bash
python launch_dashboard.py --debug
```

## 🤝 Integration

### With Existing Experiments
```python
from orthogonal_projection.dashboard_utils import ExperimentDatabase

# Store experiment results
db = ExperimentDatabase()
experiment_id = db.store_experiment(
    name="my_experiment",
    config=experiment_config,
    results=results_dict,
    embeddings=embeddings_dict
)
```

### Custom Analysis Integration
```python
from orthogonal_projection.dashboard_utils import DataProcessor

# Add custom analysis
results = load_experiment_results()
quality_scores = DataProcessor.compute_quality_scores(results)
pareto_methods = DataProcessor.get_pareto_frontier(results)
```

### Export Integration
```python
from orthogonal_projection.enhanced_dashboard import ExportManager

# Custom export
export_manager = ExportManager()
package = export_manager.create_export_package(
    results, insights, recommendations, figures
)
```

## 📚 API Reference

### Dashboard Components

#### `DashboardDataManager`
```python
manager = DashboardDataManager(results_dir="./results")
experiments = manager.get_available_experiments()
results = manager.load_experiment_results(experiment_path)
```

#### `InteractivePlotGenerator`
```python
generator = InteractivePlotGenerator()
fig = generator.create_method_comparison_scatter(results)
dashboard_fig = generator.create_performance_dashboard(results)
```

#### `EmbeddingVisualizer`
```python
visualizer = EmbeddingVisualizer()
fig = visualizer.create_embedding_scatter(embedding, method_name)
quality_fig = visualizer.create_embedding_quality_analysis(X_orig, embeddings)
```

#### `ExportManager`
```python
exporter = ExportManager()
data_bytes = exporter.export_results_data(results, format='json')
report = exporter.create_comprehensive_report(results, insights, recommendations)
package = exporter.create_export_package(results, insights, recommendations)
```

## 🔮 Future Enhancements

- **Real-time Collaboration**: Multi-user experiment sharing
- **Cloud Integration**: AWS/GCP experiment tracking
- **Advanced ML**: Automated hyperparameter optimization
- **Extended Formats**: More data import/export options
- **Mobile Support**: Responsive design for mobile devices
- **API Access**: RESTful API for programmatic access

## 📄 License

This dashboard system is part of the OrthoReduce project and follows the same licensing terms.

## 🙏 Acknowledgments

- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library
- **scikit-learn**: Machine learning utilities
- **NumPy/SciPy**: Numerical computing foundations