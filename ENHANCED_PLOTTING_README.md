# Enhanced Non-Euclidean Visualization System

This document describes the comprehensive enhancements made to the OrthoReduce visualization system for spherical and hyperbolic embeddings with mathematical rigor and publication-quality output.

## Overview

The enhanced plotting functionality provides mathematically rigorous, publication-ready visualizations for non-Euclidean embeddings with advanced geometric features and educational annotations. The system focuses on:

1. **Mathematical Accuracy**: Proper geodesics, curvature indicators, and geometric properties
2. **Educational Value**: Annotations explaining non-Euclidean concepts 
3. **Publication Quality**: High-resolution, professionally styled visualizations
4. **Research Integration**: Seamless integration with existing OrthoReduce pipeline

## Enhanced Features

### Spherical Embeddings (`plot_spherical_embedding`)

**Mathematical Rigor:**
- High-quality sphere mesh with proper lighting and shading
- Mathematically accurate geodesic paths (great circle arcs)
- Great circle visualization for geometric context
- Stereographic projection mapping from 3D sphere to 2D plane
- Proper handling of numerical precision near poles

**Visual Features:**
- Enhanced 3D rendering with optimal viewing angles
- Curvature indicators showing positive curvature effects
- North/South pole markers with coordinate system
- Configurable mesh quality and lighting parameters
- Side-by-side stereographic projection comparison

**Example Usage:**
```python
from orthogonal_projection.advanced_plotting import plot_specialized_embedding

fig = plot_specialized_embedding(
    spherical_embedding,  # 3D points on unit sphere
    'spherical',
    labels=cluster_labels,
    show_geodesics=True,      # Show great circle arcs
    show_great_circles=True,  # Show equator and meridians 
    show_stereographic=True,  # Add projection subplot
    mesh_quality=50           # High-quality mesh
)
```

### Hyperbolic Embeddings (`plot_poincare_disk`)

**Mathematical Rigor:**
- Precise hyperbolic geodesics as circular arcs orthogonal to boundary
- Horocycles (limit cycles) showing hyperbolic geometry
- Klein disk model comparison with straight-line geodesics
- Hyperbolic distance grid showing constant distance curves
- Proper handling of boundary proximity and numerical stability

**Visual Features:**
- Enhanced Poincaré disk with ideal boundary circle
- Curvature indicators showing negative curvature effects
- Klein model subplot showing geodesic differences
- Educational annotations explaining hyperbolic properties
- Mathematical accuracy indicators (curvature values)

**Example Usage:**
```python
fig = plot_specialized_embedding(
    poincare_embedding,  # 2D points in unit disk
    'hyperbolic', 
    labels=cluster_labels,
    show_geodesics=True,      # Hyperbolic geodesic arcs
    show_horocycles=True,     # Limit cycles
    show_klein_model=True,    # Klein disk comparison
    show_curvature_grid=True, # Distance contours
    curvature=1.0            # Hyperbolic curvature
)
```

### Geometric Comparison Analysis (`plot_geometric_comparison`)

**Features:**
- Side-by-side comparison of Euclidean, spherical, and hyperbolic embeddings
- Mathematical accuracy analysis with distance preservation metrics
- Curvature effect visualization showing geometric properties
- Distortion analysis with rank correlation and mean error metrics
- Integrated legends explaining geometric differences

**Example Usage:**
```python
from orthogonal_projection.advanced_plotting import AdvancedPlotter

embeddings = {
    'PCA': euclidean_2d,
    'Spherical': spherical_3d, 
    'Poincaré': hyperbolic_2d
}

embedding_types = {
    'PCA': 'euclidean',
    'Spherical': 'spherical',
    'Poincaré': 'hyperbolic'
}

plotter = AdvancedPlotter()
fig = plotter.plot_geometric_comparison(
    embeddings, embedding_types, original_data,
    show_distortion_analysis=True,
    show_curvature_effects=True
)
```

### Curvature Effects Analysis (`plot_curvature_effects_comparison`)

**Educational Features:**
- Visual demonstration of how curvature affects geometric properties
- Triangle angle sum comparison across geometries:
  - Spherical: angles > 180° (positive curvature)
  - Euclidean: angles = 180° (zero curvature)  
  - Hyperbolic: angles < 180° (negative curvature)
- Same data points embedded in different geometries
- Mathematical annotations explaining geometric principles

## Integration with OrthoReduce Pipeline

The enhanced visualizations integrate seamlessly with existing OrthoReduce functionality:

### Direct Integration
```python
# In existing dimensionality reduction pipeline
from orthogonal_projection.spherical_embeddings import adaptive_spherical_embedding
from orthogonal_projection.hyperbolic import run_poincare_optimized
from orthogonal_projection.advanced_plotting import create_geometric_analysis_report

# Generate embeddings
Y_spherical, info_sph = adaptive_spherical_embedding(X, k=3)
Y_hyperbolic, runtime_hyp = run_poincare_optimized(X, k=2)

# Create comprehensive analysis
embeddings = {'Spherical': Y_spherical, 'Poincaré': Y_hyperbolic}
embedding_types = {'Spherical': 'spherical', 'Poincaré': 'hyperbolic'}

report_files = create_geometric_analysis_report(
    embeddings, embedding_types, X,
    output_dir="geometric_analysis",
    include_curvature_analysis=True
)
```

### Convenience Functions
```python
from orthogonal_projection.advanced_plotting import (
    plot_educational_geometry_comparison,
    create_geometric_analysis_report
)

# Educational visualization
fig = plot_educational_geometry_comparison(
    sample_data,
    title="Understanding Non-Euclidean Geometries"
)

# Comprehensive analysis report  
report_files = create_geometric_analysis_report(
    embeddings, embedding_types, original_data
)
```

## Mathematical Background

### Spherical Geometry (Positive Curvature)
- **Geodesics**: Great circle arcs (shortest paths on sphere)
- **Distance Formula**: `d(x,y) = r * arccos(⟨x,y⟩/r²)` where r is radius
- **Curvature**: `K = 1/r²` (positive, constant)
- **Triangle Property**: Angle sum > π (spherical excess)

### Hyperbolic Geometry (Negative Curvature) 
- **Poincaré Model**: Unit disk with hyperbolic metric
- **Geodesics**: Circular arcs orthogonal to boundary + diameters
- **Distance Formula**: `d(x,y) = (2/√c) * artanh(√c * ||-x ⊕ y||)`  
- **Curvature**: `K = -c` (negative, constant)
- **Triangle Property**: Angle sum < π (angular defect)

### Klein Model Comparison
- **Geodesics**: Straight lines (chord distance)
- **Distortion**: Angles distorted, distances preserved differently
- **Mapping**: `Poincaré → Klein: (x,y) ↦ (2x/(1+x²+y²), 2y/(1+x²+y²))`

## Quality Assurance

### Mathematical Accuracy
- All geometric operations use numerically stable algorithms
- Proper handling of boundary conditions and singularities
- Verification against known geometric properties
- Consistent with mathematical literature

### Visual Quality
- Publication-ready 300 DPI output
- Professional color schemes and typography
- Proper aspect ratios and spatial relationships
- Clear legends and annotations

### Performance
- Efficient algorithms for large datasets
- Configurable quality vs speed trade-offs
- Graceful degradation for edge cases
- Memory-efficient batch processing

## Testing and Validation

Run the comprehensive test suite:
```bash
python3 test_enhanced_plotting.py
```

This generates 9 different visualization types demonstrating all enhanced features:
- Basic and enhanced spherical visualizations
- Basic and enhanced hyperbolic visualizations  
- Geometric comparison analysis
- Curvature effects demonstration
- Educational comparison plots

## API Reference

### Main Classes
- `AdvancedPlotter`: Core plotting class with enhanced methods
- `InteractivePlotter`: Plotly-based interactive visualizations

### Key Methods
- `plot_spherical_embedding()`: Enhanced spherical visualization
- `plot_poincare_disk()`: Enhanced hyperbolic visualization
- `plot_geometric_comparison()`: Multi-geometry analysis
- `plot_curvature_effects_comparison()`: Educational curvature demo

### Convenience Functions
- `plot_specialized_embedding()`: Unified interface for non-Euclidean plots
- `create_geometric_analysis_report()`: Comprehensive analysis pipeline
- `plot_educational_geometry_comparison()`: Teaching-focused visualization

## Future Enhancements

Potential areas for further development:
- Upper half-plane hyperbolic model support
- Spherical Voronoi diagrams and tessellations
- Interactive 3D manipulations with plotly
- Animation support for geometric transformations
- Integration with additional hyperbolic embedding methods

## References

1. Nickel & Kiela (2017): "Poincaré Embeddings for Learning Hierarchical Representations"
2. Ganea et al. (2018): "Hyperbolic Neural Networks"  
3. Wilson & Hancock (2010): "Spherical embedding and classification"
4. Classical differential geometry references for mathematical foundations

---

**Note**: All mathematical formulations have been verified for accuracy and are consistent with standard differential geometry literature. The visualizations preserve geometric properties while providing intuitive understanding of non-Euclidean concepts.