# Poincaré (Hyperbolic) Embedding Implementation

This document describes the rigorous mathematical implementation of Poincaré embeddings in OrthoReduce, following established literature and providing numerically stable algorithms for hyperbolic geometry operations.

## Mathematical Foundation

### Hyperbolic Space and the Poincaré Ball Model

The Poincaré ball model B^n_c represents n-dimensional hyperbolic space with curvature -c < 0:

```
B^n_c = {x ∈ ℝ^n : c||x||^2 < 1}
```

Key properties:
- **Negative curvature**: Enables natural representation of hierarchical/tree-like data
- **Exponential volume growth**: More "space" near the boundary than in Euclidean geometry  
- **Conformal model**: Angles are preserved, distances are distorted

### Core Operations

#### 1. Conformal Factor
```
λ_c^x = 2 / (1 - c||x||^2)
```
This factor relates the Euclidean metric to the hyperbolic metric at point x.

#### 2. Möbius Addition (⊕_c)
The gyrovector addition operation:
```
x ⊕_c y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
```

Properties:
- Associativity (up to gyration)
- Commutativity: x ⊕_c y = y ⊕_c x
- Identity: x ⊕_c 0 = x
- Inverse: x ⊕_c (-x) = 0

#### 3. Exponential Map
Maps tangent vectors to points on the manifold:
```
exp^c_x(v) = x ⊕_c (tanh(√c λ_c^x ||v|| / 2) v / (√c ||v||))
```

#### 4. Logarithmic Map  
Maps manifold points to tangent space:
```
log^c_x(y) = (2 / (√c λ_c^x)) artanh(√c ||-x ⊕_c y||) (-x ⊕_c y) / ||-x ⊕_c y||
```

#### 5. Hyperbolic Distance
```
d_B^c(x,y) = (2/√c) artanh(√c ||-x ⊕_c y||)
```

## Riemannian Optimization

### Riemannian Gradients
Convert Euclidean gradients to Riemannian:
```
grad_R f(x) = (1 - c||x||²)² / 4 * grad_E f(x)
```

### RSGD (Riemannian Stochastic Gradient Descent)
```
x_{t+1} = exp^c_x(-lr * grad_R f(x))
```

### RAdam (Riemannian Adam)
Maintains first and second moment estimates in tangent space with bias correction:
```
m_t = β₁ m_{t-1} + (1-β₁) grad_R f(x)
v_t = β₂ v_{t-1} + (1-β₂) (grad_R f(x))²
x_{t+1} = exp^c_x(-lr * m̂_t / (√v̂_t + ε))
```

## Loss Functions

### 1. Stress Loss (MDS-style)
Minimizes differences between hyperbolic and input distances:
```
L_stress = Σ_{i,j} w_{ij} (d_H(y_i, y_j) - d_{ij})²
```

### 2. Triplet Loss
Encourages similar points closer than dissimilar ones:
```
L_triplet = Σ max(0, d_H(a,p) - d_H(a,n) + margin)
```

### 3. Neighborhood Component Analysis (NCA)
Maximizes probability of correct neighbor classification:
```
L_NCA = -Σ_i log(Σ_{j: y_j = y_i} p_{ij})
```
where p_{ij} = exp(-d_H(i,j)) / Σ_k exp(-d_H(i,k))

## Numerical Stability

### Boundary Management
- **Norm clipping**: Keep ||x|| < (1-ε)/√c to avoid singularities
- **Gradient clipping**: Limit gradient norms to prevent large steps
- **Artanh clipping**: Prevent artanh from approaching ±1

### Precision Issues
- Use double precision for intermediate calculations
- Add small ε to denominators to prevent division by zero
- Monitor condition numbers in optimization

## Usage Examples

### Basic Usage
```python
from orthogonal_projection import run_poincare

# Generate hierarchical data
X = generate_hierarchical_data(n_samples=200, n_features=50)

# Run Poincaré embedding
Y, runtime = run_poincare(
    X, k=3,              # Target dimension
    c=1.0,               # Curvature parameter
    n_epochs=50,         # Optimization epochs
    lr=0.01,             # Learning rate
    optimizer='radam',   # Optimizer choice
    loss_fn='stress',    # Loss function
    seed=42
)

print(f"Embedding shape: {Y.shape}")
print(f"Point norms: {np.linalg.norm(Y, axis=1)}")
```

### Advanced Usage with Custom Parameters
```python
from orthogonal_projection.hyperbolic import HyperbolicEmbedding

embedding = HyperbolicEmbedding(
    n_components=3,
    c=1.0,                    # Higher curvature for more hierarchical data
    lr=0.01,                  # Learning rate
    n_epochs=100,             # More epochs for convergence
    optimizer='radam',        # Riemannian Adam
    loss_fn='nca',           # Supervised loss
    init_method='pca',        # PCA initialization
    regularization=0.01,      # L2 regularization
    batch_size=256,           # Mini-batch size
    seed=42
)

Y = embedding.fit_transform(X, y_labels)
```

### Parameter Recommendations

#### Curvature (c)
- **c=0.1**: Light curvature, close to Euclidean
- **c=1.0**: Standard curvature, good default
- **c=2.0**: High curvature, for very hierarchical data

#### Learning Rate
- **0.001**: Conservative, guaranteed convergence
- **0.01**: Good default balance
- **0.05**: Aggressive, faster convergence but less stable

#### Epochs
- **10-20**: Quick results, may not fully converge
- **50-100**: Good balance for most data
- **200+**: High-quality embeddings for complex data

#### Loss Functions
- **'stress'**: Unsupervised, preserves all pairwise distances
- **'nca'**: Supervised, focuses on neighborhood preservation
- **'triplet'**: Supervised, emphasizes relative comparisons

## Quality Metrics

The implementation provides comprehensive quality assessment:

1. **Rank Correlation**: Spearman correlation between input and embedded distances
2. **Mean/Max Distortion**: JLL-style distortion metrics
3. **Convergence Monitoring**: Loss history tracking
4. **Boundary Analysis**: Fraction of points near Poincaré ball boundary

## Mathematical References

1. **Nickel, M. & Kiela, D. (2017)**. "Poincaré Embeddings for Learning Hierarchical Representations". NIPS.

2. **Ganea, O., Bécigneul, G. & Hofmann, T. (2018)**. "Hyperbolic Neural Networks". NIPS.

3. **Chami, I., Ying, R., Ré, C. & Leskovec, J. (2019)**. "Hyperbolic Graph Convolutional Neural Networks". NIPS.

4. **Wilson, B. & Leimeister, M. (2018)**. "Gradient Descent in Hyperbolic Space". arXiv preprint.

5. **Ungar, A. (2008)**. "A Gyrovector Space Approach to Hyperbolic Geometry". Synthesis Lectures on Mathematics and Statistics.

## Performance Characteristics

### Computational Complexity
- **Hyperbolic operations**: O(d) per operation where d is embedding dimension
- **Optimization**: O(n × d × epochs) for n points
- **Distance computation**: O(n²d) for full distance matrix

### Memory Usage
- **Point storage**: O(nd) for embeddings
- **Optimizer state**: O(nd) additional for Adam moments
- **Gradient computation**: O(nd) temporary storage

### Scalability
The implementation scales well for:
- **Points**: Up to ~10,000 points efficiently
- **Dimensions**: Embedding dimensions up to ~50
- **Features**: Input dimensions up to ~1,000

For larger datasets, consider:
- Mini-batch optimization (default batch_size=256)
- Reduced epochs with learning rate scheduling
- Sparse gradient updates for large feature spaces

## Integration with OrthoReduce Pipeline

The Poincaré embedding integrates seamlessly with the existing OrthoReduce framework:

```python
from orthogonal_projection import run_experiment

results = run_experiment(
    n=500, d=100, k=10,
    methods=['pca', 'jll', 'poincare'],  # Include Poincaré
    use_poincare=True,                   # Enable hyperbolic methods
    seed=42
)

print("Poincaré results:", results['Poincare'])
```

The implementation maintains backward compatibility while providing advanced hyperbolic geometry capabilities for users who need them.