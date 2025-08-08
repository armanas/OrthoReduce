"""
Dashboard Utilities and Data Management

Provides utilities for loading, caching, and processing experiment data
for the interactive dashboard system.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import json
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import asdict
import hashlib
import streamlit as st

# Import OrthoReduce components
try:
    from .results_aggregator import AggregatedResults, ExperimentResult
    from .experiment_config import ExperimentConfig
    from .evaluation import compute_distortion, rank_correlation, nearest_neighbor_overlap
except ImportError:
    from results_aggregator import AggregatedResults, ExperimentResult
    from experiment_config import ExperimentConfig
    from evaluation import compute_distortion, rank_correlation, nearest_neighbor_overlap

logger = logging.getLogger(__name__)


class ExperimentDatabase:
    """SQLite-based database for storing experiment metadata and results."""
    
    def __init__(self, db_path: str = "dashboard_cache.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'completed',
                    results_path TEXT,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS method_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    method_name TEXT NOT NULL,
                    rank_correlation REAL,
                    mean_distortion REAL,
                    max_distortion REAL,
                    runtime REAL,
                    memory_usage REAL,
                    compression_ratio REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                );
                
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    method_name TEXT NOT NULL,
                    embedding_data BLOB,
                    original_data_shape TEXT,
                    embedding_shape TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name);
                CREATE INDEX IF NOT EXISTS idx_method_results_exp ON method_results(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_embeddings_exp ON embeddings(experiment_id);
            """)
    
    def store_experiment(self, name: str, config: ExperimentConfig, 
                        results: Dict[str, Any], embeddings: Optional[Dict[str, np.ndarray]] = None) -> int:
        """Store experiment results in database."""
        config_hash = config.get_hash() if config else hashlib.md5(name.encode()).hexdigest()[:8]
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert experiment
            cursor = conn.execute("""
                INSERT INTO experiments (name, config_hash, metadata, status)
                VALUES (?, ?, ?, 'completed')
            """, (name, config_hash, json.dumps(asdict(config) if config else {})))
            
            experiment_id = cursor.lastrowid
            
            # Store method results
            methods = [k for k in results.keys() if k != '_metadata']
            for method in methods:
                metrics = results[method]
                conn.execute("""
                    INSERT INTO method_results 
                    (experiment_id, method_name, rank_correlation, mean_distortion, 
                     max_distortion, runtime, memory_usage, compression_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id, method,
                    metrics.get('rank_correlation', 0),
                    metrics.get('mean_distortion', 0),
                    metrics.get('max_distortion', 0),
                    metrics.get('runtime', 0),
                    metrics.get('memory_usage', 0),
                    metrics.get('compression_ratio', 1)
                ))
            
            # Store embeddings if provided
            if embeddings:
                for method, embedding in embeddings.items():
                    conn.execute("""
                        INSERT INTO embeddings 
                        (experiment_id, method_name, embedding_data, embedding_shape)
                        VALUES (?, ?, ?, ?)
                    """, (
                        experiment_id, method,
                        pickle.dumps(embedding),
                        str(embedding.shape)
                    ))
        
        return experiment_id
    
    def get_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get list of stored experiments."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT e.*, COUNT(mr.id) as method_count
                FROM experiments e
                LEFT JOIN method_results mr ON e.id = mr.experiment_id
                GROUP BY e.id
                ORDER BY e.modified_at DESC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_experiment_results(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get results for a specific experiment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get method results
            cursor = conn.execute("""
                SELECT * FROM method_results WHERE experiment_id = ?
            """, (experiment_id,))
            
            method_results = {}
            for row in cursor.fetchall():
                method_results[row['method_name']] = {
                    'rank_correlation': row['rank_correlation'],
                    'mean_distortion': row['mean_distortion'],
                    'max_distortion': row['max_distortion'],
                    'runtime': row['runtime'],
                    'memory_usage': row['memory_usage'],
                    'compression_ratio': row['compression_ratio']
                }
            
            return method_results if method_results else None
    
    def get_embeddings(self, experiment_id: int, method_name: str) -> Optional[np.ndarray]:
        """Get embedding data for a specific method."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT embedding_data FROM embeddings 
                WHERE experiment_id = ? AND method_name = ?
            """, (experiment_id, method_name))
            
            row = cursor.fetchone()
            if row:
                return pickle.loads(row[0])
            return None


class DataProcessor:
    """Process and analyze experiment data for dashboard display."""
    
    @staticmethod
    def compute_quality_scores(results: Dict[str, Any]) -> Dict[str, float]:
        """Compute composite quality scores for methods."""
        quality_scores = {}
        methods = [k for k in results.keys() if k != '_metadata']
        
        for method in methods:
            metrics = results[method]
            correlation = metrics.get('rank_correlation', 0)
            distortion = metrics.get('mean_distortion', 1)
            
            # Quality score: higher correlation, lower distortion is better
            quality_score = correlation / (distortion + 0.01)
            quality_scores[method] = quality_score
        
        return quality_scores
    
    @staticmethod
    def get_pareto_frontier(results: Dict[str, Any], 
                           x_metric: str = 'runtime', 
                           y_metric: str = 'rank_correlation') -> List[str]:
        """Identify methods on the Pareto frontier."""
        methods = [k for k in results.keys() if k != '_metadata']
        points = []
        
        for method in methods:
            x_val = results[method].get(x_metric, 0)
            y_val = results[method].get(y_metric, 0)
            points.append((method, x_val, y_val))
        
        # Sort by x-metric (assuming lower is better for runtime)
        points.sort(key=lambda p: p[1])
        
        pareto_methods = []
        best_y = -float('inf')
        
        for method, x_val, y_val in points:
            if y_val > best_y:  # Higher y is better (correlation)
                pareto_methods.append(method)
                best_y = y_val
        
        return pareto_methods
    
    @staticmethod
    def normalize_metrics(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Normalize metrics to 0-1 scale for comparison."""
        methods = [k for k in results.keys() if k != '_metadata']
        metrics = ['rank_correlation', 'mean_distortion', 'runtime', 'compression_ratio']
        
        # Collect all values for normalization
        all_values = {metric: [] for metric in metrics}
        for method in methods:
            for metric in metrics:
                if metric in results[method]:
                    all_values[metric].append(results[method][metric])
        
        # Compute min/max for each metric
        normalization_params = {}
        for metric in metrics:
            if all_values[metric]:
                normalization_params[metric] = {
                    'min': min(all_values[metric]),
                    'max': max(all_values[metric])
                }
            else:
                normalization_params[metric] = {'min': 0, 'max': 1}
        
        # Normalize values
        normalized_results = {}
        for method in methods:
            normalized_results[method] = {}
            for metric in metrics:
                if metric in results[method]:
                    value = results[method][metric]
                    min_val = normalization_params[metric]['min']
                    max_val = normalization_params[metric]['max']
                    
                    if max_val > min_val:
                        if metric in ['mean_distortion', 'runtime']:
                            # Lower is better, so invert
                            normalized = 1 - (value - min_val) / (max_val - min_val)
                        else:
                            # Higher is better
                            normalized = (value - min_val) / (max_val - min_val)
                    else:
                        normalized = 0.5  # All values are the same
                    
                    normalized_results[method][metric] = normalized
                else:
                    normalized_results[method][metric] = 0.0
        
        return normalized_results
    
    @staticmethod
    def create_method_summary_df(results: Dict[str, Any]) -> pd.DataFrame:
        """Create a DataFrame with method summary statistics."""
        methods = [k for k in results.keys() if k != '_metadata']
        summary_data = []
        
        quality_scores = DataProcessor.compute_quality_scores(results)
        pareto_methods = DataProcessor.get_pareto_frontier(results)
        
        for method in methods:
            metrics = results[method]
            summary_data.append({
                'Method': method,
                'Rank Correlation': metrics.get('rank_correlation', 0),
                'Mean Distortion': metrics.get('mean_distortion', 0),
                'Runtime (s)': metrics.get('runtime', 0),
                'Memory (GB)': metrics.get('memory_usage', 0),
                'Compression': f"{metrics.get('compression_ratio', 1):.1f}x",
                'Quality Score': quality_scores.get(method, 0),
                'Pareto Optimal': '✅' if method in pareto_methods else '❌'
            })
        
        df = pd.DataFrame(summary_data)
        return df.sort_values('Quality Score', ascending=False)


class EmbeddingAnalyzer:
    """Analyze embedding quality and structure."""
    
    @staticmethod
    def analyze_embedding_structure(X_orig: np.ndarray, X_embed: np.ndarray) -> Dict[str, Any]:
        """Analyze the structure preservation of an embedding."""
        analysis = {}
        
        try:
            # Distance preservation analysis
            n_samples = min(1000, X_orig.shape[0])  # Sample for efficiency
            indices = np.random.choice(X_orig.shape[0], n_samples, replace=False)
            X_orig_sample = X_orig[indices]
            X_embed_sample = X_embed[indices]
            
            # Compute pairwise distances
            from scipy.spatial.distance import pdist, squareform
            orig_distances = squareform(pdist(X_orig_sample))
            embed_distances = squareform(pdist(X_embed_sample))
            
            # Correlation between distance matrices
            orig_flat = orig_distances[np.triu_indices(n_samples, k=1)]
            embed_flat = embed_distances[np.triu_indices(n_samples, k=1)]
            
            distance_correlation = np.corrcoef(orig_flat, embed_flat)[0, 1]
            analysis['distance_correlation'] = distance_correlation
            
            # Local neighborhood preservation
            k = min(10, n_samples - 1)
            neighborhood_preservation = 0.0
            
            for i in range(n_samples):
                orig_neighbors = np.argsort(orig_distances[i])[:k+1][1:]  # Exclude self
                embed_neighbors = np.argsort(embed_distances[i])[:k+1][1:]
                
                # Compute overlap
                overlap = len(set(orig_neighbors) & set(embed_neighbors)) / k
                neighborhood_preservation += overlap
            
            analysis['neighborhood_preservation'] = neighborhood_preservation / n_samples
            
            # Compute stress (Kruskal's normalized)
            stress = np.sum((orig_flat - embed_flat)**2) / np.sum(orig_flat**2)
            analysis['stress'] = stress
            
            # Intrinsic dimensionality estimation (simplified)
            analysis['effective_dimensionality'] = EmbeddingAnalyzer.estimate_intrinsic_dim(X_embed_sample)
            
        except Exception as e:
            logger.error(f"Embedding analysis failed: {e}")
            analysis = {
                'distance_correlation': 0.0,
                'neighborhood_preservation': 0.0,
                'stress': 1.0,
                'effective_dimensionality': X_embed.shape[1]
            }
        
        return analysis
    
    @staticmethod
    def estimate_intrinsic_dim(X: np.ndarray, n_neighbors: int = 10) -> float:
        """Estimate intrinsic dimensionality using correlation dimension."""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            n_samples = min(500, X.shape[0])
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_sample = X[indices]
            
            nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X_sample)
            distances, _ = nbrs.kneighbors(X_sample)
            distances = distances[:, 1:]  # Exclude self
            
            # Estimate local dimensionality using correlation dimension
            log_distances = np.log(distances + 1e-10)
            dims = []
            
            for i in range(n_samples):
                sorted_dists = np.sort(log_distances[i])
                if len(sorted_dists) > 2:
                    # Linear regression in log space
                    x = np.arange(1, len(sorted_dists) + 1)
                    slope, _ = np.polyfit(sorted_dists, np.log(x), 1)
                    dims.append(max(1, slope))
            
            return np.median(dims) if dims else X.shape[1]
            
        except Exception:
            return X.shape[1]
    
    @staticmethod
    def compute_embedding_metrics(X_orig: np.ndarray, X_embed: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive embedding quality metrics."""
        metrics = {}
        
        try:
            # Standard metrics
            mean_dist, max_dist, _, _ = compute_distortion(X_orig, X_embed)
            rank_corr = rank_correlation(X_orig, X_embed)
            nn_overlap = nearest_neighbor_overlap(X_orig, X_embed)
            
            metrics.update({
                'mean_distortion': mean_dist,
                'max_distortion': max_dist,
                'rank_correlation': rank_corr,
                'nn_overlap': nn_overlap
            })
            
            # Additional structural analysis
            structure_analysis = EmbeddingAnalyzer.analyze_embedding_structure(X_orig, X_embed)
            metrics.update(structure_analysis)
            
        except Exception as e:
            logger.error(f"Metric computation failed: {e}")
            metrics = {
                'mean_distortion': float('inf'),
                'max_distortion': float('inf'),
                'rank_correlation': 0.0,
                'nn_overlap': 0.0,
                'distance_correlation': 0.0,
                'neighborhood_preservation': 0.0,
                'stress': 1.0,
                'effective_dimensionality': X_embed.shape[1] if X_embed is not None else 0
            }
        
        return metrics


class CacheManager:
    """Manage caching for dashboard data and computations."""
    
    def __init__(self, cache_dir: str = "dashboard_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_age = timedelta(hours=24)  # Cache for 24 hours
    
    def get_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_str = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file is still valid."""
        if not cache_file.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < self.max_age
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if valid."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if self.is_cache_valid(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        
        return None
    
    def cache_result(self, cache_key: str, result: Any):
        """Cache a computation result."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")
    
    def clear_cache(self):
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception:
                pass


@st.cache_data(ttl=3600)
def load_and_process_experiments(results_dir: str) -> List[Dict[str, Any]]:
    """Load and process all experiments in directory (cached)."""
    results_path = Path(results_dir)
    experiments = []
    
    for json_file in results_path.rglob("*.json"):
        if "results" in json_file.name:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, dict) and data:
                    # Process and enhance data
                    processed_data = DataProcessor.create_method_summary_df(data)
                    
                    experiments.append({
                        "name": json_file.stem,
                        "path": str(json_file),
                        "data": data,
                        "summary": processed_data,
                        "modified": datetime.fromtimestamp(json_file.stat().st_mtime)
                    })
                    
            except Exception as e:
                logger.error(f"Failed to process {json_file}: {e}")
                continue
    
    return sorted(experiments, key=lambda x: x["modified"], reverse=True)


@st.cache_data(ttl=1800)
def compute_method_comparison_data(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute method comparison data with caching."""
    methods = [k for k in results.keys() if k != '_metadata']
    
    comparison_data = {
        'quality_scores': DataProcessor.compute_quality_scores(results),
        'pareto_methods': DataProcessor.get_pareto_frontier(results),
        'normalized_metrics': DataProcessor.normalize_metrics(results),
        'method_rankings': []
    }
    
    # Rank methods by quality score
    quality_scores = comparison_data['quality_scores']
    ranking = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
    comparison_data['method_rankings'] = ranking
    
    return comparison_data