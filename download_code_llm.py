#!/usr/bin/env python
"""
download_code_llm.py - Download and demonstrate a small code-oriented LLM with embedding space.

This script downloads a small code-oriented language model with embedding capabilities,
extracts embeddings from code snippets, and demonstrates using these embeddings with
the OrthoReduce library for dimensionality reduction.
"""

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from orthogonal_projection.projection import generate_orthogonal_basis, project_data, jll_dimension
from orthogonal_projection.evaluation import compute_distortion, nearest_neighbor_overlap

# Constants
MODEL_NAME = "microsoft/codebert-base"
CODE_SAMPLES_FILE = "code_samples.txt"
OUTPUT_DIR = "model_cache"
BATCH_SIZE = 8
MAX_LENGTH = 512

def ensure_model_downloaded():
    """Download the model and tokenizer if not already present."""
    print(f"Ensuring {MODEL_NAME} is downloaded...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Download tokenizer and model (this will cache them if not already present)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    print(f"Model and tokenizer ready at {OUTPUT_DIR}")
    return tokenizer, model

def create_example_code_samples():
    """Create example code samples if they don't exist."""
    if os.path.exists(CODE_SAMPLES_FILE):
        print(f"Using existing code samples from {CODE_SAMPLES_FILE}")
        return
    
    print(f"Creating example code samples in {CODE_SAMPLES_FILE}")
    
    # Example code snippets for embeddings
    code_samples = [
        "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)",
        "class BinarySearchTree:\n    def __init__(self):\n        self.root = None",
        "for i in range(10):\n    if i % 2 == 0:\n        print(f\"{i} is even\")",
        "import numpy as np\nX = np.random.randn(100, 50)\nX = X / np.linalg.norm(X, axis=1, keepdims=True)",
        "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
        "try:\n    with open('file.txt', 'r') as f:\n        data = f.read()\nexcept FileNotFoundError:\n    print('File not found')",
        "def generate_fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b",
        "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.next = None",
        "def memoize(func):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = func(*args)\n        return cache[args]\n    return wrapper",
        "@memoize\ndef fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "x = [1, 2, 3, 4, 5]\ny = [i * i for i in x]\nz = list(map(lambda i: i * i, x))",
        "colors = {'red': '#FF0000', 'green': '#00FF00', 'blue': '#0000FF'}\nfor color, hex_value in colors.items():\n    print(f\"{color}: {hex_value}\")",
    ]
    
    with open(CODE_SAMPLES_FILE, 'w') as f:
        for code in code_samples:
            f.write(code + "\n===\n")
    
    print(f"Created {len(code_samples)} code samples.")
    return code_samples

def load_code_samples():
    """Load code samples from file."""
    if not os.path.exists(CODE_SAMPLES_FILE):
        return create_example_code_samples()
    
    with open(CODE_SAMPLES_FILE, 'r') as f:
        content = f.read()
    
    samples = content.split("\n===\n")
    if samples and not samples[-1].strip():
        samples = samples[:-1]  # Remove last empty sample if exists
    
    print(f"Loaded {len(samples)} code samples")
    return samples

def compute_embeddings(tokenizer, model, code_samples):
    """Compute embeddings for the provided code samples."""
    print("Computing embeddings for code samples...")
    
    # Set model to evaluation mode
    model.eval()
    
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(code_samples), BATCH_SIZE):
        batch = code_samples[i:i+BATCH_SIZE]
        
        # Tokenize the batch
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        # Compute embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Extract [CLS] token embeddings (represents the entire sequence)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(embeddings)
    
    # Concatenate all batches
    embeddings_matrix = np.vstack(all_embeddings)
    print(f"Generated embeddings with shape: {embeddings_matrix.shape}")
    
    # Normalize embeddings
    embeddings_matrix = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    
    return embeddings_matrix

def demonstrate_orthogonal_reduction(embeddings):
    """Demonstrate orthogonal reduction on the embeddings."""
    print("\nDemonstrating OrthoReduce on code embeddings:")
    
    n, d = embeddings.shape
    print(f"Original embedding dimension: {d}")
    
    # Choose epsilon for JL lemma
    epsilon = 0.3
    k = jll_dimension(n, epsilon)
    k = min(k, d)
    print(f"Reduced dimension using JL lemma (epsilon={epsilon}): {k}")
    
    # Generate orthogonal basis and project
    basis = generate_orthogonal_basis(d, k, seed=42)
    reduced_embeddings = project_data(embeddings, basis)
    print(f"Reduced embedding shape: {reduced_embeddings.shape}")
    
    # Evaluate distortion and nearest neighbor preservation
    mean_distortion, max_distortion, *_ = compute_distortion(embeddings, reduced_embeddings)
    nn_overlap = nearest_neighbor_overlap(embeddings, reduced_embeddings, k=3)
    
    print(f"Mean distortion: {mean_distortion:.4f}")
    print(f"Max distortion: {max_distortion:.4f}")
    print(f"Nearest neighbor overlap (k=3): {nn_overlap:.4f}")
    
    return reduced_embeddings

def main():
    """Main function to download model and demonstrate embeddings."""
    print("=== Code LLM with Embeddings Demo ===")
    
    # Ensure model is downloaded and get tokenizer/model
    tokenizer, model = ensure_model_downloaded()
    
    # Load or create example code samples
    code_samples = load_code_samples()
    
    # Compute embeddings
    embeddings = compute_embeddings(tokenizer, model, code_samples)
    
    # Demonstrate orthogonal reduction
    reduced_embeddings = demonstrate_orthogonal_reduction(embeddings)
    
    print("\nDemonstration complete!")
    print(f"Original embedding dimension: {embeddings.shape[1]}")
    print(f"Reduced embedding dimension: {reduced_embeddings.shape[1]}")

if __name__ == "__main__":
    main()