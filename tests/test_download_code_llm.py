"""
Test the download_code_llm script functionality.
"""
import os
import sys
import numpy as np
import pytest

# Add parent directory to path so we can import the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_code_samples_creation():
    """Test that code samples can be created."""
    from download_code_llm import create_example_code_samples
    
    # Create samples file in a temporary location for testing
    original_file = "code_samples.txt"
    temp_file = "test_code_samples.txt"
    
    # Save original file name
    original_constant = "download_code_llm.CODE_SAMPLES_FILE"
    original_value = sys.modules["download_code_llm"].CODE_SAMPLES_FILE
    
    try:
        # Override constant for testing
        sys.modules["download_code_llm"].CODE_SAMPLES_FILE = temp_file
        
        samples = create_example_code_samples()
        assert samples is not None
        assert len(samples) > 0
        assert os.path.exists(temp_file)
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    finally:
        # Restore original value
        sys.modules["download_code_llm"].CODE_SAMPLES_FILE = original_value


def test_jl_dimension():
    """Test that JL dimension calculation works."""
    from download_code_llm import jll_dimension
    
    n = 50
    epsilon = 0.3
    k = jll_dimension(n, epsilon)
    assert k > 0
    assert isinstance(k, int)
    
    # With higher epsilon (more distortion allowed), dimension should be smaller
    k2 = jll_dimension(n, 0.5)
    assert k2 < k


def test_orthogonal_basis():
    """Test that orthogonal basis generation works."""
    from download_code_llm import generate_orthogonal_basis
    
    d = 768  # Typical embedding dimension
    k = 100  # Reduced dimension
    
    basis = generate_orthogonal_basis(d, k, seed=42)
    assert basis.shape == (d, k)
    
    # Check orthogonality (approximately)
    product = np.dot(basis.T, basis)
    assert np.allclose(product, np.eye(k), atol=1e-6)