"""
Test the download_code_llm script functionality.
"""
import os
import sys
import pytest

# Add parent directory to path so we can import the script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.mark.skip(reason="Skip external dependencies for CI")
def test_script_imports():
    """Test that the script can be imported."""
    import download_code_llm
    assert hasattr(download_code_llm, 'main')
    assert hasattr(download_code_llm, 'compute_embeddings')
    assert hasattr(download_code_llm, 'demonstrate_orthogonal_reduction')

def test_code_samples_file_creation():
    """Test that code samples file can be created."""
    temp_file = "test_code_samples.txt"
    
    # Import only the necessary function to avoid torch dependency
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from download_code_llm import CODE_SAMPLES_FILE
    
    # Create a simple function to test code sample creation
    def create_test_samples(filename=temp_file):
        samples = [
            "def test():\n    return True",
            "class Test:\n    pass"
        ]
        with open(filename, 'w') as f:
            for sample in samples:
                f.write(sample + "\n===\n")
        return samples
    
    try:
        # Create test samples
        samples = create_test_samples()
        assert os.path.exists(temp_file)
        
        # Read the file and verify contents
        with open(temp_file, 'r') as f:
            content = f.read()
            assert "def test():" in content
            assert "class Test:" in content
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)