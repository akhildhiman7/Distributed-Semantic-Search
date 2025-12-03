"""
Test and validate embedding generation pipeline.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import SAMPLE_DATA_DIR, EMBEDDINGS_DIR


def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
        
        # Check MPS availability
        if torch.backends.mps.is_available():
            print("  ✓ MPS (Metal Performance Shaders) available")
        else:
            print("  ⚠ MPS not available, will use CPU")
        
    except ImportError as e:
        print(f"  ✗ torch not installed: {e}")
        return False
    
    try:
        import sentence_transformers
        print(f"  ✓ sentence-transformers {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ sentence-transformers not installed: {e}")
        return False
    
    print("✓ All imports successful\n")
    return True


def test_model_loading():
    """Test model can be loaded."""
    print("Testing model loading...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="mps" if __import__("torch").backends.mps.is_available() else "cpu"
        )
        
        # Test encoding
        test_text = "This is a test sentence for semantic embedding."
        embedding = model.encode(test_text, normalize_embeddings=True)
        
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Test embedding shape: {embedding.shape}")
        print(f"  ✓ Embedding dimension: {len(embedding)}")
        print(f"  ✓ Embedding norm: {np.linalg.norm(embedding):.4f}")
        
        assert len(embedding) == 384, "Expected 384-dimensional embeddings"
        assert 0.9 < np.linalg.norm(embedding) < 1.1, "Expected normalized embeddings"
        
        print("✓ Model test passed\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def test_sample_data():
    """Test that sample data is available."""
    print("Testing sample data...")
    
    sample_file = SAMPLE_DATA_DIR / "sample.parquet"
    
    if not sample_file.exists():
        print(f"  ✗ Sample file not found: {sample_file}")
        return False
    
    try:
        df = pd.read_parquet(sample_file)
        print(f"  ✓ Sample file loaded: {len(df):,} records")
        print(f"  ✓ Columns: {', '.join(df.columns)}")
        
        # Check required columns
        required = ['paper_id', 'title', 'abstract', 'text']
        for col in required:
            if col not in df.columns:
                print(f"  ✗ Missing column: {col}")
                return False
        
        print(f"  ✓ Sample text lengths: min={df['text_length'].min()}, "
              f"max={df['text_length'].max()}, mean={df['text_length'].mean():.0f}")
        
        print("✓ Sample data valid\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading sample: {e}")
        return False


def test_embedding_output():
    """Test that embeddings were generated correctly."""
    print("Testing embedding output...")
    
    sample_embedding_file = EMBEDDINGS_DIR / "sample" / "sample_embeddings.npy"
    sample_metadata_file = EMBEDDINGS_DIR / "sample" / "sample_metadata.parquet"
    
    if not sample_embedding_file.exists():
        print(f"  ⚠ Embeddings not found: {sample_embedding_file}")
        print("  → Run: python embedding_generator.py")
        return None  # Not failed, just not ready
    
    try:
        # Load embeddings
        embeddings = np.load(sample_embedding_file)
        print(f"  ✓ Embeddings loaded: {embeddings.shape}")
        
        # Load metadata
        metadata = pd.read_parquet(sample_metadata_file)
        print(f"  ✓ Metadata loaded: {len(metadata):,} records")
        
        # Validate alignment
        if len(embeddings) != len(metadata):
            print(f"  ✗ Length mismatch: {len(embeddings)} embeddings != {len(metadata)} metadata")
            return False
        
        # Validate embedding properties
        if embeddings.shape[1] != 384:
            print(f"  ✗ Wrong dimension: {embeddings.shape[1]} != 384")
            return False
        
        # Check for NaN/Inf
        if np.any(np.isnan(embeddings)):
            print("  ✗ NaN values found in embeddings")
            return False
        
        if np.any(np.isinf(embeddings)):
            print("  ✗ Inf values found in embeddings")
            return False
        
        # Check norms
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"  ✓ Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
        
        # Verify norm alignment with metadata
        if 'embedding_norm' in metadata.columns:
            stored_norms = metadata['embedding_norm'].values
            if np.allclose(norms, stored_norms, rtol=1e-4):
                print("  ✓ Norms match metadata")
            else:
                print("  ⚠ Norm mismatch with metadata")
        
        print("✓ Embedding output valid\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Error validating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("EMBEDDING PIPELINE VALIDATION")
    print("="*60 + "\n")
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Model loading
    results.append(("Model Loading", test_model_loading()))
    
    # Test 3: Sample data
    results.append(("Sample Data", test_sample_data()))
    
    # Test 4: Embedding output (optional if not generated yet)
    embedding_result = test_embedding_output()
    if embedding_result is not None:
        results.append(("Embedding Output", embedding_result))
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print("="*60)
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        print("\nYou can now run:")
        print("  python embedding_generator.py")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    exit(main())
