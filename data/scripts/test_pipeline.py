"""
Quick test script to verify the pipeline works on a small sample.
Tests all components before running on full dataset.
"""
import json
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from text_cleaner import TextCleaner
from data_processor import ArxivDataProcessor
from config import ProcessingConfig, RAW_DATA_PATH


def test_text_cleaner():
    """Test text cleaning functions."""
    print("Testing TextCleaner...")
    
    cleaner = TextCleaner()
    
    # Test LaTeX cleaning
    latex_text = r"The function $f(x) = \alpha x^2$ where parameter is used"
    cleaned = cleaner.clean_latex(latex_text)
    assert "function" in cleaned.lower() and "parameter" in cleaned.lower()
    print("  ✓ LaTeX cleaning works")
    
    # Test HTML cleaning
    html_text = "<p>This is <strong>bold</strong> text &amp; special chars</p>"
    cleaned = cleaner.clean_html(html_text)
    assert "<" not in cleaned
    assert "&amp;" not in cleaned  # HTML entity should be decoded
    print("  ✓ HTML cleaning works")
    
    # Test full cleaning
    messy_text = r"  \textbf{Neural Networks}  are   used in  AI  "
    cleaned = cleaner.clean_text(messy_text)
    assert "Neural Networks" in cleaned
    assert "  " not in cleaned  # no double spaces
    print("  ✓ Full text cleaning works")
    
    print("✓ TextCleaner tests passed\n")


def test_processor_on_sample():
    """Test processor on first few records."""
    print("Testing ArxivDataProcessor on sample records...")
    
    if not RAW_DATA_PATH.exists():
        print(f"  ✗ Input file not found: {RAW_DATA_PATH}")
        return False
    
    processor = ArxivDataProcessor()
    
    # Process first 100 records
    count = 0
    processed_count = 0
    
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= 100:
                break
            
            try:
                record = json.loads(line.strip())
                processed = processor._process_record(record)
                
                if processed:
                    processed_count += 1
                    
                    # Validate processed record
                    assert 'paper_id' in processed
                    assert 'text' in processed
                    assert len(processed['text']) > 0
                    
                count += 1
                
            except Exception as e:
                print(f"  ! Error on record {count}: {e}")
    
    print(f"  Processed {count} records, kept {processed_count}")
    print(f"  Stats: {processor.stats['skipped_records']} skipped, "
          f"{processor.stats['duplicate_records']} duplicates")
    
    if processed_count > 0:
        print("✓ Processor tests passed\n")
        return True
    else:
        print("✗ No records were successfully processed\n")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("RUNNING PIPELINE TESTS")
    print("="*60 + "\n")
    
    try:
        test_text_cleaner()
        success = test_processor_on_sample()
        
        if success:
            print("="*60)
            print("✓ ALL TESTS PASSED")
            print("="*60)
            print("\nYou can now run the full pipeline:")
            print("  python data_processor.py")
            return 0
        else:
            print("="*60)
            print("✗ SOME TESTS FAILED")
            print("="*60)
            return 1
            
    except Exception as e:
        print(f"\n✗ Test error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
