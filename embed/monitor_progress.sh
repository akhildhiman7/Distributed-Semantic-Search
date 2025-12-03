#!/bin/bash
# Monitor embedding generation progress

echo "📊 Embedding Generation Progress Monitor"
echo "========================================"
echo ""

# Check if process is running
if pgrep -f "embedding_generator.py" > /dev/null; then
    echo "✓ Process is running"
else
    echo "⚠ Process not running"
fi

echo ""
echo "📁 Output Files:"
echo "----------------"

# Check sample output
if [ -d "embeddings/sample" ]; then
    echo "✓ Sample directory created"
    if [ -f "embeddings/sample/sample_embeddings.npy" ]; then
        size=$(ls -lh embeddings/sample/sample_embeddings.npy | awk '{print $5}')
        echo "  - sample_embeddings.npy: $size"
    fi
    if [ -f "embeddings/sample/sample_metadata.parquet" ]; then
        size=$(ls -lh embeddings/sample/sample_metadata.parquet | awk '{print $5}')
        echo "  - sample_metadata.parquet: $size"
    fi
fi

# Check full dataset output
if [ -d "embeddings/full" ]; then
    count=$(ls embeddings/full/*_embeddings.npy 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -gt 0 ]; then
        echo "✓ Full dataset: $count/9 partitions complete"
        total_size=$(du -sh embeddings/full 2>/dev/null | awk '{print $1}')
        echo "  Total size: $total_size"
    else
        echo "⏳ Full dataset: Processing..."
    fi
else
    echo "⏳ Full dataset: Not started"
fi

echo ""
echo "📈 Recent Logs (last 10 lines):"
echo "--------------------------------"
if [ -f "reports/"*.log ]; then
    tail -n 10 reports/*.log 2>/dev/null | grep -E "INFO|Complete|✓" | tail -5
fi

echo ""
echo "⏱ Estimated Time:"
echo "-----------------"
echo "Sample (10k):  ~7 minutes with MPS"
echo "Full (510k):   ~45-60 minutes with MPS"
echo ""
echo "Run this script again to check progress:"
echo "  bash monitor_progress.sh"
