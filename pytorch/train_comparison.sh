#!/bin/bash
# Train baseline, small, and medium models for comparison
# Supports named arguments and training on all available data

set -e

# Default values
DATA_DIR="training_data"
OUTPUT_DIR="trained_models/comparison"
EPOCHS=1
WANDB=""
DEVICE="cuda"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir|-d)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs|-e)
            EPOCHS="$2"
            shift 2
            ;;
        --wandb)
            WANDB="--wandb"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --data-dir, -d DIR    Training data directory (default: training_data)"
            echo "  --output-dir, -o DIR  Output directory (default: trained_models/comparison)"
            echo "  --epochs, -e N        Number of epochs (default: 1)"
            echo "  --device DEVICE       Device: cuda, cpu, auto (default: cuda)"
            echo "  --wandb               Enable wandb logging"
            echo "  --help, -h            Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Find all batch directories
BATCH_DIRS=$(find "$DATA_DIR" -maxdepth 1 -type d -name "batch_*" | sort | tr '\n' ' ')

if [ -z "$BATCH_DIRS" ]; then
    echo "Error: No batch directories found in $DATA_DIR"
    exit 1
fi

# Count batches and estimate data size
NUM_BATCHES=$(echo $BATCH_DIRS | wc -w)
echo "=========================================="
echo "Training comparison: baseline, residual-small, residual-medium"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Batch directories: $NUM_BATCHES"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Device: $DEVICE"
echo "Wandb: ${WANDB:-disabled}"
echo ""

mkdir -p "$OUTPUT_DIR"

# Train baseline
echo "=========================================="
echo "Training baseline model..."
echo "=========================================="
python3 train.py \
    -d $BATCH_DIRS \
    -n eval \
    --arch baseline \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    -o "$OUTPUT_DIR" \
    $WANDB \
    --wandb-run-name "baseline_${EPOCHS}ep"

# Train residual-small
echo ""
echo "=========================================="
echo "Training residual-small model..."
echo "=========================================="
python3 train.py \
    -d $BATCH_DIRS \
    -n eval \
    --arch residual-small \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    -o "$OUTPUT_DIR" \
    $WANDB \
    --wandb-run-name "residual-small_${EPOCHS}ep"

# Train residual-medium
echo ""
echo "=========================================="
echo "Training residual-medium model..."
echo "=========================================="
python3 train.py \
    -d $BATCH_DIRS \
    -n eval \
    --arch residual-medium \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    -o "$OUTPUT_DIR" \
    $WANDB \
    --wandb-run-name "residual-medium_${EPOCHS}ep"

echo ""
echo "=========================================="
echo "All models trained!"
echo "=========================================="
ls -la "$OUTPUT_DIR"
