#!/bin/bash
#
# Run all experiments for the paper:
# "Sequence Repetition Enhances Token Embeddings and Improves
#  Sequence Labeling with Decoder-only Language Models"
#
# This script runs:
# 1. Encoder baseline training (RoBERTa, ModernBERT)
# 2. Decoder training with sequence repetition (r=0,1,2,4,8)
# 3. Decoder training with full/middle unmasking
# 4. Early exit experiments (Mistral-7B only)
# 5. Evaluation on all models
# 6. Generate plots and tables
#
# Usage:
#   ./scripts/run_all_experiments.sh [--train-only] [--eval-only] [--plots-only]
#
# Requirements:
#   - CUDA-capable GPU with at least 24GB VRAM (40GB+ recommended)
#   - Python environment with requirements.txt installed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Configuration
SAVE_DIR="./trained_models/"
OUTPUT_DIR="./outputs/"
BATCH_SIZE=8
GRAD_ACCUM=4
NUM_EPOCHS=10
LEARNING_RATE="2e-4"
SEEDS="5 29 42 81 123"

# Datasets used in the paper
DATASETS=("conll2003" "ace" "nlupp" "absa-restaurants")

# Decoder models
DECODER_MODELS=("gemma-7b" "gemma2-2b" "gemma2-9b" "mistral-7b" "qwen3-1.7b" "qwen3-4b" "qwen3-8b")

# Encoder models
ENCODER_MODELS=("roberta" "modernbert")

# Repetition counts
REPEAT_COUNTS=(0 1 2 4 8)

# Early exit layers (for Mistral-7B, which has 32 layers)
EARLY_EXIT_LAYERS=(8 13 18 23)

# Parse arguments
TRAIN_ONLY=false
EVAL_ONLY=false
PLOTS_ONLY=false

for arg in "$@"; do
    case $arg in
        --train-only)
            TRAIN_ONLY=true
            ;;
        --eval-only)
            EVAL_ONLY=true
            ;;
        --plots-only)
            PLOTS_ONLY=true
            ;;
    esac
done

# Create directories
mkdir -p "$SAVE_DIR" "$OUTPUT_DIR"
for ds in "${DATASETS[@]}"; do
    mkdir -p "$OUTPUT_DIR/$ds"
done

# Function to train a model
train_model() {
    local model=$1
    local method=$2
    local dataset=$3
    local k_repeat=${4:-0}
    local early_exit=${5:--1}

    echo "========================================"
    echo "Training: $model ($method, r=$k_repeat, exit=$early_exit) on $dataset"
    echo "========================================"

    cmd="python scripts/train.py \
        --model $model \
        --method $method \
        --dataset $dataset \
        --save_dir $SAVE_DIR \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --seeds $SEEDS"

    if [ "$k_repeat" -gt 0 ]; then
        cmd="$cmd --k_repeat $k_repeat"
    fi

    if [ "$early_exit" -gt 0 ]; then
        cmd="$cmd --early_exit $early_exit"
    fi

    eval $cmd
}

# Function to evaluate a model
eval_model() {
    local model=$1
    local method=$2
    local dataset=$3
    local k_repeat=${4:-0}
    local early_exit=${5:--1}

    echo "========================================"
    echo "Evaluating: $model ($method, r=$k_repeat, exit=$early_exit) on $dataset"
    echo "========================================"

    cmd="python scripts/eval.py \
        --model $model \
        --method $method \
        --dataset $dataset \
        --load_dir $SAVE_DIR \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE"

    if [ "$k_repeat" -gt 0 ]; then
        cmd="$cmd --k_repeat $k_repeat"
    fi

    if [ "$early_exit" -gt 0 ]; then
        cmd="$cmd --early_exit $early_exit"
    fi

    eval $cmd
}

# Training phase
if [ "$PLOTS_ONLY" = false ] && [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "=============================================="
    echo "PHASE 1: Training encoder baselines"
    echo "=============================================="
    echo ""

    for model in "${ENCODER_MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            train_model "$model" "encoder" "$dataset"
        done
    done

    echo ""
    echo "=============================================="
    echo "PHASE 2: Training decoders with sequence repetition"
    echo "=============================================="
    echo ""

    for model in "${DECODER_MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for r in "${REPEAT_COUNTS[@]}"; do
                train_model "$model" "repeat" "$dataset" "$r"
            done
        done
    done

    echo ""
    echo "=============================================="
    echo "PHASE 3: Training decoders with unmasking"
    echo "=============================================="
    echo ""

    for model in "${DECODER_MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            # Full unmasking
            train_model "$model" "full-unmasking" "$dataset"
            # Middle unmasking
            train_model "$model" "middle-unmasking" "$dataset"
        done
    done

    echo ""
    echo "=============================================="
    echo "PHASE 4: Early exit experiments (Mistral-7B)"
    echo "=============================================="
    echo ""

    for dataset in "${DATASETS[@]}"; do
        for r in 1 2 4 8; do
            for exit_layer in "${EARLY_EXIT_LAYERS[@]}"; do
                train_model "mistral-7b" "repeat" "$dataset" "$r" "$exit_layer"
            done
        done
    done
fi

# Evaluation phase
if [ "$PLOTS_ONLY" = false ] && [ "$TRAIN_ONLY" = false ]; then
    echo ""
    echo "=============================================="
    echo "PHASE 5: Evaluating all models"
    echo "=============================================="
    echo ""

    # Evaluate encoder baselines
    for model in "${ENCODER_MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            eval_model "$model" "encoder" "$dataset"
        done
    done

    # Evaluate decoders with sequence repetition
    for model in "${DECODER_MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for r in "${REPEAT_COUNTS[@]}"; do
                eval_model "$model" "repeat" "$dataset" "$r"
            done
        done
    done

    # Evaluate decoders with unmasking
    for model in "${DECODER_MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            eval_model "$model" "full-unmasking" "$dataset"
            eval_model "$model" "middle-unmasking" "$dataset"
        done
    done

    # Evaluate early exit experiments
    for dataset in "${DATASETS[@]}"; do
        for r in 1 2 4 8; do
            for exit_layer in "${EARLY_EXIT_LAYERS[@]}"; do
                eval_model "mistral-7b" "repeat" "$dataset" "$r" "$exit_layer"
            done
        done
    done
fi

# Generate plots and tables
if [ "$TRAIN_ONLY" = false ] && [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "=============================================="
    echo "PHASE 6: Generating plots and tables"
    echo "=============================================="
    echo ""

    ./scripts/generate_plots_tables.sh
fi

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
