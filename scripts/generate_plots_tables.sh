#!/bin/bash
#
# Generate all plots and tables from evaluation results.
#
# This script generates:
# - Table 1: Main results table (Table 1 in paper)
# - Table 5: Full results table (Appendix)
# - Figure 2: Repetition count vs. F1 plots
# - Figure 5: Early exiting plots
# - Figure 3: Aggregated results
# - Figure 4: Qualitative analysis
# - Figure 6: Compute-matched comparison
#
# Usage:
#   ./scripts/generate_plots_tables.sh [--results-dir DIR] [--output-dir DIR]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Default directories
RESULTS_DIR="${RESULTS_DIR:-./outputs}"
ENCODER_RESULTS_DIR="${ENCODER_RESULTS_DIR:-./outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-./figures}"
TABLES_DIR="${TABLES_DIR:-./tables}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --encoder-results-dir)
            ENCODER_RESULTS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --tables-dir)
            TABLES_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directories
mkdir -p "$OUTPUT_DIR" "$TABLES_DIR"

echo "=============================================="
echo "Generating plots and tables"
echo "=============================================="
echo "Results directory: $RESULTS_DIR"
echo "Encoder results directory: $ENCODER_RESULTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Tables directory: $TABLES_DIR"
echo ""

# Figure 2: Repetition plots (test split)
echo "Generating Figure 2: Repetition plots (test)..."
python plots_tables/get_repetition_plots.py \
    --path_to_results "$RESULTS_DIR" \
    --output "$OUTPUT_DIR/repetition_grid_test.pdf" \
    --split test

# Figure 7: Repetition plots (validation split)
echo "Generating Figure 7: Repetition plots (validation)..."
python plots_tables/get_repetition_plots.py \
    --path_to_results "$RESULTS_DIR" \
    --output "$OUTPUT_DIR/repetition_grid_validation.pdf" \
    --split validation

# Figure 3: Aggregated results
echo "Generating Figure 3: Aggregated results..."
python plots_tables/get_aggregated_results.py \
    --path_to_results "$RESULTS_DIR" \
    --output "$OUTPUT_DIR/aggregated_results.pdf"

# Figure 4: Qualitative analysis
echo "Generating Figure 4: Qualitative analysis..."
python plots_tables/get_qualitative_results.py \
    --path_to_results "$RESULTS_DIR" \
    --output "$OUTPUT_DIR/qualitative_results.pdf"

# Figure 5: Early exiting plots
echo "Generating Figure 5: Early exiting plots..."
python plots_tables/get_early_exiting_plots.py \
    --path_to_results "$RESULTS_DIR" \
    --output "$OUTPUT_DIR/early_exiting_grid.pdf"

# Figure 6: Compute-matched comparison
echo "Generating Figure 6: Compute-matched comparison..."
python plots_tables/get_compute_matched_plot.py \
    --path_to_results "$RESULTS_DIR" \
    --path_to_encoder_results "$ENCODER_RESULTS_DIR" \
    --output "$OUTPUT_DIR/compute_matched.pdf"

# Table 1: Main results table
echo "Generating Table 1: Main results..."
python plots_tables/get_main_table.py \
    --path_to_results "$RESULTS_DIR" \
    --path_to_encoder_results "$ENCODER_RESULTS_DIR" \
    > "$TABLES_DIR/main_table.tex"

# Table 5: Full results table
echo "Generating Table 5: Full results..."
python plots_tables/get_full_table.py \
    --path_to_results "$RESULTS_DIR" \
    --path_to_encoder_results "$ENCODER_RESULTS_DIR" \
    > "$TABLES_DIR/full_table.tex"

# Generate profiling table if available
if [ -f "plots_tables/color_profiling_table.py" ]; then
    echo "Generating profiling table..."
    python plots_tables/color_profiling_table.py > "$TABLES_DIR/profiling_table.tex" 2>/dev/null || true
fi

echo ""
echo "=============================================="
echo "All plots and tables generated!"
echo "=============================================="
echo ""
echo "Figures saved to: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR"/*.pdf 2>/dev/null || echo "  (no PDF files found)"
echo ""
echo "Tables saved to: $TABLES_DIR/"
ls -la "$TABLES_DIR"/*.tex 2>/dev/null || echo "  (no TeX files found)"
