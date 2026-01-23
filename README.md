# Sequence Repetition for Sequence Labeling

This repository contains the code for the paper:

**Sequence Repetition Enhances Token Embeddings and Improves Sequence Labeling with Decoder-only Language Models**

*Matija Luka Kukić\*, Marko Čuljak\*, David Dukić\*, Martin Tutek, Jan Šnajder*

TakeLab @ Faculty of Electrical Engineering and Computing, University of Zagreb

## Abstract

Modern language models (LMs) are trained in an autoregressive manner, conditioned only on the prefix. In contrast, sequence labeling (SL) tasks assign labels to each individual input token, naturally benefiting from bidirectional context. This discrepancy has historically led SL to rely on inherently bidirectional encoder-only models. However, the rapid development of decoder-only models has raised the question of whether they can be adapted to SL. While causal mask removal has emerged as a viable technique for adapting decoder-only models to leverage the full context for SL, it requires considerable changes to the base model functionality. In this work, we explore sequence repetition (SR) as a less invasive alternative for enabling bidirectionality in decoder-only models. Through fine-tuning experiments, we show that SR inherently makes decoders bidirectional, improving the quality of token-level embeddings and surpassing encoders and unmasked decoders. Contrary to earlier claims, we find that increasing the number of repetitions does not degrade SL performance. Finally, we demonstrate that embeddings from intermediate layers are highly effective for SR, comparable to those from final layers, while being at least \david{$1.39\times$} more efficient to compute.
Our findings underscore that SR alleviates the structural limitations of decoders, enabling more efficient and adaptable LMs and broadening their applicability to other token-level tasks.

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU with at least 24GB VRAM (40GB+ recommended for larger models)
- [uv](https://docs.astral.sh/uv/) (recommended for dependency management)

### Setup

```bash
# Clone the repository
git clone https://github.com/takelab/repetition-sl.git
cd repetition-sl

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync

# Install the sequence labeling pipeline in development mode
cd sl_pipeline && uv pip install -e . && cd ..

# Install the unmasking module (custom model implementations)
cd unmasking && uv pip install -e . && cd ..
```

Alternatively, using pip:

```bash
pip install -e .
cd sl_pipeline && pip install -e . && cd ..
cd unmasking && pip install -e . && cd ..
```

### Data Preparation

The experiments use four sequence labeling datasets:

1. **CoNLL03** - Named Entity Recognition (automatically downloaded via HuggingFace)
2. **ACE05** - Event Trigger Classification (requires preprocessing)
3. **Rest14** - Aspect-Based Sentiment Analysis (SemEval-2014 Task 4)
4. **NLU++** - Slot Labeling for Task-Oriented Dialogue

For ACE05, download and preprocess using the [standard ACE preprocessing tool](https://bit.ly/ace2005-preprocessing).

## Usage

### Running All Experiments

To reproduce all results from the paper:

```bash
# Run everything (training + evaluation + plots)
./scripts/run_all_experiments.sh

# Or run individual phases:
./scripts/run_all_experiments.sh --train-only    # Only training
./scripts/run_all_experiments.sh --eval-only     # Only evaluation
./scripts/run_all_experiments.sh --plots-only    # Only generate plots/tables
```

### Training Individual Models

```bash
# Train encoder baseline (RoBERTa)
python scripts/train.py --model roberta --method encoder --dataset conll2003

# Train decoder with sequence repetition (r=2)
python scripts/train.py --model mistral-7b --method repeat --k_repeat 2 --dataset conll2003

# Train decoder with full unmasking
python scripts/train.py --model gemma2-2b --method full-unmasking --dataset ace

# Train decoder with middle unmasking
python scripts/train.py --model qwen3-4b --method middle-unmasking --dataset nlupp

# Train with early exiting (Mistral-7B, exit at layer 19)
python scripts/train.py --model mistral-7b --method repeat --k_repeat 4 --early_exit 19 --dataset absa-restaurants
```

### Evaluating Models

```bash
# Evaluate trained model
python scripts/eval.py --model mistral-7b --method repeat --k_repeat 2 \
    --dataset conll2003 --load_dir ./trained_models/
```

### Generating Plots and Tables

```bash
# Generate all figures and tables from the paper
./scripts/generate_plots_tables.sh

# Or generate individually:
python plots_tables/get_repetition_plots.py --path_to_results ./outputs --output figures/fig2.pdf
python plots_tables/get_main_table.py --path_to_results ./outputs > tables/table1.tex
```

## Supported Models

### Encoder Models (Baselines)
- `roberta` - RoBERTa-Large (355M parameters)
- `modernbert` - ModernBERT-Large

### Decoder Models
- `gemma-7b` - Gemma 7B
- `gemma2-2b` - Gemma2 2B
- `gemma2-9b` - Gemma2 9B
- `mistral-7b` - Mistral 7B v0.3
- `qwen3-0.6b` - Qwen3 0.6B
- `qwen3-1.7b` - Qwen3 1.7B
- `qwen3-4b` - Qwen3 4B
- `qwen3-8b` - Qwen3 8B

## Training Configuration

Default hyperparameters (matching the paper):
- **QLoRA**: rank=16, alpha=16, dropout=0.1
- **Learning rate**: 2e-4
- **Batch size**: 8 (with gradient accumulation of 4 → effective batch size 32)
- **Epochs**: 10
- **Weight decay**: 0.05
- **Seeds**: 5, 29, 42, 81, 123 (5 runs per configuration)

## Repository Structure

```
.
├── scripts/                    # Main experiment scripts
│   ├── train.py               # Training script
│   ├── eval.py                # Evaluation script
│   ├── run_all_experiments.sh # Run all experiments
│   └── generate_plots_tables.sh
├── sl_pipeline/               # Sequence labeling pipeline
│   └── sl_pipeline/
│       ├── train_pipeline.py  # Training utilities
│       ├── eval_pipeline.py   # Evaluation utilities
│       └── dataset_token_clf.py
├── unmasking/                 # Custom model implementations
│   └── unmasking/
│       ├── qwen/              # Qwen3 variants
│       ├── gemma/             # Gemma variants
│       ├── gemma2/            # Gemma2 variants
│       └── mistral/           # Mistral variants
├── repetition/                # Result parsing utilities
│   └── io.py                  # I/O functions for evaluation files
├── plots_tables/              # Scripts for generating figures/tables
│   ├── repetition_plots.py
│   ├── early_exiting_plots.py
│   ├── main_table.py
│   └── ...
├── results/                   # Evaluation results (JSON files)
├── pyproject.toml             # Configuration file
├── uv.lock                    # Dependencies
└── README.md
```

## Results

Our main findings:

1. **Sequence repetition outperforms unmasking**: SR achieves the best results across all models and datasets, surpassing unmasking baselines.

2. **More repetitions help**: Contrary to prior work on sentence-level tasks, we find that increasing repetitions (r > 1) improves token-level performance. Performance typically saturates around r=4.

3. **Early exiting mitigates computational overhead of repetition**: Embeddings from intermediate layers (e.g., layer 19 out of 32) perform comparably to final-layer embeddings while being 1.39× faster.

## Citation

TBD


## License

This project is licensed under the MIT License.
