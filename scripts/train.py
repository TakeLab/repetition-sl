#!/usr/bin/env python3
"""
Training script for sequence labeling experiments.

This script trains models with different configurations:
- Encoder baselines (RoBERTa, ModernBERT)
- Decoder models with sequence repetition (r=0,1,2,4,8)
- Decoder models with full/middle unmasking
- Decoder models with early exiting

Usage:
    python scripts/train.py --model mistral --method repeat --k_repeat 2 --dataset conll2003
    python scripts/train.py --model roberta --method encoder --dataset ace
    python scripts/train.py --model mistral --method repeat --k_repeat 4 --early_exit 19 --dataset nlupp
"""

import argparse
import io
import logging
import sys
import warnings
from contextlib import contextmanager

from sl_pipeline import TrainModelDataset


@contextmanager
def suppress_all(logger_names=None):
    """Suppress Python warnings, logging, and stderr output."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        logger_names = logger_names or []
        handlers = []
        for name in logger_names:
            logger = logging.getLogger(name)
            handler = logging.StreamHandler(io.StringIO())
            logger.addHandler(handler)
            logger.setLevel(logging.ERROR)
            handlers.append((logger, handler))

        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            yield
        finally:
            sys.stderr = old_stderr
            for logger, handler in handlers:
                logger.removeHandler(handler)


# Model configurations
MODEL_CONFIGS = {
    # Encoder models
    "roberta": {
        "hf_name": "FacebookAI/roberta-large",
        "model_type": "roberta",
    },
    "modernbert": {
        "hf_name": "answerdotai/ModernBERT-large",
        "model_type": "modern-bert",
    },
    # Decoder models
    "gemma-7b": {
        "hf_name": "google/gemma-7b",
        "model_type": "gemma",
    },
    "gemma2-2b": {
        "hf_name": "google/gemma-2-2b",
        "model_type": "gemma2",
    },
    "gemma2-9b": {
        "hf_name": "google/gemma-2-9b",
        "model_type": "gemma2",
    },
    "mistral-7b": {
        "hf_name": "mistralai/Mistral-7B-v0.3",
        "model_type": "mistral",
    },
    "qwen3-0.6b": {
        "hf_name": "Qwen/Qwen3-0.6B",
        "model_type": "qwen3",
    },
    "qwen3-1.7b": {
        "hf_name": "Qwen/Qwen3-1.7B",
        "model_type": "qwen3",
    },
    "qwen3-4b": {
        "hf_name": "Qwen/Qwen3-4B",
        "model_type": "qwen3",
    },
    "qwen3-8b": {
        "hf_name": "Qwen/Qwen3-8B",
        "model_type": "qwen3",
    },
}

# Supported datasets
DATASETS = ["conll2003", "ace", "nlupp", "absa-restaurants"]


def get_model_variant(model_type: str, method: str, k_repeat: int = 0) -> str:
    """Get the model variant name based on method and repeat count."""
    if method == "encoder":
        return model_type
    elif method == "repeat":
        if k_repeat == 0:
            return model_type
        else:
            return f"{model_type}-repeat-k"
    elif method == "full-unmasking":
        return f"{model_type}-unmasked"
    elif method == "middle-unmasking":
        return f"{model_type}-unmasked-middle"
    else:
        return model_type


def main():
    parser = argparse.ArgumentParser(
        description="Train sequence labeling models with various configurations."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to train",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="repeat",
        choices=["encoder", "repeat", "full-unmasking", "middle-unmasking"],
        help="Training method (default: repeat)",
    )
    parser.add_argument(
        "--k_repeat",
        type=int,
        default=0,
        help="Number of sequence repetitions (default: 0, only used with method=repeat)",
    )
    parser.add_argument(
        "--early_exit",
        type=int,
        default=-1,
        help="Early exit layer index (default: -1, no early exit)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASETS,
        help="Dataset to train on",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./trained_models/",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[5, 29, 42, 81, 123],
        help="Random seeds for training (default: 5 29 42 81 123)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show training output",
    )

    args = parser.parse_args()

    # Get model configuration
    config = MODEL_CONFIGS[args.model]
    hf_name = config["hf_name"]
    model_type = config["model_type"]

    # Get model variant
    model_variant = get_model_variant(model_type, args.method, args.k_repeat)

    # Validate method for encoder models
    if args.model in ["roberta", "modernbert"] and args.method != "encoder":
        print(f"Warning: {args.model} is an encoder model, ignoring method={args.method}")
        model_variant = model_type

    # Set k_repeat for repeat method
    k_repeat = args.k_repeat if args.method == "repeat" and args.k_repeat > 0 else -1

    print(f"Training configuration:")
    print(f"  Model: {args.model} ({hf_name})")
    print(f"  Model variant: {model_variant}")
    print(f"  Method: {args.method}")
    print(f"  K repeat: {k_repeat}")
    print(f"  Early exit: {args.early_exit if args.early_exit > 0 else 'None'}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Save dir: {args.save_dir}")
    print(f"  Seeds: {args.seeds}")
    print()

    # Create trainer
    if args.early_exit > 0:
        raise NotImplementedError(
            "Early exit training is not yet supported in the main pipeline. "
            "Use the scripts in apptainer/srce_david/run_scripts/train/ for early exit experiments."
        )

    tmd = TrainModelDataset(
        args.dataset,
        model_variant,
        hf_name,
        k_repeat,
    )

    tmd.setBnbConf4B()
    tmd.setLoraConfig()
    tmd.setTrainingArgs(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        output_dir=args.save_dir,
    )

    if args.verbose:
        tmd.trainSeeds(seeds=args.seeds, save_dir=args.save_dir)
    else:
        with suppress_all(logger_names=["transformers"]):
            tmd.trainSeeds(seeds=args.seeds, save_dir=args.save_dir)

    del tmd
    print("Training complete!")


if __name__ == "__main__":
    main()
