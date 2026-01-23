#!/usr/bin/env python3
"""
Evaluation script for sequence labeling experiments.

This script evaluates trained models on validation and test sets.

Usage:
    python scripts/eval.py --model mistral --method repeat --k_repeat 2 --dataset conll2003 --load_dir ./trained_models/
    python scripts/eval.py --model roberta --method encoder --dataset ace --load_dir ./trained_models/
"""

import argparse
import io
import json
import logging
import os
import sys
import warnings
from contextlib import contextmanager

from sl_pipeline import EvalModelDataset


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


# Model configurations (same as train.py)
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


def get_output_filename(model_name: str, dataset: str, split: str, k_repeat: int = 0, early_exit: int = -1) -> str:
    """Generate output filename for evaluation results."""
    parts = [model_name]

    if k_repeat > 0:
        parts.append(f"repeat-k-{k_repeat}")

    if early_exit > 0:
        parts.append(f"early-exit-{early_exit}")

    model_str = "-".join(parts)
    return f"eval_stats_{model_str}_{dataset}_{split}.json"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained sequence labeling models."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to evaluate",
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
        help="Number of sequence repetitions (default: 0)",
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
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        required=True,
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Evaluation batch size (default: 8)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["validation", "test"],
        help="Splits to evaluate (default: validation test)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show evaluation output",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[5, 29, 42, 81, 123],
        help="Seeds of trained models to evaluate (default: 5 29 42 81 123)",
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

    print(f"Evaluation configuration:")
    print(f"  Model: {args.model} ({hf_name})")
    print(f"  Model variant: {model_variant}")
    print(f"  Method: {args.method}")
    print(f"  K repeat: {k_repeat}")
    print(f"  Early exit: {args.early_exit if args.early_exit > 0 else 'None'}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Load dir: {args.load_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Splits: {args.splits}")
    print(f"  Seeds: {args.seeds}")
    print()

    # Create output directory
    output_dataset_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dataset_dir, exist_ok=True)

    # Evaluate on each split
    for split in args.splits:
        print(f"Evaluating on {split} split...")

        emd = EvalModelDataset(
            args.dataset,
            model_variant,
            hf_name,
            load_dir=args.load_dir,
            split=split,
            k_repeat=k_repeat,
        )
        emd.setTrainingArgs(batch_size=args.batch_size)
        emd.setBnbConf4B()

        if args.verbose:
            results = emd.evalSeeds(seeds=args.seeds)
        else:
            with suppress_all(logger_names=["transformers"]):
                results = emd.evalSeeds(seeds=args.seeds)

        del emd

        # Save results
        # Generate model name for output file
        model_name_parts = [args.model.replace("-", "-")]
        if args.method == "repeat" and args.k_repeat > 0:
            model_name_parts.append(f"repeat-k-{args.k_repeat}")
        elif args.method == "full-unmasking":
            model_name_parts.append("unmasked")
        elif args.method == "middle-unmasking":
            model_name_parts.append("unmasked-middle")

        if args.early_exit > 0:
            model_name_parts.append(f"early-exit-{args.early_exit}")

        model_output_name = "-".join(model_name_parts)
        output_filename = f"eval_stats_{model_output_name}_{args.dataset}_{split}.json"
        output_path = os.path.join(output_dataset_dir, output_filename)

        with open(output_path, "w") as f:
            json.dump({model_variant: results}, f, indent=2)

        print(f"  Saved results to {output_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
