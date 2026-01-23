"""I/O utilities for parsing evaluation result files."""

import json
import os
import re
from typing import List, Tuple, Any, Optional


def parse_model_name(filename: str) -> dict:
    """
    Parse a model name from an evaluation filename.

    Filenames follow the pattern:
    eval_stats_{model_name}_{dataset}_{split}.json

    Model names can be:
    - gemma-2-2B (base decoder, r=0)
    - gemma-2-2B-repeat-k-1 (sequence repetition, r=1)
    - gemma-2-2B-unmasked (full unmasking)
    - gemma-2-2B-unmasked-middle (middle unmasking)
    - mistral-7B-v0.3-repeat-k-1-early-exit-8 (early exit)

    Returns a dict with:
    - model: base model family (gemma, gemma-2, mistral, qwen3, etc.)
    - param_count: parameter count (2b, 7b, etc.)
    - repeat: repetition count (0 for base, 1, 2, 4, 8, etc.)
    - early_exit: early exit layer (None if not used)
    - method: "repeat", "full-unmasking", "middle-unmasking"
    """
    # Remove eval_stats_ prefix and .json suffix
    basename = os.path.basename(filename)
    if basename.startswith("eval_stats_"):
        basename = basename[len("eval_stats_"):]
    if basename.endswith(".json"):
        basename = basename[:-5]

    # Split by dataset name pattern
    parts = basename.split("_")

    # Find where the dataset name starts (it's one of the known datasets)
    known_datasets = ["conll2003", "ace", "nlupp", "absa-restaurants", "wnut2017"]
    model_parts = []
    dataset_idx = None

    for i, part in enumerate(parts):
        if part in known_datasets:
            dataset_idx = i
            break
        model_parts.append(part)

    if dataset_idx is None:
        # Try to find dataset with hyphen
        for i, part in enumerate(parts):
            for ds in known_datasets:
                if ds in "-".join(parts[i:]):
                    dataset_idx = i
                    break
            if dataset_idx is not None:
                break

    model_name = "-".join(model_parts)

    # Parse model name components
    result = {
        "model": None,
        "param_count": None,
        "repeat": 0,
        "early_exit": None,
        "method": "repeat",
    }

    # Check for early exit
    early_exit_match = re.search(r"-early-exit-(\d+)", model_name)
    if early_exit_match:
        result["early_exit"] = int(early_exit_match.group(1))
        model_name = re.sub(r"-early-exit-\d+", "", model_name)

    # Check for repeat-k
    repeat_match = re.search(r"-repeat-k-(\d+)", model_name)
    if repeat_match:
        result["repeat"] = int(repeat_match.group(1))
        model_name = re.sub(r"-repeat-k-\d+", "", model_name)
        result["method"] = "repeat"

    # Check for unmasking variants
    if "-unmasked-middle" in model_name:
        result["method"] = "middle-unmasking"
        model_name = model_name.replace("-unmasked-middle", "")
    elif "-unmasked" in model_name:
        result["method"] = "full-unmasking"
        model_name = model_name.replace("-unmasked", "")

    # Remove headwise suffix (it's a different experiment)
    if "-headwise" in model_name:
        result["method"] = "headwise"
        model_name = model_name.replace("-headwise", "")

    # Parse model family and size
    # Handle specific model patterns
    model_patterns = [
        (r"^gemma-2-(\d+[bB])(?:-v[\d.]+)?$", "gemma-2", None),
        (r"^gemma-(\d+[bB])(?:-v[\d.]+)?$", "gemma", None),
        (r"^mistral-(\d+[bB])(?:-v[\d.]+)?$", "mistral", None),
        (r"^qwen3-(\d+\.?\d*[bB])(?:-v[\d.]+)?$", "qwen3", None),
    ]

    for pattern, model_family, _ in model_patterns:
        match = re.match(pattern, model_name, re.IGNORECASE)
        if match:
            result["model"] = model_family
            result["param_count"] = match.group(1).lower()
            break

    # If no pattern matched, try to extract from model name
    if result["model"] is None:
        # Try to find size pattern
        size_match = re.search(r"(\d+\.?\d*[bB])", model_name)
        if size_match:
            result["param_count"] = size_match.group(1).lower()
            # Remove size from model name
            model_base = re.sub(r"-?\d+\.?\d*[bB]", "", model_name)
            # Remove version suffix
            model_base = re.sub(r"-v[\d.]+", "", model_base)
            result["model"] = model_base.rstrip("-")

    return result


def get_repetition_dataset_scores(dataset_path: str) -> List[Tuple]:
    """
    Parse all evaluation files in a dataset directory.

    Returns a list of tuples:
    (model, param_count, repeat, early_exit, dataset, seed, micro_f1,
     micro_precision, micro_recall, accuracy, split, method)
    """
    scores = []

    if not os.path.isdir(dataset_path):
        return scores

    # Get dataset name from path
    dataset_name = os.path.basename(dataset_path.rstrip("/"))

    for filename in os.listdir(dataset_path):
        if not filename.endswith(".json") or not filename.startswith("eval_stats_"):
            continue

        filepath = os.path.join(dataset_path, filename)

        # Parse split from filename
        if "_test.json" in filename:
            split = "test"
        elif "_validation.json" in filename:
            split = "validation"
        else:
            continue

        # Parse model info
        model_info = parse_model_name(filename)

        # Skip headwise results (different experiment)
        if model_info["method"] == "headwise":
            continue

        # Skip if model not recognized
        if model_info["model"] is None:
            continue

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # Parse results for each seed
        for model_key, seed_results in data.items():
            for seed_str, metrics in seed_results.items():
                try:
                    seed = int(seed_str)
                except ValueError:
                    continue

                scores.append((
                    model_info["model"],
                    model_info["param_count"],
                    model_info["repeat"],
                    model_info["early_exit"],
                    dataset_name,
                    seed,
                    metrics.get("micro_f1", 0),
                    metrics.get("micro_precision", 0),
                    metrics.get("micro_recall", 0),
                    metrics.get("accuracy", 0),
                    split,
                    model_info["method"],
                ))

    return scores


def get_encoder_dataset_scores(dataset_path: str) -> List[Tuple]:
    """
    Parse encoder model evaluation files in a dataset directory.

    Returns a list of tuples:
    (model_name, dataset_name, seed, micro_f1, micro_precision,
     micro_recall, accuracy, split)
    """
    scores = []

    if not os.path.isdir(dataset_path):
        return scores

    # Get dataset name from path
    dataset_name = os.path.basename(dataset_path.rstrip("/"))

    # Encoder models to look for
    encoder_models = ["roberta", "bert", "distilbert", "modern-bert", "modernbert"]

    for filename in os.listdir(dataset_path):
        if not filename.endswith(".json") or not filename.startswith("eval_stats_"):
            continue

        # Check if this is an encoder model
        filename_lower = filename.lower()
        model_name = None

        for enc in encoder_models:
            if enc in filename_lower:
                # Map to display names
                if "modern" in enc:
                    model_name = "ModernBERT"
                elif enc == "roberta":
                    model_name = "RoBERTa"
                elif enc == "bert" and "distil" not in filename_lower and "modern" not in filename_lower:
                    model_name = "BERT"
                elif enc == "distilbert":
                    model_name = "DistilBERT"
                break

        if model_name is None:
            continue

        filepath = os.path.join(dataset_path, filename)

        # Parse split from filename
        if "_test.json" in filename:
            split = "test"
        elif "_validation.json" in filename:
            split = "validation"
        else:
            continue

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # Parse results for each seed
        for model_key, seed_results in data.items():
            for seed_str, metrics in seed_results.items():
                try:
                    seed = int(seed_str)
                except ValueError:
                    continue

                scores.append((
                    model_name,
                    dataset_name,
                    seed,
                    metrics.get("micro_f1", 0),
                    metrics.get("micro_precision", 0),
                    metrics.get("micro_recall", 0),
                    metrics.get("accuracy", 0),
                    split,
                ))

    return scores
