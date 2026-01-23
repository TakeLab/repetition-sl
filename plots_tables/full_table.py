from collections import defaultdict
import os
import json
import argparse
import pandas as pd
import numpy as np
import re

from scipy import stats

from repetition.io import (
    get_repetition_dataset_scores,
    get_encoder_dataset_scores,
)

full_table_heading = """\\begin{table*}[ht]
\\centering
\\small
\\renewcommand{\\arraystretch}{1.2}
\\begin{tabular*}{\\textwidth}{l@{\\extracolsep{\\fill}}lcccccc}
\\toprule
\\textbf{Model} & \\textbf{Method} & \\textbf{$r$} & \\textbf{NLU++} & \\textbf{ACE05} & \\textbf{Rest14} & \\textbf{CoNLL03} & \\textbf{Average} \\\\
"""

full_table_footer = """\\bottomrule
\\end{tabular*}
\\caption{Micro F1 scores for all models, methods, and repetition counts. Standard deviations are subscripted. Bold indicates best overall performance per dataset.}
\\label{tab:full_res}
\\end{table*}
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_results", type=str, default="results")
    parser.add_argument(
        "--path_to_encoder_results", type=str, default="encoder_results"
    )
    args = parser.parse_args()

    results_by_model = {}

    for dataset in os.listdir(args.path_to_results):
        if not os.path.isdir(os.path.join(args.path_to_results, dataset)):
            continue
        scores = get_repetition_dataset_scores(
            os.path.join(args.path_to_results, dataset)
        )

        df = pd.DataFrame(
            scores,
            columns=[
                "model",
                "param_count",
                "repeat",
                "early_exit",
                "dataset",
                "seed",
                "micro_f1",
                "micro_precision",
                "micro_recall",
                "accuracy",
                "split",
                "method",
            ],
        )

        # Filter out early exit results
        df = df[df["early_exit"].isna()]

        std = (
            df.groupby(
                [
                    "model",
                    "param_count",
                    "dataset",
                    "split",
                    "repeat",
                    "method",
                ]
            )["micro_f1"].std()
            * 100
        )
        f1 = (
            df.groupby(
                [
                    "model",
                    "param_count",
                    "dataset",
                    "split",
                    "repeat",
                    "method",
                ]
            )["micro_f1"].mean()
            * 100
        )
        seed_count = df.groupby(
            ["model", "param_count", "dataset", "split", "repeat", "method"]
        )["seed"].count()
        t = stats.t.ppf(0.975, seed_count - 1)
        ci_size = std * t / np.sqrt(seed_count)

        merged = pd.DataFrame(
            {"f1_mean": f1, "ci_size": ci_size}
        ).reset_index()

        test = merged.query("split == 'test'")
        test = test.reset_index(drop=True)

        for row in test.itertuples():
            model_name = row.model + "-" + row.param_count
            if model_name not in results_by_model:
                results_by_model[model_name] = defaultdict(lambda: defaultdict(dict))

            # Initialize datasets dict if not exists
            if row.repeat not in results_by_model[model_name][row.method]:
                results_by_model[model_name][row.method][row.repeat] = {
                    "datasets": {}
                }

            # Store dataset results
            results_by_model[model_name][row.method][row.repeat]["datasets"][
                row.dataset
            ] = (row.f1_mean, row.ci_size)

    full_table = full_table_heading

    # Define method order
    method_order = ["repeat", "middle-unmasking", "full-unmasking"]
    method_names = {
        "repeat": "Sequence repetition",
        "middle-unmasking": "Middle unmasking",
        "full-unmasking": "Full unmasking",
    }

    # Define repeat values for sequence repetition
    repeat_values = [0, 1, 2, 4, 8]

    # Iterate through models
    for model_name, methods in sorted(results_by_model.items()):
        model_first_row = True

        for method_idx, method in enumerate(method_order):
            if method not in methods:
                continue

            method_first_row = True

            if method == "repeat":
                # For sequence repetition, iterate through all repeat values
                for repeat in repeat_values:
                    if repeat not in methods[method]:
                        continue

                    data = methods[method][repeat]["datasets"]

                    # Calculate average across datasets
                    means = []
                    cis = []
                    dataset_results = []
                    for ds in ["nlupp", "ace", "absa-restaurants", "conll2003"]:
                        if ds in data:
                            means.append(data[ds][0])
                            cis.append(data[ds][1])
                            dataset_results.append(
                                f"${data[ds][0]:.2f}_{{{{\pm {data[ds][1]:.2f}}}}}$"
                            )
                        else:
                            dataset_results.append("-")

                    if means:
                        avg_mean = np.mean(means)
                        avg_ci = np.sqrt(np.sum(np.array(cis) ** 2)) / len(cis)
                    else:
                        avg_mean = 0
                        avg_ci = 0

                    # Format row
                    row = ""
                    if model_first_row:
                        # Count total rows for this model
                        total_rows = (
                            len([r for r in repeat_values if r in methods["repeat"]])
                            if "repeat" in methods
                            else 0
                        )
                        total_rows += (
                            1 if "middle-unmasking" in methods else 0
                        )
                        total_rows += 1 if "full-unmasking" in methods else 0

                        row += "\\midrule\n"
                        row += f"\\multirow{{{total_rows}}}{{*}}{{\\textbf{{{model_name.capitalize()}}}}} "
                        model_first_row = False
                    else:
                        row += " "

                    if method_first_row:
                        repeat_rows = len(
                            [r for r in repeat_values if r in methods["repeat"]]
                        )
                        row += f"& \\multirow{{{repeat_rows}}}{{*}}{{{method_names[method]}}} "
                        method_first_row = False
                    else:
                        row += "& "

                    row += f"& {repeat} & "
                    row += " & ".join(dataset_results)
                    row += f" & ${avg_mean:.2f}_{{{{\pm {avg_ci:.2f}}}}}$ \\\\\n"

                    full_table += row

            else:
                # For middle-unmasking and full-unmasking (no repeat parameter)
                if 0 not in methods[method]:
                    continue

                data = methods[method][0]["datasets"]

                # Calculate average across datasets
                means = []
                cis = []
                dataset_results = []
                for ds in ["nlupp", "ace", "absa-restaurants", "conll2003"]:
                    if ds in data:
                        means.append(data[ds][0])
                        cis.append(data[ds][1])
                        dataset_results.append(
                            f"${data[ds][0]:.2f}_{{{{\pm {data[ds][1]:.2f}}}}}$"
                        )
                    else:
                        dataset_results.append("-")

                if means:
                    avg_mean = np.mean(means)
                    avg_ci = np.sqrt(np.sum(np.array(cis) ** 2)) / len(cis)
                else:
                    avg_mean = 0
                    avg_ci = 0

                # Format row
                row = ""
                if model_first_row:
                    # Count total rows for this model
                    total_rows = (
                        len([r for r in repeat_values if r in methods["repeat"]])
                        if "repeat" in methods
                        else 0
                    )
                    total_rows += 1 if "middle-unmasking" in methods else 0
                    total_rows += 1 if "full-unmasking" in methods else 0

                    row += "\\midrule\n"
                    row += f"\\multirow{{{total_rows}}}{{*}}{{\\textbf{{{model_name.capitalize()}}}}} "
                    model_first_row = False
                else:
                    row += " "

                row += f"& {method_names[method]} & - & "
                row += " & ".join(dataset_results)
                row += f" & ${avg_mean:.2f}_{{{{\pm {avg_ci:.2f}}}}}$ \\\\\n"

                full_table += row

    # Add encoder results
    encoder_results_by_model = {}

    for dataset in os.listdir(args.path_to_encoder_results):
        if not os.path.isdir(
            os.path.join(args.path_to_encoder_results, dataset)
        ):
            continue
        scores = get_encoder_dataset_scores(
            os.path.join(args.path_to_encoder_results, dataset)
        )
        df = pd.DataFrame(
            scores,
            columns=[
                "model_name",
                "dataset_name",
                "seed",
                "micro_f1",
                "micro_precision",
                "micro_recall",
                "accuracy",
                "split",
            ],
        )

        std = (
            df.groupby(
                [
                    "model_name",
                    "dataset_name",
                    "split",
                ]
            )["micro_f1"].std()
            * 100
        )
        f1 = (
            df.groupby(
                [
                    "model_name",
                    "dataset_name",
                    "split",
                ]
            )["micro_f1"].mean()
            * 100
        )
        seed_count = df.groupby(["model_name", "dataset_name", "split"])[
            "seed"
        ].count()
        t = stats.t.ppf(0.975, seed_count - 1)
        ci_size = std * t / np.sqrt(seed_count)

        merged = pd.DataFrame(
            {"f1_mean": f1, "ci_size": ci_size}
        ).reset_index()

        test = merged.query("split == 'test'")
        test = test.reset_index()

        for row in test.itertuples():
            if row.model_name not in encoder_results_by_model:
                encoder_results_by_model[row.model_name] = {}
            encoder_results_by_model[row.model_name][row.dataset_name] = (
                row.f1_mean,
                row.ci_size,
            )

    for model_name, datasets in sorted(encoder_results_by_model.items()):
        # Calculate average across datasets for encoder models
        means = []
        cis = []
        dataset_results = []
        for ds in ["nlupp", "ace", "absa-restaurants", "conll2003"]:
            if ds in datasets:
                means.append(datasets[ds][0])
                cis.append(datasets[ds][1])
                dataset_results.append(
                    f"${datasets[ds][0]:.2f}_{{{{\pm {datasets[ds][1]:.2f}}}}}$"
                )
            else:
                dataset_results.append("-")

        if means:
            enc_avg_mean = np.mean(means)
            enc_avg_ci = np.sqrt(np.sum(np.array(cis) ** 2)) / len(cis)
        else:
            enc_avg_mean = 0
            enc_avg_ci = 0

        full_table += "\\midrule\n"
        full_table += f"{{\\textbf{{{model_name}}}}} "
        full_table += f"& - & - & "
        full_table += " & ".join(dataset_results)
        full_table += f" & ${enc_avg_mean:.2f}_{{{{\pm {enc_avg_ci:.2f}}}}}$ \\\\\n"

    full_table += full_table_footer
    print(full_table)
