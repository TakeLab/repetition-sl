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

main_table_heading = """\\begin{table*}[ht]
\\centering
\\small
\\renewcommand{\\arraystretch}{1.2}
% \\textwidth forces the table to span the Full unmasking text width
% @{\extracolsep{\\fill}} calculates equal spacing between columns automatically
\\begin{tabular*}{\\textwidth}{l@{\\extracolsep{\\fill}}lccccc}
\\toprule
\\textbf{Model} & \\textbf{Method} & \\textbf{NLU++} & \\textbf{ACE05} & \\textbf{Rest14} & \\textbf{CoNLL03} & \\textbf{Average} \\\\
"""

main_table_footer = """\\bottomrule
\\end{tabular*}
\\caption{Micro F1 scores grouped by model. Standard deviations are subscripted. Bold indicates best overall performance per dataset.}
\\label{tab:main_res}
\\end{table*}
"""

model_to_best_r = {
    "gemma-7B": 8,
    "gemma-2-2B": 2,
    "gemma-2-9B": 2,
    "mistral-7B": 4,
    "qwen3-1.7B": 4,
    "qwen3-4B": 8,
    "qwen3-8B": 4,
}
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
        test = test.reset_index()

        best_repeat_test = test.iloc[[6, 11, 18, 26, 33, 41, 47]]

        for row in best_repeat_test.itertuples():
            model_name = row.model + "-" + row.param_count
            if model_name not in results_by_model:
                results_by_model[model_name] = defaultdict(list)

            results_by_model[model_name][row.dataset].append(
                (
                    row.f1_mean,
                    row.ci_size,
                    row.method,
                    row.repeat,
                )
            )

        for row in test.itertuples():
            model_name = row.model + "-" + row.param_count
            if model_name not in results_by_model:
                results_by_model[model_name] = defaultdict(list)

            results_by_model[model_name][row.dataset].append(
                (
                    row.f1_mean,
                    row.ci_size,
                    row.method,
                    row.repeat,
                )
            )

    main_table = main_table_heading

    for model_name, datasets in results_by_model.items():
        nlupp_results = datasets["nlupp"]
        ace05_results = datasets["ace"]
        rest14_results = datasets["absa-restaurants"]
        conll03_results = datasets["conll2003"]

        # Calculate average across datasets for each method
        # For sequence repetition (index 0)
        seq_rep_means = [
            nlupp_results[0][0],
            ace05_results[0][0],
            rest14_results[0][0],
            conll03_results[0][0],
        ]
        seq_rep_cis = [
            nlupp_results[0][1],
            ace05_results[0][1],
            rest14_results[0][1],
            conll03_results[0][1],
        ]
        seq_rep_avg_mean = np.mean(seq_rep_means)
        # Pooled CI: sqrt(sum of squared CIs / n)
        seq_rep_avg_ci = np.sqrt(np.sum(np.array(seq_rep_cis) ** 2)) / len(
            seq_rep_cis
        )

        # For middle unmasking (index 2)
        mid_means = [
            nlupp_results[2][0],
            ace05_results[2][0],
            rest14_results[2][0],
            conll03_results[2][0],
        ]
        mid_cis = [
            nlupp_results[2][1],
            ace05_results[2][1],
            rest14_results[2][1],
            conll03_results[2][1],
        ]
        mid_avg_mean = np.mean(mid_means)
        mid_avg_ci = np.sqrt(np.sum(np.array(mid_cis) ** 2)) / len(mid_cis)

        # For full unmasking (index 1)
        full_means = [
            nlupp_results[1][0],
            ace05_results[1][0],
            rest14_results[1][0],
            conll03_results[1][0],
        ]
        full_cis = [
            nlupp_results[1][1],
            ace05_results[1][1],
            rest14_results[1][1],
            conll03_results[1][1],
        ]
        full_avg_mean = np.mean(full_means)
        full_avg_ci = np.sqrt(np.sum(np.array(full_cis) ** 2)) / len(full_cis)

        main_table += "\\midrule"
        main_table += (
            f"\\multirow{{3}}{{*}}{{\\textbf{{{model_name.capitalize()}}}}} "
        )
        main_table += f"& Sequence repetition ($r = {model_to_best_r[model_name]}$)& ${nlupp_results[0][0]:.2f}_{{\\pm {nlupp_results[0][1]:.2f}}}^{{\\;r={nlupp_results[0][3]}}}$ & ${ace05_results[0][0]:.2f}_{{\\pm {ace05_results[2][1]:.2f}}}^{{\\;r={ace05_results[0][3]}}}$ & ${rest14_results[0][0]:.2f}_{{\\pm {rest14_results[0][1]:.2f}}}^{{\\;r={rest14_results[0][3]}}}$ & ${conll03_results[0][0]:.2f}_{{\\pm {conll03_results[0][1]:.2f}}}^{{\\;r={conll03_results[0][3]}}}$ & ${seq_rep_avg_mean:.2f}_{{\\pm {seq_rep_avg_ci:.2f}}}$ \\\\"
        main_table += f"& Middle unmasking & ${nlupp_results[2][0]:.2f}_{{\\pm {nlupp_results[2][1]:.2f}}}$ & ${ace05_results[2][0]:.2f}_{{\\pm {ace05_results[2][1]:.2f}}}$ & ${rest14_results[2][0]:.2f}_{{\\pm {rest14_results[2][1]:.2f}}}$ & ${conll03_results[2][0]:.2f}_{{\\pm {conll03_results[2][1]:.2f}}}$ & ${mid_avg_mean:.2f}_{{\\pm {mid_avg_ci:.2f}}}$ \\\\"
        main_table += f"& Full unmasking & ${nlupp_results[1][0]:.2f}_{{\\pm {nlupp_results[1][1]:.2f}}}$ & ${ace05_results[1][0]:.2f}_{{\\pm {ace05_results[1][1]:.2f}}}$ & ${rest14_results[1][0]:.2f}_{{\\pm {rest14_results[1][1]:.2f}}}$ & ${conll03_results[1][0]:.2f}_{{\\pm {conll03_results[1][1]:.2f}}}$ & ${full_avg_mean:.2f}_{{\\pm {full_avg_ci:.2f}}}$ \\\\"

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
        print(test)


        for row in test.itertuples():
            if row.model_name not in encoder_results_by_model:
                encoder_results_by_model[row.model_name] = defaultdict(list)
            encoder_results_by_model[row.model_name][row.dataset_name].append(
                (
                    row.f1_mean,
                    row.ci_size,
                )
            )

    for model_name, datasets in encoder_results_by_model.items():
        nlupp_results = datasets["nlupp"]
        ace05_results = datasets["ace"]
        rest14_results = datasets["absa-restaurants"]
        conll03_results = datasets["conll2003"]

        # Calculate average across datasets for encoder models
        enc_means = [
            nlupp_results[0][0],
            ace05_results[0][0],
            rest14_results[0][0],
            conll03_results[0][0],
        ]
        enc_cis = [
            nlupp_results[0][1],
            ace05_results[0][1],
            rest14_results[0][1],
            conll03_results[0][1],
        ]
        enc_avg_mean = np.mean(enc_means)
        enc_avg_ci = np.sqrt(np.sum(np.array(enc_cis) ** 2)) / len(enc_cis)

        main_table += "\\midrule"
        main_table += f"{{\\textbf{{{model_name}}}}} "
        main_table += f"& - & ${nlupp_results[0][0]:.2f}_{{\\pm {nlupp_results[0][1]:.2f}}}$ & ${ace05_results[0][0]:.2f}_{{\\pm {ace05_results[0][1]:.2f}}}$ & ${rest14_results[0][0]:.2f}_{{\\pm {rest14_results[0][1]:.2f}}}$ & ${conll03_results[0][0]:.2f}_{{\\pm {conll03_results[0][1]:.2f}}}$ & ${enc_avg_mean:.2f}_{{\\pm {enc_avg_ci:.2f}}}$ \\\\"

    main_table += main_table_footer
    print(main_table)
