import argparse
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from repetition.io import get_repetition_dataset_scores

# Fix minus sign rendering
mpl.rcParams['axes.unicode_minus'] = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_results", type=str, default="results")
    parser.add_argument("--output", type=str, default="figures/aggregated_results.pdf")
    args = parser.parse_args()

    all_dfs = []

    for dataset in os.listdir(args.path_to_results):
        if not os.path.isdir(os.path.join(args.path_to_results, dataset)):
            continue
        scores = get_repetition_dataset_scores(
            os.path.join(args.path_to_results, dataset),
        )

        df = pd.DataFrame(scores, columns=[
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
        ])

        # Filter for test split only
        df = df[df["split"] == "test"]
        df = df[df['early_exit'].isna()]

        all_dfs.append(df)

    # Combine all datasets
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Convert to percentage
    combined_df["micro_f1"] = combined_df["micro_f1"] * 100

    # Separate data by method
    repeat_df = combined_df[combined_df["method"] == "repeat"]
    full_unmask_df = combined_df[combined_df["method"] == "full-unmasking"]
    middle_unmask_df = combined_df[combined_df["method"] == "middle-unmasking"]

    # Get unique repeat values (including 0)
    repeat_values = sorted([r for r in repeat_df["repeat"].unique() if r > 0])

    # Create a lookup for middle unmasking F1 by (model, param_count, dataset, seed)
    middle_lookup = middle_unmask_df.groupby(
        ["model", "param_count", "dataset", "seed"]
    )["micro_f1"].mean().to_dict()

    # Define colors
    repeat_color = "#2E86AB"  # Blue for repetition
    full_unmask_color = "#A23B72"  # Magenta for full unmasking
    middle_unmask_color = "#F18F01"  # Orange for middle unmasking

    # Dataset configuration
    dataset_order = ["nlupp", "ace", "absa-restaurants", "conll2003"]
    dataset_labels = {
        "nlupp": "NLU++",
        "ace": "ACE05",
        "absa-restaurants": "Rest14",
        "conll2003": "CoNLL03",
        "average": "Average",
    }

    # Create figure: 2 rows x 5 columns (4 datasets + average)
    # Row 1: Means, Row 2: Standard deviations
    fig, axs = plt.subplots(2, 5, figsize=(12, 4), sharey=False)
    fig.subplots_adjust(wspace=0.20, hspace=0.15)

    for col_idx, ds in enumerate(dataset_order + ["average"]):
        ax_mean = axs[0, col_idx]
        ax_std = axs[1, col_idx]

        if ds == "average":
            # For average panel: compute across all datasets
            ds_repeat_df = repeat_df
            ds_full_df = full_unmask_df
        else:
            # Filter data for this dataset
            ds_repeat_df = repeat_df[repeat_df["dataset"] == ds]
            ds_full_df = full_unmask_df[full_unmask_df["dataset"] == ds]

        # Repetition line - compute difference from middle unmasking per (model, param_count, dataset, seed)
        repeat_means = []
        repeat_stds = []
        for r in repeat_values:
            r_df = ds_repeat_df[ds_repeat_df["repeat"] == r]
            diffs = []
            f1_values = []
            for _, row in r_df.iterrows():
                key = (row["model"], row["param_count"], row["dataset"], row["seed"])
                if key in middle_lookup:
                    diffs.append(row["micro_f1"] - middle_lookup[key])
                    f1_values.append(row["micro_f1"])
            if diffs:
                repeat_means.append(np.mean(diffs))
                repeat_stds.append(np.std(f1_values))
            else:
                repeat_means.append(np.nan)
                repeat_stds.append(np.nan)

        repeat_means = np.array(repeat_means)
        repeat_stds = np.array(repeat_stds)

        # Plot means
        ax_mean.plot(repeat_values, repeat_means, '-o', color=repeat_color,
                     markersize=5, linewidth=2.2)

        # Plot standard deviations
        ax_std.plot(repeat_values, repeat_stds, '-o', color=repeat_color,
                    markersize=5, linewidth=2.2)

        # Full unmasking - compute difference from middle unmasking per (model, param_count, dataset, seed)
        full_diffs = []
        full_f1_values = []
        for _, row in ds_full_df.iterrows():
            key = (row["model"], row["param_count"], row["dataset"], row["seed"])
            if key in middle_lookup:
                full_diffs.append(row["micro_f1"] - middle_lookup[key])
                full_f1_values.append(row["micro_f1"])
        if full_diffs:
            full_mean = np.mean(full_diffs)
            full_std = np.std(full_f1_values)
            ax_mean.axhline(y=full_mean, color=full_unmask_color, linestyle='--', linewidth=2.2)
            ax_std.axhline(y=full_std, color=full_unmask_color, linestyle='--', linewidth=2.2)

        # Middle unmasking reference line at y=0 for means
        ax_mean.axhline(y=0, color=middle_unmask_color, linestyle='--', linewidth=2.2)

        # Middle unmasking std (of raw F1 values)
        if ds == "average":
            ds_middle_df = middle_unmask_df
        else:
            ds_middle_df = middle_unmask_df[middle_unmask_df["dataset"] == ds]
        middle_std = ds_middle_df["micro_f1"].std() if len(ds_middle_df) > 0 else 0
        ax_std.axhline(y=middle_std, color=middle_unmask_color, linestyle='--', linewidth=2.2)

        # Format mean plot
        ax_mean.set_xticks(repeat_values)
        ax_mean.set_xticklabels([])  # No x labels on top row

        # Ensure y-axis includes 0 (the middle unmasking reference)
        y_min, y_max = ax_mean.get_ylim()
        y_tick_min = int(np.floor(min(y_min, 0)))
        y_tick_max = int(np.ceil(max(y_max, 0)))
        # Ensure range includes 0 and is divisible by 2 for equal spacing
        while (y_tick_max - y_tick_min) % 2 != 0:
            if abs(y_tick_min) < abs(y_tick_max):
                y_tick_min -= 1
            else:
                y_tick_max += 1
        y_tick_mid = (y_tick_min + y_tick_max) // 2
        # If 0 is not one of the ticks, adjust to include it
        ticks = [y_tick_min, y_tick_mid, y_tick_max]
        if 0 not in ticks:
            # Use 0 as middle tick and adjust others symmetrically
            max_abs = max(abs(y_tick_min), abs(y_tick_max))
            y_tick_min = -max_abs
            y_tick_max = max_abs
            ticks = [y_tick_min, 0, y_tick_max]
        ax_mean.set_yticks(ticks)
        ax_mean.set_ylim(y_tick_min - 0.5, y_tick_max + 0.5)
        ax_mean.tick_params(axis='y', labelsize=15)
        for label in ax_mean.get_yticklabels():
            label.set_fontfamily("Times New Roman")

        # Add title (dataset name) only on top row
        ax_mean.set_title(dataset_labels.get(ds, ds), fontsize=15, fontweight="bold", fontfamily="Times New Roman")

        # Add y-axis label only for leftmost column
        if col_idx == 0:
            ax_mean.set_ylabel("$\Delta$ F1 (%)", fontsize=14, fontfamily="Times New Roman")
            ax_mean.yaxis.set_label_coords(-0.15, 0.5)

        # Style mean plot
        ax_mean.spines['top'].set_visible(True)
        ax_mean.spines['right'].set_visible(True)
        ax_mean.spines['bottom'].set_color('black')
        ax_mean.spines['left'].set_color('black')
        ax_mean.spines['top'].set_color('black')
        ax_mean.spines['right'].set_color('black')
        ax_mean.tick_params(axis='both', colors='black', pad=2)

        # Format std plot
        ax_std.set_xticks(repeat_values)
        ax_std.set_xticklabels([str(int(r)) for r in repeat_values], fontsize=15, fontfamily="Times New Roman")

        y_min, y_max = ax_std.get_ylim()
        y_tick_min = int(np.floor(y_min))
        y_tick_max = int(np.ceil(y_max))
        while (y_tick_max - y_tick_min) % 2 != 0:
            y_tick_max += 1
        y_tick_mid = (y_tick_min + y_tick_max) // 2
        ax_std.set_yticks([y_tick_min, y_tick_mid, y_tick_max])
        ax_std.set_ylim(y_tick_min - 0.5, y_tick_max + 0.5)
        ax_std.tick_params(axis='y', labelsize=15)
        for label in ax_std.get_yticklabels():
            label.set_fontfamily("Times New Roman")

        # Add y-axis label only for leftmost column
        if col_idx == 0:
            ax_std.set_ylabel("Std F1 (%)", fontsize=14, fontfamily="Times New Roman")
            ax_std.yaxis.set_label_coords(-0.15, 0.5)

        # Style std plot
        ax_std.spines['top'].set_visible(True)
        ax_std.spines['right'].set_visible(True)
        ax_std.spines['bottom'].set_color('black')
        ax_std.spines['left'].set_color('black')
        ax_std.spines['top'].set_color('black')
        ax_std.spines['right'].set_color('black')
        ax_std.tick_params(axis='both', colors='black', pad=2)

    # Add legend at the top
    legend_handles = [
        Line2D([0], [0], color=repeat_color, linestyle='-', marker='o', markersize=6, linewidth=3),
        Line2D([0], [0], color=middle_unmask_color, linestyle='--', linewidth=3),
        Line2D([0], [0], color=full_unmask_color, linestyle='--', linewidth=3),
    ]
    legend_labels = ["Repetition", "Middle unmasking", "Full unmasking"]

    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=3,
               fontsize=15, frameon=False, bbox_to_anchor=(0.5, 1.05),
               prop={'family': 'Times New Roman', 'size': 15})

    # Add global x-axis label
    fig.text(0.5, -0.00, "Repetition count", ha='center', fontsize=14, fontfamily="Times New Roman")

    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {args.output}")
