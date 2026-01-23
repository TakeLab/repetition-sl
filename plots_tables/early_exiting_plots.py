import argparse
import plotnine as p9
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from repetition.io import get_repetition_dataset_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_results", type=str, default="results")
    parser.add_argument(
        "--output", type=str, default="figures/early_exiting_grid.pdf"
    )
    args = parser.parse_args()

    all_dfs = []
    baseline_dfs = []

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

        # Early exit data
        ee_df = df[df["early_exit"].notna()].query('model == "mistral"')
        ee_df["early_exit"] = ee_df["early_exit"].astype(int) + 1
        ee_df = ee_df.query('split == "test" or split == "validation"')
        ee_df = ee_df.query('method == "repeat"')

        # Aggregate with mean, std, and count for CI calculation (include repeat)
        agg_df = (
            ee_df.groupby(
                [
                    "model",
                    "param_count",
                    "dataset",
                    "split",
                    "early_exit",
                    "repeat",
                    "method",
                ]
            )["micro_f1"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg_df.columns = [
            "model",
            "param_count",
            "dataset",
            "split",
            "early_exit",
            "repeat",
            "method",
            "micro_f1",
            "std",
            "n",
        ]

        all_dfs.append(agg_df)

        # Baseline data (no early exit) for Mistral with repeat method, excluding r=0
        baseline_df = df[df["early_exit"].isna()]
        baseline_df = baseline_df.query(
            'split == "test" or split == "validation"'
        )
        baseline_df = baseline_df.query('model == "mistral"')
        baseline_df = baseline_df.query('method == "repeat"')
        baseline_df = baseline_df.query("repeat != 0")

        if len(baseline_df) > 0:
            baseline_agg = (
                baseline_df.groupby(
                    ["model", "param_count", "dataset", "split", "repeat"]
                )["micro_f1"]
                .mean()
                .reset_index()
            )
            baseline_dfs.append(baseline_agg)

    # Combine all datasets
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Combine baseline data
    if baseline_dfs:
        baseline_combined = pd.concat(baseline_dfs, ignore_index=True)
        baseline_combined["micro_f1"] = baseline_combined["micro_f1"] * 100
        baseline_combined["split_label"] = baseline_combined["split"].map(
            {
                "validation": "Validation",
                "test": "Test",
            }
        )
    else:
        baseline_combined = pd.DataFrame()

    # Ensure early_exit is numeric and sort
    combined_df["early_exit"] = pd.to_numeric(combined_df["early_exit"])
    combined_df = combined_df.sort_values(
        ["dataset", "split", "repeat", "early_exit"]
    )

    # Convert to percentage
    combined_df["micro_f1"] = combined_df["micro_f1"] * 100

    # Define order for datasets
    dataset_order = ["nlupp", "ace", "absa-restaurants", "conll2003"]
    dataset_labels = {
        "nlupp": "NLU++",
        "ace": "ACE05",
        "absa-restaurants": "Rest14",
        "conll2003": "CoNLL03",
        "average": "Average",
    }

    split_labels = {
        "validation": "Validation",
        "test": "Test",
    }

    # Map dataset names to labels
    combined_df["dataset_label"] = combined_df["dataset"].map(dataset_labels)
    combined_df["split_label"] = combined_df["split"].map(split_labels)

    # Convert repeat to string for categorical coloring
    combined_df["repeat_str"] = "r=" + combined_df["repeat"].astype(
        int
    ).astype(str)

    # Get unique repeat values and create progressively darker shades with maximum contrast
    repeat_values = sorted(combined_df["repeat"].unique())

    # Define distinct colors from very light to very dark (r=0 not in this plot)
    repeat_colors = {
        "r=1": "#E6E6FA",  # Light lavender
        "r=2": "#C8B8E6",  # Medium lavender
        "r=4": "#9B7FD9",  # Medium-dark lavender
        "r=8": "#6B4DB8",  # Dark lavender/purple
    }

    # Get x-axis breaks
    x_breaks = sorted(combined_df["early_exit"].unique())

    # Create figure with shared x-axis: 2 rows x 5 columns (4 datasets + average)
    fig, axs = plt.subplots(2, 5, figsize=(12, 4), sharex=True)
    fig.subplots_adjust(wspace=0.3, hspace=0.15)

    splits = ["Validation", "Test"]

    for row_idx, split in enumerate(splits):
        for col_idx, ds in enumerate(dataset_order + ["average"]):
            ax = axs[row_idx, col_idx]

            if ds == "average":
                # For average panel: use all datasets
                panel_df = combined_df[combined_df["split_label"] == split]
            else:
                # Filter data for this panel
                panel_df = combined_df[
                    (combined_df["split_label"] == split)
                    & (combined_df["dataset"] == ds)
                ]

            # Plot each repeat value
            for r in repeat_values:
                r_df = panel_df[panel_df["repeat"] == r].sort_values(
                    "early_exit"
                )
                color = repeat_colors[f"r={int(r)}"]
                
                if ds == "average":
                    # For average, compute mean across datasets for each early_exit value
                    avg_df = r_df.groupby("early_exit")["micro_f1"].mean().reset_index()
                    ax.plot(
                        avg_df["early_exit"],
                        avg_df["micro_f1"],
                        "-o",
                        color=color,
                        markersize=5,
                        linewidth=2.2,
                        label=f"r={int(r)}",
                    )
                else:
                    ax.plot(
                        r_df["early_exit"],
                        r_df["micro_f1"],
                        "-o",
                        color=color,
                        markersize=5,
                        linewidth=2.2,
                        label=f"r={int(r)}",
                    )

                # Add dashed baseline line (no early exit) for this repeat value
                if len(baseline_combined) > 0:
                    if ds == "average":
                        baseline_panel = baseline_combined[
                            (baseline_combined["split_label"] == split)
                            & (baseline_combined["repeat"] == r)
                        ]
                        if len(baseline_panel) > 0:
                            baseline_val = baseline_panel["micro_f1"].mean()
                            ax.axhline(
                                y=baseline_val,
                                color=color,
                                linestyle="--",
                                linewidth=1.2,
                                alpha=0.7,
                            )
                    else:
                        baseline_panel = baseline_combined[
                            (baseline_combined["split_label"] == split)
                            & (baseline_combined["dataset"] == ds)
                            & (baseline_combined["repeat"] == r)
                        ]
                        if len(baseline_panel) > 0:
                            baseline_val = baseline_panel["micro_f1"].values[0]
                            ax.axhline(
                                y=baseline_val,
                                color=color,
                                linestyle="--",
                                linewidth=1.2,
                                alpha=0.7,
                            )

            # Set x-axis ticks (shared, so only set once)
            if row_idx == 1:
                ax.set_xticks(x_breaks)
                ax.set_xticklabels(
                    [str(x) for x in x_breaks],
                    fontsize=14,
                    fontfamily="Times New Roman",
                )

            # Find min and max values for this dataset and split (including baselines)
            if ds == "average":
                # For average column, use mean values across datasets
                all_values = []
                for r in repeat_values:
                    r_df = panel_df[panel_df["repeat"] == r]
                    avg_by_ee = r_df.groupby("early_exit")["micro_f1"].mean().values
                    all_values.extend(avg_by_ee)
                if len(baseline_combined) > 0:
                    baseline_panel = baseline_combined[
                        baseline_combined["split_label"] == split
                    ]
                    if len(baseline_panel) > 0:
                        # Average baseline across datasets for each repeat value
                        baseline_avg = baseline_panel.groupby("repeat")["micro_f1"].mean().values
                        all_values.extend(baseline_avg)
            else:
                all_values = list(panel_df["micro_f1"].values)
                if len(baseline_combined) > 0:
                    baseline_panel = baseline_combined[
                        (baseline_combined["split_label"] == split)
                        & (baseline_combined["dataset"] == ds)
                    ]
                    if len(baseline_panel) > 0:
                        all_values.extend(baseline_panel["micro_f1"].values)

            y_min = min(all_values)
            y_max = max(all_values)

            # Compute 3 equally spaced integer ticks: bottom, middle, top
            # Round min down and max up to integers
            y_tick_min = int(np.floor(y_min))
            y_tick_max = int(np.ceil(y_max))

            # Ensure ylims are even numbers
            # Lower ylim: if odd, round down to lower even
            if y_tick_min % 2 == 1:
                y_tick_min -= 1
            # Upper ylim: if odd, round up to higher even
            if y_tick_max % 2 == 1:
                y_tick_max += 1

            # Calculate middle tick as the midpoint (rounded to integer)
            y_tick_mid = int(np.round((y_tick_min + y_tick_max) / 2))

            # If all three would be the same, spread them out by 1
            if y_tick_min == y_tick_mid == y_tick_max:
                y_tick_min -= 1
                y_tick_max += 1
            elif y_tick_min == y_tick_mid:
                y_tick_mid = y_tick_min + 1
            elif y_tick_mid == y_tick_max:
                y_tick_mid = y_tick_max - 1

            y_ticks = [y_tick_min, y_tick_mid, y_tick_max]

            ax.set_yticks(y_ticks)
            ax.set_yticklabels(
                [str(y) for y in y_ticks],
                fontsize=14,
                fontfamily="Times New Roman",
            )

            # Set y-axis limits to show all data with small padding
            padding = max(1, y_tick_max - y_tick_min) * 0.1
            ax.set_ylim(y_tick_min - padding, y_tick_max + padding)

            # Add title (dataset name) only for first row
            if row_idx == 0:
                ax.set_title(
                    dataset_labels.get(ds, ds),
                    fontsize=15,
                    fontweight="bold",
                    fontfamily="Times New Roman",
                )

            # Add y-axis label only for leftmost column
            if col_idx == 0:
                ylabel = f"{split} F1 (%)"
                ax.set_ylabel(
                    ylabel, fontsize=15, fontfamily="Times New Roman"
                )

            # Hide x-axis ticks for top row
            if row_idx == 0:
                ax.tick_params(
                    axis="x", which="both", bottom=False, labelbottom=False
                )

            # Style the plot - keep bounding boxes
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.spines["bottom"].set_color("black")
            ax.spines["left"].set_color("black")
            ax.spines["top"].set_color("black")
            ax.spines["right"].set_color("black")
            ax.tick_params(axis="both", colors="black")

    # Add legend at the top with two rows and row titles

    # Create handles - interleaved for column-first filling
    # First column is for row titles, then r=0, r=1, etc.
    all_handles = []
    all_labels = []

    # Add row titles as first column (invisible handles)
    empty_handle = Line2D([0], [0], color="none", linestyle="none")
    all_handles.append(empty_handle)
    all_labels.append("Early exiting")
    all_handles.append(empty_handle)
    all_labels.append("No early exiting")

    for r in repeat_values:
        color = repeat_colors[f"r={int(r)}"]
        # Solid line with marker for early exit
        solid_handle = Line2D(
            [0],
            [0],
            color=color,
            linestyle="-",
            marker="o",
            markersize=4,
            linewidth=1.5,
        )
        all_handles.append(solid_handle)
        all_labels.append(f"$r={int(r)}$")

        # Dashed line for no early exit (immediately after solid for same r)
        dashed_handle = Line2D(
            [0], [0], color=color, linestyle="--", linewidth=1.2, alpha=0.7
        )
        all_handles.append(dashed_handle)
        all_labels.append(f"$r={int(r)}$")

    legend = fig.legend(
        all_handles,
        all_labels,
        loc="upper center",
        ncol=len(repeat_values) + 1,
        fontsize=14,
        frameon=False,
        bbox_to_anchor=(0.5, 1.12),
        prop={"family": "Times New Roman", "size": 14},
        columnspacing=1.0,
        handlelength=2.5,
    )
    legend.get_title().set_fontfamily("Times New Roman")

    # Add global x-axis label
    fig.text(
        0.5,
        -0.01,
        "Early exit threshold",
        ha="center",
        fontsize=15,
        fontfamily="Times New Roman",
    )

    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {args.output}")
