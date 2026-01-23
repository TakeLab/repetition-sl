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

roberta_scores = {
    'nlupp': 72.22,
    'ace': 68.96,
    'absa-restaurants': 80.30,
    'conll2003': 92.64,
}

ee_to_model_name = {
    12: 'Qwen3-0.34B',
    28: 'Qwen3-0.6B',
}

roberta_scores['average'] = np.mean(list(roberta_scores.values()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_results", type=str, default="results")
    parser.add_argument("--output", type=str, default="figures/compute_matched_grid.pdf")
    args = parser.parse_args()

    model = 'qwen3'
    param_count = '0.6B'
    
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

        # Filter for Qwen 0.6B model and test split
        df = df[df["model"] == model]
        df = df[df["param_count"] == param_count]
        df = df[df["split"] == "test"]
        df = df[df["method"] == "repeat"]

        # Aggregate over seeds
        agg_df = df.groupby(
            ["model", "param_count", "dataset", "early_exit", "repeat"]
        )["micro_f1"].mean().reset_index()

        all_dfs.append(agg_df)

    # Combine all datasets
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Convert to percentage
    combined_df["micro_f1"] = combined_df["micro_f1"] * 100

    # Get unique early exit values (non-null)
    early_exit_values = sorted(combined_df[combined_df["early_exit"].notna()]["early_exit"].unique())
    
    # Select two early exit configurations for the two rows
    # Use the first two available early exit thresholds
    if len(early_exit_values) >= 2:
        selected_ee = [early_exit_values[0], early_exit_values[-1]]
    else:
        selected_ee = early_exit_values[:2] if early_exit_values else [None]

    # Dataset configuration
    dataset_order = ["nlupp", "ace", "absa-restaurants", "conll2003"]
    dataset_labels = {
        "nlupp": "NLU++",
        "ace": "ACE05",
        "absa-restaurants": "Rest14",
        "conll2003": "CoNLL03",
        "average": "Average",
    }

    # Calculate difference from RoBERTa for each dataset
    combined_df["diff_from_roberta"] = combined_df.apply(
        lambda row: row["micro_f1"] - roberta_scores.get(row["dataset"], 0),
        axis=1
    )

    # Filter for selected early exit configurations
    ee_df = combined_df[combined_df["early_exit"].isin(selected_ee)]

    # Get unique repeat values
    repeat_values = sorted(ee_df["repeat"].unique())

    # Define colors for early exit configurations (beige tones)
    ee_colors = {
        12: "#F5DEB3",   # Beige (wheat)
        28: "#D2A679",   # Darker beige (tan/camel)
    }

    # Create figure: 1 row x 5 columns (4 datasets + average)
    fig, axs = plt.subplots(1, 5, figsize=(12, 2), sharey=False)
    fig.subplots_adjust(wspace=0.20, hspace=0.2)

    for col_idx, ds in enumerate(dataset_order + ["average"]):
        ax = axs[col_idx]
        
        for ee_val in selected_ee:
            color = ee_colors.get(int(ee_val), "#333333")
            
            if ds == "average":
                # Calculate average across all datasets for this early exit config
                panel_df = ee_df[ee_df["early_exit"] == ee_val]
                avg_by_repeat = panel_df.groupby("repeat")["diff_from_roberta"].mean().reset_index()
                avg_by_repeat = avg_by_repeat.sort_values("repeat")
                
                ax.plot(avg_by_repeat["repeat"], avg_by_repeat["diff_from_roberta"], 
                       '-o', color=color, markersize=5, linewidth=2.2, label=f"EE={int(ee_val)}")
            else:
                # Filter data for this panel
                panel_df = ee_df[
                    (ee_df["early_exit"] == ee_val) & 
                    (ee_df["dataset"] == ds)
                ].sort_values("repeat")
                
                if len(panel_df) > 0:
                    ax.plot(panel_df["repeat"], panel_df["diff_from_roberta"], 
                           '-o', color=color, markersize=5, linewidth=2.2, label=f"EE={int(ee_val)}")
        
        # Add horizontal line at y=0 (dashed)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        
        # Set x-axis
        ax.set_xticks(repeat_values)
        ax.set_xticklabels([str(int(r)) for r in repeat_values], fontsize=15, fontfamily="Times New Roman")
        
        # Set y-axis formatting with 3 equally spaced integer ticks
        y_min, y_max = ax.get_ylim()
        y_tick_min = int(np.floor(y_min))
        y_tick_max = int(np.ceil(y_max))
        # Make range divisible by 2 for equal spacing
        while (y_tick_max - y_tick_min) % 2 != 0:
            y_tick_max += 1
        y_tick_mid = (y_tick_min + y_tick_max) // 2
        ax.set_yticks([y_tick_min, y_tick_mid, y_tick_max])
        ax.set_ylim(y_tick_min - 0.5, y_tick_max + 0.5)
        ax.tick_params(axis='y', labelsize=15)
        for label in ax.get_yticklabels():
            label.set_fontfamily("Times New Roman")
        
        # Add title (dataset name)
        ax.set_title(dataset_labels.get(ds, ds), fontsize=15, fontweight="bold", fontfamily="Times New Roman")
        
        # Add y-axis label only for leftmost column
        if col_idx == 0:
            ax.set_ylabel("$\Delta$ F1 (%)", fontsize=14, fontfamily="Times New Roman")
        
        # Style the plot
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.tick_params(axis='both', colors='black', pad=2)

    # Add legend at the top
    legend_handles = []
    legend_labels = []
    for ee_val in selected_ee:
        color = ee_colors.get(int(ee_val), "#333333")
        handle = Line2D([0], [0], color=color, linestyle='-', marker='o', markersize=5, linewidth=2)
        legend_handles.append(handle)
        legend_labels.append(ee_to_model_name[int(ee_val)])

    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=len(selected_ee),
               fontsize=15, frameon=False, bbox_to_anchor=(0.5, 1.25),
               prop={'family': 'Times New Roman', 'size': 15})

    # Add global x-axis label
    fig.text(0.5, -0.10, "Repetition count", ha='center', fontsize=14, fontfamily="Times New Roman")

    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {args.output}")
