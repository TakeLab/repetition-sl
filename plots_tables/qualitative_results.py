import argparse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from repetition.io import get_repetition_dataset_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_results", type=str, default="results")
    parser.add_argument("--output", type=str, default="figures/dominance_heatmap.pdf")
    args = parser.parse_args()

    all_dfs = []
    
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
        all_dfs.append(df)
    
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Filter: test split, no early exit
    df = df[df["split"] == "test"]
    df = df[df["early_exit"].isna()]
    
    # Create a strategy column that combines method and repeat value
    # For repeat method: use r=1, r=2, r=4, r=8
    # For other methods: use method name


    def get_strategy(row):
        if row["method"] == "repeat":
            return f"r={int(row['repeat'])}"
        elif row["method"] == "full-unmasking":
            return "FU"
        elif row["method"] == "middle-unmasking":
            return "MU"
        else:
            return row["method"]

    print(df)


    df["strategy"] = df.apply(get_strategy, axis=1)
    
    # Filter to only the strategies we care about
    strategies_of_interest = ["FU", "MU", "r=1", "r=2", "r=4", "r=8"]
    df = df[df["strategy"].isin(strategies_of_interest)]
    
    # Aggregate: mean F1 per model-dataset-strategy
    agg_df = df.groupby(["model", "param_count", "dataset", "strategy"])["micro_f1"].mean().reset_index()
    
    # Create model-dataset identifier
    agg_df["model_dataset"] = agg_df["model"] + "-" + agg_df["param_count"] + "-" + agg_df["dataset"]
    
    # Pivot to have strategies as columns
    pivot_df = agg_df.pivot(index="model_dataset", columns="strategy", values="micro_f1")
    
    # Keep only rows that have all strategies
    pivot_df = pivot_df.dropna()
    
    # Count pairwise wins: wins[i, j] = how many times strategy i outperformed strategy j
    n_strategies = len(strategies_of_interest)
    wins = np.zeros((n_strategies, n_strategies), dtype=int)
    
    for i, strat_i in enumerate(strategies_of_interest):
        for j, strat_j in enumerate(strategies_of_interest):
            if i == j:
                wins[i, j] = 0
            else:
                # Count how many times strat_i > strat_j
                wins[i, j] = (pivot_df[strat_i] > pivot_df[strat_j]).sum()
    
    # Create DataFrame for heatmap
    wins_df = pd.DataFrame(wins, index=strategies_of_interest, columns=strategies_of_interest)
    
    # Count overall wins: how many times each strategy was the best (beat ALL others)
    # For each model-dataset pair, find the strategy with the highest F1
    overall_wins = {strat: 0 for strat in strategies_of_interest}

    for idx, row in pivot_df.iterrows():
        best_strategy = row.idxmax()  # Strategy with highest F1 for this model-dataset
        overall_wins[best_strategy] += 1
    overall_wins_values = [overall_wins[strat] for strat in strategies_of_interest]
    
    # Verification: sum of overall wins should equal number of model-dataset pairs
    print(f"\nOverall wins per strategy: {overall_wins}")
    print(f"Sum of overall wins: {sum(overall_wins_values)}")
    print(f"Total model-dataset pairs: {len(pivot_df)}")
    
    # Create figure with 2 subplots: heatmap on left, horizontal bar chart on right
    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(8.5, 5), 
                                           gridspec_kw={'width_ratios': [3, 1.8], 'wspace': 0.05})
    
    # Plot heatmap with colors that complement lavender
    # Using a custom colormap with warm peach/coral tones
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#FFFFFF', '#FFE5D9', '#FFC9B3', '#FF9F80', '#FF7F5C']
    n_bins = 100
    cmap_custom = LinearSegmentedColormap.from_list('peach', colors, N=n_bins)

    sns.heatmap(wins_df, annot=True, fmt="d", cmap=cmap_custom, ax=ax_heat,
                cbar=False,
                linewidths=0.5, linecolor='white',
                annot_kws={'fontsize': 22, 'fontfamily': 'Times New Roman', 'color': 'black'},
                square=True)
    
    # Style heatmap
    ax_heat.set_xlabel("Adaptation strategy", fontsize=22, fontfamily="Times New Roman", color='black')
    ax_heat.set_ylabel("Adaptation strategy", fontsize=22, fontfamily="Times New Roman", color='black')

    # Set tick labels
    ax_heat.set_xticklabels(strategies_of_interest, fontsize=20, fontfamily="Times New Roman", rotation=0, color='black')
    ax_heat.set_yticklabels(strategies_of_interest, fontsize=20, fontfamily="Times New Roman", rotation=90, color='black')
    ax_heat.tick_params(colors='black')
    
    # Plot horizontal bar chart - bars aligned with heatmap rows
    bar_positions = np.arange(len(strategies_of_interest)) + 0.5  # Center bars on heatmap cells

    # Color bars based on win count using the same colormap as heatmap
    max_wins = max(overall_wins_values) if max(overall_wins_values) > 0 else 1
    bar_colors = [cmap_custom(val / max_wins * 0.8 + 0.2) for val in overall_wins_values]  # Scale to avoid too light colors

    bars = ax_bar.barh(bar_positions, overall_wins_values, height=0.7, color=bar_colors, edgecolor='white')
    
    # Style bar chart - remove bounding box, keep only axis lines
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(True)  # Keep left axis line
    ax_bar.spines['left'].set_color('black')
    ax_bar.spines['bottom'].set_color('black')
    ax_bar.set_ylim(len(strategies_of_interest), 0)  # Invert to match heatmap row order
    ax_bar.set_yticks([])  # No y-axis ticks (they're on the heatmap)
    ax_bar.set_xlabel("#Overall wins", fontsize=22, fontfamily="Times New Roman", color='black')
    ax_bar.tick_params(axis='x', labelsize=20, colors='black')
    for label in ax_bar.get_xticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_color('black')
    
    # Add value labels at end of bars
    for bar, val in zip(bars, overall_wins_values):
        ax_bar.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    str(int(val)), ha='left', va='center', fontsize=20, fontfamily='Times New Roman', color='black')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"\nDominance heatmap saved to {args.output}")
    print(f"\nWins matrix (row beats column):\n{wins_df}")
