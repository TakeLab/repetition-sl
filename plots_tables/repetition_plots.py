import argparse
import os
import plotnine as p9
import pandas as pd
import numpy as np
from scipy import stats

from repetition.io import get_repetition_dataset_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_results", type=str, default="results")
    parser.add_argument("--output", type=str, default="figures/repetition_grid.pdf")
    parser.add_argument("--split", type=str, default="test")
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

        df = df[df["early_exit"].isna()]

        df = df.query('method == "repeat"')
        df = df.query(f'split == "{args.split}"')

        # Aggregate with mean, std, and count for CI calculation
        agg_df = df.groupby(
            ["model", "param_count", "dataset", "split", "repeat", "method"]
        )["micro_f1"].agg(["mean", "std", "count"]).reset_index()
        agg_df.columns = ["model", "param_count", "dataset", "split", "repeat", "method", "micro_f1", "std", "n"]

        # Calculate 95% confidence interval
        agg_df["se"] = agg_df["std"] / np.sqrt(agg_df["n"])
        agg_df["t_val"] = stats.t.ppf(0.975, agg_df["n"] - 1)
        agg_df["ci"] = agg_df["se"] * agg_df["t_val"]

        all_dfs.append(agg_df)

    # Combine all datasets
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Create model name column
    combined_df["param_count"] = combined_df["param_count"].str.replace('b', 'B')
    combined_df["model"] = combined_df["model"].str.capitalize()
    combined_df["model_name"] = combined_df["model"] + "-" + combined_df["param_count"]
    combined_df["model_name"] = combined_df["model_name"].str.replace("Gemma-2", "Gemma2")

    # Convert to percentage
    combined_df["micro_f1"] = combined_df["micro_f1"] * 100
    combined_df["ci"] = combined_df["ci"] * 100
    combined_df["ymin"] = combined_df["micro_f1"] - combined_df["ci"]
    combined_df["ymax"] = combined_df["micro_f1"] + combined_df["ci"]
    print(set(combined_df['model_name']))

    # Define order for models and datasets
    model_order = [
        "Gemma-7B",
        "Gemma2-2B",
        "Gemma2-9B",
        "Mistral-7B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
    ]

    # Color palette for 7 models (colorblind-friendly)
    model_colors = {
        "Gemma-7B": "#E69F00",      # Orange
        "Gemma2-2B": "#56B4E9",    # Sky blue
        "Gemma2-9B": "#009E73",    # Teal
        "Mistral-7B": "#7B68EE",    # Medium slate blue
        "Qwen3-1.7B": "#0072B2",    # Blue
        "Qwen3-4B": "#D55E00",      # Vermillion
        "Qwen3-8B": "#CC79A7",      # Pink
    }

    dataset_order = ["nlupp", "ace", "absa-restaurants", "conll2003"]
    dataset_labels = {
        "nlupp": "NLU++",
        "ace": "ACE05",
        "absa-restaurants": "Rest14",
        "conll2003": "CoNLL03",
    }

    # Map dataset names to labels
    combined_df["dataset_label"] = combined_df["dataset"].map(dataset_labels)
    
    # Create average row across all datasets for each model
    avg_df = combined_df.groupby(["model_name", "repeat"]).agg({
        "micro_f1": "mean",
        "ci": "mean",
    }).reset_index()
    avg_df["ymin"] = avg_df["micro_f1"] - avg_df["ci"]
    avg_df["ymax"] = avg_df["micro_f1"] + avg_df["ci"]
    avg_df["dataset_label"] = "Average"
    
    # Add average rows to the dataframe
    combined_df = pd.concat([combined_df, avg_df], ignore_index=True)
    
    # Update dataset labels to include Average
    dataset_labels["average"] = "Average"

    # Set categorical ordering
    combined_df["model_name"] = pd.Categorical(
        combined_df["model_name"], categories=model_order, ordered=True
    )
    dataset_label_order = [dataset_labels[d] for d in dataset_order] + ["Average"]
    combined_df["dataset_label"] = pd.Categorical(
        combined_df["dataset_label"],
        categories=dataset_label_order,
        ordered=True,
    )

    # Create 5x7 faceted plot (4 datasets + average as rows, 7 models as columns)
    plot = (
        p9.ggplot(combined_df, p9.aes(x="repeat", y="micro_f1", color="model_name", fill="model_name"))
        + p9.geom_ribbon(p9.aes(ymin="ymin", ymax="ymax"), alpha=0.2, color=None)
        + p9.geom_line(size=1)
        + p9.geom_point(size=2)
        + p9.facet_grid("dataset_label ~ model_name")
        + p9.scale_color_manual(values=model_colors)
        + p9.scale_fill_manual(values=model_colors)
        + p9.scale_y_continuous(breaks=[50, 70, 90])
        + p9.scale_x_continuous(breaks=[0, 1, 2, 4, 8])
        + p9.labs(
            x="Repetition count",
            y="Micro F1 (%)",
        )
        + p9.theme_bw()
        + p9.theme(
            figure_size=(10, 6.25),
            text=p9.element_text(family="Times New Roman", size=14),
            strip_text_x=p9.element_text(size=13, weight="bold", family="Times New Roman", color="black"),
            strip_text_y=p9.element_text(size=13, weight="bold", family="Times New Roman", angle=-90, color="black"),
            axis_text_x=p9.element_text(size=13, family="Times New Roman", color="black"),
            axis_text_y=p9.element_text(size=13, family="Times New Roman", color="black"),
            axis_title=p9.element_text(size=14, family="Times New Roman", color="black"),
            panel_border=p9.element_rect(color="black"),
            strip_background=p9.element_blank(),
            panel_grid_major=p9.element_blank(),
            panel_grid_minor=p9.element_blank(),
            panel_spacing_x=0.0075,
            panel_spacing_y=0.0075,
            legend_position="none",
        )
    )

    plot.save(args.output, dpi=300)
    print(f"Plot saved to {args.output}")
