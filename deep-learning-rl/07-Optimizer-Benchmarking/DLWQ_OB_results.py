"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 2, 2026
Last Edit:  November 2, 2026
Progress:   Complete

Analysis:   06 - Optimizer Benchmarking
Method:     Comparison

Description:
Compares optimizer benchmarking results and identifies
the best optimizer for wine quality prediction.

Optimizers Evaluated:

- SGD
- RMSProp
- Adam
"""

# ==========================================
# Imports
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# ==========================================
# Directories
# ==========================================

results_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\07-Optimizer-Benchmarking\tables_figures"
)

comparison_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\07-Optimizer-Benchmarking\comparison"
)

comparison_dir.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# Optimizers
# ==========================================

optimizers = [
    "sgd",
    "rmsprop",
    "adam"
]

# ==========================================
# Load Results
# ==========================================

all_results = []

for optimizer in optimizers:

    df = pd.read_csv(
        results_dir /
        f"{optimizer}_results.csv"
    )

    all_results.append(df)

comparison = pd.concat(
    all_results,
    ignore_index=True
)

comparison = comparison.sort_values(
    by="RMSE"
)

# ==========================================
# Export Comparison
# ==========================================

comparison.to_csv(
    comparison_dir /
    "optimizer_comparison.csv",
    index=False
)

print(
    "\n----- OPTIMIZER COMPARISON -----"
)

print(
    comparison
    .round(4)
    .to_string(index=False)
)

# ==========================================
# Winner
# ==========================================

winner = comparison.iloc[0]

best_optimizer = winner["Optimizer"]

print(
    f"\nBest Optimizer: {best_optimizer}"
)

# ==========================================
# Plot Colors
# ==========================================

colors = {

    "SGD":
        "#1f77b4",

    "RMSProp":
        "#ff7f0e",

    "Adam":
        "#2ca02c"
}

bar_colors = [

    colors[o]

    for o in comparison["Optimizer"]

]

# ==========================================
# MAE Comparison
# ==========================================

plt.figure(
    figsize=(8,6)
)

plt.bar(
    comparison["Optimizer"],
    comparison["MAE"],
    color=bar_colors
)

plt.title(
    "MAE Comparison"
)

plt.ylabel("MAE")

plt.grid(
    axis="y"
)

plt.savefig(
    comparison_dir /
    "mae_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# RMSE Comparison
# ==========================================

plt.figure(
    figsize=(8,6)
)

plt.bar(
    comparison["Optimizer"],
    comparison["RMSE"],
    color=bar_colors
)

plt.title(
    "RMSE Comparison"
)

plt.ylabel("RMSE")

plt.grid(
    axis="y"
)

plt.savefig(
    comparison_dir /
    "rmse_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Adjusted R² Comparison
# ==========================================

plt.figure(
    figsize=(8,6)
)

plt.bar(
    comparison["Optimizer"],
    comparison["Adjusted_R2"],
    color=bar_colors
)

plt.title(
    "Adjusted R² Comparison"
)

plt.ylabel(
    "Adjusted R²"
)

plt.grid(
    axis="y"
)

plt.savefig(
    comparison_dir /
    "adjusted_r2_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Training Time Comparison
# ==========================================

plt.figure(
    figsize=(8,6)
)

plt.bar(
    comparison["Optimizer"],
    comparison["Training_Time_Seconds"],
    color=bar_colors
)

plt.title(
    "Training Time Comparison"
)

plt.ylabel(
    "Training Time (Seconds)"
)

plt.grid(
    axis="y"
)

plt.savefig(
    comparison_dir /
    "training_time_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Combined Metrics Figure
# ==========================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14,10)
)

axes[0,0].bar(
    comparison["Optimizer"],
    comparison["MAE"],
    color=bar_colors
)

axes[0,0].set_title(
    "MAE"
)

axes[0,0].grid(
    axis="y"
)

axes[0,1].bar(
    comparison["Optimizer"],
    comparison["RMSE"],
    color=bar_colors
)

axes[0,1].set_title(
    "RMSE"
)

axes[0,1].grid(
    axis="y"
)

axes[1,0].bar(
    comparison["Optimizer"],
    comparison["Adjusted_R2"],
    color=bar_colors
)

axes[1,0].set_title(
    "Adjusted R²"
)

axes[1,0].grid(
    axis="y"
)

axes[1,1].bar(
    comparison["Optimizer"],
    comparison["Training_Time_Seconds"],
    color=bar_colors
)

axes[1,1].set_title(
    "Training Time"
)

axes[1,1].grid(
    axis="y"
)

plt.suptitle(
    "Optimizer Benchmark Comparison"
)

plt.tight_layout()

plt.savefig(
    comparison_dir /
    "optimizer_metrics_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Validation Loss Comparison
# ==========================================

plt.figure(
    figsize=(10,6)
)

for optimizer in optimizers:

    history = pd.read_csv(
        results_dir /
        f"{optimizer}_history.csv"
    )

    plt.plot(
        history["val_loss"],
        linewidth=2,
        label=optimizer.upper()
    )

plt.title(
    "Validation Loss Comparison"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Validation Loss"
)

plt.legend()

plt.grid(True)

plt.savefig(
    comparison_dir /
    "optimizer_learning_curve_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Winner History
# ==========================================

winner_history = pd.read_csv(
    results_dir /
    f"{best_optimizer.lower()}_history.csv"
)

# ==========================================
# Winner Learning Curve
# ==========================================

plt.figure(
    figsize=(10,6)
)

plt.plot(
    winner_history["loss"],
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    winner_history["val_loss"],
    linewidth=2,
    label="Validation Loss"
)

plt.title(
    f"{best_optimizer} Winner Learning Curve"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(True)

plt.savefig(
    comparison_dir /
    "winner_learning_curve.png",
    bbox_inches="tight"
)

plt.show()

plt.close()

# ==========================================
# Ranking
# ==========================================

ranking = comparison.sort_values(
    by="RMSE"
)

# ==========================================
# Summary Export
# ==========================================

with open(
    comparison_dir /
    "optimizer_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Optimizer Benchmarking Study\n"
    )

    f.write(
        "=" * 50
    )

    f.write(
        "\n\nResults\n\n"
    )

    f.write(
        comparison
        .round(4)
        .to_string(index=False)
    )

    f.write(
        f"\n\nWinning Optimizer: {best_optimizer}\n"
    )
    
    f.write(
        "\nGenerated Files\n"
    )

    f.write(
        "- optimizer_comparison.csv\n"
    )

    f.write(
        "- mae_comparison.png\n"
    )

    f.write(
        "- rmse_comparison.png\n"
    )

    f.write(
        "- adjusted_r2_comparison.png\n"
    )

    f.write(
        "- training_time_comparison.png\n"
    )

    f.write(
        "- optimizer_metrics_comparison.png\n"
    )

    f.write(
        "- optimizer_learning_curve_comparison.png\n"
    )

    f.write(
        "- winner_learning_curve.png\n"
    )

    f.write(
        "\nFinal Ranking (RMSE)\n"
    )

    for i, row in enumerate(
        ranking.itertuples(),
        start=1
    ):

        f.write(
            f"\n{i}. {row.Optimizer}"
        )

# ==========================================
# Complete
# ==========================================

print(
    "\nAnalysis 6 Comparison Complete"
)

print(
    f"\nWinning Optimizer: {best_optimizer}"
)