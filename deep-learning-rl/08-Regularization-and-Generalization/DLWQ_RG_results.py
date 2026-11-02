"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 2, 2026
Last Edit:  November 2, 2026
Progress:   Complete

Analysis:   07 - Regularization and Generalization
Method: Comparison
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
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\08-Regularization-and-Generalization\tables_figures"
)

comparison_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\08-Regularization-and-Generalization\comparison"
)

comparison_dir.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# Methods
# ==========================================

methods = [
    "baseline",
    "l1",
    "l2",
    "dropout",
    "earlystopping"
]

# ==========================================
# Load Results
# ==========================================

all_results = []

for method in methods:

    df = pd.read_csv(
        results_dir /
        f"{method}_results.csv"
    )

    all_results.append(df)

comparison = pd.concat(
    all_results,
    ignore_index=True
)

comparison = comparison.sort_values(
    "RMSE"
)

comparison.to_csv(
    comparison_dir /
    "regularization_comparison.csv",
    index=False
)

print(
    "\n----- REGULARIZATION COMPARISON -----"
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

best_method = winner["Method"]

print(
    f"\nWinning Method: {best_method}"
)

# ==========================================
# MAE Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison["Method"],
    comparison["MAE"]
)

plt.title(
    "MAE Comparison"
)

plt.ylabel("MAE")

plt.grid(axis="y")

plt.savefig(
    comparison_dir /
    "mae_comparison.png",
    bbox_inches="tight"
)

plt.close()

# ==========================================
# RMSE Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison["Method"],
    comparison["RMSE"]
)

plt.title(
    "RMSE Comparison"
)

plt.ylabel("RMSE")

plt.grid(axis="y")

plt.savefig(
    comparison_dir /
    "rmse_comparison.png",
    bbox_inches="tight"
)

plt.close()

# ==========================================
# Adjusted R² Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison["Method"],
    comparison["Adjusted_R2"]
)

plt.title(
    "Adjusted R² Comparison"
)

plt.ylabel("Adjusted R²")

plt.grid(axis="y")

plt.savefig(
    comparison_dir /
    "adjusted_r2_comparison.png",
    bbox_inches="tight"
)

plt.close()

# ==========================================
# Training Time Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison["Method"],
    comparison["Training_Time_Seconds"]
)

plt.title(
    "Training Time Comparison"
)

plt.ylabel(
    "Seconds"
)

plt.grid(axis="y")

plt.savefig(
    comparison_dir /
    "training_time_comparison.png",
    bbox_inches="tight"
)

plt.close()

# ==========================================
# Generalization Gap Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison["Method"],
    comparison["Generalization_Gap"]
)

plt.title(
    "Generalization Gap Comparison"
)

plt.ylabel(
    "Validation Loss - Training Loss"
)

plt.grid(axis="y")

plt.savefig(
    comparison_dir /
    "generalization_gap_comparison.png",
    bbox_inches="tight"
)

plt.close()

# ==========================================
# Combined Metrics
# ==========================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14,10)
)

axes[0,0].bar(
    comparison["Method"],
    comparison["MAE"]
)

axes[0,0].set_title(
    "MAE"
)

axes[0,1].bar(
    comparison["Method"],
    comparison["RMSE"]
)

axes[0,1].set_title(
    "RMSE"
)

axes[1,0].bar(
    comparison["Method"],
    comparison["Adjusted_R2"]
)

axes[1,0].set_title(
    "Adjusted R²"
)

axes[1,1].bar(
    comparison["Method"],
    comparison["Generalization_Gap"]
)

axes[1,1].set_title(
    "Generalization Gap"
)

plt.suptitle(
    "Regularization Comparison"
)

plt.tight_layout()

plt.savefig(
    comparison_dir /
    "regularization_metrics_comparison.png",
    bbox_inches="tight"
)

plt.close()

# ==========================================
# All Validation Curves
# ==========================================

plt.figure(
    figsize=(12,7)
)

for method in methods:

    history = pd.read_csv(
        results_dir /
        f"{method}_history.csv"
    )

    plt.plot(
        history["val_loss"],
        linewidth=2,
        label=method.title()
    )

plt.title(
    "Regularization Comparison: Validation Loss"
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
    "regularization_learning_curve_comparison.png",
    bbox_inches="tight"
)

plt.show()

plt.close()

# ==========================================
# Winner Learning Curve
# ==========================================

winner = comparison.iloc[0]

best_method = winner["Method"]

winner_history = pd.read_csv(
    results_dir /
    f"{best_method.lower()}_history.csv"
)

plt.figure(
    figsize=(12,7)
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
    f"{best_method} Winner Learning Curve"
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
# Summary Export
# ==========================================

with open(
    comparison_dir /
    "regularization_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        comparison
        .round(4)
        .to_string(index=False)
    )

    f.write(
        f"\n\nWinning Method: {best_method}"
    )
    
    f.write(
        "\n\nGenerated Files:\n"
    )

    f.write(
        "- regularization_comparison.csv\n"
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
        "- generalization_gap_comparison.png\n"
    )

    f.write(
        "- regularization_metrics_comparison.png\n"
    )

    f.write(
        "- regularization_learning_curve_comparison.png\n"
    )

    f.write(
        "- winner_learning_curve.png\n"
    )

print(
    "\nAnalysis 7 Comparison Complete"
)

print(
    f"\nWinning Method: {best_method}"
)