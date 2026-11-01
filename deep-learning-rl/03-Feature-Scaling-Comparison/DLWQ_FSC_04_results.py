"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Analysis:   02 - Feature Scaling Comparison
Method:     Final Comparison

Description:
Loads results from No Scaling, StandardScaler, and MinMaxScaler,
creates comparison tables and comparison visualizations, and
identifies the best scaling method.
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
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\03-Feature-Scaling-Comparison\tables_figures"
)

comparison_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\03-Feature-Scaling-Comparison\comparison"
)

comparison_dir.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# Load Results
# ==========================================

ns = pd.read_csv(
    results_dir / "no_scaling_results.csv"
)

ss = pd.read_csv(
    results_dir / "standardscaler_results.csv"
)

mms = pd.read_csv(
    results_dir / "minmaxscaler_results.csv"
)

# ==========================================
# Load Training Histories
# ==========================================

ns_history = pd.read_csv(
    results_dir / "no_scaling_history.csv"
)

ss_history = pd.read_csv(
    results_dir / "standardscaler_history.csv"
)

mms_history = pd.read_csv(
    results_dir / "minmaxscaler_history.csv"
)

# ==========================================
# Combine Results
# ==========================================

comparison = pd.concat(
    [ns, ss, mms],
    ignore_index=True
)

comparison = comparison.sort_values(
    by="RMSE"
)

print("\n----- FEATURE SCALING COMPARISON -----")
print(comparison.round(4))

# ==========================================
# Export Combined Results
# ==========================================

comparison.to_csv(
    comparison_dir / "feature_scaling_comparison.csv",
    index=False
)

# ==========================================
# Best Method
# ==========================================

best_method = comparison.iloc[0]["Scaling_Method"]

print("\n----- BEST SCALING METHOD -----")
print(best_method)

# ==========================================
# Colors
# ==========================================

colors = {
    "No Scaling": "#d62728",       # Red
    "StandardScaler": "#1f77b4",   # Blue
    "MinMaxScaler": "#2ca02c"      # Green
}

bar_colors = [
    colors[m]
    for m in comparison["Scaling_Method"]
]

plt.figure(figsize=(8,6))

plt.bar(
    comparison["Scaling_Method"],
    comparison["MAE"],
    color=bar_colors
)

plt.title("MAE Comparison")
plt.ylabel("MAE")
plt.grid(axis="y")

plt.savefig(
    comparison_dir / "mae_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# RMSE Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison["Scaling_Method"],
    comparison["RMSE"],
    color=bar_colors
)

plt.title("RMSE Comparison")
plt.ylabel("RMSE")
plt.grid(axis="y")

plt.savefig(
    comparison_dir / "rmse_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Adjusted R² Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison["Scaling_Method"],
    comparison["Adjusted_R2"],
    color=bar_colors
)

plt.title("Adjusted R² Comparison")
plt.ylabel("Adjusted R²")
plt.grid(axis="y")

plt.savefig(
    comparison_dir / "adjusted_r2_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Combined Metrics Figure
# ==========================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18,6)
)

axes[0].bar(
    comparison["Scaling_Method"],
    comparison["MAE"],
    color=bar_colors
)
axes[0].set_title("MAE")

axes[1].bar(
    comparison["Scaling_Method"],
    comparison["RMSE"],
    color=bar_colors
)
axes[1].set_title("RMSE")

axes[2].bar(
    comparison["Scaling_Method"],
    comparison["Adjusted_R2"],
    color=bar_colors
)
axes[2].set_title("Adjusted R²")

plt.suptitle(
    "Feature Scaling Comparison"
)

plt.tight_layout()

plt.savefig(
    comparison_dir / "feature_scaling_metrics_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Training Loss Comparison
# ==========================================

plt.figure(figsize=(12,6))

plt.plot(
    ns_history["loss"],
    color=colors["No Scaling"],
    linewidth=2,
    label="No Scaling"
)

plt.plot(
    ss_history["loss"],
    color=colors["StandardScaler"],
    linewidth=2,
    label="StandardScaler"
)

plt.plot(
    mms_history["loss"],
    color=colors["MinMaxScaler"],
    linewidth=2,
    label="MinMaxScaler"
)

plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    comparison_dir /
    "training_loss_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Validation Loss Comparison
# ==========================================

plt.figure(figsize=(12,6))

plt.plot(
    ns_history["val_loss"],
    color=colors["No Scaling"],
    linewidth=2,
    label="No Scaling"
)

plt.plot(
    ss_history["val_loss"],
    color=colors["StandardScaler"],
    linewidth=2,
    label="StandardScaler"
)

plt.plot(
    mms_history["val_loss"],
    color=colors["MinMaxScaler"],
    linewidth=2,
    label="MinMaxScaler"
)

plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    comparison_dir /
    "validation_loss_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Feature Scaling Learning Curve Comparison
# ==========================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,6)
)

# Training Loss

axes[0].plot(
    ns_history["loss"],
    color=colors["No Scaling"],
    linewidth=2,
    label="No Scaling"
)

axes[0].plot(
    ss_history["loss"],
    color=colors["StandardScaler"],
    linewidth=2,
    label="StandardScaler"
)

axes[0].plot(
    mms_history["loss"],
    color=colors["MinMaxScaler"],
    linewidth=2,
    label="MinMaxScaler"
)

axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# Validation Loss

axes[1].plot(
    ns_history["val_loss"],
    color=colors["No Scaling"],
    linewidth=2,
    label="No Scaling"
)

axes[1].plot(
    ss_history["val_loss"],
    color=colors["StandardScaler"],
    linewidth=2,
    label="StandardScaler"
)

axes[1].plot(
    mms_history["val_loss"],
    color=colors["MinMaxScaler"],
    linewidth=2,
    label="MinMaxScaler"
)

axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.suptitle(
    "Feature Scaling Learning Curve Comparison"
)

plt.tight_layout()

plt.savefig(
    comparison_dir /
    "feature_scaling_learning_curve_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Export Summary
# ==========================================

with open(
    comparison_dir / "feature_scaling_comparison_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("Feature Scaling Comparison\n")
    f.write("=" * 50)

    f.write("\n\nResults\n\n")

    f.write(
        comparison.round(4).to_string(index=False)
    )

    f.write(
        f"\n\nBest Scaling Method:\n{best_method}"
    )

    f.write(
        "\n\nGenerated Files:\n"
    )

    f.write(
        "- feature_scaling_comparison.csv\n"
    )

    f.write(
        "- training_loss_comparison.png\n"
    )

    f.write(
        "- validation_loss_comparison.png\n"
    )

    f.write(
        "- feature_scaling_learning_curve_comparison.png\n"
    )

print("\nAnalysis 2 Comparison Complete")
