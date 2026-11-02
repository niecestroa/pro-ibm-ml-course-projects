"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Analysis:   04 - Architecture Comparison
Method:     Final Comparison

Description:
Compares Model A, Model B, Model C, and Model D neural
network architectures and identifies the best architecture
for wine quality prediction.
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
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\05-Architecture Comparison\tables_figures"
)

comparison_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\05-Architecture Comparison\comparison"
)

comparison_dir.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# Colors
# ==========================================

colors = {
    "Model A": "#d62728",
    "Model B": "#ff7f0e",
    "Model C": "#2ca02c",
    "Model D": "#1f77b4"
}

# ==========================================
# Load Results
# ==========================================

model_a = pd.read_csv(
    results_dir / "model_a_results.csv"
)

model_b = pd.read_csv(
    results_dir / "model_b_results.csv"
)

model_c = pd.read_csv(
    results_dir / "model_c_results.csv"
)

model_d = pd.read_csv(
    results_dir / "model_d_results.csv"
)

# ==========================================
# Load Histories
# ==========================================

model_a_history = pd.read_csv(
    results_dir / "model_a_history.csv"
)

model_b_history = pd.read_csv(
    results_dir / "model_b_history.csv"
)

model_c_history = pd.read_csv(
    results_dir / "model_c_history.csv"
)

model_d_history = pd.read_csv(
    results_dir / "model_d_history.csv"
)

# ==========================================
# RMSE Comparison (Best Model First)
# ==========================================

# ==========================================
# Combine Results
# ==========================================

comparison = pd.concat(
    [model_a, model_b, model_c, model_d],
    ignore_index=True
)

comparison = comparison.sort_values(
    by="RMSE"
)

print("\n----- ARCHITECTURE COMPARISON -----")
print(comparison.round(4))

# ==========================================
# Export Results
# ==========================================

comparison.to_csv(
    comparison_dir / "architecture_comparison.csv",
    index=False
)

# ==========================================
# Determine Winner
# ==========================================

winner = comparison.iloc[0]["Architecture"]

print("\n----- BEST ARCHITECTURE -----")
print(winner)

# ==========================================
# Plot Colors
# ==========================================

bar_colors = [
    colors[a]
    for a in comparison["Architecture"]
]

# ==========================================
# MAE Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison["Architecture"],
    comparison["MAE"],
    color=bar_colors
)

plt.title("MAE Comparison")
plt.ylabel("MAE")

plt.grid(axis="y")

plt.savefig(
    comparison_dir /
    "mae_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# RMSE Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison["Architecture"],
    comparison["RMSE"],
    color=bar_colors
)

plt.title("RMSE Comparison")
plt.ylabel("RMSE")

plt.grid(axis="y")

plt.savefig(
    comparison_dir /
    "rmse_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Adjusted R² Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison["Architecture"],
    comparison["Adjusted_R2"],
    color=bar_colors
)

plt.title("Adjusted R² Comparison")
plt.ylabel("Adjusted R²")

plt.grid(axis="y")

plt.savefig(
    comparison_dir /
    "adjusted_r2_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Training Time Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison["Architecture"],
    comparison["Training_Time_Seconds"],
    color=bar_colors
)

plt.title("Training Time Comparison")
plt.ylabel("Training Time (Seconds)")

plt.grid(axis="y")

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
    1,
    4,
    figsize=(24,6)
)

axes[0].bar(
    comparison["Architecture"],
    comparison["MAE"],
    color=bar_colors
)

axes[0].set_title("MAE")
axes[0].grid(axis="y")

axes[1].bar(
    comparison["Architecture"],
    comparison["RMSE"],
    color=bar_colors
)

axes[1].set_title("RMSE")
axes[1].grid(axis="y")

axes[2].bar(
    comparison["Architecture"],
    comparison["Adjusted_R2"],
    color=bar_colors
)

axes[2].set_title("Adjusted R²")
axes[2].grid(axis="y")

axes[3].bar(
    comparison["Architecture"],
    comparison["Training_Time_Seconds"],
    color=bar_colors
)

axes[3].set_title("Training Time")
axes[3].grid(axis="y")

plt.suptitle(
    "Architecture Comparison"
)

plt.tight_layout()

plt.savefig(
    comparison_dir /
    "architecture_metrics_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Learning Curve Comparison
# ==========================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,6)
)

# Training Loss

axes[0].plot(
    model_a_history["loss"],
    color=colors["Model A"],
    label="Model A",
    linewidth=2
)

axes[0].plot(
    model_b_history["loss"],
    color=colors["Model B"],
    label="Model B",
    linewidth=2
)

axes[0].plot(
    model_c_history["loss"],
    color=colors["Model C"],
    label="Model C",
    linewidth=2
)

axes[0].plot(
    model_d_history["loss"],
    color=colors["Model D"],
    label="Model D",
    linewidth=2
)

axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# Validation Loss

axes[1].plot(
    model_a_history["val_loss"],
    color=colors["Model A"],
    label="Model A",
    linewidth=2
)

axes[1].plot(
    model_b_history["val_loss"],
    color=colors["Model B"],
    label="Model B",
    linewidth=2
)

axes[1].plot(
    model_c_history["val_loss"],
    color=colors["Model C"],
    label="Model C",
    linewidth=2
)

axes[1].plot(
    model_d_history["val_loss"],
    color=colors["Model D"],
    label="Model D",
    linewidth=2
)

axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.suptitle(
    "Architecture Learning Curve Comparison"
)

plt.tight_layout()

plt.savefig(
    comparison_dir /
    "architecture_learning_curve_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Ordered Plots
# ==========================================

# ==========================================
# Plot Order
# ==========================================

plot_order = [
    "Model A",
    "Model B",
    "Model C",
    "Model D"
]

comparison_plot = (
    comparison
    .set_index("Architecture")
    .loc[plot_order]
    .reset_index()
)

# ==========================================
# Plot Colors
# ==========================================

bar_colors = [
    colors[a]
    for a in comparison_plot["Architecture"]
]

# ==========================================
# MAE Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison_plot["Architecture"],
    comparison_plot["MAE"],
    color=bar_colors
)

plt.title("MAE Comparison")
plt.ylabel("MAE")
plt.grid(axis="y")

plt.show()

# ==========================================
# RMSE Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison_plot["Architecture"],
    comparison_plot["RMSE"],
    color=bar_colors
)

plt.title("RMSE Comparison")
plt.ylabel("RMSE")
plt.grid(axis="y")

plt.show()

# ==========================================
# Adjusted R² Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison_plot["Architecture"],
    comparison_plot["Adjusted_R2"],
    color=bar_colors
)

plt.title("Adjusted R² Comparison")
plt.ylabel("Adjusted R²")
plt.grid(axis="y")

plt.show()

# ==========================================
# Training Time Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison_plot["Architecture"],
    comparison_plot["Training_Time_Seconds"],
    color=bar_colors
)

plt.title("Training Time Comparison")
plt.ylabel("Training Time (Seconds)")

plt.grid(axis="y")

plt.show()

# ==========================================
# Combined Metrics Figure
# ==========================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(24,6)
)

axes[0].bar(
    comparison_plot["Architecture"],
    comparison_plot["MAE"],
    color=bar_colors
)

axes[0].set_title("MAE")
axes[0].grid(axis="y")

axes[1].bar(
    comparison_plot["Architecture"],
    comparison_plot["RMSE"],
    color=bar_colors
)

axes[1].set_title("RMSE")
axes[1].grid(axis="y")

axes[2].bar(
    comparison_plot["Architecture"],
    comparison_plot["Adjusted_R2"],
    color=bar_colors
)

axes[2].set_title("Adjusted R²")
axes[2].grid(axis="y")

axes[3].bar(
    comparison_plot["Architecture"],
    comparison_plot["Training_Time_Seconds"],
    color=bar_colors
)

axes[3].set_title("Training Time")
axes[3].grid(axis="y")

plt.suptitle(
    "Architecture Comparison"
)

plt.tight_layout()

plt.show()

# ==========================================
# Learning Curve Comparison
# ==========================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,6)
)

# ------------------------------------------
# Training Loss
# ------------------------------------------

axes[0].plot(
    model_a_history["loss"],
    color=colors["Model A"],
    label="Model A",
    linewidth=2
)

axes[0].plot(
    model_b_history["loss"],
    color=colors["Model B"],
    label="Model B",
    linewidth=2
)

axes[0].plot(
    model_c_history["loss"],
    color=colors["Model C"],
    label="Model C",
    linewidth=2
)

axes[0].plot(
    model_d_history["loss"],
    color=colors["Model D"],
    label="Model D",
    linewidth=2
)

axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# ------------------------------------------
# Validation Loss
# ------------------------------------------

axes[1].plot(
    model_a_history["val_loss"],
    color=colors["Model A"],
    label="Model A",
    linewidth=2
)

axes[1].plot(
    model_b_history["val_loss"],
    color=colors["Model B"],
    label="Model B",
    linewidth=2
)

axes[1].plot(
    model_c_history["val_loss"],
    color=colors["Model C"],
    label="Model C",
    linewidth=2
)

axes[1].plot(
    model_d_history["val_loss"],
    color=colors["Model D"],
    label="Model D",
    linewidth=2
)

axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.suptitle(
    "Architecture Learning Curve Comparison"
)

plt.tight_layout()

plt.savefig(
    comparison_dir /
    "architecture_learning_curve_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Export Summary
# ==========================================

with open(
    comparison_dir /
    "architecture_comparison_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Architecture Comparison\n"
    )

    f.write(
        "=" * 50
    )

    f.write(
        "\n\nResults\n\n"
    )

    f.write(
        comparison.round(4).to_string(index=False)
    )

    f.write(
        f"\n\nBest Architecture:\n{winner}"
    )

# ==========================================
# Complete
# ==========================================

print("\nAnalysis 4 Comparison Complete")
print(f"\nWinning Architecture: {winner}")