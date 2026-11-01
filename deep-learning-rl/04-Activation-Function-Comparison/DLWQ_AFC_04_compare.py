"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Analysis:   03 - Activation Function Comparison
Method:     Final Comparison

Description:
Compares ReLU, Sigmoid, and Tanh activation functions and
identifies the best activation function for wine quality prediction.
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
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\04-Activation-Function-Comparison\tables_figures"
)

comparison_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\04-Activation-Function-Comparison\comparison"
)

comparison_dir.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# Colors
# ==========================================

colors = {
    "ReLU": "#2ca02c",      # Green
    "Tanh": "#1f77b4",      # Blue
    "Sigmoid": "#d62728"    # Red
}

# ==========================================
# Load Results
# ==========================================

relu = pd.read_csv(
    results_dir / "relu_results.csv"
)

sigmoid = pd.read_csv(
    results_dir / "sigmoid_results.csv"
)

tanh = pd.read_csv(
    results_dir / "tanh_results.csv"
)

# ==========================================
# Load Training Histories
# ==========================================

relu_history = pd.read_csv(
    results_dir / "relu_history.csv"
)

sigmoid_history = pd.read_csv(
    results_dir / "sigmoid_history.csv"
)

tanh_history = pd.read_csv(
    results_dir / "tanh_history.csv"
)

# ==========================================
# Combine Results
# ==========================================

comparison = pd.concat(
    [relu, sigmoid, tanh],
    ignore_index=True
)

comparison = comparison.sort_values(
    by="RMSE"
)

print("\n----- ACTIVATION FUNCTION COMPARISON -----")
print(comparison.round(4))

# ==========================================
# Export Combined Results
# ==========================================

comparison.to_csv(
    comparison_dir / "activation_function_comparison.csv",
    index=False
)

# ==========================================
# Determine Winner
# ==========================================

winner = comparison.iloc[0]["Activation_Function"]

print("\n----- BEST ACTIVATION FUNCTION -----")
print(winner)

# ==========================================
# Plot Colors
# ==========================================

bar_colors = [
    colors[a]
    for a in comparison["Activation_Function"]
]

# ==========================================
# MAE Comparison
# ==========================================

plt.figure(figsize=(8,6))

plt.bar(
    comparison["Activation_Function"],
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
    comparison["Activation_Function"],
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
    comparison["Activation_Function"],
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
# Combined Metrics Plot
# ==========================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18,6)
)

axes[0].bar(
    comparison["Activation_Function"],
    comparison["MAE"],
    color=bar_colors
)

axes[0].set_title("MAE")
axes[0].set_ylabel("Value")
axes[0].grid(axis="y")

axes[1].bar(
    comparison["Activation_Function"],
    comparison["RMSE"],
    color=bar_colors
)

axes[1].set_title("RMSE")
axes[1].grid(axis="y")

axes[2].bar(
    comparison["Activation_Function"],
    comparison["Adjusted_R2"],
    color=bar_colors
)

axes[2].set_title("Adjusted R²")
axes[2].grid(axis="y")

plt.suptitle(
    "Activation Function Comparison"
)

plt.tight_layout()

plt.savefig(
    comparison_dir /
    "activation_function_metrics_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Training Loss Comparison
# ==========================================

plt.figure(figsize=(12,6))

plt.plot(
    relu_history["loss"],
    color=colors["ReLU"],
    linewidth=2,
    label="ReLU"
)

plt.plot(
    tanh_history["loss"],
    color=colors["Tanh"],
    linewidth=2,
    label="Tanh"
)

plt.plot(
    sigmoid_history["loss"],
    color=colors["Sigmoid"],
    linewidth=2,
    label="Sigmoid"
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
    relu_history["val_loss"],
    color=colors["ReLU"],
    linewidth=2,
    label="ReLU"
)

plt.plot(
    tanh_history["val_loss"],
    color=colors["Tanh"],
    linewidth=2,
    label="Tanh"
)

plt.plot(
    sigmoid_history["val_loss"],
    color=colors["Sigmoid"],
    linewidth=2,
    label="Sigmoid"
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
# Combined Learning Curve Figure
# ==========================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,6)
)

# Training Loss

axes[0].plot(
    relu_history["loss"],
    color=colors["ReLU"],
    linewidth=2,
    label="ReLU"
)

axes[0].plot(
    tanh_history["loss"],
    color=colors["Tanh"],
    linewidth=2,
    label="Tanh"
)

axes[0].plot(
    sigmoid_history["loss"],
    color=colors["Sigmoid"],
    linewidth=2,
    label="Sigmoid"
)

axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# Validation Loss

axes[1].plot(
    relu_history["val_loss"],
    color=colors["ReLU"],
    linewidth=2,
    label="ReLU"
)

axes[1].plot(
    tanh_history["val_loss"],
    color=colors["Tanh"],
    linewidth=2,
    label="Tanh"
)

axes[1].plot(
    sigmoid_history["val_loss"],
    color=colors["Sigmoid"],
    linewidth=2,
    label="Sigmoid"
)

axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.suptitle(
    "Activation Function Learning Curve Comparison"
)

plt.tight_layout()

plt.savefig(
    comparison_dir /
    "activation_learning_curve_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Export Summary
# ==========================================

with open(
    comparison_dir /
    "activation_function_comparison_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Activation Function Comparison\n"
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
        f"\n\nBest Activation Function:\n{winner}"
    )

    f.write(
        "\n\nGenerated Files:\n"
    )

    f.write(
        "- activation_function_comparison.csv\n"
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
        "- activation_function_metrics_comparison.png\n"
    )

    f.write(
        "- training_loss_comparison.png\n"
    )

    f.write(
        "- validation_loss_comparison.png\n"
    )

    f.write(
        "- activation_learning_curve_comparison.png\n"
    )

# ==========================================
# Complete
# ==========================================

print("\nAnalysis 3 Comparison Complete")
print(f"\nWinning Activation Function: {winner}")