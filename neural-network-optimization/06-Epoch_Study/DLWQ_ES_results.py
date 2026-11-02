"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 2, 2026
Last Edit:  November 2, 2026
Progress:   Complete

Analysis:   05 - Epoch Optimization
Method:     Comparison

Description:
Compares all epoch configurations and identifies the
optimal training duration for wine quality prediction.
"""

# ==========================================
# Imports
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# ------------------------------------------
# Reproducibility
# ------------------------------------------

import os
import numpy as np
import random
import tensorflow as tf

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ==========================================
# Directories
# ==========================================

results_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\06-Epoch_Study\tables_figures"
)

comparison_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\06-Epoch_Study\comparison"
)

comparison_dir.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# Epoch Values
# ==========================================

epochs = [
    25,
    50,
    75,
    100,
    125,
    150,
    175,
    200
]

# ==========================================
# Load Results
# ==========================================

all_results = []

for epoch in epochs:

    df = pd.read_csv(
        results_dir /
        f"{epoch}_epochs_results.csv"
    )

    all_results.append(df)

comparison = pd.concat(
    all_results,
    ignore_index=True
)

comparison = comparison.sort_values(
    by="Epochs"
)

# ==========================================
# Export Comparison
# ==========================================

comparison.to_csv(
    comparison_dir /
    "epoch_study_comparison.csv",
    index=False
)

print("\n----- EPOCH COMPARISON -----")
print(comparison.round(4))

# ==========================================
# Best Epoch
# ==========================================

winner = comparison.loc[
    comparison["RMSE"].idxmin()
]

best_epochs = int(
    winner["Epochs"]
)

print(
    f"\nBest Epochs: {best_epochs}"
)

# ==========================================
# MAE Comparison
# ==========================================

plt.figure(figsize=(10,6))

plt.plot(
    comparison["Epochs"],
    comparison["MAE"],
    marker="o",
    linewidth=2
)

plt.title(
    "MAE by Epoch Count"
)

plt.xlabel("Epochs")
plt.ylabel("MAE")

plt.grid(True)

plt.savefig(
    comparison_dir /
    "mae_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# RMSE Comparison
# ==========================================

plt.figure(figsize=(10,6))

plt.plot(
    comparison["Epochs"],
    comparison["RMSE"],
    marker="o",
    linewidth=2
)

plt.title(
    "RMSE by Epoch Count"
)

plt.xlabel("Epochs")
plt.ylabel("RMSE")

plt.grid(True)

plt.savefig(
    comparison_dir /
    "rmse_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Adjusted R² Comparison
# ==========================================

plt.figure(figsize=(10,6))

plt.plot(
    comparison["Epochs"],
    comparison["Adjusted_R2"],
    marker="o",
    linewidth=2
)

plt.title(
    "Adjusted R² by Epoch Count"
)

plt.xlabel("Epochs")
plt.ylabel("Adjusted R²")

plt.grid(True)

plt.savefig(
    comparison_dir /
    "adjusted_r2_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Training Time Comparison
# ==========================================

plt.figure(figsize=(10,6))

plt.plot(
    comparison["Epochs"],
    comparison["Training_Time_Seconds"],
    marker="o",
    linewidth=2
)

plt.title(
    "Training Time by Epoch Count"
)

plt.xlabel("Epochs")
plt.ylabel("Training Time (Seconds)")

plt.grid(True)

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

axes[0,0].plot(
    comparison["Epochs"],
    comparison["MAE"],
    marker="o"
)

axes[0,0].set_title("MAE")
axes[0,0].grid(True)

axes[0,1].plot(
    comparison["Epochs"],
    comparison["RMSE"],
    marker="o"
)

axes[0,1].set_title("RMSE")
axes[0,1].grid(True)

axes[1,0].plot(
    comparison["Epochs"],
    comparison["Adjusted_R2"],
    marker="o"
)

axes[1,0].set_title("Adjusted R²")
axes[1,0].grid(True)

axes[1,1].plot(
    comparison["Epochs"],
    comparison["Training_Time_Seconds"],
    marker="o"
)

axes[1,1].set_title("Training Time")
axes[1,1].grid(True)

plt.suptitle(
    "Epoch Optimization Comparison"
)

plt.tight_layout()

plt.savefig(
    comparison_dir /
    "epoch_metrics_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Learning Curve Comparison
# ==========================================

plt.figure(figsize=(12,8))

for epoch in epochs:

    history = pd.read_csv(
        results_dir /
        f"{epoch}_epochs_history.csv"
    )

    plt.plot(
        history["val_loss"],
        label=f"{epoch} Epochs"
    )

plt.title(
    "Validation Loss Comparison"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    comparison_dir /
    "epoch_learning_curve_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Winner History
# ==========================================

winner_history = pd.read_csv(
    results_dir /
    f"{best_epochs}_epochs_history.csv"
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
    f"{best_epochs} Epochs Winner Learning Curve"
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
# Epoch Ranking
# ==========================================

ranking = comparison.sort_values(
    "RMSE"
)

# ==========================================
# Export Summary
# ==========================================

with open(
    comparison_dir /
    "epoch_study_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Epoch Optimization Study\n"
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
        f"\n\nWinning Epoch Count: {best_epochs}\n"
    )
    
    f.write(
        f"\nMAE: {winner['MAE']:.4f}"
    )

    f.write(
        f"\nRMSE: {winner['RMSE']:.4f}"
    )

    f.write(
        f"\nAdjusted R²: {winner['Adjusted_R2']:.4f}"
    )

    f.write(
        f"\nTraining Time: "
        f"{winner['Training_Time_Seconds']:.2f} seconds"
    )
    
    f.write(
        "\nGenerated Files\n"
    )

    f.write(
        "- epoch_study_comparison.csv\n"
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
        "- epoch_metrics_comparison.png\n"
    )

    f.write(
        "- epoch_learning_curve_comparison.png\n"
    )

    f.write(
        "- winner_learning_curve.png\n"
    )

    f.write(
        "\nFinal Ranking (RMSE)\n"
    )
    
    f.write(
        "\n\nSelected Configuration Moving Forward\n"
    )

    f.write(
        "\n- Scaling Method: StandardScaler"
    )

    f.write(
        "\n- Activation Function: Sigmoid"
    )

    f.write(
        "\n- Architecture: 64 -> 32 -> 16"
    )

    f.write(
        "\n- Optimizer: Adam"
    )

    f.write(
        "\n- Epochs: 125"
    )

    for i, row in enumerate(
        ranking.itertuples(),
        start=1
    ):
        f.write(
            f"\n{i}. {row.Epochs} Epochs"
        )
        

# ==========================================
# Complete
# ==========================================

print(
    "\nAnalysis 5 Comparison Complete"
)

print(
    "\n----- WINNING PERFORMANCE -----"
)

print(
    f"\nWinning Epoch Count: {best_epochs}"
)

print(
    f"MAE: {winner['MAE']:.4f}"
)

print(
    f"RMSE: {winner['RMSE']:.4f}"
)

print(
    f"Adjusted R²: "
    f"{winner['Adjusted_R2']:.4f}"
)

print(
    f"Training Time: "
    f"{winner['Training_Time_Seconds']:.2f} seconds"
)

print(
    f"Final Training Loss: "
    f"{winner['Final_Training_Loss']:.4f}"
)

print(
    f"Final Validation Loss: "
    f"{winner['Final_Validation_Loss']:.4f}"
)
