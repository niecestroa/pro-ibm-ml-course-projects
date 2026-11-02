"""
Author: Aaron Niecestro
Project: Deep Learning for Wine Quality (DLWQ)

Analysis: 07 - Regularization and Generalization
Method: Winner Visualization

Description:
Loads the regularization comparison results,
identifies the winning method, and generates
a standalone winner learning curve visualization.
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
    by="RMSE"
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
# Winner History
# ==========================================

winner_history = pd.read_csv(
    results_dir /
    f"{best_method.lower()}_history.csv"
)

# ==========================================
# File Name
# ==========================================

winner_file = (
    f"{best_method.lower()}_winner_learning_curve.png"
)

# ==========================================
# Learning Curve
# ==========================================

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
    winner_file,
    bbox_inches="tight"
)

plt.show()

plt.close()

# ==========================================
# Export Summary
# ==========================================

with open(
    comparison_dir /
    "winner_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"Winning Method: {best_method}\n"
    )

    f.write(
        f"RMSE: {winner['RMSE']:.4f}\n"
    )

    f.write(
        f"MAE: {winner['MAE']:.4f}\n"
    )

    f.write(
        f"Adjusted R2: {winner['Adjusted_R2']:.4f}\n"
    )

    f.write(
        f"Learning Curve: {winner_file}\n"
    )

print(
    "\nWinner Visualization Complete"
)

print(
    f"Generated: {winner_file}"
)
