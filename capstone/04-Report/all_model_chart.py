# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:56:43 2026

@author: aniec
"""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# Recommendation Model Performance Comparison
# ============================================================

comparison_df = pd.DataFrame({
    "Model": [
        "Baseline",
        "KNN\nPearson",
        "KNN\nCosine",
        "KNN\nMSD",
        "Linear\nRegression",
        "Ridge",
        "Lasso",
        "Elastic\nNet",
        "NMF",
        "ANN\n(Emb=4)",
        "ANN\n(Emb=8)",
        "ANN\n(Emb=16)",
        "ANN\n(Emb=32)",
        "ANN\n(Emb=64)"
    ],
    "RMSE": [
        0.8187,    # Baseline
        1.2321,    # KNN Pearson
        0.9048,    # KNN Cosine
        0.8951,    # KNN MSD
        0.8170,    # Linear Regression
        0.8170,    # Ridge
        0.8170,    # Lasso
        0.8170,    # Elastic Net
        3.2763,    # NMF
        0.4248,    # ANN Emb=4
        0.4219,    # ANN Emb=8
        0.4237,    # ANN Emb=16
        0.4271,    # ANN Emb=32
        0.4584     # ANN Emb=64
    ]
})

# ------------------------------------------------------------
# Colors by Model Family
# ------------------------------------------------------------

colors = [
    "#7f7f7f",  # Baseline

    "#ff7f0e",  # KNN Pearson
    "#ff7f0e",  # KNN Cosine
    "#ff7f0e",  # KNN MSD

    "#2ca02c",  # Linear
    "#2ca02c",  # Ridge
    "#2ca02c",  # Lasso
    "#2ca02c",  # Elastic

    "#d62728",  # NMF

    "#6baed6",  # ANN 4
    "#08306b",  # ANN 8 (BEST)
    "#6baed6",  # ANN 16
    "#6baed6",  # ANN 32
    "#6baed6"   # ANN 64
]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(14, 7))

ax = sns.barplot(
    data=comparison_df,
    x="Model",
    y="RMSE",
    palette=colors
)

# ------------------------------------------------------------
# RMSE Labels
# ------------------------------------------------------------

for i, value in enumerate(comparison_df["RMSE"]):

    ax.text(
        i,
        value + 0.04,
        f"{value:.3f}",
        ha="center",
        fontsize=9,
        fontweight="bold"
    )

# ------------------------------------------------------------
# Best Model Label
# ------------------------------------------------------------

best_idx = comparison_df["RMSE"].idxmin()

ax.text(
    best_idx,
    comparison_df.loc[best_idx, "RMSE"] + 0.28,
    "★ Best Model",
    ha="center",
    fontsize=11,
    fontweight="bold",
    color="#08306b"
)

# ------------------------------------------------------------
# Title and Axes
# ------------------------------------------------------------

plt.title(
    "Collaborative Filtering Model Performance Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.ylabel(
    "RMSE (Lower is Better)",
    fontsize=12
)

plt.xlabel("")

plt.xticks(rotation=30)

# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#7f7f7f", label="Baseline"),
    Patch(facecolor="#ff7f0e", label="KNN Models"),
    Patch(facecolor="#2ca02c", label="Regression Models"),
    Patch(facecolor="#d62728", label="NMF"),
    Patch(facecolor="#6baed6", label="Neural Networks")
]

plt.legend(
    handles=legend_elements,
    title="Model Family",
    loc="upper right"
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "recommendation_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()