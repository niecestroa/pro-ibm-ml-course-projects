"""
Author: Aaron Niecestro

Created:    October 19, 2026
Edited:     October 22, 2026
Status:     Completed

Description

"This started as a linkage-method comparison and finished by explaining why 
Ward was the winning linkage method and how its hierarchy evolved."

3. Linkage Methods (Hierarchical Clustering)

This analysis evaluates and compares the major hierarchical
linkage strategies used in Agglomerative Clustering:

    • Single Linkage
        - Preserves elongated cluster structures
        - Susceptible to chaining effects

    • Complete Linkage
        - Produces compact clusters
        - Resistant to chaining and noise

    • Average Linkage
        - Balanced compromise between Single and Complete

    • Ward Linkage
        - Minimizes within-cluster variance
        - Produces compact, balanced clusters
        - Requires Euclidean distance

Analysis includes:

    • Linkage method comparison
    • Silhouette Score evaluation
    • Davies-Bouldin Index evaluation
    • Cluster-balance analysis
    • Dendrogram comparison
    • Merge-distance analysis
    • Identification of practical clustering solutions

Ward Linkage Deep Dive:

    • Comparison of k=2, k=3, and k=4 solutions
    • Hierarchy evolution analysis
    • Cluster transition analysis
    • Cluster profile interpretation
    • Wine-type recovery analysis
    • Quality distribution analysis
    • Feature drivers of hierarchical structure

Objectives:

    1. Understand how linkage selection influences
       hierarchical clustering behavior.

    2. Demonstrate why internal validation metrics
       alone can be misleading.

    3. Explain why Ward Linkage produced the most
       meaningful clustering hierarchy for the
       Wine Quality dataset.

    4. Investigate how the Ward hierarchy evolves
       from broad wine populations into increasingly
       specialized wine subgroups.
"""

# =========================================================
# Load Results
# =========================================================

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(
    style="whitegrid",
    context="talk"
)

output_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\03-Linkage-Methods\03-linkage-results"
)

linkage_comparison = pd.read_csv(
    output_dir / "linkage_method_comparison.csv"
)

balanced_solutions = pd.read_csv(
    output_dir / "linkage_balanced_solutions.csv"
)

merge_distance = pd.read_csv(
    output_dir / "merge_distance_ranking.csv"
)

ward_models = pd.read_csv(
    output_dir / "ward_model_comparison.csv"
)

ward_k2_profile = pd.read_csv(
    output_dir / "ward_k2_profile.csv"
)

ward_k3_profile = pd.read_csv(
    output_dir / "ward_k3_profile.csv"
)

ward_k4_profile = pd.read_csv(
    output_dir / "ward_k4_profile.csv"
)

ward_feature_sep = pd.read_csv(
    output_dir / "ward_feature_separation.csv",
    index_col=0
)

ward_transition_k2_k3 = pd.read_csv(
    output_dir / "ward_k2_k3_transition.csv"
)

ward_transition_k3_k4 = pd.read_csv(
    output_dir / "ward_k3_k4_transition.csv"
)

# =========================================================
# Best Linkage Solutions
# =========================================================

print(
    "\n----- LINKAGE COMPARISON -----"
)

print(
    balanced_solutions
    .sort_values(
        by="silhouette_score",
        ascending=False
    )
)

# =========================================================
# Ward Comparison
# =========================================================

print(
    "\n----- WARD MODEL COMPARISON -----"
)

print(
    ward_models
)

# =========================================================
# Merge Distance Ranking
# =========================================================

linkage_colors = {
    "single": "#1f77b4",     # Blue
    "complete": "#2ca02c",   # Green
    "average": "#ff7f0e",    # Orange
    "ward": "#d62728"        # Red
}

plt.figure(figsize=(8,5))

ax = sns.barplot(
    data=merge_distance,
    x="linkage",
    y="max_merge_distance",
    hue="linkage",
    palette=linkage_colors,
    legend=False
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f"
    )

plt.title(
    "Maximum Merge Distance by Linkage Method",
    fontsize=18
)

plt.xlabel(
    "Linkage Method"
)

plt.ylabel(
    "Maximum Merge Distance"
)

plt.tight_layout()

plt.show()

# =========================================================
# Ward Metrics Comparison
# =========================================================

ward_colors = {
    2: "#3A86FF",   # Blue
    3: "#FFBE0B",   # Gold
    4: "#FB5607"    # Orange-Red
}

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14,5)
)

# Silhouette

sns.barplot(
    data=ward_models,
    x="clusters",
    y="silhouette_score",
    hue="clusters",
    palette=ward_colors,
    legend=False,
    ax=axes[0]
)

for container in axes[0].containers:
    axes[0].bar_label(
        container,
        fmt="%.3f"
    )

axes[0].set_title(
    "Ward Silhouette Score"
)

axes[0].set_xlabel(
    "Clusters (k)"
)

axes[0].set_ylabel(
    "Silhouette Score"
)

# Davies-Bouldin

sns.barplot(
    data=ward_models,
    x="clusters",
    y="davies_bouldin_index",
    hue="clusters",
    palette=ward_colors,
    legend=False,
    ax=axes[1]
)

for container in axes[1].containers:
    axes[1].bar_label(
        container,
        fmt="%.3f"
    )

axes[1].set_title(
    "Ward Davies-Bouldin Index"
)

axes[1].set_xlabel(
    "Clusters (k)"
)

axes[1].set_ylabel(
    "Davies-Bouldin Index"
)

plt.tight_layout()

plt.show()

# =========================================================
# Hierarchy Evolution
# =========================================================

print(
    "\n----- k=2 -> k=3 -----"
)

print(
    ward_transition_k2_k3
)

print(
    "\n----- k=3 -> k=4 -----"
)

print(
    ward_transition_k3_k4
)

# =========================================================
# Features Driving Ward Hierarchy
# =========================================================

feature_plot = (
    ward_feature_sep
    .sort_values(
        by="separation",
        ascending=False
    )
    .reset_index()
)

feature_plot.rename(
    columns={"index": "feature"},
    inplace=True
)

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=feature_plot,
    x="separation",
    y="feature",
    color="steelblue"
)

for i, v in enumerate(
    feature_plot["separation"]
):
    ax.text(
        v + 0.02,
        i,
        f"{v:.2f}",
        va="center"
    )

plt.title(
    "Features Driving Ward Hierarchy"
)

plt.xlabel(
    "Standardized Separation"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.show()

# =========================================================
# Ward Profiles
# =========================================================

print(
    "\n----- WARD K=2 PROFILE -----"
)

print(
    ward_k2_profile
)

print(
    "\n----- WARD K=3 PROFILE -----"
)

print(
    ward_k3_profile
)

print(
    "\n----- WARD K=4 PROFILE -----"
)

print(
    ward_k4_profile
)

# =========================================================
# Final Results Summary
# =========================================================

print(
    """
FINAL RESULTS

Best Linkage Method:
    Ward

Best Balanced Solution:
    Ward k=2

Hierarchy Evolution:

    k=2
        White Wines
        Red Wines

    k=3
        Dry White Wines
        Sweet White Wines
        Red Wines

    k=4
        Dry White Wines
        Sweet White Wines
        Structured Red Wines
        High Volatile-Acidity Red Wines

Key Finding:

    Internal clustering metrics alone favored
    Single, Complete, and Average Linkage.

    Cluster balance, dendrogram structure,
    and hierarchy evolution demonstrated that
    Ward Linkage produced the most meaningful
    segmentation of the Wine Quality dataset.
"""
)
