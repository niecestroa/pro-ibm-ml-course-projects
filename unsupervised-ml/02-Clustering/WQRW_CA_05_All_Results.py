# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October 21, 2026
Status:     Completed

Description

1. Clustering Algorithms Comparison Table and Plots

    K‑Means (Euclidean; spherical clusters)
    Gaussian Mixture Models (GMM) (probabilistic clusters)
    Agglomerative Clustering (hierarchical structure; multiple linkages)
    DBSCAN (density‑based; detects noise/outliers)

Each algorithm is evaluated using:

    Silhouette Score
    Davies–Bouldin Index
    Cluster stability
    PCA/t‑SNE/UMAP visualization
"""

# =========================================================
# Imports & Configuration
# =========================================================

from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid", context="talk")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

RANDOM_STATE = 42

# =========================================================
# Load Clustering Summaries
# =========================================================

file_path = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\02-Clustering\02-clustering-results"
)


comparison_df = pd.concat([
    pd.read_csv(file_path / "kmeans_summary.csv"),
    pd.read_csv(file_path / "gmm_summary.csv"),
    pd.read_csv(file_path / "agglomerative_summary.csv"),
    pd.read_csv(file_path / "dbscan_summary.csv")
],
ignore_index=True)

print("\n----- FINAL CLUSTERING COMPARISON -----")

print(
    comparison_df.round(4)
)

comparison_df.to_csv(
    file_path / "clustering_comparison_summary.csv",
    index=False
)

# =========================================================
# Consistent Algorithm Colors
# =========================================================

algorithm_colors = {
    "K-Means": "#1f77b4",        # Blue
    "GMM": "#ff7f0e",            # Orange
    "Agglomerative": "#2ca02c", # Green
    "DBSCAN": "#d62728"          # Red
}

palette=algorithm_colors

# =========================================================
# Automatic Comparison Charts
# =========================================================

# =========================================================
# Silhouette Comparison
# =========================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=comparison_df,
    x="algorithm",
    y="silhouette_score",
    palette=algorithm_colors
)

plt.title("Silhouette Score Comparison")
plt.ylabel("Score")

plt.tight_layout()
plt.show()

# =========================================================
# Davies-Bouldin Comparison
# =========================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=comparison_df,
    x="algorithm",
    y="davies_bouldin_index",
    palette=algorithm_colors
)

plt.title("Davies-Bouldin Index Comparison")
plt.ylabel("Score")

plt.tight_layout()
plt.show()

# =========================================================
# Cluster Count
# =========================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=comparison_df,
    x="algorithm",
    y="cluster_count",
    palette=algorithm_colors
)

plt.title("Number of Clusters Found")
plt.ylabel("Clusters")

plt.tight_layout()
plt.show()

# =========================================================
# Noise Percentage Comparison
# =========================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=comparison_df,
    x="algorithm",
    y="noise_pct",
    palette=algorithm_colors
)

plt.title("Noise Percentage")
plt.ylabel("Percent")

plt.tight_layout()
plt.show()

# =========================================================
# Algorithm Rankings
# =========================================================

silhouette_ranking = (
    comparison_df
    .sort_values(
        by="silhouette_score",
        ascending=False
    )
    [["algorithm", "silhouette_score"]]
)

print("\n----- SILHOUETTE RANKING -----")
print(silhouette_ranking)

db_ranking = (
    comparison_df
    .sort_values(
        by="davies_bouldin_index"
    )
    [["algorithm", "davies_bouldin_index"]]
)

print("\n----- DAVIES-BOULDIN RANKING -----")
print(db_ranking)

# =========================================================
# Overall Metric Comparison
# =========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

sns.barplot(
    data=comparison_df,
    x="algorithm",
    y="silhouette_score",
    palette=algorithm_colors,
    ax=axes[0]
)

axes[0].set_title("Silhouette Score")
axes[0].set_ylabel("Score")

sns.barplot(
    data=comparison_df,
    x="algorithm",
    y="davies_bouldin_index",
    palette=algorithm_colors,
    ax=axes[1]
)

axes[1].set_title("Davies-Bouldin Index")
axes[1].set_ylabel("Score")

plt.tight_layout()

plt.show()

# =========================================================
# Final Clustering Comparison
# =========================================================

print(
    """
FINAL COMPARISON

Agglomerative Clustering (Ward k=2) achieved the highest
Silhouette Score and strongest recovery of the dominant
wine-population structure.

K-Means (k=4) produced the richest and most interpretable
wine-profile segmentation.

Gaussian Mixture Models provided a strong probabilistic
representation and closely reproduced the wine-type structure.

DBSCAN was most useful for identifying sparse and transitional
observations rather than maximizing cluster separation.

Goal	WinnerBest overall cluster separation	Agglomerative (Ward k=2)
Best wine profile discovery	K-Means (k=4)
Best probabilistic model	GMM (2-diag)
Best outlier detection	DBSCAN

And for a business recommendation:

K-Means (k=4) is recommended for customer-facing segmentation and interpretation
because it produced four stable and chemically distinct wine profiles. 
Agglomerative Clustering (Ward k=2) most effectively recovered the dominant 
natural structure of the dataset, while GMM confirmed the existence of two 
underlying wine populations. DBSCAN was most useful for identifying sparse 
and transitional observations.


FINAL REPORT:
    
- Best Overall Clustering Model: Agglomerative Clustering (Ward k=2)

Agglomerative Clustering provided the strongest recovery of the underlying 
structure of the Wine Quality dataset, achieving the highest silhouette score 
and approximately 96.94% agreement with the known wine type categories.

- Best Business Segmentation Model: K-Means (k=4)

K-Means generated four highly stable and interpretable wine profiles, providing 
the most actionable segmentation for business analysis and decision-making.

- Best Probabilistic Model: Gaussian Mixture Model (2 Components, Diagonal Covariance)

GMM confirmed the existence of two dominant wine populations while providing a 
probabilistic framework for cluster membership.

- Best Outlier Detection Model: DBSCAN

DBSCAN proved most useful for identifying sparse and transitional wines that 
did not belong strongly to the major wine populations.


QUESTIONS AND ANSWERS

Q: "Which clustering model should we deploy?"
A: My recommendation would be: K-Means (k=4)
This is because it provides the most useful segmentation for business decisions.

Q: "Which model best captures the true underlying structure of the wine data?"
A: My recommendation would be: Agglomerative Clustering (Ward k=2).
This is because it produced the strongest overall clustering evidence.
"""
)