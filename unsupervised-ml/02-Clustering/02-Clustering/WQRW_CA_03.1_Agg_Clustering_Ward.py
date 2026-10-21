# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 21, 2026
Editted:    October 21, 2026
Status:     Completed

Description

Agglomerative Clustering (hierarchical structure; multiple linkages)
Ward Only for k = 2, 3, 4

Evaluated using:

- Silhouette Score
- Davies–Bouldin Index
- Cluster stability
- PCA/t‑SNE/UMAP visualization
"""

# =========================================================
# 0. Imports & Configuration
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
# 1. Load Data
# =========================================================

file_path = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\00-data\clean-data\winequality-final.csv"
)

output_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\01-EDA\output_tables"
)

output_dir.mkdir(parents=True, exist_ok=True)

wqrw = pd.read_csv(file_path)

# Clean column names.
wqrw.columns = wqrw.columns.str.strip()

print("----- DATA SHAPE (MASTER) -----")
print(f"Rows: {wqrw.shape[0]:,}")
print(f"Columns: {wqrw.shape[1]}")

print("\n----- FIRST FIVE ROWS -----")
print(wqrw.head())

# =========================================================
# 2. Column Type Setup
# =========================================================

required_cols = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
    "type",
]

missing_cols = [
    col for col in required_cols
    if col not in wqrw.columns
]

if missing_cols:
    raise ValueError(
        f"Missing expected Wine Quality columns: {missing_cols}"
    )

# Categorical variable.
categorical_cols = ["type"]

# Numeric columns, including quality.
numeric_cols = wqrw.select_dtypes(
    include=["number"]
).columns.tolist()

# Exclude quality from the clustering features.
target = "quality"

feature_cols = [
    col for col in numeric_cols
    if col != target
]

print("\n----- NUMERIC COLUMNS -----")
print(numeric_cols)

print("\n----- CATEGORICAL COLUMNS -----")
print(categorical_cols)

print("\n----- CLUSTERING FEATURES -----")
print(feature_cols)

print("\n----- EXCLUDED INTERPRETATION VARIABLE -----")
print(target)

# =========================================================
# 3. Feature Matrix for Clustering
# =========================================================

from sklearn.preprocessing import StandardScaler

X = wqrw[feature_cols].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature matrix shape:", X_scaled.shape)

# =========================================================
# Ward Comparison for k=2, k=3, and k=4
# =========================================================

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap.umap_ as umap

# Define the Ward cluster counts to compare.

ward_k_values = [2, 3, 4]

# Store model evaluation results.

ward_results = []

# Fit each Ward model and save its cluster assignments.

for k in ward_k_values:

    ward_model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    ward_labels = ward_model.fit_predict(X_scaled)

    cluster_column = f"ward_k{k}_cluster"

    wqrw[cluster_column] = ward_labels

    cluster_sizes = (
        wqrw[cluster_column]
        .value_counts()
        .sort_index()
    )

    ward_results.append(
        {
            "clusters": k,
            "silhouette_score": silhouette_score(
                X_scaled,
                ward_labels
            ),
            "davies_bouldin_index": davies_bouldin_score(
                X_scaled,
                ward_labels
            ),
            "smallest_cluster": cluster_sizes.min(),
            "largest_cluster": cluster_sizes.max(),
            "smallest_cluster_pct": (
                cluster_sizes.min() / len(wqrw) * 100
            ),
            "largest_cluster_pct": (
                cluster_sizes.max() / len(wqrw) * 100
            )
        }
    )

    print(f"\n----- WARD k={k} CLUSTER SIZES -----")
    print(cluster_sizes)

# Create and save the Ward comparison table.

ward_results_df = (
    pd.DataFrame(ward_results)
    .sort_values("clusters")
)

print("\n----- WARD k=2, k=3, AND k=4 COMPARISON -----")
print(ward_results_df.round(4))

ward_results_df.to_csv(
    output_dir / "ward_k2_k3_k4_comparison.csv",
    index=False
)

# Calculate and save the cluster profile for each Ward model.

for k in ward_k_values:

    cluster_column = f"ward_k{k}_cluster"

    ward_profile = (
        wqrw
        .groupby(cluster_column)[feature_cols + ["quality"]]
        .mean()
        .round(4)
    )

    print(f"\n----- WARD k={k} PROFILE -----")
    print(ward_profile)

    ward_profile.to_csv(
        output_dir / f"ward_k{k}_cluster_profile.csv"
    )

# Compare each Ward model with the excluded wine type variable.

for k in ward_k_values:

    cluster_column = f"ward_k{k}_cluster"

    type_counts = pd.crosstab(
        wqrw[cluster_column],
        wqrw["type"],
        margins=True
    )

    type_percentages = (
        pd.crosstab(
            wqrw[cluster_column],
            wqrw["type"],
            normalize="index"
        )
        .mul(100)
        .round(2)
    )

    print(f"\n----- WARD k={k} TYPE COUNTS -----")
    print(type_counts)

    print(f"\n----- WARD k={k} TYPE PERCENTAGES -----")
    print(type_percentages)

    type_counts.to_csv(
        output_dir / f"ward_k{k}_type_counts.csv"
    )

    type_percentages.to_csv(
        output_dir / f"ward_k{k}_type_percentages.csv"
    )

# Calculate PCA coordinates once for all Ward models.

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(X_scaled)

# Plot PCA results side by side.

fig, axes = plt.subplots(
    1,
    3,
    figsize=(24, 6)
)

for axis, k in zip(axes, ward_k_values):

    cluster_column = f"ward_k{k}_cluster"

    scatter = axis.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=wqrw[cluster_column],
        cmap="tab10",
        alpha=0.7
    )

    axis.set_title(f"Ward k={k} - PCA")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")

    fig.colorbar(
        scatter,
        ax=axis,
        ticks=range(k)
    )

plt.tight_layout()
plt.show()

# Calculate t-SNE coordinates once for all Ward models.

tsne = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    random_state=RANDOM_STATE
)

X_tsne = tsne.fit_transform(X_scaled)

# Plot t-SNE results side by side.

fig, axes = plt.subplots(
    1,
    3,
    figsize=(24, 6)
)

for axis, k in zip(axes, ward_k_values):

    cluster_column = f"ward_k{k}_cluster"

    scatter = axis.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=wqrw[cluster_column],
        cmap="tab10",
        alpha=0.7
    )

    axis.set_title(f"Ward k={k} - t-SNE")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")

    fig.colorbar(
        scatter,
        ax=axis,
        ticks=range(k)
    )

plt.tight_layout()
plt.show()

# Calculate UMAP coordinates once for all Ward models.

reducer = umap.UMAP(
    n_components=2,
    random_state=RANDOM_STATE
)

X_umap = reducer.fit_transform(X_scaled)

# Plot UMAP results side by side.

fig, axes = plt.subplots(
    1,
    3,
    figsize=(24, 6)
)

for axis, k in zip(axes, ward_k_values):

    cluster_column = f"ward_k{k}_cluster"

    scatter = axis.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=wqrw[cluster_column],
        cmap="tab10",
        alpha=0.7
    )

    axis.set_title(f"Ward k={k} - UMAP")
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")

    fig.colorbar(
        scatter,
        ax=axis,
        ticks=range(k)
    )

plt.tight_layout()
plt.show()

# Create a sample Ward dendrogram.

sample_size = min(250, len(wqrw))

sample_indices = (
    pd.Series(
        np.arange(len(wqrw))
    )
    .sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )
    .to_numpy()
)

X_sample = X_scaled[sample_indices]

sample_linkage = linkage(
    X_sample,
    method="ward"
)

plt.figure(figsize=(16, 7))

dendrogram(
    sample_linkage,
    truncate_mode="level",
    p=5,
    color_threshold=None
)

plt.title(
    f"Ward Dendrogram - Sample of {sample_size} Wines"
)
plt.xlabel("Sampled Observations")
plt.ylabel("Ward Distance")
plt.tight_layout()
plt.show()

# Create the full-data Ward dendrogram.

full_linkage = linkage(
    X_scaled,
    method="ward"
)

plt.figure(figsize=(18, 8))

dendrogram(
    full_linkage,
    no_labels=True,
    color_threshold=None
)

plt.title("Ward Dendrogram - Full Dataset")
plt.xlabel("Observations")
plt.ylabel("Ward Distance")
plt.tight_layout()
plt.show()

# Preserve the current official model as Ward k=2.

wqrw["agg_cluster"] = wqrw["ward_k2_cluster"]

# Save the dataset with all Ward solutions.

wqrw.to_csv(
    output_dir / "winequality_ward_k2_k3_k4_clusters.csv",
    index=False
)

# =========================================================
# Interpretation
# =========================================================

"""
Best overall Ward solution: k = 2

Based on your stated evaluation framework, Ward k=2 remains the best final Agglomerative model.

Why k=2 wins
Highest Silhouette Score: 0.2607
Reasonably balanced sizes: 3,842 and 1,478
Clearest broad separation
Approximately 96.94% agreement with wine type after aligning the two clusters:
Cluster 0 is 99.43% white
Cluster 1 is 90.46% red
Simplest and most reproducible interpretation
What k=3 and k=4 add
Ward k=3: best hierarchical alternative
Silhouette: 0.2145
Best Davies-Bouldin Index: 1.5702
Balanced sizes: 1,478, 2,662, and 1,180
Preserves the red-dominated group
Divides the white population into:
drier, higher-alcohol, higher-quality wines
sweeter, high-sulfur, lower-alcohol wines
Corresponds well with the three prominent dendrogram branches

This is a meaningful exploratory model, but its separation is weaker than k=2.

Ward k=4: best detailed profile model
Silhouette: 0.2229
Davies-Bouldin: 1.5977
Balanced sizes: 685, 2,662, 1,180, and 793
Divides both broad wine populations:
two predominantly white profiles
two predominantly red profiles
Provides the richest chemical interpretation

Interestingly, k=4 performs slightly better than k=3 on silhouette, while k=3 performs slightly better on Davies-Bouldin.

Recommended conclusion

Ward linkage with two clusters was selected as the final Agglomerative model because it achieved the highest silhouette score among the balanced solutions and produced two meaningful, well-sized clusters. The assignments also showed approximately 96.94% agreement with the known red and white wine types, despite wine type being excluded from clustering. Ward solutions with three and four clusters revealed meaningful nested substructure, with k=3 separating two white-wine profiles and k=4 additionally separating two red-wine profiles. These models were retained as exploratory hierarchical segmentations.

Final designation
Official Agglomerative model: Ward k=2
Hierarchical exploratory model: Ward k=3
Detailed profile model: Ward k=4


One caution: the three colors in the dendrogram are generated using a distance threshold. They support exploring k=3, but they do not by themselves prove that k=3 is superior. Your quantitative and type-agreement evidence still favors k=2.
"""

# =========================================================
# Story and more information
# =========================================================

"""
For Agglomerative Clustering specifically, I would choose:

1. 🏆 Best Overall: Ward k = 2

Why?

Metrics
Silhouette Score = 0.2607  ← Highest
Davies-Bouldin  = 1.7164

Cluster Balance
Cluster 0 = 3,842 wines (72.2%)
Cluster 1 = 1,478 wines (27.8%)


Both clusters are large and meaningful.

External Validation
Cluster 0 = 99.43% White
Cluster 1 = 90.46% Red


This produced approximately:

96.94% agreement with wine type


which is extremely strong considering type was excluded from training.

2. 🥈 Most Interesting Hierarchical Solution: Ward k = 3

The dendrogram makes a strong case for considering k=3.

Metrics
Silhouette Score = 0.2145
Davies-Bouldin  = 1.5702  ← Best DB

What happens?

The red wines largely remain together while the white wines split into:

Dry / higher alcohol / higher quality whites
Sweeter / higher sulfur dioxide / lower alcohol whites

This is actually a very meaningful business interpretation.

3. 🥉 Best Detailed Segmentation: Ward k = 4
Metrics
Silhouette Score = 0.2229
Davies-Bouldin  = 1.5977

What happens?

The two white-wine groups from k=3 remain intact, and the red wines split into:

Structured high-acidity reds
Higher volatile-acidity reds

This gives:

2 white profiles
2 red profiles


which is arguably the richest interpretation.

4. Final report

I'd write:

Selected Model
Ward Linkage
k = 2


because it:

Has the highest silhouette score
Produces balanced clusters
Achieves ~97% agreement with wine type
Represents the strongest natural separation in the data
Additional Insight
Ward k = 3 and k = 4 revealed meaningful hierarchical substructure.


Specifically:

k=3 splits white wines into two distinct profiles.
k=4 further separates red wines into two distinct profiles.
My actual ranking
Best Overall Clustering Structure     → Ward k=2
Best Hierarchical Interpretation      → Ward k=3
Best Detailed Wine Profiling          → Ward k=4


If your goal is model selection, choose k=2.

If your goal is telling the most interesting wine story, k=4 is arguably the most informative because it uncovers four chemically distinct wine profiles while still maintaining reasonable cluster quality.
"""

# =========================================================
# Key Insights
# =========================================================

"""
• Ward linkage with two clusters was selected as the final
  Agglomerative model because it achieved the highest silhouette
  score among the balanced solutions (0.2607), maintained
  reasonable cluster sizes, and reproduced the strongest natural
  separation within the dataset.

• Comparison against the excluded wine type variable showed that
  the two-cluster Ward solution achieved approximately 96.94%
  agreement with the known red and white wine categories,
  despite wine type not being used during clustering.

• Dendrogram analysis revealed additional hierarchical structure
  beyond the final two-cluster solution.

• Ward linkage with three clusters produced the lowest
  Davies-Bouldin Index (1.5702) and aligned closely with the
  three major branches visible in the dendrogram.

• The three-cluster solution preserved the predominantly red-wine
  group while separating white wines into two distinct profiles:
  a drier, higher-alcohol, higher-quality subgroup and a sweeter,
  higher-sulfur, lower-alcohol subgroup.

• Ward linkage with four clusters provided the most detailed
  wine-profile segmentation.

• The four-cluster solution further divided the red-wine
  population into two chemically distinct profiles while
  preserving the two white-wine profiles identified in the
  three-cluster solution.

• Although Ward linkage with three and four clusters revealed
  meaningful substructure, neither solution exceeded the
  two-cluster model in overall cluster separation.

• Therefore, Ward linkage with two clusters was retained as the
  final Agglomerative model, while the three- and four-cluster
  solutions were interpreted as hierarchical refinements that
  provide additional insight into the underlying wine
  populations.

FINAL MODEL

Algorithm: Agglomerative Clustering
Linkage Method: Ward
Clusters: 2

Silhouette Score: 0.2607
Davies-Bouldin Index: 1.7164

Cluster Sizes:
Cluster 0 = 3,842 wines (72.2%)
Cluster 1 = 1,478 wines (27.8%)

Additional Hierarchical Findings:
Ward k=3 → Best Davies-Bouldin Index (1.5702)
Ward k=4 → Richest Wine Profile Segmentation
"""