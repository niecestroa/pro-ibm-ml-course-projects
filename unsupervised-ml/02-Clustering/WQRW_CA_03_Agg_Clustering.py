# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October 21, 2026
Status:     Completed

Description

Agglomerative Clustering (hierarchical structure; multiple linkages)

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
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\02-Clustering\02-clustering-results"
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
# Part 1: Starting Agglomerative Clustering
# =========================================================

# =========================================================
# 4. Agglomerative Evaluation
# =========================================================

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

linkage_methods = [
    "ward",
    "complete",
    "average"
]

agg_results = []

for linkage in linkage_methods:

    for k in range(2, 21):

        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage
        )

        labels = model.fit_predict(X_scaled)

        agg_results.append(
            {
                "linkage": linkage,
                "clusters": k,
                "silhouette_score": silhouette_score(
                    X_scaled,
                    labels
                ),
                "davies_bouldin_index": davies_bouldin_score(
                    X_scaled,
                    labels
                )
            }
        )

agg_results_df = pd.DataFrame(
    agg_results
)

print("\n----- AGGLOMERATIVE RESULTS -----")

print(
    agg_results_df
    .sort_values(
        by="silhouette_score",
        ascending=False
    )
    .head(20)
)

# =========================================================
# 5. Agglomerative Evaluation Plots
# =========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 5)
)

for linkage in linkage_methods:

    subset = agg_results_df[
        agg_results_df["linkage"] == linkage
    ]

    axes[0].plot(
        subset["clusters"],
        subset["silhouette_score"],
        marker="o",
        label=linkage
    )

    axes[1].plot(
        subset["clusters"],
        subset["davies_bouldin_index"],
        marker="o",
        label=linkage
    )

axes[0].set_title("Silhouette Score")
axes[0].set_xlabel("Clusters")
axes[0].legend()

axes[1].set_title("Davies-Bouldin Index")
axes[1].set_xlabel("Clusters")
axes[1].legend()

plt.tight_layout()

plt.show()

"""
Agglomerative Clustering: Preliminary Findings
Linkage Methods Evaluated

Three linkage methods were tested:

Ward
Complete
Average

Across cluster counts:

k = 2 through 20

Evaluation metrics:

Silhouette Score
Davies-Bouldin Index
Best Performing Models

Complete Linkage
Clusters = 2
Silhouette = 0.7897
Davies-Bouldin = 0.1492

Average Linkage
Clusters = 2
Silhouette = 0.7897
Davies-Bouldin = 0.1492


Both methods produced essentially identical top results.

Ward Linkage Performance

Ward linkage performed substantially worse:

Best Silhouette ≈ 0.26
Davies-Bouldin ≈ 1.47+

These values are very similar to the K-Means results.

This suggests Ward is not capturing the strongest structure present in the data.

Comparison with Previous Algorithms
Algorithm	Best Silhouette	Best Davies-BouldinK-Means	0.2428	1.4844
GMM	0.2516	1.7782
Agglomerative	0.7897	0.1492

Agglomerative Clustering clearly outperforms both K-Means and GMM on the 
evaluation metrics examined so far.

Main Observation

The strongest solutions occur at:

k = 2


for both:

Complete linkage
Average linkage

As cluster counts increase:

Silhouette declines steadily
Davies-Bouldin increases steadily

This indicates the dataset naturally favors a small number of highly separated groups.

Relationship to Previous Findings

These results strongly support what was discovered with GMM:

Two dominant wine populations exist.


Recall that the GMM two-component solution achieved:

95.13% agreement with wine type


without using the wine type variable during clustering.

The Agglomerative metrics suggest it may be detecting the same underlying 
structure even more clearly.
"""

"""
Current Best Candidate
Algorithm: Agglomerative Clustering
Linkage: Complete
Clusters: 2
Silhouette Score: 0.7897
Davies-Bouldin Index: 0.1492
"""

# =========================================================
# 6. Final Agglomerative Model
# =========================================================

# Fit the final Agglomerative model.

best_linkage = "complete"
best_clusters = 2

agg_final = AgglomerativeClustering(
    n_clusters=best_clusters,
    linkage=best_linkage
)

wqrw["agg_cluster"] = agg_final.fit_predict(
    X_scaled
)

# Display cluster sizes.

print("\n----- FINAL AGGLOMERATIVE CLUSTER SIZES -----")

print(
    wqrw["agg_cluster"]
    .value_counts()
    .sort_index()
)

# Display final metrics.

print("\n----- FINAL AGGLOMERATIVE METRICS -----")

print(
    f"Silhouette Score: "
    f"{silhouette_score(X_scaled, wqrw['agg_cluster']):.4f}"
)

print(
    f"Davies-Bouldin Index: "
    f"{davies_bouldin_score(X_scaled, wqrw['agg_cluster']):.4f}"
)

# =========================================================
# 7. Compare Against Wine Type
# =========================================================

# Compare clusters against wine type.

agg_type_counts = pd.crosstab(
    wqrw["agg_cluster"],
    wqrw["type"],
    margins=True
)

agg_type_percent = pd.crosstab(
    wqrw["agg_cluster"],
    wqrw["type"],
    normalize="index"
).mul(100).round(2)

print("\n----- AGGLOMERATIVE TYPE COUNTS -----")
print(agg_type_counts)

print("\n----- AGGLOMERATIVE TYPE PERCENTAGES -----")
print(agg_type_percent)

"""
Revised Interpretation
Complete Linkage (k=2)

Pros

Highest Silhouette Score
Lowest Davies-Bouldin Index

Cons

Produced a trivial solution
One cluster contains only a single observation
Does not provide meaningful segmentation
Does not reveal underlying wine structure

This is a classic example of:

Good metric values can still produce a bad clustering solution.
"""

# Next Check

for linkage in ["complete", "average"]:

    for k in [3, 4]:

        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage
        )

        labels = model.fit_predict(X_scaled)

        print(
            f"\n{linkage.upper()}  k={k}"
        )

        print(
            pd.Series(labels)
            .value_counts()
            .sort_index()
        )
        
# =========================================================
# Part 2: Checking Agglomerative Cluster since problems found
# =========================================================
        
# =========================================================
# 8. Agglomerative Cluster Balance Evaluation
# =========================================================

"""
Likely direction

Based on your earlier table, Ward linkage is now the most promising candidate, 
even though its raw silhouette scores are lower. Ward is more likely to produce 
clusters useful for segmentation rather than one enormous cluster plus isolated 
observations.

Do not finalize complete or average linkage. Use the balanced-results table to 
select the best nontrivial, reasonably sized solution, then proceed to profiles 
and visualization.

Report-ready finding so far

Complete and average linkage initially produced exceptionally high silhouette 
scores and low Davies-Bouldin indices. However, inspection of cluster sizes 
revealed highly imbalanced solutions in which nearly all 5,320 observations 
were assigned to one cluster and the remaining clusters contained only one to 
21 observations. These configurations were therefore interpreted as outlier-isolation 
solutions rather than meaningful wine segmentations. Final model selection incorporated 
cluster-size balance in addition to internal validation metrics.
"""

# Store evaluation results with cluster-size diagnostics.

balance_results = []

for linkage in linkage_methods:

    for k in range(2, 21):

        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage
        )

        labels = model.fit_predict(X_scaled)

        cluster_counts = pd.Series(labels).value_counts()

        balance_results.append(
            {
                "linkage": linkage,
                "clusters": k,
                "silhouette_score": silhouette_score(
                    X_scaled,
                    labels
                ),
                "davies_bouldin_index": davies_bouldin_score(
                    X_scaled,
                    labels
                ),
                "smallest_cluster": cluster_counts.min(),
                "largest_cluster": cluster_counts.max(),
                "smallest_cluster_pct": (
                    cluster_counts.min() / len(labels) * 100
                ),
                "largest_cluster_pct": (
                    cluster_counts.max() / len(labels) * 100
                )
            }
        )

# Create the cluster-balance results table.

balance_df = pd.DataFrame(balance_results)

# Retain solutions where every cluster contains at least five percent of the data.

valid_balance_df = balance_df[
    balance_df["smallest_cluster_pct"] >= 5
].copy()

# Display valid solutions ranked by silhouette score.

print("\n----- BALANCED AGGLOMERATIVE SOLUTIONS -----")

print(
    valid_balance_df
    .sort_values(
        by="silhouette_score",
        ascending=False
    )
    .round(4)
    .head(20)
)

"""
This is actually a very important result and a good lesson in clustering.

What happened?

The metrics initially suggested:

Complete Linkage + k=2
Silhouette = 0.7897
Davies-Bouldin = 0.1492

which looked incredible.

But when you examined cluster sizes:

Cluster 0 = 5319 wines
Cluster 1 = 1 wine

discovered the model was simply isolating a single extreme observation.

1. Revised Agglomerative Conclusion

A. Reject Complete Linkage

Reasons:

Creates trivial clusters
Isolates only 1-2 observations
Not useful for segmentation
Metrics are artificially inflated

Evidence:

k=2 -> 5319 / 1
k=3 -> 5317 / 1 / 2
k=4 -> 5316 / 1 / 2 / 1

B. Reject Average Linkage

Reasons:

Similar behavior
Produces highly imbalanced clusters

Evidence:

k=2 -> 5319 / 1
k=3 -> 5318 / 1 / 1
k=4 -> 5297 / 21 / 1 / 1

C. Meaningful Solutions

After imposing a minimum cluster-size requirement, only Ward linkage remained:

Ward k=2
Silhouette = 0.2607
DB Index = 1.7164
Cluster Sizes:
27.8% / 72.2%

Ward k=3
Silhouette = 0.2145
DB Index = 1.5702

Ward k=4
Silhouette = 0.2229
DB Index = 1.5977

D. Recommended Final Agglomerative Model

AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)


Why?
- Highest silhouette among balanced solutions
- Reasonable cluster sizes
- Not dominated by outlier detection
- Similar structure to the successful GMM model
"""


# =========================================================
# Part 3: Changed Algorithm for Ward Only
# =========================================================

# =========================================================
# Final Ward Model Selection
# =========================================================

# Fit the selected Ward model.

agg_final = AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)

final_cluster_col = "agg_cluster"

# Replace previous cluster assignments.

wqrw["agg_cluster"] = agg_final.fit_predict(
    X_scaled
)

# Display final Ward cluster sizes.

print("\n----- FINAL WARD CLUSTER SIZES -----")

print(
    wqrw["agg_cluster"]
    .value_counts()
    .sort_index()
)

# Display final Ward metrics.

print("\n----- FINAL WARD METRICS -----")

print(
    f"Silhouette Score: "
    f"{silhouette_score(X_scaled, wqrw['agg_cluster']):.4f}"
)

print(
    f"Davies-Bouldin Index: "
    f"{davies_bouldin_score(X_scaled, wqrw['agg_cluster']):.4f}"
)

# =========================================================
# 9. Agglomerative Cluster Profile Table 
# =========================================================

agg_profile = (
    wqrw
    .groupby("agg_cluster")
    [feature_cols + ["quality"]]
    .mean()
    .round(4)
)

print("\n----- AGGLOMERATIVE PROFILE FOR WARD -----")
print(agg_profile)

agg_profile.to_csv(
    output_dir / "agglomerative_cluster_profile.csv"
)

# =========================================================
# 10. Agglomerative PCA Visualization
# =========================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=wqrw["agg_cluster"],
    cmap="tab10",
    alpha=0.7
)

plt.title("Agglomerative Clustering for WARD - PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(scatter)

plt.show()

# =========================================================
# 11. Agglomerative t-SNE Visualization
# =========================================================

from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=RANDOM_STATE
)

X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=wqrw["agg_cluster"],
    cmap="tab10",
    alpha=0.7
)

plt.title("Agglomerative Clustering for WARD - t-SNE")

plt.show()

# =========================================================
# 12. Agglomerative UMAP Visualization
# =========================================================

import umap.umap_ as umap # Import UMAP

reducer = umap.UMAP(
    n_components=2,
    random_state=RANDOM_STATE
)

X_umap = reducer.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=wqrw["agg_cluster"],
    cmap="tab10",
    alpha=0.7
)

plt.title("Agglomerative Clustering for WARD - UMAP")

plt.show()

"""
Agglomerative Clustering Interpretation

After correcting for the outlier problem in Complete and Average linkage, the 
final model was:

AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)

with:

Silhouette Score = 0.2607
Davies-Bouldin Index = 1.7164

and cluster sizes:

Cluster 0 = 3,842 wines (72.2%)
Cluster 1 = 1,478 wines (27.8%)

1. Cluster Profile Interpretation

A. Cluster 0

Characteristics:

Lower fixed acidity (6.84)
Lower volatile acidity (0.28)
Higher residual sugar (5.98)
Higher sulfur dioxide
Lower chlorides
Lower sulphates
Slightly higher quality (5.87)

This cluster appears to represent:

Sweeter, lower-acidity wines with higher sulfur dioxide concentrations and slightly 
higher quality scores.

B. Cluster 1

Characteristics:

Higher fixed acidity (8.18)
Higher volatile acidity (0.52)
Lower residual sugar (2.63)
Lower sulfur dioxide
Higher chlorides
Higher sulphates
Slightly lower quality (5.59)

This cluster represents:

More acidic and structured wines with lower sugar content and stronger mineral 
characteristics.

2. PCA Interpretation

The PCA plot shows:

Two broad groups are visible.
Some overlap exists in the center.
Separation is present but not perfect.

Interpretation:

PCA suggests Agglomerative Clustering identified two meaningful wine populations, 
although the separation is moderate when projected onto only two principal components.

This aligns with the silhouette score of:

0.2607


which indicates usable but imperfect separation.

3. t-SNE Interpretation

The t-SNE plot is much stronger.

Observations:

Two distinct regions emerge.
The groups occupy mostly different parts of the embedding.
Only limited overlap occurs near the boundary.

Interpretation:

The t-SNE visualization reveals a substantially clearer separation between clusters 
than PCA, indicating that the distinction between groups becomes more apparent when 
nonlinear relationships are preserved.

This supports the existence of two dominant latent wine populations.

4. UMAP Interpretation

The UMAP visualization is arguably the strongest.

Observations:

Two large blocks are visible.
Separation between clusters is clear.
Only a small transition region appears between groups.

Interpretation:

UMAP provides strong visual evidence that the dataset contains two major naturally 
occurring populations. The separation observed in UMAP closely matches the 
clustering structure identified by the Ward linkage algorithm.

Overall Interpretation

Agglomerative Clustering produced a different type of solution than K-Means.

5. Comparison

K-Means
4 clusters

Focused on multiple wine profiles.
Captured differences in sugar, alcohol, acidity, and sulphates.
Produced richer segmentation.
Agglomerative (Ward)
2 clusters

Focused on the strongest overall split in the dataset.
Separated wines into:
sweeter/lower-acidity wines
more acidic/structured wines

Rather than discovering numerous wine styles, Agglomerative Clustering identified 
two broad wine populations.

Final Takeaway

Ward linkage produced the strongest balanced Agglomerative solution. 
Cluster profiling revealed two chemically distinct wine populations characterized 
primarily by differences in acidity, residual sugar, sulfur dioxide concentrations, 
chlorides, and sulphates. PCA showed moderate separation, while t-SNE and UMAP 
revealed much clearer cluster boundaries, supporting the presence of two dominant 
latent structures within the wine dataset.
"""

# =========================================================
# 13. Agglomerative Dendrograms
# =========================================================

from scipy.cluster.hierarchy import (
    linkage,
    dendrogram
)

# Generate Ward linkage matrix for a random sample.

sample_df = (
    pd.DataFrame(X_scaled)
    .sample(
        n=250,
        random_state=RANDOM_STATE
    )
)

sample_linkage = linkage(
    sample_df,
    method="ward"
)

# Plot sample dendrogram.

plt.figure(figsize=(14, 6))

dendrogram(
    sample_linkage,
    truncate_mode="level",
    p=5
)

plt.title(
    "Ward Dendrogram (Sample of 250 Wines)"
)

plt.xlabel("Sampled Observations")
plt.ylabel("Distance")

plt.show()

# Generate Ward linkage matrix for all observations.

full_linkage = linkage(
    X_scaled,
    method="ward"
)

# Plot full dendrogram.

plt.figure(figsize=(16, 8))

dendrogram(
    full_linkage,
    no_labels=True
)

plt.title(
    "Ward Dendrogram (Full Dataset)"
)

plt.xlabel("Observations")
plt.ylabel("Distance")

plt.show()

"""
1. Sample Dendrogram Interpretation

The sample dendrogram of 250 wines is much easier to read.

Key observations:

Several smaller groups form at low distances (roughly 0-10).
Those groups merge into larger structures around distances of 15-25.
The final large merge occurs around distance 35.

This suggests:

The wine dataset contains a hierarchical structure where smaller groups combine into broader wine populations.

A horizontal cut near the upper portion of the dendrogram would naturally produce 2-3 major clusters, 
which is consistent with the balanced Ward solutions you evaluated.

2. Full Dataset Dendrogram Interpretation

The full dendrogram is dense, which is expected with 5,320 observations.

Key observations:

Two very large branches remain separated until a relatively high distance.
The final merge occurs around distance 150+.
Several intermediate branches exist before the final merge.

This indicates:

The wine dataset contains two dominant groups with additional substructure nested within each group.

This finding aligns closely with:

Ward clustering selecting 2 clusters as the strongest balanced solution.
GMM identifying 2 dominant wine populations.
t-SNE and UMAP showing two major regions.

3. Report 

Dendrogram analysis revealed a clear hierarchical structure within the wine dataset. 
Both the sample and full-data dendrograms showed two dominant branches that remained 
separated until relatively large linkage distances. Smaller groups were observed
within each major branch, indicating meaningful substructure. These findings 
support the final Ward linkage solution with two clusters and reinforce the 
broader conclusion that the strongest latent structure in the dataset consists 
of two major wine populations.
"""

# =========================================================
# Final Model and Interpretation
# =========================================================

"""
1. FINAL MODEL

Algorithm: Agglomerative Clustering
Linkage Method: Ward
Clusters: 2
Silhouette Score: 0.2607
Davies-Bouldin Index: 1.7164

Cluster Sizes:
Cluster 0 = 3,842 wines (72.2%)
Cluster 1 = 1,478 wines (27.8%)

Comparison So Far
Algorithm	Final Model	Silhouette	DB Index	
Key FindingK-Means	k=4	0.2428	1.4844	Best wine-profile segmentation
GMM	2-diag	0.2516	1.7782	~95% agreement with wine type
Agglomerative	Ward k=2	0.2607	1.7164	Strong two-population structure

A really interesting pattern is emerging:

K-Means is best at finding multiple wine profiles.
GMM is best at recovering the hidden red/white structure.
Agglomerative Ward also favors a two-group solution, reinforcing the idea 
that the strongest natural structure in the dataset is a broad split between 
two wine populations.

At this point I would consider the Agglomerative section complete and move on to
DBSCAN, which will likely reveal whether there are natural density-based clusters 
and outliers in the wine data.

2. INTERPRETATION

The initial Complete and Average linkage solutions appeared exceptionally 
strong based on traditional clustering metrics. However, further inspection 
revealed that these solutions were driven by extreme outlier isolation, 
assigning nearly all observations to a single cluster while separating 
only a handful of wines into tiny clusters.

To address this issue, cluster-size balance was incorporated into model selection. 
After applying a minimum cluster-size threshold, Ward linkage produced the 
strongest practical solution.

The final Ward model identified two broad wine populations:

Cluster 0 (72.2%)

Higher residual sugar
Higher sulfur dioxide concentrations
Lower volatile acidity
Slightly higher average quality

Cluster 1 (27.8%)

Higher fixed acidity
Higher volatile acidity
Higher chlorides
Higher sulphates
Lower residual sugar

The PCA visualization showed moderate separation between clusters, while 
t-SNE and UMAP revealed much clearer segmentation. The UMAP visualization in 
particular demonstrated two distinct and well-formed groups with minimal overlap.

Overall, Agglomerative Clustering favored a broad two-population structure 
rather than the finer four-profile segmentation produced by K-Means.
"""

# =========================================================
# Export Model Summary
# =========================================================

agg_summary = pd.DataFrame({
    "algorithm": ["Agglomerative"],
    "final_model": ["Ward k=2"],
    "silhouette_score": [
        silhouette_score(
            X_scaled,
            wqrw[final_cluster_col]
        )
    ],
    "davies_bouldin_index": [
        davies_bouldin_score(
            X_scaled,
            wqrw[final_cluster_col]
        )
    ],
    "stability_ari": [np.nan],
    "cluster_count": [2],
    "noise_pct": [0.0]
})

agg_summary.to_csv(
    output_dir / "agglomerative_summary.csv",
    index=False
)

# =========================================================
# Agglomerative Clustering: Key Insights
# =========================================================

"""
These Ward results are actually much better and much more believable than 
the earlier Complete/Average solutions.

KEY INSIGHTS

• Complete and Average linkage initially produced extremely strong
  validation metrics, with silhouette scores approaching 0.79 and
  Davies-Bouldin indices near 0.15.

• Inspection of cluster sizes revealed these solutions were highly
  imbalanced, often assigning more than 99% of observations to a
  single cluster while isolating only one or a few wines.

• These solutions were therefore interpreted as outlier-detection
  behavior rather than meaningful wine segmentation.

• A cluster-balance filter requiring each cluster to contain at
  least 5% of the dataset was introduced to identify more practical
  clustering solutions.

• After applying this constraint, Ward linkage produced the strongest
  valid clustering solutions.

• Ward linkage with two clusters achieved the highest silhouette score
  among the balanced solutions (0.2607) while maintaining reasonable
  cluster proportions of 72.2% and 27.8%.

• Cluster profiles showed two chemically distinct wine populations.

• Cluster 0 contained wines with:
  - higher residual sugar
  - higher sulfur dioxide concentrations
  - lower volatile acidity
  - slightly higher quality scores

• Cluster 1 contained wines with:
  - higher fixed acidity
  - higher volatile acidity
  - higher chlorides
  - higher sulphates
  - lower residual sugar

• PCA revealed moderate separation between the two Ward clusters.

• t-SNE showed substantially stronger separation, with the two groups
  forming distinct regions in the projected feature space.

• UMAP also displayed a clear two-group structure with limited overlap,
  supporting the existence of two dominant latent wine populations.

• Unlike K-Means, which identified four wine profiles, Agglomerative
  Clustering favored a broader two-group segmentation of the dataset.
"""
