# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October 21, 2026
Status:     Completed

Description

DBSCAN (density‑based; detects noise/outliers)

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

"""
Good candidates generally have:

✅ At least 2 clusters

✅ Reasonable amount of noise

Plain Text
5%-25%

✅ High silhouette score

✅ Low Davies-Bouldin index

Avoid models that produce:

Plain Text
1 cluster
0 clusters
5000 noise points

because those aren't useful segmentations.
"""

# =========================================================
# 4. DBSCAN Evaluation
# =========================================================

from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

dbscan_results = []

eps_values = [
    0.3,
    0.5,
    0.7,
    0.9,
    1.1,
    1.3,
    1.5
]

min_samples_values = [
    3,
    5,
    10,
    15
]

for eps in eps_values:

    for min_samples in min_samples_values:

        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        labels = dbscan.fit_predict(X_scaled)

        n_clusters = len(
            set(labels)
        ) - (
            1 if -1 in labels else 0
        )

        n_noise = (
            labels == -1
        ).sum()

        if n_clusters >= 2:

            silhouette = silhouette_score(
                X_scaled,
                labels
            )

            dbi = davies_bouldin_score(
                X_scaled,
                labels
            )

        else:

            silhouette = np.nan
            dbi = np.nan

        dbscan_results.append(
            {
                "eps": eps,
                "min_samples": min_samples,
                "clusters": n_clusters,
                "noise_points": n_noise,
                "silhouette_score": silhouette,
                "davies_bouldin_index": dbi
            }
        )

dbscan_results_df = pd.DataFrame(
    dbscan_results
)

print("\n----- DBSCAN RESULTS -----")

print(
    dbscan_results_df
    .sort_values(
        by="silhouette_score",
        ascending=False
    )
    .round(4)
)

# =========================================================
# 5. DBSCAN Evaluation Summary
# =========================================================

print(
    dbscan_results_df
    .sort_values(
        by="silhouette_score",
        ascending=False
    )
    .head(15)
)

"""
DBSCAN is giving a very different message than the previous three algorithms.

Most DBSCAN solutions are poor:

Negative silhouette scores
Huge numbers of noise points
Thousands of observations classified as noise

Examples:

eps=0.7, min_samples=10
Clusters = 3
Noise = 5,245 out of 5,320 wines


That means DBSCAN considered almost the entire dataset to be noise.

Best DBSCAN candidate

Your strongest result is:

eps = 1.3
min_samples = 10

Clusters = 2
Noise Points = 1,424

Silhouette = 0.1505
Davies-Bouldin = 3.3824


Although the silhouette is positive, compare it to your other models:

Algorithm	Best SilhouetteAgglomerative	0.2607
GMM	0.2516
K-Means	0.2428
DBSCAN	0.1505

DBSCAN is clearly weaker on this dataset.

What this suggests:

The Wine Quality dataset does not contain strong density-based clusters.

Instead, it seems to contain:

Broad continuous groups

DBSCAN shines when data contains:

irregular cluster shapes
dense pockets
strong outliers

The wine chemistry data appears relatively continuous and well-behaved.

Select:

eps = 1.3
min_samples = 10

as the final DBSCAN candidate because it:

Has the highest silhouette score
Produces 2 clusters
Produces a reasonable amount of noise
"""

# =========================================================
# 6. Final DBSCAN Model
# =========================================================

# Fit the selected DBSCAN model.

best_eps = 1.3
best_min_samples = 10

dbscan_final = DBSCAN(
    eps=best_eps,
    min_samples=best_min_samples
)

wqrw["dbscan_cluster"] = dbscan_final.fit_predict(
    X_scaled
)

# Display cluster sizes including noise points.

print("\n----- FINAL DBSCAN CLUSTER SIZES -----")

print(
    wqrw["dbscan_cluster"]
    .value_counts()
    .sort_index()
)

# Calculate the number of clusters.

n_clusters = len(
    set(wqrw["dbscan_cluster"])
) - (
    1 if -1 in wqrw["dbscan_cluster"].unique() else 0
)

# Calculate the number of noise observations.

n_noise = (
    wqrw["dbscan_cluster"] == -1
).sum()

# Display final metrics.

print("\n----- FINAL DBSCAN METRICS -----")

print(f"Clusters Found: {n_clusters}")
print(f"Noise Points: {n_noise}")

print(
    f"Silhouette Score: "
    f"{silhouette_score(X_scaled, wqrw['dbscan_cluster']):.4f}"
)

print(
    f"Davies-Bouldin Index: "
    f"{davies_bouldin_score(X_scaled, wqrw['dbscan_cluster']):.4f}"
)

"""
Good. This is actually a fairly typical DBSCAN outcome.

Your final model:

Plain Text
eps = 1.3
min_samples = 10
 
produced:

Plain Text
Noise Points = 1,424 (26.8%)
Cluster 0 = 834
Cluster 1 = 3,062
 
Total = 5,320

**Interpretation**

DBSCAN is telling us:

26.8% of wines do not belong strongly to any dense region and are 
therefore classified as: Noise (Cluster = -1)

The remaining: 73.2% of wines form two density-based clusters.

Unlike:

K-Means
GMM
Agglomerative

DBSCAN is not trying to force every wine into a cluster.

That's actually the major strength of DBSCAN.
"""

# =========================================================
# 7. DBSCAN Cluster Profile Table
# =========================================================

# Calculate feature averages by DBSCAN cluster.

dbscan_profile = (
    wqrw
    .groupby("dbscan_cluster")
    [feature_cols + ["quality"]]
    .mean()
    .round(4)
)

print("\n----- DBSCAN PROFILE -----")
print(dbscan_profile)

dbscan_profile.to_csv(
    output_dir / "dbscan_cluster_profile.csv"
)

# =========================================================
# 8. DBSCAN Noise Summary
# =========================================================

noise_count = (
    wqrw["dbscan_cluster"] == -1
).sum()

noise_pct = (
    noise_count / len(wqrw) * 100
)

print("\n----- DBSCAN NOISE SUMMARY -----")

print(f"Noise Points: {noise_count}")

print(
    f"Noise Percentage: "
    f"{noise_pct:.2f}%"
)

"""
This actually gives you a very clear DBSCAN story.

DBSCAN Interpretation
Cluster 1 (3,062 wines)

This is the dominant dense region in the dataset.

Characteristics:

Lowest volatile acidity (0.2648)
Lowest chlorides (0.0423)
Lowest sulphates (0.4823)
Highest alcohol (10.62)
Highest quality (5.92)

Interpretation:

This cluster represents the highest-quality dense wine population identified 
by DBSCAN. Wines in this group tend to have lower volatile acidity and 
chlorides while exhibiting slightly higher alcohol content and quality scores.

Cluster 0 (834 wines)

Characteristics:

Higher fixed acidity (8.05)
Higher volatile acidity (0.5267)
Higher chlorides (0.0786)
Higher sulphates (0.6142)
Lower residual sugar (2.30)

Interpretation:

This cluster represents a more acidic and structured wine profile characterized 
by higher acidity, chlorides, and sulphate concentrations.

Noise Group (-1)
1,424 wines
26.77% of dataset

Characteristics:

Intermediate values across most variables
Quality near the dataset average (5.65)
Not obviously extreme wines

Interesting finding:

DBSCAN did not identify the noise observations as extreme outliers. 
Instead, many wines appear to occupy transitional regions between the 
two dense clusters and therefore were not assigned to either cluster.

This is common when a dataset contains gradual transitions rather than 
sharply separated density regions.
"""

"""
FINAL MODEL

Algorithm: DBSCAN
eps = 1.3
min_samples = 10

Clusters Found: 2
Noise Points: 1,424
Noise Percentage: 26.77%

Silhouette Score: 0.1505
Davies-Bouldin Index: 3.3824
"""

# =========================================================
# 9. DBSCAN PCA Visualization
# =========================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=wqrw["dbscan_cluster"],
    cmap="tab10",
    alpha=0.7
)

plt.title("DBSCAN - PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(scatter)

plt.show()

"""
From the PCA plot alone, DBSCAN looks exactly like the metrics suggested.

- PCA Interpretation

The PCA visualization shows:

A large blue group (noise, -1) spread throughout the center of the feature space.
A brown cluster (0) concentrated toward the left side.
A cyan cluster (1) concentrated toward the right side.

However, there is substantial overlap between all three groups.

Most Important Observation

The noise points are not isolated on the edges of the plot.

Instead:

Noise points sit between the two clusters.


This supports what we saw in the profile table:

Noise not equal extreme outliers

Rather:

Noise = transitional wines


that do not belong strongly to either dense region.

- Cluster 0

Characteristics:

Higher acidity
Higher volatile acidity
Higher chlorides
Higher sulphates
Lower residual sugar

This resembles the red-dominated group identified by:

GMM
Agglomerative
K-Means

- Cluster 1

Characteristics:

Lower volatile acidity
Lower chlorides
Higher alcohol
Higher quality

This resembles the white-dominated / higher-quality population found previously.


- PCA Conclusion

PCA reveals two modest density-based clusters surrounded by a substantial 
number of transitional observations. Unlike K-Means, GMM, and Agglomerative 
Clustering, DBSCAN does not force all wines into clusters and instead 
classifies approximately 26.8% of observations as noise. The identified 
clusters correspond to two chemically distinct wine populations, while 
the noise observations occupy overlapping regions between those populations.

What to expect next:

If PCA is telling the truth, then:

t-SNE should show two dense islands with a large boundary region of noise.
UMAP should show two major populations with noise scattered around their borders.

If t-SNE and UMAP confirm that pattern, then the final DBSCAN conclusion becomes:

DBSCAN functioned more effectively as an outlier and transitional-observation 
detector than as a primary clustering algorithm for the Wine Quality dataset.
"""

# =========================================================
# 10. DBSCAN t-SNE Visualization
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
    c=wqrw["dbscan_cluster"],
    cmap="tab10",
    alpha=0.7
)

plt.title("DBSCAN - t-SNE")

plt.show()

"""
The t-SNE plot is actually much stronger than the PCA plot.

- t-SNE Interpretation

What I see:

Cluster 0 (brown)

Forms a fairly compact group on the left side.
Occupies a distinct region of the embedding.
Relatively little overlap with Cluster 1.

This supports the profile interpretation:

Higher acidity
Higher volatile acidity
Higher chlorides
Higher sulphates
Lower residual sugar

Cluster 1 (cyan)

Forms the large dense population on the right.
Occupies most of the embedding space.
Contains the higher-quality wines identified in the profile table.

Characteristics:

Lower volatile acidity
Lower chlorides
Higher alcohol
Higher quality

Noise Points (-1)

The most important finding:

The blue noise observations are not randomly scattered everywhere.

Instead:

Many occur around the boundaries of the two clusters.
Many sit between Cluster 0 and Cluster 1.
Others are embedded within the larger structures.

This confirms what we suspected from the profile table:

DBSCAN is not primarily detecting extreme outliers.

Instead:

DBSCAN is identifying wines that do not belong strongly to either dense population.

This is a subtle but important distinction.

- Comparison with PCA

PCA Suggested:

Moderate separation
Considerable overlap

t-SNE

Shows:

Two recognizable dense populations
Boundary region between them

The nonlinear structure is much clearer.

- Report:
    
The t-SNE visualization revealed two primary density-based wine populations 
separated by a substantial transition region. Unlike traditional partitioning 
algorithms, DBSCAN classified many wines located between the major dense 
regions as noise observations rather than forcing them into clusters. 
The visualization suggests that the majority of noise points represent 
transitional wines occupying overlapping areas of feature space rather than 
extreme outliers.
"""

# =========================================================
# 11. DBSCAN UMAP Visualization
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
    c=wqrw["dbscan_cluster"],
    cmap="tab10",
    alpha=0.7
)

plt.title("DBSCAN - UMAP")

plt.show()

"""
The UMAP plot confirms the DBSCAN story much more clearly than either PCA or t-SNE.

- UMAP Interpretation

Cluster 0 (brown)

Forms a compact, dense group on the left.
Clearly separated from the larger cluster.
Represents the higher-acidity wine population identified in the profile table.

Characteristics:

Higher fixed acidity
Higher volatile acidity
Higher chlorides
Higher sulphates
Lower residual sugar

Cluster 1 (cyan)

Forms a large, dense region on the right.
Contains the majority of clustered observations.
Represents the higher-quality wine population.

Characteristics:

Lower volatile acidity
Lower chlorides
Higher alcohol
Higher quality

Noise Points (-1)

This is the most important DBSCAN finding.

Unlike PCA:

Noise points are not scattered randomly. Instead:

Many surround the edges of the two clusters.
Many fill sparse regions between the clusters.
A few appear as isolated pockets.

This suggests:
DBSCAN is functioning primarily as a boundary detector rather than a pure 
outlier detector.

The noise wines are often transitional observations that do not belong strongly to either dense population.
"""

# =========================================================
# Overall DBSCAN Interpretation
# =========================================================

"""
Overall DBSCAN Interpretation

The combination of:

Metrics
Silhouette Score: 0.1505
Davies-Bouldin Index: 3.3824

Cluster Sizes
Cluster 0 = 834 wines
Cluster 1 = 3,062 wines
Noise     = 1,424 wines

PCA => Moderate overlap.

t-SNE => Two major dense populations emerge.

UMAP +. Two well-defined dense regions separated by a large set of border observations.

supports the following conclusion:

DBSCAN successfully identified two dense wine populations but did not provide 
cluster separation as strong as K-Means, GMM, or Agglomerative Clustering. 
Its primary contribution was the identification of approximately 27% of wines 
as transitional or sparsely distributed observations rather than assigning 
every wine to a cluster.
"""

# =========================================================
# Export Model Summary
# =========================================================

dbscan_summary = pd.DataFrame({
    "algorithm": ["DBSCAN"],
    "final_model": ["eps=1.3 min_samples=10"],
    "silhouette_score": [
        silhouette_score(
            X_scaled,
            wqrw["dbscan_cluster"]
        )
    ],
    "davies_bouldin_index": [
        davies_bouldin_score(
            X_scaled,
            wqrw["dbscan_cluster"]
        )
    ],
    "stability_ari": [np.nan],
    "cluster_count": [2],
    "noise_pct": [noise_pct]
})

dbscan_summary.to_csv(
    output_dir / "dbscan_summary.csv",
    index=False
)

# =========================================================
# DBSCAN Key Insights
# =========================================================

"""
DBSCAN Key Insights
KEY INSIGHTS

• Multiple combinations of eps and min_samples were evaluated.

• The best-performing DBSCAN configuration used:

  eps = 1.3
  min_samples = 10

• The final model identified two density-based clusters and
  classified 1,424 observations (26.77% of the dataset) as noise.

• Cluster 0 represented a more acidic and structured wine profile
  characterized by higher fixed acidity, volatile acidity,
  chlorides, and sulphates.

• Cluster 1 represented the largest dense wine population and
  exhibited lower volatile acidity, lower chlorides, higher
  alcohol content, and the highest average quality score.

• Unlike K-Means, GMM, and Agglomerative Clustering, DBSCAN did not
  force all observations into clusters.

• PCA showed moderate overlap between groups.

• t-SNE revealed two recognizable dense wine populations with a
  substantial transition region between them.

• UMAP showed two clearly separated dense regions surrounded by
  boundary observations and sparse pockets.

• Many of the observations classified as noise were not extreme
  outliers but instead occupied transitional regions between the
  major wine populations.

• DBSCAN functioned more effectively as a detector of sparse and
  transitional observations than as a primary clustering method
  for this dataset.

FINAL MODEL

Algorithm: DBSCAN
eps = 1.3
min_samples = 10

Clusters Found: 2
Noise Points: 1,424
Noise Percentage: 26.77%

Silhouette Score: 0.1505
Davies-Bouldin Index: 3.3824
"""
