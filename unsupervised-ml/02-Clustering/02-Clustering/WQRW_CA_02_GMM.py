# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October 21, 2026
Status:     Completed

Description

Gaussian Mixture Models (GMM) (probabilistic clusters)

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
# 4. GMM Evaluation
# =========================================================

from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

gmm_results = []

component_range = range(2, 21)

for n in component_range:

    gmm = GaussianMixture(
        n_components=n,
        covariance_type="full",
        random_state=RANDOM_STATE
    )

    labels = gmm.fit_predict(X_scaled)

    gmm_results.append(
        {
            "components": n,
            "silhouette_score": silhouette_score(
                X_scaled,
                labels
            ),
            "davies_bouldin_index": davies_bouldin_score(
                X_scaled,
                labels
            ),
            "bic": gmm.bic(X_scaled),
            "aic": gmm.aic(X_scaled)
        }
    )

gmm_results_df = pd.DataFrame(gmm_results)

print("\n----- GMM EVALUATION -----")
print(gmm_results_df.round(4))

# =========================================================
# 5. GMM Evaluation Plots
# =========================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(24, 5)
)

# Plot silhouette scores.

axes[0].plot(
    gmm_results_df["components"],
    gmm_results_df["silhouette_score"],
    marker="o"
)

axes[0].set_title("Silhouette Score")
axes[0].set_xlabel("Components")

# Plot Davies-Bouldin index.

axes[1].plot(
    gmm_results_df["components"],
    gmm_results_df["davies_bouldin_index"],
    marker="o"
)

axes[1].set_title("Davies-Bouldin Index")
axes[1].set_xlabel("Components")

# Plot BIC values.

axes[2].plot(
    gmm_results_df["components"],
    gmm_results_df["bic"],
    marker="o"
)

axes[2].set_title("BIC")
axes[2].set_xlabel("Components")

# Plot AIC values.

axes[3].plot(
    gmm_results_df["components"],
    gmm_results_df["aic"],
    marker="o"
)

axes[3].set_title("AIC")
axes[3].set_xlabel("Components")

plt.tight_layout()

plt.show()

"""
This is actually a very interesting result, because GMM is not clearly 
outperforming K-Means on your dataset.

Initial Interpretation

1. Silhouette Score (Higher is Better)

Best result:

2 components = 0.2454


Compare to K-Means:

k = 2 = 0.2704
k = 4 = 0.2428


GMM's best silhouette is basically comparable to your final K-Means solution.

2. Davies-Bouldin Index (Lower is Better)

Best result:

6 components = 2.5543


Compare to K-Means:

k = 4 = 1.4844


This is a huge difference.

K-Means produced much more compact and separated clusters.

3. BIC

Lower is better.

Your BIC minimum appears around:

13 components = 104545.83


After that it starts increasing.

This suggests:

The statistical model fit improves until about 13 components, then begins overfitting.

4. AIC

AIC keeps decreasing:

20 components = 95973.44

That is common because AIC tends to favor more complex models.

I would trust BIC more for component selection.

What this tells us
K-Means
4 clusters
Silhouette = 0.2428
DB = 1.4844

GMM

Best silhouette:    2 components     0.2454
Best BIC:           13 components    105563.0999
Best DB:            6 component      2.5543


There is no strong consensus here.

Unlike K-Means where all metrics pointed toward 4 clusters, GMM's metrics 
disagree substantially.
"""

# =========================================================
# 6. Covariance Type Comparison
# =========================================================

from sklearn.mixture import GaussianMixture

covariance_types = [
    "full",
    "diag",
    "spherical"
]

covariance_results = []

# =========================================================
# Individual Component
# =========================================================

best_components = 13

for cov_type in covariance_types:

    gmm = GaussianMixture(
        n_components=best_components,
        covariance_type=cov_type,
        random_state=RANDOM_STATE
    )

    gmm.fit(X_scaled)

    covariance_results.append(
        {
            "covariance_type": cov_type,
            "bic": gmm.bic(X_scaled),
            "aic": gmm.aic(X_scaled)
        }
    )

covariance_df = pd.DataFrame(
    covariance_results
)

print("\n----- COVARIANCE COMPARISON -----")
print(
    covariance_df.sort_values("bic")
)

# =========================================================
# Covariance Comparison
# =========================================================

component_options = [2, 6, 13]

covariance_types = [
    "full",
    "diag",
    "spherical"
]

comparison_results = []

for n_components in component_options:

    for cov_type in covariance_types:

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=cov_type,
            random_state=RANDOM_STATE
        )

        labels = gmm.fit_predict(X_scaled)

        comparison_results.append(
            {
                "components": n_components,
                "covariance_type": cov_type,
                "silhouette_score": silhouette_score(
                    X_scaled,
                    labels
                ),
                "davies_bouldin_index": davies_bouldin_score(
                    X_scaled,
                    labels
                ),
                "bic": gmm.bic(X_scaled),
                "aic": gmm.aic(X_scaled)
            }
        )

comparison_df = pd.DataFrame(
    comparison_results
)

print("\n----- GMM COMPARISON -----")

print(
    comparison_df.sort_values(
        by="bic"
    )
)

"""
1. 2 Components: diag covariance

2. 6 Components: full covariance
    
3. 13 Components:
    
Excellent. This gives us a very clear answer.

Covariance Type Winner
Covariance Type	BIC	AICfull	104,545.83	97,881.08
diag	120,163.70	118,203.09
spherical	132,801.18	131,695.87

Both BIC and AIC strongly favor:

Conclusion => covariance_type = "full"

The difference is massive:

full vs diag       ≈ 15,600 BIC points
full vs spherical  ≈ 28,300 BIC points

That's not a close contest.

Interpretation

A Gaussian Mixture Model with:

covariance_type="full" allows:
different variances for each feature
correlations between features
ellipsoidal cluster shapes

Meanwhile:

- diagnol assumes features are independent.
- spherical assumes round clusters similar to K-Means.

Your wine data clearly prefers the more flexible representation.
"""

"""
This comparison confirms that there is no single GMM winner across every metric. 
More importantly, after testing covariance types, the earlier statement that 
six components had the best Davies-Bouldin score is no longer true.

What each metric selects
- Best Silhouette: 2 components, diag covariance, 0.2516
- Best Davies-Bouldin: 2 components, spherical covariance, 1.6765
- Best BIC: 13 components, full covariance, 104,545.83
- Best AIC among these candidates: 13 components, full covariance, 97,881.08

Practical interpretation
- 2-diag gives the clearest separation.
- 2-spherical gives the best compactness/separation balance according to Davies-Bouldin.
- 13-full gives the best probabilistic fit, but its silhouette of 0.0194 
indicates substantial overlap among the 13 assigned groups.
- 6-full is an intermediate-complexity candidate, but it does not win any metric 
in the expanded comparison.
"""

# =========================================================
# 7. GMM Candidate Stability Comparison
# =========================================================

from sklearn.metrics import adjusted_rand_score

# Define candidate GMM configurations.

gmm_candidates = [
    {
        "model": "2_diag",
        "components": 2,
        "covariance_type": "diag"
    },
    {
        "model": "6_full",
        "components": 6,
        "covariance_type": "full"
    },
    {
        "model": "13_full",
        "components": 13,
        "covariance_type": "full"
    }
]

# Define random seeds for repeated model fitting.

seeds = [1, 11, 21, 31, 41]

stability_results = []

# Evaluate assignment stability for each candidate.

for candidate in gmm_candidates:

    candidate_labels = []

    for seed in seeds:

        gmm = GaussianMixture(
            n_components=candidate["components"],
            covariance_type=candidate["covariance_type"],
            n_init=5,
            random_state=seed
        )

        labels = gmm.fit_predict(X_scaled)

        candidate_labels.append(labels)

    # Compare every run against the first run.

    ari_scores = [
        adjusted_rand_score(
            candidate_labels[0],
            candidate_labels[i]
        )
        for i in range(1, len(candidate_labels))
    ]

    stability_results.append(
        {
            "model": candidate["model"],
            "components": candidate["components"],
            "covariance_type": candidate["covariance_type"],
            "minimum_ari": np.min(ari_scores),
            "average_ari": np.mean(ari_scores),
            "maximum_ari": np.max(ari_scores)
        }
    )

# Create and display the stability table.

stability_df = pd.DataFrame(stability_results)

print("\n----- GMM STABILITY COMPARISON -----")
print(
    stability_df
    .sort_values("average_ari", ascending=False)
    .round(4)
)

"""
----- GMM STABILITY COMPARISON -----
     model  components covariance_type  minimum_ari  average_ari  maximum_ari
0   2_diag           2            diag       0.9923       0.9946       1.0000
1   6_full           6            full       0.7764       0.8491       0.9902
2  13_full          13            full       0.4684       0.5635       0.6077

1. Interpretation:

2-diag: average ARI = 0.9946
- Exceptionally stable. Different random seeds produce nearly identical assignments.

6-full: average ARI = 0.8491
- Reasonably stable, although the minimum ARI of 0.7764 shows some sensitivity to initialization.

13-full: average ARI = 0.5635
- Unstable. The model changes substantially across random seeds, despite having the lowest BIC.

2. Best final candidate

n_components = 2
covariance_type = "diag"

The justification is strong:

Highest tested silhouette score: 0.2516
Nearly perfect stability: 0.9946 average ARI
Simple and interpretable
Better separation than the 6- and 13-component candidates

Although 13-full has the best BIC, the low silhouette and weak stability suggest 
that the model is fitting statistical detail rather than discovering reproducible, 
clearly separated clusters.

3. What about 6-full?

6-full remains a useful secondary candidate because it provides more detailed 
segmentation and reasonable stability. However, it does not beat 2-diag on 
silhouette, Davies-Bouldin, stability, BIC, or AIC individually. It is therefore 
a compromise rather than a metric winner.

4. Report-ready conclusion

Three GMM configurations were evaluated to balance cluster separation, 
probabilistic fit, complexity, and stability. The 13-component full-covariance 
model achieved the lowest BIC but produced weak separation and unstable 
assignments, with a silhouette score of 0.0194 and an average ARI of 0.5635. 
The 6-component full-covariance model demonstrated reasonable stability with 
an average ARI of 0.8491, but its silhouette score remained low at 0.0642. 
The 2-component diagonal-covariance model achieved the strongest overall balance, 
producing the highest silhouette score of 0.2516 and exceptional stability with 
an average ARI of 0.9946. Therefore, the 2-component diagonal-covariance model 
was selected as the final GMM solution.
"""

"""
This is actually the key decision point of the GMM section.

Choose 2 components if your goal is...
Strongest cluster separation
Highest stability
Easy interpretation
Following the same philosophy as K-Means

Your metrics:

Silhouette = 0.2516  (best)
Average ARI = 0.9946 (best)


Advantages:

✅ Extremely stable

✅ Easy to explain

✅ Clearer clusters

✅ Better reproducibility

✅ More likely to produce clean PCA/t-SNE/UMAP plots

Disadvantages:

❌ Very coarse segmentation

❌ May hide meaningful subgroups

❌ Less detailed than K-Means (which found 4 clusters)
"""

"""
Choose 6 components if your goal is...
Discover more nuanced wine profiles
Balance detail and stability
Create richer cluster interpretations

Your metrics:

Silhouette = 0.0642
Average ARI = 0.8491

Advantages:

✅ More detailed segmentation

✅ Reasonably stable

✅ May uncover specialty wine groups

✅ More comparable to K-Means' 4-cluster solution

Disadvantages:

❌ Much weaker separation

❌ More overlap between clusters

❌ Harder to explain

❌ Not the best silhouette or stability
"""

"""
Why I'm hesitant about 13

Your stability results basically rule it out:

Average ARI = 0.5635
Minimum ARI = 0.4684


That means:

Change the random seed and cluster assignments change substantially.
"""

"""
Because the project requirements emphasize:

Silhouette
Davies-Bouldin
Stability
PCA
t-SNE
UMAP

I would personally choose:

n_components = 2
covariance_type = "diag"

because it wins the metrics that the project explicitly cares about.

However...

There is a reasonable argument for 6

If your goal is to compare clustering methods and discover wine subtypes, you 
could justify:

n_components = 6
covariance_type = "full"

because:

Stability is still acceptable (0.8491)
More informative than only 2 groups
Produces richer profile tables
My recommendation


Use 2-diag as the official final GMM model.

Discussion section note:

A six-component solution was also evaluated and offered greater segmentation detail, 
but the two-component diagonal covariance model demonstrated substantially stronger 
cluster separation and stability.

That gives you the strongest methodological justification while still showing you 
explored alternative solutions.
"""

# =========================================================
# 8. Final GMM Models
# =========================================================

# Fit the official two-component GMM model.

gmm_2 = GaussianMixture(
    n_components=2,
    covariance_type="diag",
    n_init=5,
    random_state=RANDOM_STATE
)

wqrw["gmm_2_cluster"] = gmm_2.fit_predict(X_scaled)

# Fit the exploratory six-component GMM model.

gmm_6 = GaussianMixture(
    n_components=6,
    covariance_type="full",
    n_init=5,
    random_state=RANDOM_STATE
)

wqrw["gmm_6_cluster"] = gmm_6.fit_predict(X_scaled)

# Display cluster sizes for the two-component model.

print("\n----- GMM (2 COMPONENTS) CLUSTER SIZES -----")

print(
    wqrw["gmm_2_cluster"]
    .value_counts()
    .sort_index()
)

# Display metrics for the two-component model.

print("\n----- GMM (2 COMPONENTS) METRICS -----")

print(
    f"Silhouette Score: "
    f"{silhouette_score(X_scaled, wqrw['gmm_2_cluster']):.4f}"
)

print(
    f"Davies-Bouldin Index: "
    f"{davies_bouldin_score(X_scaled, wqrw['gmm_2_cluster']):.4f}"
)

print(
    f"BIC: {gmm_2.bic(X_scaled):.4f}"
)

print(
    f"AIC: {gmm_2.aic(X_scaled):.4f}"
)

# Display cluster sizes for the six-component model.

print("\n----- GMM (6 COMPONENTS) CLUSTER SIZES -----")

print(
    wqrw["gmm_6_cluster"]
    .value_counts()
    .sort_index()
)

# Display metrics for the six-component model.

print("\n----- GMM (6 COMPONENTS) METRICS -----")

print(
    f"Silhouette Score: "
    f"{silhouette_score(X_scaled, wqrw['gmm_6_cluster']):.4f}"
)

print(
    f"Davies-Bouldin Index: "
    f"{davies_bouldin_score(X_scaled, wqrw['gmm_6_cluster']):.4f}"
)

print(
    f"BIC: {gmm_6.bic(X_scaled):.4f}"
)

print(
    f"AIC: {gmm_6.aic(X_scaled):.4f}"
)


"""
best_components = 2
best_covariance = "diag"

----- FINAL GMM CLUSTER SIZES -----
gmm_cluster
0    1584
1    3736
Name: count, dtype: int64

----- FINAL GMM METRICS -----
Silhouette Score: 0.2516
Davies-Bouldin Index: 1.7782
BIC: 146649.5631
AIC: 146353.4978


best_components = 6
best_covariance = "full"

----- FINAL GMM CLUSTER SIZES -----
gmm_cluster
0     566
1    1259
2     193
3    1790
4    1015
5     497
Name: count, dtype: int64

----- FINAL GMM METRICS -----
Silhouette Score: 0.0642
Davies-Bouldin Index: 2.5543
BIC: 107624.9429
AIC: 104552.4432
"""


"""
1. What that means

2 components provide:

Much clearer separation
More compact clusters
Almost perfectly reproducible assignments
Easier interpretation
A coarse division into two broad wine populations

6 components provide:

A much better likelihood-based statistical fit
More detailed segmentation
Six potential wine profiles instead of two
Greater overlap and more sensitivity to initialization

2. Recommendation

Keep 2-diag as the official final GMM model because your project’s stated 
framework prioritizes:

Silhouette Score
Davies-Bouldin Index
Stability
Visual separation

Keep 6-full as an exploratory alternative because BIC and AIC strongly prefer it. 
This gives you a useful finding:

The two-component model prioritized separation, stability, and interpretability, 
whereas the six-component model provided a stronger likelihood-based fit and greater 
segmentation detail at the cost of substantial cluster overlap.
"""

# =========================================================
# 9. GMM Cluster Profile Tables
# =========================================================

# Calculate average feature values for the two-component GMM model.

gmm_2_profile = (
    wqrw
    .groupby("gmm_2_cluster")
    [feature_cols + ["quality"]]
    .mean()
    .round(4)
)

print("\n----- GMM (2 COMPONENTS) PROFILE -----")
print(gmm_2_profile)

gmm_2_profile.to_csv(
    output_dir / "gmm_2_cluster_profile.csv"
)

# Calculate average feature values for the six-component GMM model.

gmm_6_profile = (
    wqrw
    .groupby("gmm_6_cluster")
    [feature_cols + ["quality"]]
    .mean()
    .round(4)
)

print("\n----- GMM (6 COMPONENTS) PROFILE -----")
print(gmm_6_profile)

gmm_6_profile.to_csv(
    output_dir / "gmm_6_cluster_profile.csv"
)

# Calculate cluster membership counts for the two-component model.

gmm_2_sizes = (
    wqrw["gmm_2_cluster"]
    .value_counts()
    .sort_index()
)

print("\n----- GMM (2 COMPONENTS) CLUSTER SIZES -----")
print(gmm_2_sizes)

# Calculate cluster membership counts for the six-component model.

gmm_6_sizes = (
    wqrw["gmm_6_cluster"]
    .value_counts()
    .sort_index()
)

print("\n----- GMM (6 COMPONENTS) CLUSTER SIZES -----")
print(gmm_6_sizes)

# Save cluster size summaries.

gmm_2_sizes.to_csv(
    output_dir / "gmm_2_cluster_sizes.csv"
)

gmm_6_sizes.to_csv(
    output_dir / "gmm_6_cluster_sizes.csv"
)

"""
These profile tables show that both GMM solutions are meaningful, but they 
answer different questions. The 2-component model finds two broad populations, 
while the 6-component model divides those populations into more specific wine profiles.

Two-component interpretation
Cluster 0: Higher-acidity, structured wines
1,584 wines, about 29.8%
Higher fixed acidity: 8.13
Higher volatile acidity: 0.51
Higher chlorides: 0.09
Higher sulphates: 0.64
Lower residual sugar: 2.60
Lower average quality: 5.56

This group represents wines with a stronger acidity and mineral structure.

Cluster 1: Sweeter, sulfur-managed wines
3,736 wines, about 70.2%
Lower volatile acidity: 0.27
Higher residual sugar: 6.09
Higher free sulfur dioxide: 35.35
Higher total sulfur dioxide: 138.13
Slightly higher alcohol: 10.62
Higher average quality: 5.90

This group represents wines with more residual sugar, greater sulfur dioxide levels, and somewhat higher average quality.

**Conclusion for two components**

The two-component solution creates a clean and interpretable broad split. 
Based only on the numeric profile, one group is more acidic and structured, 
while the other is sweeter and more sulfur-managed.

Do not label these as “red” and “white” yet. The results strongly suggest that 
possibility, but you should verify it using the excluded type column.
"""

"""
Six-component interpretation
Cluster 0: Very sweet, high-sulfur, low-alcohol wines
566 wines
Highest residual sugar: 10.58
Highest free sulfur dioxide: 51.40
Highest total sulfur dioxide: 186.60
Lowest alcohol: 9.39
Average quality: 5.46

This is a very distinctive profile and appears to be a meaningful subgroup.

Cluster 1: Dry, higher-alcohol wines
1,259 wines
Lowest residual sugar: 1.47
Low chlorides: 0.04
Alcohol: 10.97
Average quality: 5.92

This group represents a relatively dry and higher-alcohol profile.

Cluster 2: High-chloride, high-sulphate specialty group
193 wines, about 3.6%
Highest chlorides: 0.14
Highest sulphates: 0.85
Highest fixed acidity: 8.91
Highest alcohol: 11.01
Average quality: 5.80

Although this cluster is small, its profile is clearly different. It may represent 
a legitimate specialty group rather than random over-segmentation.

Cluster 3: Sweet, higher-quality wines
1,790 wines
Residual sugar: 7.70
Low volatile acidity: 0.27
Alcohol: 10.79
Highest average quality: 6.03

This is the largest cluster and has the highest average quality.

Cluster 4: Structured, low-sulfur wines
1,015 wines
Fixed acidity: 8.29
Volatile acidity: 0.51
Total sulfur dioxide: 39.43, the lowest value
Sulphates: 0.64
Average quality: 5.67

This resembles the structured population from the 2-component solution, but 
with relatively low sulfur dioxide.

Cluster 5: Moderate-sugar, lower-quality wines
497 wines
Residual sugar: 4.87
Volatile acidity: 0.50
Total sulfur dioxide: 123.12
Lowest average quality: 5.27
Alcohol: 9.98

This group appears to combine moderate sugar with higher volatile acidity and 
lower average quality.

What the profiles change

The 6-component model is not simply producing meaningless fragments. Several 
groups have recognizable differences, especially:

Cluster 0: very sweet and high sulfur
Cluster 1: dry and higher alcohol
Cluster 2: high chloride and high sulphate
Cluster 3: sweet and higher quality
Cluster 4: structured and low sulfur
Cluster 5: moderate sugar and lower quality

However, the poor silhouette score means these profiles probably overlap 
considerably in the full 11-dimensional feature space.
"""

"""
Best project conclusion
- 2-diag is the official final GMM solution because it has much stronger separation 
and stability.
- 6-full is a meaningful exploratory solution because it offers detailed 
segmentation and much better BIC/AIC.
- The profile results support retaining both in the discussion, but not treating 
both as equally strong final models.
"""

# Before visualization, the most useful next check is whether the clusters 
# correspond to wine type:
    
# Compare two-component clusters with wine type.

gmm_2_type_table = pd.crosstab(
    wqrw["gmm_2_cluster"],
    wqrw["type"],
    margins=True
)

gmm_2_type_percent = pd.crosstab(
    wqrw["gmm_2_cluster"],
    wqrw["type"],
    normalize="index"
).mul(100).round(2)

print("\n----- GMM (2 COMPONENTS) TYPE COUNTS -----")
print(gmm_2_type_table)

print("\n----- GMM (2 COMPONENTS) TYPE PERCENTAGES -----")
print(gmm_2_type_percent)

"""
The two-component diagonal-covariance Gaussian Mixture Model closely reproduced 
the known distinction between red and white wines, despite wine type being 
excluded from model training. Cluster 0 contained 84.72% red wines and 
captured approximately 98.75% of all red observations, while Cluster 1 
contained 99.54% white wines and captured approximately 93.89% of all white 
observations. After aligning cluster labels with wine type, the unsupervised 
assignments agreed with the known wine categories for approximately 95.13% of 
observations. This result indicates that the physicochemical measurements 
contain a strong latent structure corresponding to wine type.


This is a major finding. The two-component GMM independently recovered the 
red/white wine distinction with very high purity, even though type was
excluded from the clustering features.

Interpretation
GMM Cluster 0
1,584 wines
84.72% red
15.28% white
Contains 1,342 of the dataset’s 1,359 red wines, approximately 98.75% of all red wines

This can reasonably be described as a predominantly red-wine cluster, not a perfectly red-only cluster.

GMM Cluster 1
3,736 wines
99.54% white
0.46% red
Contains 3,719 of the dataset’s 3,961 white wines, approximately 93.89% of all white wines

This is an exceptionally pure white-wine cluster.

Overall agreement with wine type

The diagonal entries are the correctly aligned observations:

1,342 red wines in Cluster 0
3,719 white wines in Cluster 1


So the cluster-to-type agreement is:

(1,342 + 3,719) / 5,320 = 95.13%


Report-ready interpretation

The two-component diagonal-covariance Gaussian Mixture Model closely reproduced 
the known distinction between red and white wines, despite wine type being 
excluded from model training. Cluster 0 contained 84.72% red wines and 
captured approximately 98.75% of all red observations, while Cluster 1 
contained 99.54% white wines and captured approximately 93.89% of all white 
observations. After aligning cluster labels with wine type, the unsupervised 
assignments agreed with the known wine categories for approximately 95.13% of 
observations. This result indicates that the physicochemical measurements 
contain a strong latent structure corresponding to wine type.

Important wording

Avoid saying:
- GMM classified wine type with 95.13% accuracy.

GMM was not trained as a classifier, and the type labels were not used during fitting. 

Better language is:
- The GMM cluster assignments showed 95.13% agreement with the known wine types after cluster-label alignment.

This result strongly supports keeping 2-diag as the official GMM solution. 
It is highly stable, reasonably separated, interpretable, and independently 
identifies a meaningful known structure in the data.
"""

# =========================================================
# 10. PCA Visualization
# =========================================================

# Project data into two principal components.

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# Plot two-component GMM clusters.

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=wqrw["gmm_2_cluster"],
    cmap="tab10",
    alpha=0.7
)

plt.title("GMM (2 Components) - PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(scatter)

plt.show()

# Plot six-component GMM clusters.

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=wqrw["gmm_6_cluster"],
    cmap="tab10",
    alpha=0.7
)

plt.title("GMM (6 Components) - PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.colorbar(scatter)

plt.show()

# =========================================================
# 11. t-SNE Visualization
# =========================================================

# Project data using t-SNE.

from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=RANDOM_STATE
)

X_tsne = tsne.fit_transform(X_scaled)

# Plot two-component GMM clusters.

plt.figure(figsize=(8, 6))

plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=wqrw["gmm_2_cluster"],
    cmap="tab10",
    alpha=0.7
)

plt.title("GMM (2 Components) - t-SNE")

plt.show()

# Plot six-component GMM clusters.

plt.figure(figsize=(8, 6))

plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=wqrw["gmm_6_cluster"],
    cmap="tab10",
    alpha=0.7
)

plt.title("GMM (6 Components) - t-SNE")

plt.show()

# =========================================================
# 11. t-SNE Visualization
# =========================================================

# Project data using t-SNE.

from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=RANDOM_STATE
)

X_tsne = tsne.fit_transform(X_scaled)

# Plot two-component GMM clusters.

plt.figure(figsize=(8, 6))

plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=wqrw["gmm_2_cluster"],
    cmap="tab10",
    alpha=0.7
)

plt.title("GMM (2 Components) - t-SNE")

plt.show()

# Plot six-component GMM clusters.

plt.figure(figsize=(8, 6))

plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=wqrw["gmm_6_cluster"],
    cmap="tab10",
    alpha=0.7
)

plt.title("GMM (6 Components) - t-SNE")

plt.show()

"""
These t-SNE plots are actually more informative than the PCA plots.

GMM (2 Components) ✅

The t-SNE visualization is extremely convincing.

What we see:

One large cluster on the left
One large cluster on the right
Very little mixing
Clear boundary region

This matches perfectly with:

Silhouette = 0.2516
ARI = 0.9946
95.13% agreement with wine type


The visualization and metrics tell the same story.

Interpretation

The two-component GMM identified two highly distinct latent wine populations. t-SNE revealed strong separation between clusters, supporting the silhouette score of 0.2516 and the exceptional stability of the clustering solution (ARI = 0.9946). Comparison with the excluded wine type variable showed approximately 95% agreement between cluster assignments and known wine categories.

GMM (6 Components) ✅

This plot is actually better than I expected.

What I notice:

Left side

The large yellow group is fairly distinct.

Bottom center

The green group dominates a separate region.

Right side

The pink group occupies much of the right-half.

Top right

The blue group forms another reasonably coherent area.

Other groups

Purple and cyan appear to fill transition regions between larger structures.

Important observation

The 6-component model does not look like random noise.

Honestly, when I saw:

Silhouette = 0.0642


I expected a complete mess.

Instead, t-SNE suggests:

there are still larger super-groups
but the six clusters subdivide those large groups

So the GMM isn't making nonsense clusters.

It is finding:

Population A
 ├── Subgroup 1
 ├── Subgroup 2

Population B
 ├── Subgroup 3
 ├── Subgroup 4
 ├── Subgroup 5
 ├── Subgroup 6


which aligns with the profile-table interpretations.

If I were writing the report
Two-Component Solution

The two-component Gaussian Mixture Model produced the most stable and interpretable clustering solution. PCA and t-SNE visualizations revealed two clearly separated groups that corresponded closely to the known wine type categories despite type being excluded from model training.

Six-Component Solution

The six-component model produced substantially finer segmentation and identified several chemically distinct wine profiles characterized by differences in residual sugar, sulfur dioxide concentrations, acidity, alcohol content, and sulphate levels. Although cluster overlap reduced quantitative separation metrics, t-SNE visualizations demonstrated meaningful substructure within the broader wine populations.

**
My final GMM conclusion
Official GMM model
2 Components
Diagonal Covariance
**


Reasons:

Best silhouette
Best stability
Near-perfect reproducibility
Strong alignment with known wine types
Clean t-SNE separation
Exploratory GMM model
6 Components
Full Covariance


Reasons:

Better BIC/AIC
More detailed wine segmentation
Meaningful chemical profiles
Visible substructure in t-SNE
"""

# =========================================================
# Export Model Summary
# =========================================================

gmm_summary = pd.DataFrame({
    "algorithm": ["GMM"],
    "final_model": ["2 Components (diag)"],
    "silhouette_score": [
        silhouette_score(
            X_scaled,
            wqrw["gmm_2_cluster"]
        )
    ],
    "davies_bouldin_index": [
        davies_bouldin_score(
            X_scaled,
            wqrw["gmm_2_cluster"]
        )
    ],
    "stability_ari": [
        stability_df.loc[
            stability_df["model"] == "2_diag",
            "average_ari"
        ].iloc[0]
    ],
    "cluster_count": [2],
    "noise_pct": [0.0]
})

gmm_summary.to_csv(
    output_dir / "gmm_summary.csv",
    index=False
)

# =========================================================
# Gaussian Mixture Models (GMM) Summary
# =========================================================

"""
KEY INSIGHTS

• Gaussian Mixture Models were evaluated using multiple component counts and
  covariance structures.

• Component counts of 2, 6, and 13 were investigated as candidate solutions.

• Full covariance produced the strongest likelihood-based model fit, while
  diagonal covariance produced the strongest cluster separation and
  stability.

• Although the 13-component full-covariance model achieved the lowest BIC,
  it exhibited poor cluster separation and weak stability.

• The 6-component full-covariance model identified several chemically
  distinct wine profiles but displayed considerable overlap between groups.

• The 2-component diagonal-covariance model achieved the highest silhouette
  score (0.2516) and exceptional stability (ARI = 0.9946).

• PCA and t-SNE visualizations revealed two clearly separated latent wine
  populations.

• Comparison against the excluded wine type variable demonstrated that the
  two-component solution reproduced the red-versus-white wine distinction
  with approximately 95.13% agreement.

• The final GMM solution was selected because it provided the strongest
  balance of separation, stability, interpretability, and reproducibility.

FINAL MODEL

Algorithm: Gaussian Mixture Model
Components: 2
Covariance Type: Diagonal
Silhouette Score: 0.2516
Davies-Bouldin Index: 1.7782
Average ARI: 0.9946
Agreement with Wine Type: 95.13%
"""
