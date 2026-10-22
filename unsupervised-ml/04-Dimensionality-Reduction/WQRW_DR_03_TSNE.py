# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October __, 2026
Status:     Completed

Description

t-Distributed Stochastic Neighbor Embedding (t-SNE)

This analysis applies t-SNE to the Wine Quality dataset to
visualize local neighborhood structure and investigate whether
nonlinear embedding reveals stronger cluster separation than
PCA or Kernel PCA.

Unlike PCA and Kernel PCA, t-SNE prioritizes preservation of
local relationships between observations rather than variance
explanation. This makes it particularly useful for exploratory
analysis and cluster visualization.

Analysis Includes

    • 2D t-SNE embedding
    • Wine-type visualization
    • Quality visualization
    • PCA vs t-SNE comparison
    • Kernel PCA vs t-SNE comparison
    • Dimensionality-reduction method evaluation

Objectives

    1. Investigate local cluster structure within the wine
       chemistry data.

    2. Evaluate separation between red and white wine
       populations.

    3. Assess whether wine quality forms identifiable
       low-dimensional patterns.

    4. Compare t-SNE against PCA and Kernel PCA.

Key Findings

    • t-SNE produced the strongest visual separation among
      all dimensionality-reduction techniques evaluated.

    • Red and white wine populations formed largely distinct
      regions of the embedding.

    • Quality remained highly overlapping despite the
      nonlinear transformation.

    • The dominant dataset structure remained wine type.
"""

# =========================================================
# 0. Imports & Configuration
# =========================================================

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
# Output Directory
# =========================================================

output_dir = Path(
    r"C:/Users/aniec/Desktop/ibm-ml-project/unsupervised-ml/04-Dimensionality-Reduction/04-dimensionality-results"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# Part 1: t-SNE Embedding

# =========================================================
# 4. t-SNE Model
# =========================================================

from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=RANDOM_STATE
)

X_tsne = tsne.fit_transform(
    X_scaled
)

tsne_df = pd.DataFrame(
    {
        "TSNE1": X_tsne[:,0],
        "TSNE2": X_tsne[:,1],
        "type": wqrw["type"],
        "quality": wqrw["quality"]
    }
)

print(
    "\n----- t-SNE SHAPE -----"
)

print(
    tsne_df.shape
)

# =========================================================
# 5. t-SNE Projection by Wine Type
# =========================================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=tsne_df,
    x="TSNE1",
    y="TSNE2",
    hue="type",
    alpha=0.6
)

plt.title(
    "t-SNE Projection by Wine Type"
)

plt.tight_layout()

plt.show()

# =========================================================
# 6. t-SNE Projection by Quality
# =========================================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=tsne_df,
    x="TSNE1",
    y="TSNE2",
    hue="quality",
    palette="viridis",
    alpha=0.6
)

plt.title(
    "t-SNE Projection by Quality"
)

plt.tight_layout()

plt.show()

# =========================================================
# Part 1 Results Interpretation
# =========================================================

"""
These are excellent t-SNE results.
Honestly, t-SNE just became best visualization so far.
"""

"""
1. t-SNE by Wine Type

The separation is dramatic:

Left cluster => Mostly Red Wines

Right cluster => Mostly White Wines

Unlike PCA and even Kernel PCA:

PCA         => Some overlap
Kernel PCA  => Better separation
t-SNE       => Near-complete separation

There are only a few boundary observations between the groups.

Interpretation

t-SNE preserved local neighborhoods exceptionally well.

The result suggests:

Red Wines and White Wines occupy distinct regions of wine-chemistry space.

This reinforces what PCA, Kernel PCA, and Ward Clustering all found:

Wine Type
=
Dominant Structure

2. t-SNE by Quality

The clustering structure remains visible.

However:

Quality labels remain distributed throughout the embedding.

The colors are mixed across clusters.

Interpretation

Even after strong nonlinear transformation:

Quality does not form isolated groups.

Instead:

Quality appears continuous rather than clustered.

This is consistent with everything observed so far:

PCA

Kernel PCA

Ward Linkage

Comparison So Far
PCA
Pros
✓ Variance explained
✓ Interpretable
✓ Feature loadings

Cons
✗ Moderate overlap

Kernel PCA
Pros
✓ Nonlinear structure
✓ Better separation

Cons
✗ Difficult interpretation

t-SNE
Pros
✓ Best visual separation
✓ Strong local structure
✓ Clear wine-type clusters

Cons
✗ No variance explanation
✗ No feature interpretation
✗ Distances not globally meaningful
"""

# =========================================================
# Key Insights
# =========================================================

"""
KEY INSIGHTS

• t-SNE produced the strongest visual separation among all dimensionality-reduction 
methods evaluated thus far.

• Red and white wines formed largely distinct regions within the embedding.

• The resulting structure suggests that wine type is the dominant source of 
variation in the dataset.

• Wine quality remained widely distributed across the embedding and did not 
form clearly separated groups.

• The agreement between PCA, Kernel PCA, t-SNE, and Ward Clustering provides 
strong evidence that the primary structure of the dataset is the distinction 
between red and white wines.

• t-SNE successfully preserved local relationships and revealed fine-grained 
cluster structure not visible in PCA.

Final Takeaway

PCA explained the variation.

Kernel PCA revealed nonlinear structure.

t-SNE produced the clearest visualization of the underlying
wine populations.

All three methods independently identified wine type as the
dominant organizational structure of the Wine Quality dataset.
"""

# =========================================================
# Part 1 Exports
# =========================================================

tsne_df.to_csv(
    output_dir / "tsne_projection.csv",
    index=False
)

tsne_summary = pd.DataFrame(
    {
        "Method": ["t-SNE"],
        "Perplexity": [30],
        "Dimensions": [2]
    }
)

tsne_summary.to_csv(
    output_dir / "tsne_summary.csv",
    index=False
)

# =========================================================
# Part 2: Method Comparisons
# =========================================================

# =========================================================
# 7. PCA vs t-SNE
# =========================================================

pca_df = pd.read_csv(
    output_dir / "pca_2d_projection.csv"
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,6)
)

# PCA

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="type",
    alpha=0.6,
    ax=axes[0]
)

axes[0].set_title(
    "PCA"
)

# t-SNE

sns.scatterplot(
    data=tsne_df,
    x="TSNE1",
    y="TSNE2",
    hue="type",
    alpha=0.6,
    ax=axes[1]
)

axes[1].set_title(
    "t-SNE"
)

plt.tight_layout()

plt.show()

# =========================================================
# 8. Kernel PCA vs t-SNE
# =========================================================

kpca_df = pd.read_csv(
    output_dir / "kernel_pca_projection.csv"
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,6)
)

# Kernel PCA

sns.scatterplot(
    data=kpca_df,
    x="KPC1",
    y="KPC2",
    hue="type",
    alpha=0.6,
    ax=axes[0]
)

axes[0].set_title(
    "Kernel PCA (RBF)"
)

# t-SNE

sns.scatterplot(
    data=tsne_df,
    x="TSNE1",
    y="TSNE2",
    hue="type",
    alpha=0.6,
    ax=axes[1]
)

axes[1].set_title(
    "t-SNE"
)

plt.tight_layout()

plt.show()

# =========================================================
# 9. Method Comparison
# =========================================================

dr_comparison = pd.DataFrame(
    {
        "Method": [
            "PCA",
            "Kernel PCA",
            "t-SNE"
        ],

        "Interpretability": [
            "High",
            "Low",
            "Low"
        ],

        "Variance Explained": [
            "Yes",
            "No",
            "No"
        ],

        "Feature Loadings": [
            "Yes",
            "No",
            "No"
        ],

        "Visual Separation": [
            "Good",
            "Excellent",
            "Outstanding"
        ],

        "Nonlinear Structure": [
            "No",
            "Yes",
            "Yes"
        ]
    }
)

print(
    dr_comparison
)

# =========================================================
# Part 2 Results Interpretation
# =========================================================

"""
1. PCA vs t-SNE

PCA

Preserved nearly 50% of variance using only two components.
Clearly separated red and white wines.
Maintained interpretability through loadings and explained variance.
Still exhibited substantial overlap between wine populations.

t-SNE

Produced dramatically stronger separation.
Nearly isolated red and white wines into distinct regions.
Preserved local neighborhood structure.
Revealed substructure within wine populations.

Conclusion

PCA explains the data.
t-SNE visualizes the data.

The tradeoff is:

Interpretability
    ↓
Visual Separation

2. Kernel PCA vs t-SNE

Kernel PCA

Improved separation relative to PCA.
Revealed nonlinear manifold structure.
Preserved a continuous embedding.

t-SNE

Produced the strongest cluster formation.
Sharpened boundaries between wine types.
Better highlighted local group structure.

Conclusion

Kernel PCA  => Better nonlinear representation

t-SNE       => Better cluster visualization

3. Dimensionality Reduction Ranking

Based on the results so far:

Category                    Winner
Explained Variance          PCA
Feature Interpretation      PCA
Component Loadings          PCA
Nonlinear Structure	Kernel  PCA
Wine-Type Separation        t-SNE
Cluster Visualization       t-SNE
"""

# =========================================================
# Part 2 Key Insights
# =========================================================

"""
KEY INSIGHTS

• PCA provided the most interpretable low-dimensional
  representation through explained variance and
  component loadings.

• Kernel PCA improved visual separation and revealed
  nonlinear manifold structure not fully captured by
  standard PCA.

• t-SNE produced the strongest separation between
  red and white wine populations.

• All dimensionality-reduction methods consistently
  identified wine type as the dominant source of
  variation within the dataset.

• Wine quality remained highly overlapping regardless
  of the dimensionality-reduction technique applied.

• The agreement between PCA, Kernel PCA, and t-SNE
  strengthens confidence that the primary structure
  of the Wine Quality dataset is the distinction
  between red and white wines.

FINAL TAKEAWAY

PCA provided the strongest interpretability, Kernel PCA
revealed nonlinear relationships, and t-SNE produced the
clearest visualization of wine population structure.
"""

# =========================================================
# Part 2 Exports
# =========================================================

dr_comparison.to_csv(
    output_dir / "dr_method_comparison.csv",
    index=False
)

tsne_comparison_summary = pd.DataFrame(
    {
        "Method": [
            "PCA",
            "Kernel PCA",
            "t-SNE"
        ],
        "Best Use": [
            "Interpretation",
            "Nonlinear Structure",
            "Visualization"
        ]
    }
)

tsne_comparison_summary.to_csv(
    output_dir / "dr_best_use_summary.csv",
    index=False
)
