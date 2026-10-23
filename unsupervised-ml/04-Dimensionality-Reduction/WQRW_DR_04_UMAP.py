# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October 22, 2026
Status:     Completed

Description

Uniform Manifold Approximation and Projection (UMAP)

This analysis applies UMAP to the Wine Quality dataset to
investigate low-dimensional manifold structure and compare
its performance against PCA, Kernel PCA, and t-SNE.

UMAP preserves both local and global relationships within
the data, often providing stronger visual separation than
PCA while maintaining more global structure than t-SNE.

Analysis Includes

    • 2D UMAP embedding
    • Wine-type visualization
    • Quality visualization
    • t-SNE vs UMAP comparison
    • PCA, Kernel PCA, t-SNE, and UMAP comparison
    • Dimensionality-reduction method evaluation

Objectives

    1. Visualize the underlying structure of wine chemistry
       data in a low-dimensional space.

    2. Evaluate separation between red and white wine
       populations.

    3. Investigate whether wine quality forms meaningful
       low-dimensional patterns.

    4. Compare UMAP against PCA, Kernel PCA, and t-SNE.

Key Findings

    • UMAP produced exceptionally strong separation between
      red and white wine populations.

    • Red wines formed a compact and highly concentrated
      cluster.

    • White wines formed a larger structured manifold with
      visible internal subgroup structure.

    • Wine quality remained highly overlapping throughout
      the embedding and did not form clearly separated
      groups.

    • UMAP achieved separation comparable to t-SNE while
      maintaining a cleaner and more organized global
      structure.

    • Multiple dimensionality-reduction techniques
      consistently identified wine type as the dominant
      source of variation within the dataset.

Method Comparison

    PCA
        • Highest interpretability
        • Explained variance
        • Feature loadings
        • Good separation

    Kernel PCA
        • Revealed nonlinear structure
        • Improved separation over PCA

    t-SNE
        • Outstanding local separation
        • Strong cluster visualization

    UMAP
        • Outstanding separation
        • Better global organization
        • Best overall balance between separation
          and structure preservation

Final Takeaway

    UMAP provided one of the strongest visualizations in the
    dimensionality-reduction section, combining excellent
    cluster separation with a highly organized embedding.
    Among the methods evaluated, UMAP offered the best
    overall balance between visualization quality, structure
    preservation, and interpretability of the resulting
    manifold.
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
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\04-Dimensionality-Reduction\04-dimensionality-results"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

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
# 4. UMAP Model
# =========================================================

import umap

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=RANDOM_STATE
)

X_umap = umap_model.fit_transform(
    X_scaled
)

umap_df = pd.DataFrame(
    {
        "UMAP1": X_umap[:,0],
        "UMAP2": X_umap[:,1],
        "type": wqrw["type"],
        "quality": wqrw["quality"]
    }
)

print(
    "\n----- UMAP SHAPE -----"
)

print(
    umap_df.shape
)

# =========================================================
# 5. UMAP Projection by Wine Type
# =========================================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=umap_df,
    x="UMAP1",
    y="UMAP2",
    hue="type",
    alpha=0.6
)

plt.title(
    "UMAP Projection by Wine Type"
)

plt.tight_layout()

plt.show()

# =========================================================
# 6. UMAP Projection by Quality
# =========================================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=umap_df,
    x="UMAP1",
    y="UMAP2",
    hue="quality",
    palette="viridis",
    alpha=0.6
)

plt.title(
    "UMAP Projection by Quality"
)

plt.tight_layout()

plt.show()

# =========================================================
# Part 1 Interpertation
# =========================================================

"""
Results Interpretation

Wine Type Projection

• UMAP produced exceptionally strong separation between red
  and white wine populations.

• Red wines formed a compact and well-defined cluster on the
  left side of the embedding.

• White wines formed a larger structured manifold on the
  right side of the embedding.

• Only a very small overlap region existed between the two
  wine types.

• The resulting separation was comparable to or stronger than
  the separation observed using t-SNE.

• UMAP preserved distinct substructure within both wine
  populations while maintaining clear global organization.

Quality Projection

• Quality classes remained heavily mixed throughout the
  embedding.

• No isolated quality-specific clusters emerged.

• Higher-quality and lower-quality wines appeared throughout
  both major wine populations.

• These results suggest that quality does not represent the
  dominant source of variation within the dataset.

Conclusion

UMAP successfully captured the dominant chemical structure of
the Wine Quality dataset and produced one of the clearest
representations of the underlying wine populations. The
embedding strongly supports previous findings that wine type
drives most of the observed variation.
"""

# =========================================================
# Part 1 Key Insights
# =========================================================

"""
KEY INSIGHTS

• UMAP produced extremely strong separation between red and
  white wine populations.

• Red wines formed a compact and highly concentrated region.

• White wines formed a larger manifold with visible internal
  structure.

• Wine quality remained broadly distributed across the
  embedding.

• The dominant pattern recovered by UMAP was wine type
  rather than wine quality.

• UMAP preserved both local neighborhood structure and
  larger global organization.

FINAL TAKEAWAY

UMAP provided one of the clearest visualizations of wine
population structure observed in the project and strongly
confirmed that wine type is the dominant source of variation.
"""

# =========================================================
# Part 1 Exports
# =========================================================

umap_df.to_csv(
    output_dir / "umap_projection.csv",
    index=False
)

umap_summary = pd.DataFrame(
    {
        "Method": ["UMAP"],
        "Neighbors": [15],
        "Min_Dist": [0.1],
        "Dimensions": [2]
    }
)

umap_summary.to_csv(
    output_dir / "umap_summary.csv",
    index=False
)

# =========================================================
# Part 2: UMAP Comparisons
# =========================================================

# =========================================================
# 7. t-SNE vs UMAP
# =========================================================

tsne_df = pd.read_csv(
    output_dir / "tsne_projection.csv"
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,6)
)

# t-SNE

sns.scatterplot(
    data=tsne_df,
    x="TSNE1",
    y="TSNE2",
    hue="type",
    alpha=0.6,
    ax=axes[0]
)

axes[0].set_title(
    "t-SNE"
)

# UMAP

sns.scatterplot(
    data=umap_df,
    x="UMAP1",
    y="UMAP2",
    hue="type",
    alpha=0.6,
    ax=axes[1]
)

axes[1].set_title(
    "UMAP"
)

plt.tight_layout()

plt.show()

# =========================================================
# 8. All Methods Comparison
# =========================================================

pca_df = pd.read_csv(
    output_dir / "pca_2d_projection.csv"
)

kpca_df = pd.read_csv(
    output_dir / "kernel_pca_projection.csv"
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14,12)
)

# PCA

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="type",
    alpha=0.5,
    ax=axes[0,0]
)

axes[0,0].set_title(
    "PCA"
)

# Kernel PCA

sns.scatterplot(
    data=kpca_df,
    x="KPC1",
    y="KPC2",
    hue="type",
    alpha=0.5,
    ax=axes[0,1]
)

axes[0,1].set_title(
    "Kernel PCA"
)

# t-SNE

sns.scatterplot(
    data=tsne_df,
    x="TSNE1",
    y="TSNE2",
    hue="type",
    alpha=0.5,
    ax=axes[1,0]
)

axes[1,0].set_title(
    "t-SNE"
)

# UMAP

sns.scatterplot(
    data=umap_df,
    x="UMAP1",
    y="UMAP2",
    hue="type",
    alpha=0.5,
    ax=axes[1,1]
)

axes[1,1].set_title(
    "UMAP"
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
            "t-SNE",
            "UMAP"
        ],

        "Interpretability": [
            "High",
            "Low",
            "Low",
            "Medium"
        ],

        "Variance Explained": [
            "Yes",
            "No",
            "No",
            "No"
        ],

        "Feature Loadings": [
            "Yes",
            "No",
            "No",
            "No"
        ],

        "Visual Separation": [
            "Good",
            "Excellent",
            "Outstanding",
            "Outstanding"
        ],

        "Nonlinear Structure": [
            "No",
            "Yes",
            "Yes",
            "Yes"
        ]
    }
)

print(dr_comparison)

# =========================================================
# Part 2 Interpretation
# =========================================================

"""
Part 2 Results Interpretation

1. PCA vs Kernel PCA vs t-SNE vs UMAP

PCA

Produced clear but overlapping separation between red and white wines.
Retained the highest interpretability through explained variance and component loadings.
Best method for understanding why structure exists.

Kernel PCA

Improved separation relative to PCA.
Revealed nonlinear structure in the wine chemistry data.
Reduced overlap but still maintained a continuous manifold.

t-SNE

Produced extremely strong local separation.
Created clear boundaries between red and white wine populations.
Revealed local neighborhood structure more effectively than PCA and Kernel PCA.

UMAP

Produced separation comparable to t-SNE.
Preserved clearer overall organization of the wine populations.
Generated the cleanest and most compact representation of red and white wines.
Maintained visible internal structure within the white-wine population.

t-SNE vs UMAP

t-SNE

Strengths
• Excellent local cluster separation
• Clear class boundaries
• Strong subgroup discovery

Limitations
• Global distances are less meaningful
• Overall shape can be difficult to interpret

UMAP

Strengths
• Excellent separation
• Better global structure preservation
• More compact and organized embedding
• Easier to interpret visually

Limitations
• Less directly interpretable than PCA

Conclusion

The progression across methods was:

PCA
    ↓
Kernel PCA
    ↓
t-SNE
    ↓
UMAP

in terms of visual separation and structure discovery.

PCA provided the strongest interpretability.

Kernel PCA revealed nonlinear relationships.

t-SNE generated the strongest local separation.

UMAP provided comparable separation while preserving a cleaner global organization of the dataset.
"""

# =========================================================
# Part 2 Key Insights
# =========================================================

"""
Part 2 Key Insights

KEY INSIGHTS

• PCA provided the most interpretable representation of
  wine chemistry through explained variance and feature
  loadings.

• Kernel PCA revealed nonlinear structure and improved
  separation between wine populations.

• t-SNE produced outstanding local separation and displayed
  clearly defined wine populations.

• UMAP matched the strong separation produced by t-SNE while
  maintaining a more organized global structure.

• All dimensionality-reduction techniques consistently
  identified wine type as the dominant source of variation.

• Red and white wines occupied distinct regions regardless
  of the embedding method applied.

• Quality remained highly overlapping across all methods,
  suggesting that quality contributes substantially less
  structure than wine type.

FINAL TAKEAWAY

PCA explained the structure, Kernel PCA revealed nonlinear
relationships, t-SNE highlighted local cluster separation,
and UMAP provided the strongest overall balance between
separation, structure preservation, and visualization.

"""

# =========================================================
# Overall UMAP Key Insights
# =========================================================

"""
Overall UMAP Key Insights
KEY INSIGHTS

• UMAP produced one of the strongest visual representations
  of the Wine Quality dataset.

• Red and white wines formed highly distinct and compact
  clusters.

• Internal structure remained visible within the larger
  white-wine population.

• Quality ratings did not form separate groups and remained
  widely distributed throughout the embedding.

• UMAP successfully preserved both local neighborhood
  structure and larger global organization.

• Multiple dimensionality-reduction techniques independently
  confirmed that wine type is the dominant source of
  variation within the dataset.

OVERALL INTERPRETATION

The Wine Quality dataset contains strong underlying
structure driven by wine type rather than quality.
UMAP provided a highly organized low-dimensional
representation of this structure and delivered one of the
best overall visualizations observed in the project.

FINAL TAKEAWAY

PCA was the most interpretable method, t-SNE produced the
strongest local separation, and UMAP provided the best
combination of separation, organization, and visualization
quality among all dimensionality-reduction techniques
evaluated.
"""