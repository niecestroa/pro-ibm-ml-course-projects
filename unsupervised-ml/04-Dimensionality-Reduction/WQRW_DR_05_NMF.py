# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October __, 2026
Status:     Completed

Description

Non-Negative Matrix Factorization (NMF)

This analysis applies Non-Negative Matrix Factorization (NMF)
to the Wine Quality dataset to identify latent additive
chemistry components and compare NMF against other
dimensionality-reduction techniques.

Unlike PCA, NMF constrains all components and coefficients
to non-negative values, producing interpretable additive
factors that describe how wine chemistry variables combine
to form underlying wine profiles.

Analysis Includes

    • NMF decomposition
    • Component loadings analysis
    • Component heatmap visualization
    • Feature contribution ranking
    • Wine-type visualization
    • Quality visualization
    • NMF vs PCA comparison
    • NMF vs t-SNE comparison
    • NMF vs UMAP comparison
    • Dimensionality-reduction method comparison

Objectives

    1. Identify interpretable latent chemistry components
       underlying the Wine Quality dataset.

    2. Determine which wine chemistry variables contribute
       most strongly to each latent factor.

    3. Evaluate whether wine populations separate within
       NMF factor space.

    4. Compare NMF against PCA, Kernel PCA, t-SNE,
       and UMAP.

Part 1: NMF Decomposition

NMF identified two major latent chemistry components.

Component 1

    • Alcohol
    • pH
    • Fixed Acidity
    • Volatile Acidity
    • Sulphates

This component represents a chemistry profile associated
with alcohol content, acidity structure, and wine body.

Component 2

    • Total Sulfur Dioxide
    • Density
    • Citric Acid
    • Free Sulfur Dioxide
    • Residual Sugar

This component represents a chemistry profile associated
with sweetness, preservation chemistry, density, and sulfur
compounds.

Part 2: Method Comparison

Comparison against PCA, Kernel PCA, t-SNE, and UMAP showed
that NMF serves a different purpose than most other
dimensionality-reduction techniques.

Where PCA, Kernel PCA, t-SNE, and UMAP focus primarily on
representation and separation of wine populations, NMF
focuses on identifying latent additive chemistry factors.

Method Comparison Summary

    PCA
        • Best variance explanation
        • Most interpretable mathematically

    Kernel PCA
        • Revealed nonlinear chemistry structure

    t-SNE
        • Strongest local separation

    UMAP
        • Best overall visualization
        • Strongest balance of separation and structure

    NMF
        • Best latent-factor interpretation
        • Strongest chemistry component analysis

Key Findings

    • NMF identified two highly interpretable latent wine
      chemistry components.

    • Component 1 was dominated by alcohol and acidity
      characteristics.

    • Component 2 was dominated by sulfur chemistry,
      density, and residual sugar.

    • Wine quality remained highly overlapping within
      latent-factor space.

    • Wine-type separation was weaker than PCA,
      Kernel PCA, t-SNE, and UMAP.

    • NMF provided the strongest chemistry interpretation
      among all dimensionality-reduction methods.

Final Takeaway

    PCA explained variance.

    Kernel PCA revealed nonlinear structure.

    t-SNE maximized local separation.

    UMAP provided the strongest overall visualization.

    NMF revealed the latent chemistry components underlying
    wine composition.

Together, these methods provided a comprehensive view of
both the structure and composition of the Wine Quality
dataset.
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
# Output Directory
# =========================================================

output_dir = Path(
    r"C:/Users/aniec/Desktop/ibm-ml-project/unsupervised-ml/04-Dimensionality-Reduction/04-dimensionality-results"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# 3. Prepare Non-negative Data
# =========================================================

from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()

X_nmf = mm.fit_transform(
    wqrw[feature_cols]
)

# =========================================================
# Part 1 
# =========================================================

n_components = 2

from sklearn.decomposition import NMF

nmf = NMF(
    n_components=2,
    random_state=RANDOM_STATE
)

W = nmf.fit_transform(
    X_nmf
)

H = nmf.components_

# =========================================================
# Component DataFrame
# =========================================================

nmf_df = pd.DataFrame(
    {
        "Component1": W[:,0],
        "Component2": W[:,1],
        "type": wqrw["type"],
        "quality": wqrw["quality"]
    }
)

print(
    "\n----- NMF SHAPE -----"
)

print(
    nmf_df.shape
)

# =========================================================
# Feature Contributions
# Most important NMF output.
# =========================================================

component_loadings = pd.DataFrame(
    H,
    columns=feature_cols,
    index=[
        "Component1",
        "Component2"
    ]
)

print(
    component_loadings.round(4)
)

# Heatmap
plt.figure(figsize=(12,6))

sns.heatmap(
    component_loadings,
    cmap="YlGnBu",
    annot=True,
    fmt=".2f"
)

plt.title(
    "NMF Component Loadings"
)

plt.tight_layout()

plt.show()

# =========================================================
# NMF by Wine Type
# =========================================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=nmf_df,
    x="Component1",
    y="Component2",
    hue="type",
    alpha=0.6
)

plt.title(
    "NMF Components by Wine Type"
)

plt.tight_layout()

plt.show()

# NMF by Quality

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=nmf_df,
    x="Component1",
    y="Component2",
    hue="quality",
    palette="viridis",
    alpha=0.6
)

plt.title(
    "NMF Components by Quality"
)

plt.tight_layout()

plt.show()

# =========================================================
# Component 1 Drivers
# =========================================================

comp1 = (
    component_loadings
    .loc["Component1"]
    .sort_values(
        ascending=False
    )
)

print(
    "\n----- COMPONENT 1 DRIVERS -----"
)

print(
    comp1.round(4)
)

# =========================================================
# Component 2 Drivers
# =========================================================

comp2 = (
    component_loadings
    .loc["Component2"]
    .sort_values(
        ascending=False
    )
)

print(
    "\n----- COMPONENT 2 DRIVERS -----"
)

print(
    comp2.round(4)
)

# =========================================================
# Exports
# =========================================================

component_loadings.to_csv(
    output_dir / "nmf_component_loadings.csv"
)

nmf_df.to_csv(
    output_dir / "nmf_projection.csv",
    index=False
)

nmf_summary = pd.DataFrame(
    {
        "Method": ["NMF"],
        "Components": [2]
    }
)

nmf_summary.to_csv(
    output_dir / "nmf_summary.csv",
    index=False
)

# =========================================================
# Part 1: NMF Decomposition & Interpretation 
# NMF Interpretation
# =========================================================

"""
Results Interpretation

Component Analysis

NMF decomposed wine chemistry into two latent additive
components.

Component 1

• Strongly influenced by alcohol, pH, fixed acidity,
  volatile acidity, and sulphates.

• Residual sugar contributed almost nothing to this
  component.

• Represents a chemistry axis associated with alcohol
  content, acidity structure, and wine body.

• Can be interpreted as a Dry/Structured Chemistry
  Component.

Component 2

• Strongly influenced by total sulfur dioxide, density,
  citric acid, residual sugar, and free sulfur dioxide.

• Alcohol contributed almost nothing to this component.

• Represents a chemistry axis associated with sweetness,
  density, preservation chemistry, and sulfur compounds.

• Can be interpreted as a Sweetness/Preservation Chemistry
  Component.

Wine Type Projection

• Red and white wines occupied similar regions within the
  latent-factor space.

• Some tendency toward separation was visible, but the
  distinction was weaker than with PCA, Kernel PCA,
  t-SNE, or UMAP.

Quality Projection

• Quality ratings remained heavily intermixed.

• No distinct low-dimensional quality groups emerged.

• The result reinforces previous findings that quality is
  not the primary source of variation within the dataset.

Conclusion

NMF explained the composition of wine chemistry more
effectively than it separated wine populations. The method
revealed interpretable chemistry factors while confirming
that wine quality does not form naturally distinct groups.
"""

# =========================================================
# Part 1: NMF Decomposition & Interpretation 
# NMF Key Insights
# =========================================================

"""
KEY INSIGHTS

• NMF identified two chemically meaningful latent
  components.

• Component 1 captured alcohol, acidity, pH, and sulphate
  structure.

• Component 2 captured sulfur chemistry, density,
  residual sugar, and preservation characteristics.

• The resulting factors describe how wine chemistry is
  assembled rather than how wines separate.

• Wine quality remained broadly distributed throughout
  the latent-factor space.

• Wine type was visible but less strongly separated than
  in other dimensionality-reduction techniques.

FINAL TAKEAWAY

NMF provided highly interpretable additive chemistry
components but was less effective as a visualization method
than PCA, Kernel PCA, t-SNE, or UMAP.
"""

# =========================================================
# Part 2: Method Comparison 
# =========================================================

# =========================================================
# 11. NMF vs PCA
# =========================================================

pca_df = pd.read_csv(
    output_dir / "pca_2d_projection.csv"
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,6)
)

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="type",
    alpha=0.5,
    ax=axes[0]
)

axes[0].set_title(
    "PCA"
)

sns.scatterplot(
    data=nmf_df,
    x="Component1",
    y="Component2",
    hue="type",
    alpha=0.5,
    ax=axes[1]
)

axes[1].set_title(
    "NMF"
)

plt.tight_layout()

plt.show()

# =========================================================
# 12. NMF vs UMAP
# =========================================================

umap_df = pd.read_csv(
    output_dir / "umap_projection.csv"
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,6)
)

sns.scatterplot(
    data=nmf_df,
    x="Component1",
    y="Component2",
    hue="type",
    alpha=0.5,
    ax=axes[0]
)

axes[0].set_title(
    "NMF"
)

sns.scatterplot(
    data=umap_df,
    x="UMAP1",
    y="UMAP2",
    hue="type",
    alpha=0.5,
    ax=axes[1]
)

axes[1].set_title(
    "UMAP"
)

plt.tight_layout()

plt.show()

# =========================================================
# 13. t-SNE vs NMF
# =========================================================

tsne_df = pd.read_csv(
    output_dir / "tsne_projection.csv"
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,6)
)

sns.scatterplot(
    data=tsne_df,
    x="TSNE1",
    y="TSNE2",
    hue="type",
    alpha=0.5,
    ax=axes[0]
)

axes[0].set_title(
    "t-SNE"
)

sns.scatterplot(
    data=nmf_df,
    x="Component1",
    y="Component2",
    hue="type",
    alpha=0.5,
    ax=axes[1]
)

axes[1].set_title(
    "NMF"
)

plt.tight_layout()

plt.show()

# =========================================================
# 14. Method Comparison
# =========================================================

dr_comparison = pd.DataFrame(
    {
        "Method": [
            "PCA",
            "Kernel PCA",
            "t-SNE",
            "UMAP",
            "NMF"
        ],

        "Interpretability": [
            "High",
            "Low",
            "Low",
            "Medium",
            "High"
        ],

        "Variance Explained": [
            "Yes",
            "No",
            "No",
            "No",
            "No"
        ],

        "Feature Interpretation": [
            "Excellent",
            "Poor",
            "Poor",
            "Poor",
            "Excellent"
        ],

        "Visual Separation": [
            "Good",
            "Excellent",
            "Outstanding",
            "Outstanding",
            "Moderate"
        ],

        "Primary Strength": [
            "Variance",
            "Nonlinear Structure",
            "Local Separation",
            "Global + Local Structure",
            "Latent Factors"
        ]
    }
)

print(
    dr_comparison
)

# =========================================================
# Part 2 Exports
# =========================================================

dr_comparison.to_csv(
    output_dir / "nmf_method_comparison.csv",
    index=False
)

nmf_method_rankings = pd.DataFrame(
    {
        "Category": [
            "Variance Explanation",
            "Feature Interpretation",
            "Nonlinear Structure",
            "Local Separation",
            "Global Structure",
            "Overall Visualization"
        ],

        "Best Method": [
            "PCA",
            "NMF",
            "Kernel PCA",
            "t-SNE",
            "UMAP",
            "UMAP"
        ]
    }
)

nmf_method_rankings.to_csv(
    output_dir / "dr_method_rankings.csv",
    index=False
)

# =========================================================
# Part 2: Method Comparison 
# Interpretation
# =========================================================

"""
Results Interpretation

PCA vs NMF

• PCA produced clearer separation between red and white
  wine populations.

• PCA captured the major directions of variance within the
  wine chemistry data.

• NMF produced weaker separation but generated more
  interpretable additive chemistry factors.

• PCA explained how the data varies, while NMF explained
  how the data is composed.

UMAP vs NMF

• UMAP produced substantially stronger separation between
  red and white wines.

• Distinct wine populations were immediately visible within
  the UMAP embedding.

• NMF formed a continuous latent-factor space rather than
  clearly separated groups.

• UMAP emphasized population structure, while NMF
  emphasized chemistry composition.

t-SNE vs NMF

• t-SNE produced the strongest local cluster separation of
  all dimensionality-reduction methods evaluated.

• Red and white wines formed highly distinct regions within
  the t-SNE embedding.

• NMF showed considerable overlap between wine types.

• The contrast highlights the different goals of the two
  methods:

      t-SNE
          → Visualization and cluster discovery

      NMF
          → Latent-factor interpretation

Method Comparison

• PCA retained the highest interpretability through
  explained variance and component loadings.

• Kernel PCA improved separation through nonlinear
  transformation.

• t-SNE maximized local cluster separation.

• UMAP balanced separation and global structure
  preservation.

• NMF provided the strongest latent chemistry
  interpretation.

Conclusion

The dimensionality-reduction methods each emphasized a
different aspect of the Wine Quality dataset. Together,
they provided complementary views of both wine chemistry
composition and wine population structure
"""

# =========================================================
# # Part 2: Method Comparison 
# Key Insights
# =========================================================

"""
KEY INSIGHTS

• PCA was the strongest variance-explanation method.

• Kernel PCA revealed nonlinear chemistry structure not
  visible in standard PCA.

• t-SNE produced the strongest local separation between
  wine populations.

• UMAP produced outstanding separation while maintaining
  a cleaner global organization of the data.

• NMF identified interpretable latent chemistry factors
  underlying wine composition.

• Wine type remained the dominant source of variation
  across all dimensionality-reduction methods.

• Quality ratings remained highly overlapping regardless
  of the method applied.

• The agreement among PCA, Kernel PCA, t-SNE, UMAP, and
  NMF strengthens confidence that the primary structure
  of the dataset is the distinction between red and white
  wines.

FINAL TAKEAWAY

No single dimensionality-reduction technique was best at
every task.

PCA explained variance.

Kernel PCA revealed nonlinear relationships.

t-SNE maximized local separation.

UMAP provided the strongest overall visualization.

NMF explained the latent chemistry components underlying
the Wine Quality dataset.
"""

# =========================================================
# Overall NMF Interpretation
# =========================================================

"""
OVERALL INTERPRETATION

NMF approached dimensionality reduction from a fundamentally
different perspective than PCA, Kernel PCA, t-SNE, and UMAP.

Rather than maximizing variance retention or cluster
separation, NMF identified additive chemistry components
that describe how wines are constructed from underlying
chemical characteristics.

The method revealed two major latent factors:

• Alcohol and Acidity Component

• Sulfur, Density, and Preservation Component

Although visual separation was weaker than other methods,
NMF provided the most interpretable chemistry-based view of
the dataset.

FINAL TAKEAWAY

PCA explained variance.

Kernel PCA revealed nonlinear structure.

t-SNE maximized local separation.

UMAP provided the strongest overall visualization.

NMF revealed the latent chemistry components underlying the
wine populations.

Together, these methods provided a comprehensive view of
both the structure and composition of the Wine Quality
dataset.
"""

# =========================================================
# Overall NMF Key Insights
# =========================================================

"""
KEY INSIGHTS

• NMF identified two latent additive chemistry components
  underlying the Wine Quality dataset.

• Component 1 represented alcohol and acidity-related
  chemistry.

• Component 2 represented sulfur dioxide, density,
  residual sugar, and preservation-related chemistry.

• NMF provided the strongest chemistry interpretation among
  all dimensionality-reduction methods evaluated.

• Red and white wines remained visible within the latent
  factors but exhibited weaker separation than in PCA,
  Kernel PCA, t-SNE, and UMAP.

• Quality remained highly overlapping and did not form
  naturally distinct groups.

OVERALL INTERPRETATION

NMF revealed how wine chemistry variables combine to form
latent additive factors, providing a complementary view of
the dataset. Rather than maximizing separation, NMF focused
on explaining the composition of wine chemistry.

FINAL TAKEAWAY

PCA explained variance.

Kernel PCA revealed nonlinear structure.

t-SNE maximized local separation.

UMAP provided the strongest overall visualization.

NMF revealed the latent chemistry components underlying the
Wine Quality dataset.

Together, these methods provided a comprehensive view of
both the structure and composition of wine chemistry.
"""
