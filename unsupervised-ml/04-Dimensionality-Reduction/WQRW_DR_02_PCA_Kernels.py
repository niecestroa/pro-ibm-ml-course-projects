# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October __, 2026
Status:     Completed

Description

For Kernel PCA, the goal is not:
- "Can I reduce dimensions?"
PCA already answered that.

The question becomes:
- "Does a nonlinear transformation reveal structure that standard PCA misses?"

Kernel Principal Component Analysis (Kernel PCA)

This analysis applies Kernel PCA using an RBF (Radial Basis Function)
kernel to investigate nonlinear structure within the Wine Quality
dataset.

Unlike standard PCA, Kernel PCA projects data into a higher-dimensional
feature space before performing dimensionality reduction, allowing
curved and non-Euclidean relationships to be represented.

Analysis Includes

    • 2D Kernel PCA embedding
    • Wine-type visualization
    • Quality visualization
    • PCA vs Kernel PCA comparison
    • Nonlinear structure evaluation

Objectives

    1. Determine whether nonlinear dimensionality reduction
       reveals structure not visible in PCA.

    2. Evaluate separation between red and white wine
       populations.

    3. Investigate whether wine quality exhibits improved
       separation under a nonlinear transformation.

    4. Compare the interpretability and visualization
       capabilities of PCA and Kernel PCA.

Key Findings

    • Kernel PCA produced stronger visual separation between
      red and white wine populations than standard PCA.

    • The embedding revealed curved manifold structure that
      was not visible in the linear PCA representation.

    • Wine quality remained highly overlapping, indicating
      that quality is not the dominant source of variation
      in the dataset.

    • Kernel PCA confirmed the primary findings from PCA,
      while revealing additional nonlinear structure within
      the wine chemistry space.
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
# 4. Kernel PCA (RBF)
# =========================================================

from sklearn.decomposition import KernelPCA

kpca = KernelPCA(
    n_components=2,
    kernel="rbf",
    gamma=0.1,
    random_state=RANDOM_STATE
)

X_kpca = kpca.fit_transform(
    X_scaled
)

kpca_df = pd.DataFrame(
    {
        "KPC1": X_kpca[:, 0],
        "KPC2": X_kpca[:, 1],
        "type": wqrw["type"],
        "quality": wqrw["quality"]
    }
)

print(
    "\n----- KERNEL PCA SHAPE -----"
)

print(
    kpca_df.shape
)

# =========================================================
# 5. Kernel PCA by Wine Type
# =========================================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=kpca_df,
    x="KPC1",
    y="KPC2",
    hue="type",
    alpha=0.6
)

plt.title(
    "Kernel PCA Projection by Wine Type"
)

plt.tight_layout()

plt.show()

# =========================================================
# 6. Kernel PCA by Quality
# =========================================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=kpca_df,
    x="KPC1",
    y="KPC2",
    hue="quality",
    palette="viridis",
    alpha=0.6
)

plt.title(
    "Kernel PCA Projection by Quality"
)

plt.tight_layout()

plt.show()

"""
Interpretation

Kernel PCA suggests there are nonlinear relationships among:

sulfur chemistry
acidity
residual sugar
alcohol

that standard PCA cannot fully capture.

The separation appears cleaner because Kernel PCA maps observations into a higher-dimensional feature space before performing PCA.
"""
# =========================================================
# 7. PCA vs Kernel PCA
# =========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,6)
)

# PCA

pca_df = pd.read_csv(
    output_dir / "pca_2d_projection.csv"
)

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

# Kernel PCA

sns.scatterplot(
    data=kpca_df,
    x="KPC1",
    y="KPC2",
    hue="type",
    alpha=0.6,
    ax=axes[1]
)

axes[1].set_title(
    "Kernel PCA (RBF)"
)

plt.tight_layout()

plt.show()

# =========================================================
# Kernel PCA Results Interpretation
# =========================================================

"""
Does Kernel PCA separate red/white wines better?

Does Kernel PCA reveal curved structure?

Does Kernel PCA reveal new quality patterns?

Is PCA already sufficient?
"""

"""
PCA

Clear separation between red and white wines.
Separation occurs mostly along PC1.
Significant overlap in the center.
Structure appears approximately linear.

Kernel PCA (RBF)

Separation between red and white wines becomes noticeably stronger.
The embedding forms a curved manifold rather than a simple linear cloud.
Red wines occupy a more concentrated region in the lower portion of the embedding.
White wines occupy the upper and right portions of the embedding.
The overlap region is smaller than with standard PCA.
Updated Interpretation

Updated Interpretation from before in part 6

PCA

✅ Easier to interpret
✅ Provides explained variance
✅ Provides loadings
✅ Clearly separates red and white wines

Kernel PCA

✅ Stronger red/white separation
✅ Reveals nonlinear structure
✅ Better visualization of wine-type populations
❌ No explained variance
❌ No component loadings
❌ More difficult to interpret
"""

# =========================================================
# Kernel PCA Key Insights
# =========================================================

"""
KEY INSIGHTS

• Kernel PCA successfully captured nonlinear relationships
  within the wine chemistry data.

• Compared with standard PCA, Kernel PCA produced stronger
  separation between red and white wine populations.

• The embedding revealed a curved manifold structure that
  was not visible in the linear PCA representation.

• Red wines formed a more concentrated subgroup in the
  lower portion of the embedding.

• White wines occupied a broader nonlinear manifold and
  remained clearly separated from most red wines.

• Wine quality continued to exhibit substantial overlap,
  indicating that quality is not the dominant source of
  variation in the dataset.

• The agreement between PCA and Kernel PCA confirms that
  wine type is the strongest underlying structure within
  the Wine Quality dataset.

• Kernel PCA suggests that portions of the wine chemistry
  space contain nonlinear relationships that are not fully
  captured by standard PCA.

Comparison With PCA

Winner in Interpretability                  => PCA
Winner in Variance Analysis                 => PCA
Winner in Feature Understanding             => PCA
Winner: Visual Separation                   => Kernel PCA
Winner: Nonlinear Structure Detection       => Kernel PCA

Final Takeaway

PCA explained the data.
Kernel PCA visualized the data.
PCA provided interpretable principal components and variance analysis, while 
Kernel PCA revealed stronger nonlinear separation between red and white wines.

Together, the two methods indicate that wine type is the
dominant underlying structure in the dataset and that this
structure contains both linear and nonlinear components.
"""

# =========================================================
# Exports
# =========================================================

output_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\04-Dimensionality-Reduction\04-dimensionality-results"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

kpca_df.to_csv(
    output_dir / "kernel_pca_projection.csv",
    index=False
)

kpca_summary = pd.DataFrame(
    {
        "Method": ["Kernel PCA"],
        "Kernel": ["RBF"],
        "Gamma": [0.1],
        "Components": [2]
    }
)

kpca_summary.to_csv(
    output_dir / "kernel_pca_summary.csv",
    index=False
)

# =========================================================
# PCA vs Kernel PCA Summary
# =========================================================

kpca_comparison = pd.DataFrame(
    {
        "Method": [
            "PCA",
            "Kernel PCA"
        ],
        "Interpretability": [
            "High",
            "Low"
        ],
        "Variance Explained": [
            "Yes",
            "No"
        ],
        "Feature Loadings": [
            "Yes",
            "No"
        ],
        "Visual Separation": [
            "Good",
            "Excellent"
        ],
        "Nonlinear Structure": [
            "No",
            "Yes"
        ]
    }
)

kpca_comparison.to_csv(
    output_dir / "pca_vs_kernel_pca.csv",
    index=False
)
