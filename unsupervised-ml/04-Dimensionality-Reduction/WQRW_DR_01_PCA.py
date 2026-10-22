"""
Author: Aaron Niecestro

Created:    October 19, 2026
Edited:     October 22, 2026
Status:     Completed

Description

4.1 Principal Component Analysis (PCA)

This analysis applies Principal Component Analysis (PCA) to the
combined Red and White Wine Quality dataset.

Dataset Summary

    • Observations: 5,320 wines
    • Features Used: 11 chemistry variables
    • Target Excluded: quality
    • Categories Available: red, white

Features Included

    • fixed acidity
    • volatile acidity
    • citric acid
    • residual sugar
    • chlorides
    • free sulfur dioxide
    • total sulfur dioxide
    • density
    • pH
    • sulphates
    • alcohol

Objectives

    1. Reduce dimensionality while preserving variance.

    2. Identify the dominant sources of variation in wine
       chemistry.

    3. Determine how many principal components are required
       to retain most of the information contained in the
       dataset.

    4. Investigate whether red and white wines naturally
       separate in PCA space.

    5. Explore whether wine quality produces meaningful
       structure within the principal-component projections.

Analysis Includes

    • Explained variance analysis
    • Cumulative variance analysis
    • Scree plot evaluation
    • PCA component loadings
    • Feature contribution analysis
    • PCA loadings heatmap
    • 2D PCA visualization by wine type
    • 2D PCA visualization by wine quality

Key Findings

    • PC1 and PC2 explained 49.67% of total variance.
    • Approximately 85% of variance was retained using six
      principal components.
    • Wine type was the dominant source of variation.
    • PCA clearly separated red and white wine populations.
    • Quality showed considerably less structure than wine
      type within PCA space.
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
# 4. Full PCA Model
# =========================================================

from sklearn.decomposition import PCA

pca = PCA()

pca.fit(X_scaled)

explained_variance = pd.DataFrame(
    {
        "Component":
            range(
                1,
                len(feature_cols)+1
            ),

        "Explained Variance":
            pca.explained_variance_ratio_,

        "Cumulative Variance":
            pca.explained_variance_ratio_
            .cumsum()
    }
)

print(
    explained_variance
    .round(4)
)

# =========================================================
# 5. Scree Plot
# =========================================================

plt.figure(figsize=(10,6))

sns.barplot(
    x=explained_variance["Component"],
    y=explained_variance["Explained Variance"],
    color="steelblue"
)

plt.title(
    "PCA Explained Variance"
)

plt.ylabel(
    "Variance Ratio"
)

plt.xlabel(
    "Principal Component"
)

plt.tight_layout()

plt.show()

# =========================================================
# 6. Cumulative Variance
# =========================================================

plt.figure(figsize=(10,6))

plt.plot(
    explained_variance["Component"],
    explained_variance["Cumulative Variance"],
    marker="o"
)

plt.axhline(
    y=0.80,
    color="red",
    linestyle="--"
)

plt.axhline(
    y=0.90,
    color="green",
    linestyle="--"
)

plt.title(
    "Cumulative Explained Variance"
)

plt.ylabel(
    "Cumulative Variance"
)

plt.xlabel(
    "Principal Component"
)

plt.tight_layout()

plt.show()

# =========================================================
# 7. PCA Loadings
# =========================================================

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[
        f"PC{i+1}"
        for i in range(
            len(feature_cols)
        )
    ],
    index=feature_cols
)

print(
    loadings[
        ["PC1", "PC2"]
    ]
    .round(4)
)

"""
This is actually a very strong PCA result. For wine chemistry, I'd call this a success.

PCA Interpretation
Explained Variance
PC	VariancePC1	27.17%
PC2	22.50%
PC3	14.43%
PC4	8.67%

Together:

PC1 + PC2 = 49.67%


Almost:

50% of all variation


is captured by only two components.

Variance Retention
80% Variance → 6 Components
90% Variance → 8 Components


From your cumulative table:

PC5 = 79.51%
PC6 = 85.21%
PC8 = 94.57%


Meaning:

6 components retain roughly 85%
8 components retain roughly 95%

This suggests:

The dataset contains meaningful dimensional structure,
but cannot be reduced to only 2 variables without losing information.
"""

# =========================================================
# 8. PC1 Feature Importance
# =========================================================

pc1_importance = (
    loadings["PC1"]
    .abs()
    .sort_values(
        ascending=False
    )
)

print(pc1_importance)

"""
PC1 Interpretation

Largest contributors:

Total Sulfur Dioxide
Free Sulfur Dioxide
Volatile Acidity
Chlorides
Sulphates
Residual Sugar

PC1 appears to separate wines based on:

Sulfur Chemistry
+
Acidity
+
Preservation Characteristics


This is likely the main driver behind: Red vs White separation.
"""
# =========================================================
# 9. PC2 Feature Importance
# =========================================================

pc2_importance = (
    loadings["PC2"]
    .abs()
    .sort_values(
        ascending=False
    )
)

print(pc2_importance)

"""
PC2 Interpretation

Largest contributors:

Density
Alcohol
Residual Sugar
Fixed Acidity

PC2 appears to represent:

Sweetness / Body Axis

because:

Residual Sugar
Density

increase together while:

Alcohol

moves strongly in the opposite direction.

This is exactly what we'd expect in wine chemistry.
"""

# =========================================================
# 10. PCA 2D Projection
# =========================================================

pca_2d = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

X_pca = pca_2d.fit_transform(
    X_scaled
)

pca_df = pd.DataFrame(
    {
        "PC1": X_pca[:,0],
        "PC2": X_pca[:,1],
        "type": wqrw["type"],
        "quality": wqrw["quality"]
    }
)

# =========================================================
# 11. PCA by Wine Type
# =========================================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="type",
    alpha=0.6
)

plt.title(
    "PCA Projection by Wine Type"
)

plt.tight_layout()

plt.show()

"""
PCA by Wine Type

This is the strongest result in the file.

The PCA projection shows:

Red Wines primarily on the left side of PC1

and

White Wines primarily on the right side of PC1.

There is overlap:

Expected because PCA is unsupervised. But the separation is still quite clear.

Interpretation:

The dominant source of variation in the Wine Quality
dataset corresponds closely to wine type.

which matches your clustering findings:

Ward k=2

White Wines
vs
Red Wines
"""

# =========================================================
# 12. PCA by Quality
# =========================================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="quality",
    palette="viridis",
    alpha=0.6
)

plt.title(
    "PCA Projection by Wine Quality"
)

plt.tight_layout()

plt.show()

"""
PCA by Quality

Very different outcome.

Quality classes:

3
4
5
6
7
8
9

are heavily mixed. No clear quality clusters emerge.

Interpretation:

Quality is not a dominant source of variance.

Instead:

Wine Type drives more variation than:

Quality

which is actually a useful finding.

Most Interesting PCA Finding

The most important discovery is:

Unsupervised PCA recovered wine type much more clearly than quality.

and:

Ward Linkage recovered the same pattern.

Two completely different unsupervised methods reached the same conclusion:

Primary Structure

White Wines
vs
Red Wines

That consistency increases confidence that the pattern is real.
"""

# =========================================================
# PCA Loadings Heatmap
# =========================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    loadings[
        ["PC1", "PC2", "PC3", "PC4", "PC5"]
    ],
    cmap="RdBu_r",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

plt.title(
    "PCA Component Loadings"
)

plt.ylabel(
    "Features"
)

plt.xlabel(
    "Principal Components"
)

plt.tight_layout()

plt.show()

"""
That heatmap is excellent. 

PCA Loadings Heatmap Interpretation

1. PC1 (27.17% Variance)

Strongest contributors:

+ Total Sulfur Dioxide   (+0.48)
+ Free Sulfur Dioxide    (+0.43)

- Volatile Acidity       (-0.39)
- Chlorides              (-0.32)
- Sulphates              (-0.32)

Interpretation

PC1 appears to separate wines using:

sulfur dioxide levels
preservation characteristics
acidity-related chemistry

This aligns closely with the separation observed between:

White Wines
vs
Red Wines

in the PCA scatterplot.

2. PC2 (22.50% Variance)

Strongest contributors:

+ Density            (+0.57)
+ Residual Sugar     (+0.37)

- Alcohol            (-0.47)

Interpretation

PC2 clearly captures a:

Sweetness / Body Axis

because:

sweeter wines typically have higher residual sugar
sweeter wines tend to have higher density
drier wines generally have higher alcohol

This component explains why white wines appeared to spread vertically in PCA space.

3. PC3 (14.43% Variance)

Strongest contributors:

+ Citric Acid        (+0.58)
- pH                 (-0.46)

Interpretation

PC3 appears to represent an:

Acidity Structure Axis

contrasting:

citric acid
pH
fixed acidity

This component likely captures subtle chemistry differences within wine types.

4. PC4 (8.67% Variance)

Strongest contributors:

+ Sulphates          (+0.63)
+ pH                 (+0.39)
+ Free SO₂           (+0.34)

Interpretation

PC4 is largely driven by:

Sulphates and secondary preservation chemistry variables.

5. PC5 (6.75% Variance)

Strongest contributors:

+ Chlorides          (+0.57)
- pH                 (-0.45)
- Residual Sugar     (-0.40)

Interpretation

PC5 captures additional mineral and acidity characteristics not explained by 
the earlier components.

6. Most Important Finding

The heatmap confirms something very interesting:

PCA is not dominated by one feature.

Instead:

PC1 -> Sulfur Chemistry
PC2 -> Sweetness / Density / Alcohol
PC3 -> Acidity Structure
PC4 -> Sulphates
PC5 -> Mineral & Acidity Effects

Every major component is driven by a different combination of wine chemistry variables.

7. Additional PCA Insight

This also explains why:

PC1 + PC2 ≈ 50%

of the dataset variance.

Wine chemistry is inherently multidimensional:

sulfur compounds
sugar content
alcohol content
acidity
preservation chemistry

all contribute meaningful variation.

PCA Key Insight (Heatmap)
• The PCA loadings heatmap revealed that different principal
  components capture distinct aspects of wine chemistry.

• PC1 was primarily driven by sulfur dioxide, volatile acidity,
  chlorides, and sulphates.

• PC2 represented a sweetness/body dimension dominated by
  density, residual sugar, and alcohol content.

• PC3 captured acidity structure through citric acid, pH,
  and fixed acidity.

• Later components highlighted additional chemistry patterns
  involving sulphates, chlorides, and preservation variables.

• No single feature dominated the PCA representation.

• The wine dataset exhibits meaningful multidimensional
  structure across several chemistry measurements.

• These results help explain why PCA successfully separated
  red and white wines while preserving much of the underlying
  chemical variation.
"""

# =========================================================
# PCA Results Interpretation
# =========================================================

"""
Results Interpretation

Variance Analysis

• The first principal component (PC1) explained 27.17% of
  total variance.

• The second principal component (PC2) explained 22.50%
  of total variance.

• Together, PC1 and PC2 captured 49.67% of all variation
  within the Wine Quality dataset.

• Approximately 85% of variance was retained using the
  first six principal components.

• Approximately 95% of variance was retained using the
  first eight principal components.

Component Interpretation

PC1

• Primarily driven by sulfur dioxide concentrations,
  volatile acidity, chlorides, sulphates, and residual
  sugar.

• Represents a chemistry axis strongly associated with
  differences between red and white wines.

PC2

• Primarily driven by density, alcohol content, residual
  sugar, and fixed acidity.

• Represents a sweetness/body dimension where sweeter,
  denser wines contrast with higher-alcohol wines.

PC3

• Dominated by citric acid, pH, and fixed acidity.

• Captures additional acidity-related structure within
  wines.

Projection Analysis

• PCA revealed clear separation between red and white
  wine populations.

• The dominant source of variation within the dataset
  corresponds to wine type rather than wine quality.

• Although overlap exists, red and white wines occupy
  distinct regions of PCA space.

• Quality levels showed considerably greater overlap,
  indicating that quality explains less variance than
  wine chemistry characteristics.

Loadings Analysis

• Different principal components were driven by different
  groups of chemistry variables.

• No single feature dominated the PCA representation.

• The dataset exhibits meaningful multidimensional
  structure across acidity, sulfur compounds, density,
  sugar content, and alcohol content.

Conclusion

PCA successfully reduced the dimensionality of the Wine
Quality dataset while preserving much of its underlying
structure. The analysis showed that wine type represents
the strongest source of variation, while multiple chemistry
variables contribute to the overall organization of the
data. These findings provide a foundation for comparing
nonlinear dimensionality-reduction techniques such as
Kernel PCA, t-SNE, and UMAP.
"""

# =========================================================
# PCA Key Insights
# =========================================================

"""
KEY INSIGHTS

• PCA reduced the 11-dimensional wine chemistry space into
  a smaller set of orthogonal components.

• The first two principal components explained 49.67% of
  total dataset variance.

• Approximately 85% of variance was retained using only
  six principal components.

• PC1 was primarily driven by sulfur dioxide, volatile
  acidity, chlorides, sulphates, and residual sugar.

• PC2 was primarily driven by density, alcohol, residual
  sugar, and fixed acidity.

• PCA revealed clear separation between red and white wines,
  indicating that wine type is the dominant source of
  variation in the dataset.

• Quality levels showed substantial overlap within PCA space,
  suggesting that quality contributes less variance than
  wine chemistry differences.

• The PCA embedding confirmed findings from previous Ward
  clustering analyses, where red and white wines formed the
  highest-level hierarchy.

• PCA provides a useful low-dimensional representation of
  the dataset while preserving much of the underlying
  chemical structure.
  
• PC1 and PC2 explained 49.67% of total variance.

• Six principal components retained approximately 85%
  of dataset information.

• Eight principal components retained approximately 95%
  of dataset information.

• PCA naturally separated red and white wines despite
  wine type not being used during dimensional reduction.

• Wine type contributed more variation than quality
  ratings.

• Sulfur dioxide, volatile acidity, density, alcohol,
  and residual sugar were among the most influential
  variables driving principal-component structure.

• PCA revealed meaningful low-dimensional structure
  while preserving much of the underlying wine chemistry.

FINAL TAKEAWAY

The dominant source of variation in the Wine Quality
dataset is the distinction between red and white wines.
PCA successfully captured this structure and provided an
interpretable low-dimensional representation of the data.

FINAL TAKEAWAY

The dominant structure of the Wine Quality dataset is driven
by differences between red and white wines. PCA successfully
captured this relationship and demonstrated that wine type
contributes more variation than quality ratings.
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

# Explained Variance
explained_variance.to_csv(
    output_dir / "pca_explained_variance.csv",
    index=False
)

# PCA Loadings
loadings.to_csv(
    output_dir / "pca_loadings.csv"
)

# PC1 Importance
pc1_importance.to_csv(
    output_dir / "pca_pc1_importance.csv"
)

# PC2 Importance
pc2_importance.to_csv(
    output_dir / "pca_pc2_importance.csv"
)

# PCA Embedding
# This is probably the most important export.

pca_df.to_csv(
    output_dir / "pca_2d_projection.csv",
    index=False
)

"""
Contains:

PC1
PC2
type
quality

and will allow you to rebuild:

PCA by Type
PCA by Quality

without rerunning PCA.
"""

# PCA Summary Table
# Useful later when comparing PCA, Kernel PCA, t-SNE, and UMAP.

pca_summary = pd.DataFrame(
    {
        "Method": ["PCA"],
        "PC1_Variance": [
            explained_variance.loc[
                0,
                "Explained Variance"
            ]
        ],
        "PC2_Variance": [
            explained_variance.loc[
                1,
                "Explained Variance"
            ]
        ],
        "PC1_PC2_Total": [
            explained_variance.loc[
                1,
                "Cumulative Variance"
            ]
        ],
        "Components_80pct": [6],
        "Components_90pct": [8]
    }
)

pca_summary.to_csv(
    output_dir / "pca_summary.csv",
    index=False
)
