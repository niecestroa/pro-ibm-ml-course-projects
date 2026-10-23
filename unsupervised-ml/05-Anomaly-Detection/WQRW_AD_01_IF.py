# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October 23, 2026
Status:     Ongoing

Description

5.1 Isolation Forest

This analysis applies Isolation Forest to identify anomalous
wine observations within the Wine Quality dataset.

Isolation Forest is an unsupervised anomaly-detection
algorithm that isolates observations using random feature
splits. Observations requiring fewer splits to isolate are
considered more anomalous.

Analysis Includes

    • Isolation Forest fitting
    • Anomaly score evaluation
    • Anomaly identification
    • Anomaly score distribution
    • PCA anomaly visualization
    • Wine-type anomaly analysis

Objectives

    1. Identify unusual wine observations based on
       chemistry characteristics.

    2. Evaluate the prevalence of anomalies within the
       dataset.

    3. Investigate whether anomalies are concentrated
       within specific wine populations.

    4. Determine whether detected anomalies represent
       meaningful chemistry extremes.

Part 1: Anomaly Detection

Isolation Forest identified 266 anomalous wines,
representing 5.0% of the dataset.

The most anomalous observations were dominated by red wines
and exhibited extreme chemistry profiles relative to the
broader wine population.

Part 2: Anomaly Structure & Visualization

PCA visualization showed that anomalies primarily occupied
boundary and low-density regions of feature space rather
than dense population centers.

Wine-type analysis revealed that red wines exhibited a
substantially higher anomaly rate than white wines.

Anomaly Rates

    • Red Wines:
        164 / 1,359 ≈ 12.1%

    • White Wines:
        102 / 3,961 ≈ 2.6%

Red wines were nearly five times more likely to be
classified as anomalous.

Key Findings

    • Isolation Forest identified 266 anomalous wines.

    • Most anomalous observations occurred within the
      red wine population.

    • Anomalies primarily occupied low-density and
      boundary regions of wine chemistry space.

    • The anomaly-score distribution displayed a clear
      lower-tail population corresponding to unusual wines.

    • Red wines exhibited substantially greater chemistry
      variability than white wines.

Final Takeaway

    Isolation Forest successfully identified globally
    unusual wine observations and revealed that rare
    chemistry profiles occur disproportionately within the
    red wine population.
"""

# =========================================================
# 0. Imports & Configuration
# =========================================================

from pathlib import Path

import pandas as pd
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
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\05-Anomaly-Detection\05-anomaly-results"
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

print(
    "\n----- FEATURE MATRIX -----"
)

print(
    f"Observations: {X_scaled.shape[0]:,}"
)

print(
    f"Features: {X_scaled.shape[1]}"
)

print("Feature matrix shape:", X_scaled.shape)



# =========================================================
# Part 1: Isolation Forest Detection
# =========================================================

# =========================================================
# 4. Isolation Forest
# =========================================================

from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    contamination=0.05,
    random_state=RANDOM_STATE
)

iso.fit(
    X_scaled
)

anomaly_labels = iso.predict(
    X_scaled
)

anomaly_scores = iso.decision_function(
    X_scaled
)

# =========================================================
# 5. Isolation Forest Results
# =========================================================

anomaly_df = wqrw.copy()

anomaly_df["anomaly_label"] = anomaly_labels

anomaly_df["anomaly_score"] = anomaly_scores

# Summary

print(
    "\n----- ANOMALY SUMMARY -----"
)

print(
    anomaly_df[
        "anomaly_label"
    ]
    .value_counts()
)

total_obs = len(
    anomaly_df
)

# Anomaly Percentage

total_anomalies = (
    anomaly_df["anomaly_label"]
    == -1
).sum()

print(
    f"Anomalies: "
    f"{total_anomalies:,}"
)

print(
    f"Percent: "
    f"{100 * total_anomalies / total_obs:.2f}%"
)

# =========================================================
# 6. Anomaly Score Distribution
# =========================================================

plt.figure(figsize=(10,6))

sns.histplot(
    anomaly_df["anomaly_score"],
    bins=50
)

plt.title(
    "Isolation Forest Anomaly Scores"
)

plt.xlabel(
    "Anomaly Score"
)

plt.tight_layout()

plt.show()

# =========================================================
# 7. Most Anomalous Wines
# =========================================================

most_anomalous = (
    anomaly_df
    .sort_values(
        by="anomaly_score"
    )
    .head(20)
)

print(
    most_anomalous[
        [
            "type",
            "quality",
            "anomaly_score"
        ]
    ]
)

# =========================================================
# 8. Anomalies by Wine Type
# =========================================================

anomaly_type = pd.crosstab(
    anomaly_df["type"],
    anomaly_df["anomaly_label"]
)

print(
    anomaly_type
)

# =========================================================
# Exports
# =========================================================

anomaly_df.to_csv(
    output_dir / "isolation_forest_results.csv",
    index=False
)

most_anomalous.to_csv(
    output_dir / "isolation_forest_top20.csv",
    index=False
)

anomaly_type.to_csv(
    output_dir / "isolation_forest_type_breakdown.csv"
)

"""
Initial Findings

1. Total Anomalies

Observations: 5,320
Anomalies: 266
Percent: 5.00%

Exactly what we'd expect because contamination=0.05 tells Isolation Forest to 
classify approximately 5% of observations as anomalous.

So: 266 anomalies is not surprising.

2. Most Anomalous Wines

The strongest anomalies are dominated by:

Red Wines

Looking at the Top 20:

18 Red Wines
2 White Wines

The most anomalous wine in the entire dataset is:

Red Wine
Quality = 4
Score = -0.1477

This immediately suggests that:

Red wines contain more extreme chemistry profiles than white wines.

3. Anomalies by Wine Type

Raw Counts:

Red
    164 anomalies

White
    102 anomalies

At first glance that looks like:

More red anomalies but we need proportions.

Red Wines
164 / 1359

≈ 12.1%

White Wines
102 / 3961

≈ 2.6%

This is huge.

4. Most Important Finding

Red wines are nearly:

12.1 / 2.6 ≈ 4.7x

more likely to be flagged as anomalous than white wines.

That is probably the most important result you've found so far.

5. Why Might This Occur?

The red wine dataset has already shown greater variability in:

Volatile Acidity
Sulphates
Alcohol
Chlorides

through:

PCA
Ward Linkage
NMF

So Isolation Forest appears to be detecting:

Rare Chemistry Profiles

primarily within the red wine population.

6. Score Distribution

The histogram looks very healthy. You can clearly see:

Large Normal Population with Small Left Tail containing the anomalies.

This suggests:
    
No major model issues
No strange score behavior
Reasonable anomaly detection
"""

# =========================================================
# Part 1: Isolation Forest Detection 
# Results Interpretation
# =========================================================

"""
Results Interpretation

Isolation Forest identified 266 anomalous observations, representing 5.0% of 
the Wine Quality dataset.

The anomaly score distribution showed a clear separation between the 
majority of wines and a small population of unusual observations located 
within the lower tail of the score distribution.

The most anomalous observations were dominated by red wines,suggesting that 
red wine chemistry exhibits greater extreme variation than white wine chemistry.

Although white wines represent the majority of observations, red wines 
accounted for most of the extreme anomaly scores.

Wine-Type Analysis

• Red Wines:
    164 anomalies

• White Wines:
    102 anomalies

When adjusted for population size, red wines exhibited a substantially higher 
anomaly rate than white wines.

This indicates that unusual chemistry profiles are more commonly observed 
within the red wine population.

Conclusion

Isolation Forest successfully identified a small subset of wines exhibiting 
highly unusual chemistry characteristics. The detected anomalies appear 
concentrated within red wines and likely represent rare chemical profiles 
rather than random noise.
"""

# =========================================================
# Part 1: Isolation Forest Detection 
# Key Insights
# =========================================================

"""
Key Insights
KEY INSIGHTS

• Isolation Forest identified 266 anomalous wines,
  representing 5.0% of the dataset.

• The anomaly-score distribution displayed a distinct
  lower-tail population corresponding to unusual wine
  observations.

• Red wines accounted for most of the strongest anomaly
  scores.

• Red wines exhibited a substantially higher anomaly rate
  than white wines.

• The most anomalous wines were primarily low- and
  medium-quality red wines.

• The detected anomalies likely represent rare chemistry
  profiles rather than simple measurement noise.

• The findings suggest that the red wine population exhibits
  greater chemical diversity than the white wine population.

FINAL TAKEAWAY

Isolation Forest identified a small group of wines with highly unusual 
chemistry characteristics. These anomalies occur disproportionately within 
red wines and may represent rare production styles, unusual chemistry 
combinations, or extreme observations that influence clustering structure.
"""

# =========================================================
# Part 2: Anomaly Structure & Visualization
# =========================================================

"""
Questions need to answer in this part :

Where are the anomalies?
How do they relate to wine populations?
Do they appear meaningful or random?
"""

# =========================================================
# 9. PCA Anomaly Visualization
# =========================================================

pca = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

X_pca = pca.fit_transform(
    X_scaled
)

anomaly_pca = pd.DataFrame(
    {
        "PC1": X_pca[:,0],
        "PC2": X_pca[:,1],
        "anomaly_label": anomaly_labels
    }
)

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=anomaly_pca,
    x="PC1",
    y="PC2",
    hue="anomaly_label",
    palette={
        1: "#1f77b4",
        -1: "#d62728"
    },
    alpha=0.6
)

plt.title(
    "Isolation Forest Anomalies in PCA Space"
)

plt.xlabel(
    "PC1"
)

plt.ylabel(
    "PC2"
)

plt.tight_layout()

plt.show()

"""
What the PCA plot shows

Most anomalies (red points) occur:

✅ Around the outer boundaries
✅ At the top of the distribution
✅ At the extreme left and right tails
✅ In sparse regions

This is exactly what we'd hope Isolation Forest would detect.

The anomalies are not randomly scattered throughout the dense center.

Instead, they tend to occupy:

Low-density regions 
Boundary regions
Extreme chemistry profiles

which is strong evidence that Isolation Forest is finding genuine outliers.
"""

# =========================================================
# 10. Anomalies by Wine Type
# =========================================================

anomaly_counts = (
    anomaly_df
    .groupby(
        ["type", "anomaly_label"]
    )
    .size()
    .reset_index(
        name="count"
    )
)

plt.figure(figsize=(8,5))

sns.barplot(
    data=anomaly_counts,
    x="type",
    y="count",
    hue="anomaly_label"
)

plt.title(
    "Isolation Forest Anomalies by Wine Type"
)

plt.tight_layout()

plt.show()

# =========================================================
# Part 2 Exports
# =========================================================

anomaly_pca.to_csv(
    output_dir / "isolation_forest_pca_projection.csv",
    index=False
)

anomaly_counts.to_csv(
    output_dir / "isolation_forest_counts_by_type.csv",
    index=False
)


# =========================================================
# Part 2: Anomaly Structure & Visualization
# Results Interpretation
# =========================================================

"""
Results Interpretation

PCA visualization showed that anomalous wines primarily
occupy boundary and low-density regions of the wine
chemistry space.

Most detected anomalies occurred near the outer edges of
the PCA distribution rather than within dense wine
populations.

A small number of observations appeared as extreme
isolated points located far from the majority of wines,
indicating highly unusual chemistry profiles.

Wine-Type Analysis

• Red Wines:
    164 anomalies

• White Wines:
    102 anomalies

Although white wines represent most of the observations in
the dataset, red wines accounted for most of the detected
anomalies.

When adjusted for population size:

• Red Wines:
    12.1% anomalous

• White Wines:
    2.6% anomalous

Red wines were nearly five times more likely to be
classified as anomalous.

Conclusion

Isolation Forest successfully identified wines occupying
low-density and boundary regions of feature space. These
results suggest that unusual chemistry profiles occur more
frequently within the red wine population than within the
white wine population.
"""

# =========================================================
# Part 2: Anomaly Structure & Visualization
# Key Insights
# =========================================================

"""
KEY INSIGHTS

• Isolation Forest identified 266 anomalous wines,
  representing 5.0% of the dataset.

• PCA visualization showed that anomalies primarily
  occupy low-density and boundary regions rather than
  dense population centers.

• Most extreme anomalies occurred within the red wine
  population.

• Red wines exhibited substantially higher anomaly rates
  than white wines.

• Approximately 12.1% of red wines were classified as
  anomalous compared with 2.6% of white wines.

• The detected anomalies likely represent rare chemistry
  profiles rather than random noise.

• The findings suggest that red wine chemistry exhibits
  greater variability than white wine chemistry.

FINAL TAKEAWAY

Isolation Forest successfully identified a small set of
wines with unusual chemistry characteristics. These
anomalies occur disproportionately within red wines and
appear concentrated around the boundaries of the wine
chemistry space, indicating meaningful chemical extremes
rather than random variation.
"""

# =========================================================
# Isolation Forest
# OVERALL INTERPRETATION
# =========================================================

"""
Isolation Forest identified a small subset of wines with chemistry profiles 
that differ substantially from the broader wine population.

The anomaly-score distribution revealed a clear distinction between most wines 
and a small set of unusual observations.

PCA visualization demonstrated that detected anomalies primarily occupy 
low-density and boundary regions rather than the centers of wine populations.

Wine-type analysis showed that anomalies occur disproportionately within 
red wines, indicating greater chemical diversity and more extreme chemistry 
profiles than those observed within white wines.

These findings suggest that the detected anomalies likely represent meaningful 
rare wine profiles rather than random measurement noise.
"""

# =========================================================
# Isolation Forest
# OVERALL INTERPRETATION
# =========================================================

"""
KEY INSIGHTS

• Isolation Forest identified 266 anomalous wines,
  representing 5.0% of the dataset.

• The anomaly-score distribution displayed a distinct lower
  tail associated with unusual observations.

• PCA visualization showed that anomalies primarily occupy
  sparse and boundary regions of feature space.

• Most extreme anomalies occurred within the red wine
  population.

• Red wines exhibited a substantially higher anomaly rate
  than white wines.

• Approximately 12.1% of red wines were classified as
  anomalous compared with 2.6% of white wines.

• The detected anomalies likely represent meaningful
  chemistry extremes rather than measurement noise.

• Red wine chemistry appears substantially more variable
  than white wine chemistry.

FINAL TAKEAWAY

Isolation Forest successfully identified globally unusual wine observations 
and demonstrated that rare chemistry profiles occur disproportionately within
red wines. These anomalies represent meaningful extremes that may influence 
clustering structure and population boundaries throughout the dataset.
"""
