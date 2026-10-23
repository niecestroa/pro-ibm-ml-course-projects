# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October __, 2026
Status:     Completed

Description

5.2 Local Outlier Factor (LOF)

This analysis applies Local Outlier Factor (LOF) to identify
anomalous wine observations based on local neighborhood
density within the Wine Quality dataset.

Unlike Isolation Forest, which detects globally unusual
observations, LOF identifies wines whose local density
differs substantially from nearby observations. This allows
LOF to detect rare local patterns that may not appear
globally extreme.

Analysis Includes

    • LOF anomaly detection
    • Outlier score evaluation
    • Anomaly score distribution
    • Most anomalous wine identification
    • PCA anomaly visualization
    • Wine-type anomaly analysis
    • Isolation Forest vs LOF comparison

Objectives

    1. Identify locally unusual wine observations.

    2. Evaluate anomaly prevalence within the dataset.

    3. Investigate whether local anomalies occur more
       frequently within specific wine populations.

    4. Compare local and global anomaly-detection
       approaches.

Part 1: LOF Detection

LOF identified 266 anomalous wines, representing 5.0% of
the dataset.

The most extreme anomaly scores were dominated by white
wines, suggesting the presence of locally unusual chemistry
profiles within the white wine population.

Part 2: Local Anomaly Structure & Method Comparison

PCA visualization revealed that LOF anomalies were
distributed throughout the wine chemistry space rather than
appearing only at the outer boundaries.

LOF identified:

    • 87 red wine anomalies
    • 179 white wine anomalies

Comparison against Isolation Forest demonstrated that the
two techniques capture different forms of anomalous
behavior.

Isolation Forest emphasized globally unusual chemistry
profiles, while LOF emphasized locally unusual wines.

Key Findings

    • LOF identified 266 anomalous wines.

    • White wines accounted for most detected anomalies.

    • LOF anomalies were distributed throughout PCA space.

    • Several anomalies occupied populated regions and were
      unusual relative to nearby observations rather than
      globally isolated.

    • Isolation Forest and LOF identified substantially
      different anomaly populations.

Final Takeaway

    LOF successfully identified locally unusual wine
    observations and provided a complementary perspective
    to Isolation Forest. Together, the methods revealed both
    globally extreme and locally unusual chemistry profiles
    within the Wine Quality dataset.
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

print("Feature matrix shape:", X_scaled.shape)

# =========================================================
# Part 1: LOF Detection
# =========================================================

# =========================================================
# 4. Local Outlier Factor
# =========================================================

from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05
)

anomaly_labels = lof.fit_predict(
    X_scaled
)

anomaly_scores = (
    lof.negative_outlier_factor_
)

# Summary

# =========================================================
# 5. LOF Results
# =========================================================

lof_df = wqrw.copy()

lof_df["anomaly_label"] = anomaly_labels

lof_df["anomaly_score"] = anomaly_scores

# Summary
print(
    "\n----- LOF ANOMALY SUMMARY -----"
)

print(
    lof_df[
        "anomaly_label"
    ]
    .value_counts()
)

# Anomoly Percentage
total_obs = len(
    lof_df
)

total_anomalies = (
    lof_df["anomaly_label"]
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
# 6. LOF Score Distribution
# =========================================================

plt.figure(figsize=(10,6))

sns.histplot(
    lof_df["anomaly_score"],
    bins=50
)

plt.title(
    "LOF Outlier Scores"
)

plt.tight_layout()

plt.show()

# =========================================================
# 7. Most Anomalous Wines
# =========================================================

"""
For LOF: More negative = More anomalous
"""

most_anomalous = (
    lof_df
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
    lof_df["type"],
    lof_df["anomaly_label"]
)

print(
    anomaly_type
)

"""
Interesting! LOF is telling a somewhat different story than Isolation Forest.

1. Isolation Forest

Found:

Red anomalies:   164
White anomalies: 102

Rates:

Red   ≈ 12.1%
White ≈ 2.6%

Conclusion:

Red wines contain more globally unusual chemistry profiles.

2. LOF

Found:

Red anomalies:    87
White anomalies: 179

Rates:

Red   ≈ 6.4%
White ≈ 4.5%

The gap is much smaller.
Even more interesting: Most extreme LOF anomalies => dominated by White Wines
Top 10 is almost entirely white wines.

3. Why the Difference?

This is exactly why running both methods is valuable.

A. Isolation Forest

Looks for: Globally unusual observations

Examples:

Extreme alcohol
Extreme sulphates
Extreme acidity

far away from most wines.

Result:

Red wines have more extreme chemistry.

B. LOF

Looks for: Locally unusual observations

A wine can be: Globally normal but Locally strange compared to nearby wines.

Result:

White wines contain more local anomalies.

The huge score:

White wine
score = -8.13

is a strong sign that at least one white wine sits in an extremely sparse 
local neighborhood.

Before visualization, I'd already write down:

Isolation Forest
    Red wines dominate anomalies

LOF
    White wines dominate extreme local anomalies

That suggests:

Red wines
    More globally extreme

White wines
    More locally unusual

which is a genuinely interesting result and exactly the sort of insight 
anomaly detection is supposed to uncover.
"""

# =========================================================
# Part 1 Exports
# =========================================================

lof_df.to_csv(
    output_dir / "lof_results.csv",
    index=False
)

most_anomalous.to_csv(
    output_dir / "lof_top20.csv",
    index=False
)

anomaly_type.to_csv(
    output_dir / "lof_type_breakdown.csv"
)

# =========================================================
# Part 1: LOF Detection
# Results Interpretation
# =========================================================

"""
Results Interpretation

Local Outlier Factor (LOF) identified 266 anomalous wines,
representing 5.0% of the Wine Quality dataset.

The distribution of LOF scores showed that most wines
occupied dense local neighborhoods, while a small subset
of observations exhibited substantially lower local
densities and were classified as anomalies.

Unlike Isolation Forest, which primarily identified red
wines as globally unusual observations, LOF detected a
larger number of anomalies within the white wine
population.

Most Anomalous Observations

• The most extreme LOF anomaly was a white wine with a
  quality score of 6.

• Many of the strongest anomaly scores belonged to white
  wines, indicating the presence of locally unusual white
  wine chemistry profiles.

• Several red wines also appeared among the strongest
  anomalies, but less frequently than in the Isolation
  Forest analysis.

Wine-Type Analysis

• Red Wines:
    87 anomalies

• White Wines:
    179 anomalies

Although white wines dominate the overall dataset, they also
represent the majority of LOF-detected anomalies.

Anomaly Rates

• Red Wines:
    87 / 1,359 ≈ 6.4%

• White Wines:
    179 / 3,961 ≈ 4.5%

While red wines still exhibit a slightly higher anomaly
rate, LOF identified substantially more local anomalies
within white wines than Isolation Forest did.

Comparison with Isolation Forest

• Isolation Forest emphasized globally unusual chemistry
  profiles and identified red wines as the dominant source
  of anomalies.

• LOF emphasized local density differences and identified
  many unusual white wines that appeared relatively normal
  from a global perspective.

Conclusion

LOF successfully identified wines located in sparse local
regions of feature space. Its results suggest that white
wines contain a larger number of locally unusual chemistry
profiles, while red wines contain more globally extreme
chemistry observations.
"""

# =========================================================
# Part 1: LOF Detection
# Key Insights
# =========================================================

"""
KEY INSIGHTS

• Local Outlier Factor identified 266 anomalous wines,
  representing 5.0% of the dataset.

• The most extreme anomaly scores were primarily associated
  with white wines.

• White wines accounted for 179 detected anomalies,
  compared with 87 red wine anomalies.

• LOF revealed locally unusual chemistry profiles that were
  not emphasized by Isolation Forest.

• The score distribution showed a small subset of wines
  occupying sparse local neighborhoods.

• White wines appear to contain a greater number of local
  chemistry irregularities, while red wines exhibit more
  globally extreme profiles.

• LOF provides complementary information to Isolation
  Forest by focusing on neighborhood density rather than
  overall distance from the wine population.

FINAL TAKEAWAY

Local Outlier Factor successfully identified wines that
differ from their immediate neighbors. The results suggest
that white wines contain a larger number of locally unusual
chemistry profiles, while red wines remain more likely to
exhibit globally extreme chemistry characteristics.
"""

# =========================================================
# Part 2: Local Anomaly Structure
# =========================================================

# =========================================================
# 9. PCA LOF Visualization
# =========================================================

pca = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

X_pca = pca.fit_transform(
    X_scaled
)

lof_pca = pd.DataFrame(
    {
        "PC1": X_pca[:,0],
        "PC2": X_pca[:,1],
        "anomaly_label": anomaly_labels
    }
)

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=lof_pca,
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
    "LOF Anomalies in PCA Space"
)

plt.tight_layout()

plt.show()

"""

For Isolation Forest, anomalies tended to concentrate around:

Outer boundaries
Extreme edges
Sparse regions

For LOF:

Anomalies appear much more distributed throughout the space.
Red points are not only on the edges. Many occur inside regions that appear 
populated.
That is exactly what we'd expect from a local density detector.

LOF is essentially saying:

"This wine isn't globally unusual, but it looks unusual relative to its nearby 
neighbors."

That makes the PCA plot a strong visual validation of LOF's behavior.
"""

# =========================================================
# 10. LOF Anomalies by Wine Type
# =========================================================

anomaly_counts = (
    lof_df
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
    "LOF Anomalies by Wine Type"
)

plt.tight_layout()

plt.show()

# =========================================================
# 11. Isolation Forest vs LOF
# =========================================================

comparison = pd.DataFrame(
    {
        "Method": [
            "Isolation Forest",
            "LOF"
        ],
        "Total Anomalies": [
            266,
            266
        ],
        "Red Anomalies": [
            164,
            87
        ],
        "White Anomalies": [
            102,
            179
        ]
    }
)

print(comparison)

# =========================================================
# Part 2 Exports
# =========================================================

lof_pca.to_csv(
    output_dir / "lof_pca_projection.csv",
    index=False
)

anomaly_counts.to_csv(
    output_dir / "lof_counts_by_type.csv",
    index=False
)

comparison.to_csv(
    output_dir / "if_vs_lof_comparison.csv",
    index=False
)

comparison.to_csv(
    output_dir / "if_vs_lof_comparison.csv",
    index=False
)

# =========================================================
# Part 2: Local Anomaly Structure
# Results Interpretation
# =========================================================

"""
Results Interpretation

PCA visualization revealed important differences between
LOF and Isolation Forest anomaly detection.

Unlike Isolation Forest, which primarily identified
observations along the outer boundaries of wine chemistry
space, LOF detected anomalies throughout the PCA
projection.

Many LOF anomalies occurred within regions that appeared
densely populated at a global scale. This indicates that
the identified wines are locally unusual relative to their
nearest neighbors rather than globally isolated from the
dataset.

Wine-Type Analysis

• Red Wines:
    87 anomalies

• White Wines:
    179 anomalies

White wines accounted for most LOF-detected anomalies,
contrasting with Isolation Forest where most anomalies
originated from the red wine population.

Method Comparison

Isolation Forest

• 266 anomalies detected
• 164 red wine anomalies
• 102 white wine anomalies
• Emphasized globally unusual wines

LOF

• 266 anomalies detected
• 87 red wine anomalies
• 179 white wine anomalies
• Emphasized locally unusual wines

The differing anomaly populations indicate that the two
methods capture different forms of unusual wine behavior.

Conclusion

LOF successfully identified wines that differ from their
immediate neighborhoods even when those wines occupy
otherwise populated regions of chemistry space. This
suggests that white wines contain a greater number of
locally unusual chemistry profiles, while red wines
contain more globally extreme chemistry observations.
"""

# =========================================================
# Part 2: Local Anomaly Structure
# Key Insights
# =========================================================

"""
KEY INSIGHTS

• LOF anomalies were distributed throughout the PCA
  projection rather than appearing only along outer
  boundaries.

• White wines accounted for most LOF-detected anomalies.

• Isolation Forest and LOF identified substantially
  different anomaly populations.

• Isolation Forest emphasized globally unusual chemistry
  profiles.

• LOF emphasized locally unusual chemistry profiles.

• The strongest LOF anomalies were dominated by white wines.

• The results suggest that white wines contain more local
  chemistry irregularities than red wines.

• Anomaly detection results depend strongly on whether
  global or local definitions of outlier behavior are used.

FINAL TAKEAWAY

LOF and Isolation Forest provided complementary views of
anomaly structure within the Wine Quality dataset.

Isolation Forest revealed globally extreme wines,
particularly within the red wine population, while LOF
identified locally unusual wines, particularly within the
white wine population. Together, the two methods provide a
more complete understanding of rare and unusual wine
chemistry profiles.
"""

# =========================================================
# Local Outlier Factor (LOF) 
# OVERALL INTERPRETATION
# =========================================================

"""
Local Outlier Factor identified a small subset of wines
whose local chemistry characteristics differ substantially
from nearby observations.

Unlike Isolation Forest, which primarily identified global
chemistry extremes, LOF detected wines that appeared
unusual relative to their local feature-space
neighborhoods.

PCA visualization demonstrated that LOF anomalies were
distributed throughout the wine chemistry space rather than
being concentrated solely along outer boundaries.

White wines accounted for most LOF-detected anomalies,
suggesting that local chemistry irregularities occur more
frequently within the white wine population.

Comparison with Isolation Forest revealed that the two
methods identify different forms of unusual behavior.

These results indicate that anomaly structure within the
Wine Quality dataset depends strongly on whether anomalies
are defined globally or locally.
"""

# =========================================================
# Local Outlier Factor (LOF)
# KEY INSIGHTS
# =========================================================

"""
KEY INSIGHTS

• LOF identified 266 anomalous wines, representing 5.0% of
  the dataset.

• The strongest LOF anomalies were predominantly white
  wines.

• White wines accounted for 179 anomalies compared with
  87 red wine anomalies.

• LOF anomalies appeared throughout the PCA projection
  rather than only along extreme boundaries.

• The detected observations represent locally unusual wine
  chemistry profiles.

• Isolation Forest and LOF detected substantially
  different anomaly populations.

• Isolation Forest emphasized globally unusual wines.

• LOF emphasized locally unusual wines.

• The combined findings suggest that red wines contain more
  global chemistry extremes while white wines contain more
  local chemistry irregularities.

FINAL TAKEAWAY

LOF successfully identified locally unusual wine
observations that were not necessarily globally extreme.
When combined with Isolation Forest, the analysis provides
a comprehensive view of anomaly structure within the Wine
Quality dataset by capturing both global and local forms of
unusual wine chemistry.
"""
