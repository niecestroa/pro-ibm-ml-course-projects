# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October 23, 2026
Status:     Completed

Description 

Anomaly Detection Results

This analysis consolidates and compares the anomaly-
detection techniques evaluated throughout the project.

Methods Evaluated

    • Isolation Forest
    • Local Outlier Factor (LOF)

Objectives

    1. Compare global and local anomaly-detection methods.

    2. Evaluate anomaly prevalence across wine populations.

    3. Assess agreement between anomaly-detection methods.

    4. Identify meaningful rare wine chemistry profiles.

Analysis Includes

    • Isolation Forest summary
    • LOF summary
    • Method comparison
    • Overlap analysis
    • PCA anomaly visualizations
    • Wine-type comparison
    • Overall anomaly-detection evaluation

Final Goal

    Determine how globally and locally anomalous wine
    observations influence the structure of the Wine
    Quality dataset.
"""

# =========================================================
# 0. Imports & Configuration
# =========================================================

from pathlib import Path

import pandas as pd
import seaborn as sns

sns.set_theme(
    style="whitegrid",
    context="talk"
)

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    140
)

RANDOM_STATE = 42

# =========================================================
# 1. Paths
# =========================================================

output_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\05-Anomaly-Detection\05-anomaly-results"
)

# =========================================================
# 2. Load Results
# =========================================================

if_df = pd.read_csv(
    output_dir / "isolation_forest_results.csv"
)

lof_df = pd.read_csv(
    output_dir / "lof_results.csv"
)

if_top20 = pd.read_csv(
    output_dir / "isolation_forest_top20.csv"
)

lof_top20 = pd.read_csv(
    output_dir / "lof_top20.csv"
)

if_vs_lof = pd.read_csv(
    output_dir / "if_vs_lof_comparison.csv"
)

# =========================================================
# 3. Results Summary
# =========================================================

print(
    "\n----- ISOLATION FOREST -----"
)

print(
    if_df["anomaly_label"]
    .value_counts()
)

print(
    "\n----- LOF -----"
)

print(
    lof_df["anomaly_label"]
    .value_counts()
)

# =========================================================
# 4. Overlap Analysis
# =========================================================

overlap = (
    (if_df["anomaly_label"] == -1)
    &
    (lof_df["anomaly_label"] == -1)
)

overlap_count = overlap.sum()

print(
    "\n----- OVERLAP ANALYSIS -----"
)

print(
    f"Shared Anomalies: "
    f"{overlap_count:,}"
)

print(
    f"Percent Shared: "
    f"{100 * overlap_count / 266:.2f}%"
)

"""
High overlap (>70%)
    Methods agree strongly.

Moderate overlap (40-70%)
    Methods capture related but distinct anomalies.

Low overlap (<40%)
    Methods identify substantially different anomaly types.
"""

# =========================================================
# Overall Results Interpretation
# =========================================================

print(
    "\n----- Overall Results Interpretation -----"
)
print(
"""
Two complementary anomaly-detection methods were applied to
the Wine Quality dataset: Isolation Forest and Local
Outlier Factor (LOF).

Both methods identified 266 anomalous wines, representing
5.0% of the dataset. However, the populations identified
as anomalous differed substantially between the methods.

Isolation Forest

• Isolation Forest primarily detected globally unusual
  wine observations.

• The majority of detected anomalies occurred within the
  red wine population.

• Red wines accounted for 164 anomalies compared with
  102 white wine anomalies.

• PCA visualization showed that most anomalies occupied
  sparse and boundary regions of wine chemistry space.

These results suggest that red wines contain a greater
number of globally extreme chemistry profiles.

Local Outlier Factor (LOF)

• LOF identified locally unusual observations based on
  neighborhood density.

• White wines accounted for 179 anomalies compared with
  87 red wine anomalies.

• PCA visualization showed that anomalies occurred
  throughout the feature space rather than only along
  extreme boundaries.

These results suggest that white wines contain more locally
unusual chemistry profiles.

Method Agreement Analysis

• Isolation Forest identified 266 anomalies.

• LOF identified 266 anomalies.

• Only 96 wines were identified by both methods.

• Agreement between methods was 36.09%.

The relatively low overlap demonstrates that the two
methods capture different forms of anomalous behavior.

Isolation Forest focuses primarily on globally unusual
chemistry profiles, while LOF focuses on local density
differences.

Conclusion

Anomaly structure within the Wine Quality dataset is not
defined by a single type of outlier behavior.

Red wines appear more likely to exhibit globally extreme
chemistry characteristics, whereas white wines appear more
likely to exhibit locally unusual chemistry profiles.

The combination of Isolation Forest and LOF provides a more
complete understanding of rare wine observations than
either method alone.
"""
)

# =========================================================
# Overall Key Insights
# =========================================================


print(
    "\n----- Overall Results Interpretation -----"
)

print(
"""
• Both Isolation Forest and LOF identified 266 anomalous
  wines, representing 5.0% of the dataset.

• Isolation Forest primarily identified globally unusual
  wines concentrated within the red wine population.

• LOF primarily identified locally unusual wines
  concentrated within the white wine population.

• Red wines accounted for 164 Isolation Forest anomalies
  but only 87 LOF anomalies.

• White wines accounted for 102 Isolation Forest anomalies
  but 179 LOF anomalies.

• Only 96 wines were identified by both methods.

• Overall agreement between Isolation Forest and LOF was
  36.09%.

• The low overlap indicates that different anomaly-
  detection methods reveal different aspects of unusual
  wine chemistry.

• Isolation Forest anomalies were primarily located along
  low-density boundaries of wine chemistry space.

• LOF anomalies were distributed throughout the feature
  space and reflected local chemistry irregularities.

FINAL TAKEAWAY

Isolation Forest and LOF provided complementary views of
anomaly structure within the Wine Quality dataset.

Isolation Forest revealed globally extreme chemistry
profiles, particularly among red wines, while LOF revealed
locally unusual chemistry patterns, particularly among
white wines.

Together, the methods demonstrated that anomalous wine
behavior exists at both global and local scales and that
multiple anomaly-detection techniques are necessary to
fully characterize rare wine chemistry profiles.
"""
)

# =========================================================
# Final Recommendation
# =========================================================


print(
    "\n----- Final Recommendation -----"
)

print(
"""
Best Global Anomaly Detector:   Isolation Forest

Best Local Anomaly Detector:    Local Outlier Factor (LOF)

Most Informative Finding:       Only 36.09% agreement between methods

Implication: 

Global and local anomaly detectors identify substantially different types of 
unusual wines.

Overall Winner:                 Use Both Methods Together

Isolation Forest and LOF capture complementary anomaly structures and provide 
a more comprehensive assessment of rare wine chemistry profiles when used 
jointly than when used independently.
"""
)