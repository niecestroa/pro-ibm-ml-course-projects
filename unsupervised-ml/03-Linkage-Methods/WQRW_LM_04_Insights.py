# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Edited:     October 22, 2026
Status:     Completed
"""

# =========================================================
# Overall Project Takeaway
# =========================================================

"""
OVERALL PROJECT TAKEAWAY

This project demonstrated that hierarchical clustering results are
highly dependent on linkage selection.

Single, Complete, and Average Linkage produced superior internal
validation metrics but failed to generate meaningful segmentation due
to severe cluster imbalance and outlier-driven solutions.

Ward Linkage consistently produced balanced cluster structures,
preserved meaningful hierarchical organization, and recovered known
wine-type patterns without access to class labels.

Further investigation revealed a logical hierarchy consisting of:

    White Wines
    Red Wines

        ↓

    Dry White Wines
    Sweet White Wines

        ↓

    Structured Red Wines
    High Volatile-Acidity Red Wines

These results indicate that Ward Linkage captured genuine
multidimensional structure within the Wine Quality dataset and
provided the most interpretable hierarchical clustering solution.
"""

# =========================================================
# File 1 WQRW_LM_01_Single.py 
# =========================================================

# =========================================================
# Single Linkage
# =========================================================

"""
KEY INSIGHTS

• Single Linkage was evaluated to demonstrate how minimum-distance
  cluster merging influences hierarchical clustering structure.

• Single Linkage achieved the highest Silhouette Score (0.7897)
  and lowest Davies-Bouldin Index (0.1492) among evaluated
  hierarchical solutions.

• Despite these strong internal validation metrics, cluster-size
  analysis revealed severe imbalance.

• The best solution assigned 5,319 wines to one cluster and only
  a single wine to the other cluster.

• Similar behavior persisted across all tested values of k.

• Dendrogram analysis confirmed the classic chaining effect
  associated with Single Linkage.

• The hierarchy collapsed into one dominant cluster while
  repeatedly isolating individual observations.

• This study demonstrated that internal clustering metrics can
  appear excellent even when the resulting segmentation is not
  meaningful.

FINAL TAKEAWAY

Single Linkage successfully illustrated the chaining phenomenon but
failed to produce practical segmentation of the Wine Quality dataset.
"""

# =========================================================
# Linkage Methods Comparison 
# =========================================================

"""
KEY INSIGHTS

• Four linkage methods were evaluated:

      Single
      Complete
      Average
      Ward

• Single, Complete, and Average Linkage achieved substantially
  stronger Silhouette Scores and Davies-Bouldin Indices than
  Ward Linkage.

• Cluster-balance analysis revealed that these apparently strong
  solutions were dominated by outlier isolation and highly
  imbalanced clusters.

• Single, Complete, and Average Linkage repeatedly assigned
  more than 99% of observations to a single cluster.

• Ward Linkage was the only method that consistently generated
  balanced cluster structures.

• Dendrogram analysis showed major differences in hierarchical
  behavior across linkage methods.

• Single Linkage exhibited chaining behavior.

• Complete Linkage emphasized outlier separation.

• Average Linkage behaved as a compromise between Single and
  Complete.

• Ward Linkage preserved meaningful hierarchical structure over
  substantially larger merge distances.

• Merge-distance analysis showed:

      Single    =  6.16
      Average   =  8.71
      Complete  = 12.86
      Ward      = 34.99

• Ward maintained cluster separation much longer than the other
  linkage methods.

FINAL TAKEAWAY

Best metrics did not correspond to the most meaningful clustering
solution. Ward Linkage emerged as the preferred strategy because it
produced the only balanced and interpretable hierarchy.
"""

# =========================================================
# File 2: WQRE_LM_02_Ward_Deep_Dive.py
# =========================================================

# =========================================================
# Ward Linkage Deep Dive
# =========================================================

"""
KEY INSIGHTS

• Ward Linkage was selected for deeper analysis because it was
  the only linkage method that consistently produced balanced
  clustering solutions.

• Ward k=2 recovered the dominant structure of the dataset by
  separating wines into primarily white and red populations.

• Ward k=3 revealed additional structure within the white-wine
  population.

• The white-wine hierarchy separated into:

      Dry, higher-alcohol whites

      Sweet, lower-alcohol whites

• Ward k=4 further subdivided the red-wine population.

• The resulting hierarchy revealed:

      Structured red wines

      High volatile-acidity red wines

      Dry white wines

      Sweet white wines

• Transition analysis showed that the hierarchy evolved
  logically as additional clusters were introduced.

• Previously discovered groups remained stable while new
  subgroup structure emerged.

• Wine-type recovery analysis demonstrated that Ward nearly
  reconstructed the known red-versus-white classification
  without using wine-type labels.

• Quality analysis showed meaningful differences among the
  resulting clusters despite quality not being included during
  clustering.

• Feature-driver analysis revealed that the hierarchy was
  influenced by many chemical characteristics simultaneously
  rather than a single dominant variable.

• Alcohol, citric acid, pH, quality, acidity measures, sulfur
  dioxide, and residual sugar all contributed to cluster
  separation.

FINAL TAKEAWAY

Ward Linkage uncovered a stable and interpretable hierarchy that
progressed naturally from broad wine populations to increasingly
specialized wine subgroups.
"""