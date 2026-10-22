# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October 22, 2026
Status:     Completed

Description

This analysis consolidates and compares the dimensionality-
reduction techniques evaluated throughout the project.

Methods Evaluated

    • PCA
    • Kernel PCA
    • t-SNE
    • UMAP
    • NMF

Objectives

    1. Compare linear and nonlinear dimensionality-
       reduction techniques.

    2. Evaluate cluster separation and latent structure
       discovery.

    3. Compare interpretability across methods.

    4. Determine which technique provides the most useful
       representation of the Wine Quality dataset.

Analysis Includes

    • PCA summary
    • Kernel PCA summary
    • t-SNE summary
    • UMAP summary
    • NMF summary
    • Method-comparison framework
    • Overall dimensionality-reduction ranking

Final Goal

    Identify the strengths, limitations, and practical use
    cases of each dimensionality-reduction technique.
"""

# =========================================================
# Imports
# =========================================================

from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# Plot Settings
# =========================================================

sns.set_theme(
    style="whitegrid",
    context="talk"
)

plt.rcParams[
    "figure.figsize"
] = (10, 6)

# =========================================================
# Output Directory
# =========================================================

output_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\04-Dimensionality-Reduction\04-dimensionality-results"
)

# =========================================================
# Load Results
# =========================================================

pca_summary = pd.read_csv(
    output_dir / "pca_summary.csv"
)

kpca_summary = pd.read_csv(
    output_dir / "kernel_pca_summary.csv"
)

tsne_summary = pd.read_csv(
    output_dir / "tsne_summary.csv"
)

umap_summary = pd.read_csv(
    output_dir / "umap_summary.csv"
)

nmf_summary = pd.read_csv(
    output_dir / "nmf_summary.csv"
)

dr_comparison = pd.read_csv(
    output_dir / "nmf_method_comparison.csv"
)

dr_rankings = pd.read_csv(
    output_dir / "dr_method_rankings.csv"
)

pca_df = pd.read_csv(
    output_dir / "pca_2d_projection.csv"
)

kpca_df = pd.read_csv(
    output_dir / "kernel_pca_projection.csv"
)

tsne_df = pd.read_csv(
    output_dir / "tsne_projection.csv"
)

umap_df = pd.read_csv(
    output_dir / "umap_projection.csv"
)

nmf_df = pd.read_csv(
    output_dir / "nmf_projection.csv"
)

# =========================================================
# Method Comparison
# =========================================================

print(
    dr_comparison
)

# =========================================================
# Rankings
# =========================================================

print(
    dr_rankings
)

# =========================================================
# Five Method Comparison
# =========================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18,12)
)

# PCA

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="type",
    alpha=0.4,
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
    alpha=0.4,
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
    alpha=0.4,
    ax=axes[0,2]
)

axes[0,2].set_title(
    "t-SNE"
)

# UMAP

sns.scatterplot(
    data=umap_df,
    x="UMAP1",
    y="UMAP2",
    hue="type",
    alpha=0.4,
    ax=axes[1,0]
)

axes[1,0].set_title(
    "UMAP"
)

# NMF

sns.scatterplot(
    data=nmf_df,
    x="Component1",
    y="Component2",
    hue="type",
    alpha=0.4,
    ax=axes[1,1]
)

axes[1,1].set_title(
    "NMF"
)

axes[1,2].axis("off")

plt.tight_layout()

plt.show()

print(
"""
Results Interpretation

PCA successfully captured the dominant variance structure of
the dataset and provided the strongest mathematical
interpretation through explained variance and feature
loadings.

Kernel PCA revealed nonlinear chemistry relationships and
produced stronger separation than standard PCA.

t-SNE generated the strongest local cluster separation and
produced highly distinct wine populations.

UMAP matched t-SNE's separation while preserving a cleaner
global organization of the data.

NMF revealed interpretable latent chemistry factors and
provided insight into how wine characteristics combine to
form wine populations.

All methods consistently identified wine type as the
dominant source of variation while quality remained highly
overlapping.
"""
)

print(
"""
KEY INSIGHTS

• Red and white wines formed distinct populations across
  all dimensionality-reduction methods.

• Quality ratings did not form naturally separated groups.

• PCA provided the strongest variance-based interpretation.

• Kernel PCA revealed nonlinear wine chemistry structure.

• t-SNE produced the strongest local separation.

• UMAP provided the best overall visualization.

• NMF revealed the most interpretable chemistry factors.

• Multiple independent methods reinforced the same
  conclusion regarding wine-type structure.

FINAL TAKEAWAY

PCA explained variance.

Kernel PCA revealed nonlinear structure.

t-SNE maximized local separation.

UMAP provided the strongest overall visualization.

NMF explained the latent chemistry components underlying
wine composition.
"""
)

print(
"""
FINAL RECOMMENDATION

Best Variance Explanation:
    PCA

Best Nonlinear Representation:
    Kernel PCA

Best Local Separation:
    t-SNE

Best Overall Visualization:
    UMAP

Best Chemistry Interpretation:
    NMF

Overall Winner:
    UMAP

UMAP delivered the strongest balance between cluster separation, structure 
preservation, and visual interpretability while consistently revealing the 
dominant wine-type structure within the dataset.
"""
)
    
print(
"""
Warning and Note:

The nonlinear dimensionality-reduction techniques revealed evidence of 
additional internal structure within both red and white wine populations.

Kernel PCA, t-SNE, and UMAP suggested the presence of multiple subgroups 
rather than a simple binary separation of red and white wines.

These patterns are consistent with previous Ward clustering results, which 
identified meaningful subpopulations within both major wine types.

While dimensionality-reduction embeddings alone cannot confirm the existence 
of true clusters, the agreement between several embedding techniques and 
hierarchical clustering provides evidence of underlying subgroup structure 
within the dataset.
"""
)

print(
"""
Honest Assessment:

For the entire project so far:
    
2 obvious groups?
    Absolutely.

3 groups?
    Likely.

4 groups?
    Supported by Ward clustering.

5+ groups?
    Not enough evidence yet.
    
The fact that Ward, Kernel PCA, t-SNE, and UMAP are all hinting at internal 
subdivisions is actually one of the strongest findings from the project. 
It suggests the data is not simply "red wine vs white wine" but contains 
meaningful nested subgroups within those populations.
"""
)