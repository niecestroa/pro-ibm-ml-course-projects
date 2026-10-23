# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October 22, 2026
Status:     Completed

Description

Agglomerative Clustering answered:
- "Which hierarchical clustering solution works best?"

Linkage Analysis answers:
- "Why does Ward work best, and how do linkage choices affect hierarchical clustering?"

# =========================================================
# Single Linkage Analysis
# =========================================================

Purpose

Evaluates Single Linkage as a hierarchical clustering method and
demonstrates the chaining effect.

What the code does

• Tests Single Linkage across multiple cluster counts.
• Calculates Silhouette Score and Davies-Bouldin Index.
• Evaluates cluster-size balance.
• Creates a dendrogram showing how observations merge.
• Identifies the best-performing Single Linkage solution.

Why it matters

Although Single Linkage produced excellent validation metrics, the
resulting clusters were extremely imbalanced, with nearly all wines
assigned to a single cluster. The analysis demonstrates how clustering
metrics can be misleading when cluster balance is ignored.

# =========================================================
# Linkage Method Comparison
# =========================================================

Purpose

Compares the four major hierarchical linkage strategies.

What the code does

• Evaluates Single, Complete, Average, and Ward Linkage.
• Compares Silhouette Score and Davies-Bouldin Index.
• Measures cluster-size balance.
• Identifies practical versus misleading clustering solutions.

Why it matters

Single, Complete, and Average Linkage achieved strong validation scores
but produced highly imbalanced clusters. Ward Linkage was the only
method that consistently generated balanced and interpretable cluster
structures.

# =========================================================
# Dendrogram Comparison
# =========================================================

Purpose

Visualizes how different linkage methods construct hierarchical
cluster structures.

What the code does

• Generates dendrograms for Single, Complete, Average, and Ward
  Linkage using the same wine sample.
• Compares branch structure and merge behavior.
• Measures maximum merge distance for each linkage method.

Why it matters

The dendrograms reveal why linkage methods behave differently.
Single Linkage exhibits chaining, Complete Linkage isolates outliers,
Average Linkage behaves as a compromise, and Ward preserves meaningful
hierarchical structure over larger merge distances.
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
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\03-Linkage-Methods\03-linkage-results"
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
    r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\03-Linkage-Methods\03-linkage-results"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# Part 1: Single Linkage
# =========================================================

"What happens when Single Linkage is applied to the Wine Quality dataset?"

# =========================================================
# 4. Single Linkage Evaluation
# =========================================================

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

single_results = []

for k in range(2, 21):

    single_model = AgglomerativeClustering(
        n_clusters=k,
        linkage="single"
    )

    labels = single_model.fit_predict(
        X_scaled
    )

    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
    )

    single_results.append(
        {
            "clusters": k,
            "silhouette_score": silhouette_score(
                X_scaled,
                labels
            ),
            "davies_bouldin_index": davies_bouldin_score(
                X_scaled,
                labels
            ),
            "smallest_cluster": cluster_sizes.min(),
            "largest_cluster": cluster_sizes.max(),
            "smallest_cluster_pct":
                cluster_sizes.min()
                / len(labels)
                * 100,
            "largest_cluster_pct":
                cluster_sizes.max()
                / len(labels)
                * 100
        }
    )

single_results_df = pd.DataFrame(
    single_results
)

print(
    "\n----- SINGLE LINKAGE RESULTS -----"
)

print(
    single_results_df
    .sort_values(
        by="silhouette_score",
        ascending=False
    )
    .round(4)
)

# =========================================================
# 5. Single Linkage Performance
# =========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,6)
)

sns.lineplot(
    data=single_results_df,
    x="clusters",
    y="silhouette_score",
    marker="o",
    ax=axes[0]
)

axes[0].set_title(
    "Single Linkage Silhouette Score"
)

sns.lineplot(
    data=single_results_df,
    x="clusters",
    y="davies_bouldin_index",
    marker="o",
    ax=axes[1]
)

axes[1].set_title(
    "Single Linkage Davies-Bouldin Index"
)

plt.tight_layout()

plt.show()

# =========================================================
# 6. Single Linkage Cluster Balance
# =========================================================

plt.figure(figsize=(10,5))

sns.lineplot(
    data=single_results_df,
    x="clusters",
    y="smallest_cluster_pct",
    marker="o"
)

plt.title(
    "Single Linkage Smallest Cluster Percentage"
)

plt.ylabel(
    "% of Dataset"
)

plt.tight_layout()

plt.show()

print(
    "\n----- SINGLE LINKAGE BALANCE -----"
)

print(
    single_results_df[
        [
            "clusters",
            "smallest_cluster_pct",
            "largest_cluster_pct"
        ]
    ]
)

# =========================================================
# 7. Single Linkage Dendrogram
# =========================================================

from scipy.cluster.hierarchy import (
    linkage,
    dendrogram
)

sample_df = (
    pd.DataFrame(X_scaled)
    .sample(
        n=250,
        random_state=RANDOM_STATE
    )
)

single_linkage = linkage(
    sample_df,
    method="single"
)

plt.figure(figsize=(14,6))

dendrogram(
    single_linkage,
    truncate_mode="level",
    p=5
)

plt.title(
    "Single Linkage Dendrogram"
)

plt.xlabel(
    "Sampled Wines"
)

plt.ylabel(
    "Distance"
)

plt.tight_layout()

plt.show()

# =========================================================
# 8. Best Single Linkage Solution
# =========================================================

best_single = (
    single_results_df
    .sort_values(
        by="silhouette_score",
        ascending=False
    )
    .iloc[0]
)

print(
    "\n----- BEST SINGLE LINKAGE MODEL -----"
)

print(best_single)

# =========================================================
# Part 1 Insights
# =========================================================

"""
This is actually a fantastic result and exactly why the linkage-method project 
is worth doing.

Single Linkage completely exposed one of the biggest weaknesses of hierarchical 
clustering.

What happened?

The metrics look incredible:

Silhouette = 0.7897
Davies-Bouldin = 0.1492

Those are better than:

K-Means
GMM
Ward
DBSCAN

But the cluster sizes reveal the truth:

k=2

Cluster 0 = 5319 wines
Cluster 1 =    1 wine

and:

smallest_cluster_pct = 0.0188%
largest_cluster_pct  = 99.9812%

The model is essentially saying:

"All wines belong together except this one weird wine."

Why Single Linkage Failed

Single linkage merges clusters using:

minimum distance between observations.

Because only one nearby point is needed to connect two groups:

A ---- B ---- C ---- D ---- E

eventually becomes one giant chain.

This is called: Chaining Effect

Instead of finding: Compact groups

Single Linkage creates:

One enormous cluster
+
Tiny isolated observations

Exactly what happened here.

The Smallest Cluster Graph. The graph is actually one of the strongest pieces 
of evidence. The smallest cluster size remains: 

0.0188% for:

k=2
through
k=20


The same wine remains isolated the entire time. That's not clustering.

That's: Repeated outlier isolation

The Dendrogram

The dendrogram is also revealing.

Note: Almost everything merges together very early

Then:

A couple observations remain detached until very high linkage distances.

That's a classic signature of:

Single Linkage Chaining

What This Proves

Before running this experiment we knew:


- Complete Linkage was isolating outliers.

Now we've learned:

Single Linkage does essentially the same thing on this dataset.

The difference is:

Single
→ connects everything into one chain

Complete
→ isolates distant points

Average
→ middle ground

Ward
→ balanced compact groups

Report-Ready Interpretation

Single Linkage produced the highest Silhouette Score
(0.7897) and lowest Davies-Bouldin Index (0.1492)
among all linkage methods. However, cluster-size
analysis revealed that these apparently excellent
metrics were driven by trivial solutions.

For k=2, Single Linkage assigned 5,319 wines to a
single cluster while isolating only one observation
into a separate cluster. Similar behavior persisted
for all evaluated cluster counts. This outcome
illustrates the classic chaining effect associated
with Single Linkage, in which observations become
connected through sequences of nearby neighbors,
resulting in one dominant cluster and a small number
of isolated observations.

Although the internal validation metrics appeared
strong, the resulting solutions were not useful for
segmentation or interpretation. Therefore Single
Linkage was not considered a practical clustering
strategy for the Wine Quality dataset.
"""

# =========================================================
# Part 2: Linkage Method Comparison
# =========================================================

"""
Perfect. Now we move from:

"How does Single Linkage behave?"

to

"How do all linkage methods compare?"
"""

# =========================================================
# 9. Linkage Method Evaluation
# =========================================================

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

linkage_methods = [
    "single",
    "complete",
    "average",
    "ward"
]

linkage_results = []

for linkage in linkage_methods:

    for k in range(2, 21):

        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage
        )

        labels = model.fit_predict(
            X_scaled
        )

        cluster_sizes = (
            pd.Series(labels)
            .value_counts()
        )

        linkage_results.append(
            {
                "linkage": linkage,
                "clusters": k,
                "silhouette_score":
                    silhouette_score(
                        X_scaled,
                        labels
                    ),

                "davies_bouldin_index":
                    davies_bouldin_score(
                        X_scaled,
                        labels
                    ),

                "smallest_cluster":
                    cluster_sizes.min(),

                "largest_cluster":
                    cluster_sizes.max(),

                "smallest_cluster_pct":
                    (
                        cluster_sizes.min()
                        / len(labels)
                        * 100
                    ),

                "largest_cluster_pct":
                    (
                        cluster_sizes.max()
                        / len(labels)
                        * 100
                    )
            }
        )

linkage_results_df = pd.DataFrame(
    linkage_results
)

print(
    "\n----- LINKAGE METHOD RESULTS -----"
)

print(
    linkage_results_df
    .sort_values(
        by="silhouette_score",
        ascending=False
    )
    .round(4)
    .head(25)
)

# =========================================================
# 10. Metric Comparison
# =========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,6)
)

sns.lineplot(
    data=linkage_results_df,
    x="clusters",
    y="silhouette_score",
    hue="linkage",
    marker="o",
    ax=axes[0]
)

axes[0].set_title(
    "Silhouette Score by Linkage Method"
)

sns.lineplot(
    data=linkage_results_df,
    x="clusters",
    y="davies_bouldin_index",
    hue="linkage",
    marker="o",
    ax=axes[1]
)

axes[1].set_title(
    "Davies-Bouldin Index by Linkage Method"
)

plt.tight_layout()

plt.show()

# =========================================================
# 11. Cluster Balance Summary
# =========================================================

balance_summary = (
    linkage_results_df
    .groupby("linkage")
    .agg({
        "smallest_cluster_pct": "max",
        "silhouette_score": "max",
        "davies_bouldin_index": "min"
    })
)

print(
    "\n----- LINKAGE BALANCE SUMMARY -----"
)

print(
    balance_summary
    .round(4)
)

# =========================================================
# 12. Balanced Linkage Solutions
# =========================================================

balanced_df = (
    linkage_results_df[
        linkage_results_df[
            "smallest_cluster_pct"
        ] >= 5
    ]
)

print(
    "\n----- BALANCED LINKAGE SOLUTIONS -----"
)

print(
    balanced_df
    .sort_values(
        by="silhouette_score",
        ascending=False
    )
    .round(4)
)

# =========================================================
# Part 2 Insights
# =========================================================

"""
This is an excellent result, and honestly it already tells most of the story 
of the project.

Your table revealed something very important:

Linkage	Best Silhouette	Best DB Index	

Single      0.7897      0.1492      99.98%
Complete	0.7897      0.1492      99.98%
Average	    0.7897      0.1492      99.98%
Ward        0.2607      1.4726      72.22%

At first glance:

Single
Complete
Average

appear dramatically better than Ward. But this is completely misleading.

The Real Discovery

Your balance table proves:

Single Linkage
Complete Linkage
Average Linkage

all suffer from essentially the same problem.

They produce:

One giant cluster
+
One or more tiny clusters

Example:

5319 wines
1 wine

or

5297 wines
21 wines
1 wine
1 wine

These are not useful cluster solutions.


The Biggest Surprise

Single Linkage behaved almost identically to Complete and Average Linkage.

Before running this I expected:

Single -> chaining
Complete -> compact
Average -> compromise

But for this dataset:

Single
Complete
Average

all ultimately ended up producing highly imbalanced solutions.

That's actually a really strong finding.

What the Graphs Show
Silhouette Plot

The blue, orange, and green curves:

Single
Complete
Average

dominate the chart.

But now we know why:

They are rewarded for isolating tiny groups.

The metrics look fantastic because:

1 outlier
vs
5319 wines

is extremely easy to separate.

Davies-Bouldin Plot has the same issue. Lower is better.

Single Linkage appears amazing: 0.149 compared to: Ward ≈ 1.7

Yet Ward is the only method producing meaningful segmentation.

BALANCED LINKAGE SOLUTIONS

Only:

Ward k=2
Ward k=3
Ward k=4

survives. Everything else gets eliminated.

That is a very powerful result.

Conclusion:

Although Single, Complete, and Average linkage achieved substantially higher 
Silhouette Scores and lower Davies-Bouldin Indices than Ward linkage, cluster 
balance analysis revealed that these methods consistently produced trivial 
solutions. In most cases, more than 99% of observations were assigned to a 
single cluster while only one or a few observations formed separate clusters. 
These outcomes were interpreted as outlier-isolation behavior rather than 
meaningful segmentation. After applying a minimum cluster-size requirement, 
Ward linkage was the only method that consistently produced balanced and 
interpretable cluster structures. Therefore, Ward linkage was selected as 
the preferred hierarchical clustering strategy for the Wine Quality dataset.
"""

# =========================================================
# Part 3: Dendrogram Comparison
# =========================================================

# =========================================================
# 13. Dendrogram Comparison
# =========================================================

from scipy.cluster.hierarchy import (
    linkage,
    dendrogram
)

sample_df = (
    pd.DataFrame(X_scaled)
    .sample(
        n=250,
        random_state=RANDOM_STATE
    )
)

# =========================================================
# 14. Dendrogram Comparison
# =========================================================

linkage_methods = [
    "single",
    "complete",
    "average",
    "ward"
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(18, 12)
)

axes = axes.flatten()

for ax, method in zip(
    axes,
    linkage_methods
):

    Z = linkage(
        sample_df,
        method=method
    )

    dendrogram(
        Z,
        truncate_mode="level",
        p=5,
        ax=ax
    )

    ax.set_title(
        f"{method.capitalize()} Linkage"
    )

    ax.set_xlabel(
        "Sampled Wines"
    )

    ax.set_ylabel(
        "Distance"
    )

plt.tight_layout()

plt.show()

# =========================================================
# 15. Dendrogram Distance Comparison
# =========================================================

distance_summary = []

for method in linkage_methods:

    Z = linkage(
        sample_df,
        method=method
    )

    distance_summary.append(
        {
            "linkage": method,
            "max_merge_distance": Z[:, 2].max(),
            "mean_merge_distance": Z[:, 2].mean()
        }
    )

distance_df = pd.DataFrame(
    distance_summary
)

print(
    "\n----- DENDROGRAM DISTANCE SUMMARY -----"
)

print(
    distance_df
    .round(4)
)

"""
Overall Objective:

The purpose of dendrogram analysis was to move beyond clustering metrics and 
directly examine how different linkage methods construct hierarchical cluster 
structures. While Silhouette Score and Davies-Bouldin Index provide useful 
information about cluster separation, dendrograms reveal how observations merge 
throughout the hierarchy and whether those merges produce meaningful groupings.

1. Single Linkage

Maximum Merge Distance: 6.16

The Single Linkage dendrogram displayed the shallowest hierarchical structure 
among all evaluated linkage methods.

Key observations:

Observations merged rapidly at very small linkage distances.
Most wines became connected through a sequence of nearby observations.
Few major branches remained separated for long periods.
The hierarchy collapsed into a single dominant structure early in the clustering process.

Interpretation:

Single Linkage exhibited the classic chaining effect, where clusters grow through 
a series of nearest-neighbor connections rather than forming compact groups. 
As a result, the method generated one extremely large cluster while repeatedly 
isolating a handful of observations into tiny clusters.

This behavior explains why Single Linkage achieved extremely strong validation 
metrics while simultaneously producing highly imbalanced clustering solutions.

2. Complete Linkage

Maximum Merge Distance: 12.86

The Complete Linkage dendrogram displayed substantially stronger separation 
between branches than Single Linkage.

Key observations:

Distinct groups remained separated for longer distances.
Outlying observations merged only near the top of the hierarchy.
Larger gaps appeared between major branch mergers.

Interpretation:

Complete Linkage prioritizes maximum pairwise distance between clusters, 
encouraging compact and highly separated structures. However, this behavior 
also caused isolated wines to remain detached from the primary population, 
producing trivial cluster solutions in which one or two observations formed 
their own clusters.

Although the resulting Silhouette Scores were very high, the cluster-size 
analysis demonstrated that these solutions were not practically useful for 
segmentation.

3. Average Linkage

Maximum Merge Distance: 8.71

The Average Linkage dendrogram exhibited behavior between Single and Complete Linkage.

Key observations:

Branch structure was more balanced than Single Linkage.
Cluster separation was less extreme than Complete Linkage.
Several intermediate mergers occurred throughout the hierarchy.

Interpretation:

Average Linkage acts as a compromise between Single and Complete Linkage by 
considering the average pairwise distance between clusters. While this produced 
a more balanced hierarchy than Single Linkage, cluster-size analysis revealed 
that the method continued to isolate extremely small groups.

Consequently, Average Linkage did not resolve the fundamental imbalance problem 
observed with the other non-Ward methods.

4. Ward Linkage

Maximum Merge Distance: 34.99

The Ward Linkage dendrogram displayed dramatically different behavior from the 
other linkage methods.

Key observations:

Large branches remained separated until substantially larger distances.
Multiple levels of hierarchical structure were visible.
Two dominant branches persisted throughout much of the hierarchy.
Several meaningful subgroups were visible within each major branch.

Interpretation:

Unlike the other linkage methods, Ward Linkage minimizes increases in 
within-cluster variance rather than relying solely on pairwise distances 
between observations. This produced larger and more balanced cluster 
structures that remained distinct across much of the hierarchy.

The dendrogram revealed a clear hierarchical organization consisting of two 
dominant wine populations with additional substructure nested within each branch.

This finding aligns closely with previous Ward clustering results:

Ward k=2
→ Two dominant wine populations

Ward k=3
→ Additional white-wine subgroup structure

Ward k=4
→ Further subdivision within the major populations
"""

# =========================================================
# 16. Merge Distance Comparison
# =========================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=distance_df,
    x="linkage",
    y="max_merge_distance"
)

plt.title(
    "Maximum Merge Distance by Linkage"
)

plt.ylabel(
    "Distance"
)

plt.tight_layout()

plt.show()

"""
5. Merge Distance Comparison

The comparison of maximum merge distances provided additional evidence that the 
linkage methods behaved fundamentally differently.

Linkage	Maximum Merge DistanceSingle	6.16
Average	8.71
Complete	12.86
Ward	34.99

Key implication:

Single, Average, and Complete Linkage merged observations rapidly.
Ward preserved major structures for much longer periods before merging.
The substantial increase in merge distance observed for Ward indicates a 
stronger and more meaningful hierarchical organization within the dataset.
"""

# =========================================================
# 17. Merge Distance Ranking
# =========================================================

print(
    "\n----- MERGE DISTANCE RANKING -----"
)

print(
    distance_df
    .sort_values(
        by="max_merge_distance",
        ascending=False
    )
)

"""
Section 17 Conclusion

The merge-distance ranking reinforces the conclusions reached through cluster-balance 
evaluation and dendrogram analysis. Single, Average, and Complete Linkage merged 
observations into large structures relatively quickly despite achieving strong 
internal validation scores. Ward Linkage preserved meaningful hierarchical structure 
across much larger distances, resulting in balanced clusters and interpretable 
subgroups. The substantial gap between Ward and the remaining linkage methods 
provides strong evidence that Ward is the most suitable linkage strategy for the 
Wine Quality dataset.
"""

"""
Section 16 and 17 Purpose

Maximum merge distance measures how long distinct hierarchical structures remain 
separated before being merged into a larger cluster. Larger values indicate 
that substantial cluster structures persist throughout more of the hierarchical 
clustering process.

Merge Distance Ranking
Ward      : 34.99
Complete  : 12.86
Average   :  8.71
Single    :  6.16

1. Single Linkage

Maximum Merge Distance: 6.16

Single Linkage produced the smallest maximum merge distance among all evaluated 
methods.

Key observations:

Most observations merged rapidly.
Large structures formed early in the hierarchy.
Few distinct branches remained separated for long periods.

Interpretation:

The small maximum merge distance provides additional evidence of the chaining 
effect observed earlier. Observations were connected through nearest-neighbor 
relationships, causing the hierarchy to collapse quickly into one dominant cluster. 
This behavior explains the highly imbalanced solutions observed during 
cluster-balance analysis.

2. Average Linkage

Maximum Merge Distance: 8.71

Average Linkage produced a slightly deeper hierarchy than Single Linkage.

Key observations:

Some additional branch structure was preserved.
Clusters remained distinct somewhat longer.
Merging still occurred relatively early.

Interpretation:

Average Linkage moderated the chaining behavior of Single Linkage but ultimately 
produced similar cluster-imbalance issues. Although more hierarchical structure 
was preserved, meaningful segmentation remained limited.

3. Complete Linkage

Maximum Merge Distance: 12.86

Complete Linkage produced substantially larger merge distances.

Key observations:

Distinct groups remained separated longer.
Larger linkage distances were required before major mergers occurred.
Outlying observations stayed detached throughout much of the hierarchy.

Interpretation:

Complete Linkage encouraged strong separation between groups and generated 
deeper hierarchical structures than Single or Average Linkage. However, this 
 primarily reflected outlier isolation rather than practical dataset segmentation.

4. Ward Linkage

Maximum Merge Distance: 34.99

Ward Linkage produced dramatically larger merge distances than every other method.

Key observations:

Maximum merge distance was nearly three times larger than Complete Linkage.
Large independent branches remained separated throughout much of the hierarchy.
Multiple levels of nested structure were preserved before final mergers occurred.

Interpretation:

Ward Linkage maintained meaningful cluster structure for substantially longer 
than the other linkage methods. Rather than rapidly merging observations or 
isolating individual outliers, Ward preserved broad population structure while 
simultaneously retaining important subgroups.

This behavior aligns closely with previous findings:

Ward k=2
→ Two dominant wine populations

Ward k=3
→ Additional white-wine subgroup structure

Ward k=4
→ Additional red-wine subgroup structure

At this point, your linkage-method project has a very clean narrative:

Part 1
Single Linkage
    ↓
Part 2
Single vs Complete vs Average vs Ward
    ↓
Part 3
Dendrogram & Merge-Distance Analysis
    ↓
Part 4
Why Ward Wins

which is honestly stronger from an educational standpoint than the original 
Agglomerative Clustering section.
"""

# =============================================
# Part 3 Conclulsion
# =============================================

"""
Part 3 Conclusions

Dendrogram analysis provided critical context for interpreting the results
of the linkage-method evaluation. Although Single, Complete, and Average
Linkage achieved substantially stronger Silhouette Scores and lower
Davies-Bouldin Indices than Ward Linkage, visual inspection of the
dendrograms and cluster-balance analysis revealed that these solutions
were largely driven by outlier isolation and highly imbalanced clusters.

Single Linkage produced the smallest maximum merge distance (6.16) and
exhibited the classic chaining effect. Observations merged rapidly into
one dominant cluster, resulting in trivial clustering solutions where
more than 99% of wines belonged to a single group.

Complete Linkage produced larger merge distances (12.86) and stronger
apparent separation between branches. However, cluster-size analysis
showed that this method frequently isolated only one or a few wines while
assigning nearly all observations to a single cluster.

Average Linkage displayed behavior between Single and Complete Linkage,
with a maximum merge distance of 8.71. Although its hierarchy appeared
somewhat more balanced, it ultimately produced similarly imbalanced
clustering solutions and failed to generate practical segmentation.

Ward Linkage behaved fundamentally differently. It generated the largest
maximum merge distance (34.99), preserved major cluster structures for
substantially longer periods, and revealed two dominant branches with
additional nested subgroup structure. Unlike the other linkage methods,
Ward consistently produced balanced and interpretable cluster solutions.

The comparison demonstrates that hierarchical clustering evaluation
should not rely exclusively on internal validation metrics. Cluster
balance, dendrogram structure, and merge-distance behavior provide
essential information that can expose misleading clustering solutions.
When these factors were considered together, Ward Linkage emerged as the
most appropriate linkage strategy for the Wine Quality dataset.
"""

# =========================================================
# Key Findings
# =========================================================

"""
Key Findings

• Single Linkage demonstrated severe chaining behavior and produced one
  dominant cluster with isolated observations.

• Complete Linkage generated compact clusters but primarily isolated
  outliers rather than uncovering meaningful population structure.

• Average Linkage provided intermediate behavior but still failed to
  produce balanced segmentation.

• Ward Linkage preserved meaningful hierarchy, balanced cluster sizes,
  and interpretable subgroup structure.

• The largest maximum merge distance was observed for Ward Linkage
  (34.99), indicating substantially stronger hierarchical organization
  than Single (6.16), Average (8.71), or Complete (12.86) Linkage.

• Internal validation metrics alone would have incorrectly favored
  Single, Complete, and Average Linkage.

• Combining cluster-balance analysis, dendrogram interpretation, and
  merge-distance evaluation identified Ward Linkage as the preferred
  hierarchical clustering strategy for this dataset.
"""

# =========================================================
# Linkage Methods: Key Insights
# =========================================================

"""
• Single, Complete, Average, and Ward Linkage methods were
  evaluated to understand how linkage selection influences
  hierarchical clustering structure.

• Single Linkage, Complete Linkage, and Average Linkage all
  achieved substantially higher Silhouette Scores and lower
  Davies-Bouldin Indices than Ward Linkage.

• At first glance, these results suggested that Single,
  Complete, and Average Linkage were superior clustering
  strategies.

• Cluster-size analysis revealed the opposite conclusion.

• Single Linkage produced extreme cluster imbalance across
  all evaluated cluster counts, repeatedly assigning more
  than 99% of wines to a single cluster while isolating one
  observation into a separate cluster.

• This behavior is consistent with the classic chaining
  effect associated with Single Linkage, where observations
  become connected through a series of nearest-neighbor
  relationships.

• Complete Linkage and Average Linkage produced similar
  outcomes, frequently isolating only one or a few wines
  while assigning the overwhelming majority of observations
  to one dominant cluster.

• Although these methods achieved excellent internal
  validation metrics, the resulting solutions were not
  meaningful for practical segmentation.

• Balanced-cluster filtering requiring each cluster to
  contain at least 5% of the dataset eliminated all Single,
  Complete, and Average Linkage solutions.

• Ward Linkage was the only linkage strategy that consistently
  produced balanced and interpretable cluster structures.

• The strongest Ward solutions were:

      Ward k = 2
      Ward k = 3
      Ward k = 4

• Dendrogram analysis provided additional evidence of
  fundamental differences between linkage methods.

• Single Linkage produced the shallowest hierarchy and the
  smallest maximum merge distance (6.16), indicating that
  observations merged into large structures quickly.

• Average Linkage produced a slightly deeper hierarchy with a
  maximum merge distance of 8.71.

• Complete Linkage generated larger merge distances (12.86)
  and stronger separation between branches, but this
  separation was largely driven by outlier isolation.

• Ward Linkage produced a dramatically larger maximum merge
  distance (34.99), preserving major branches throughout
  much more of the hierarchical clustering process.

• The Ward dendrogram revealed two dominant wine populations
  together with meaningful nested subgroup structure.

• The large separation between Ward and the remaining
  linkage methods indicates that Ward preserved meaningful
  hierarchical organization that was not captured by Single,
  Complete, or Average Linkage.

• This study demonstrates that internal validation metrics
  alone can be misleading when evaluating hierarchical
  clustering methods.

• Cluster balance, dendrogram structure, and merge-distance
  behavior provided critical context that exposed the
  weaknesses of Single, Complete, and Average Linkage.

• Considering all evaluation criteria together, Ward
  Linkage emerged as the most effective and interpretable
  linkage strategy for the Wine Quality dataset.

FINAL RECOMMENDATION

Preferred Linkage Method: Ward

Reasons:
- Only linkage method producing balanced cluster solutions
- Preserved meaningful hierarchical structure
- Largest maximum merge distance (34.99)
- Revealed dominant wine populations and subgroup structure
- Most interpretable clustering hierarchy

Highest Metric Scores:
- Single Linkage
- Complete Linkage
- Average Linkage

Most Practical Linkage:
- Ward Linkage
"""

"""
**Most important Finding:**

Best Metrics does not always mean the Best Clustering
"""

# Exports

linkage_results_df.to_csv(
    output_dir / "linkage_method_comparison.csv",
    index=False
)

balanced_df.to_csv(
    output_dir / "linkage_balanced_solutions.csv",
    index=False
)

distance_df.to_csv(
    output_dir / "merge_distance_ranking.csv",
    index=False
)

distance_df.to_csv(
    output_dir / "merge_distance_ranking.csv",
    index=False
)