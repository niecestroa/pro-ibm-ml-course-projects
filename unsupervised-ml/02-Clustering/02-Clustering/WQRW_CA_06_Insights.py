# -*- coding: utf-8 -*-
"""
Author:     Aaron Niecestro
Created:    October 21, 2026
Editted:    October 21, 2026
Status:     Completed

Key Insights of the 4 section of Unsupervised Clustering
"""

# =========================================================
# K-Means: Key Insights
# =========================================================

print(
    """
KEY INSIGHTS

• K-Means was evaluated across k values ranging from 2 to 20.

• The highest silhouette score (0.2704) occurred at k=2, but this solution
  was considered overly simplistic for meaningful segmentation.

• The lowest Davies-Bouldin Index (1.4844) occurred at k=4, indicating the
  best balance between cluster compactness and separation.

• The elbow plot showed diminishing reductions in inertia beyond four
  clusters, supporting k=4 as the optimal solution.

• Cluster stability was exceptionally high, achieving an average Adjusted
  Rand Index (ARI) of 0.9986 across multiple random seeds.

• PCA, t-SNE, and UMAP visualizations all revealed a clear four-cluster
  structure with limited overlap between groups.

• Cluster profiling identified distinct wine populations characterized by
  differences in alcohol content, residual sugar, acidity, sulphates, and
  sulfur dioxide concentrations.

• The final K-Means solution consisted of four stable and interpretable wine
  profiles.

FINAL MODEL

Algorithm: K-Means
Clusters: 4
Silhouette Score: 0.2428
Davies-Bouldin Index: 1.4844
Average ARI: 0.9986
"""
)

# =========================================================
# Gaussian Mixture Models (GMM) Summary
# =========================================================

print(
    """
KEY INSIGHTS

• Gaussian Mixture Models were evaluated using multiple component counts and
  covariance structures.

• Component counts of 2, 6, and 13 were investigated as candidate solutions.

• Full covariance produced the strongest likelihood-based model fit, while
  diagonal covariance produced the strongest cluster separation and
  stability.

• Although the 13-component full-covariance model achieved the lowest BIC,
  it exhibited poor cluster separation and weak stability.

• The 6-component full-covariance model identified several chemically
  distinct wine profiles but displayed considerable overlap between groups.

• The 2-component diagonal-covariance model achieved the highest silhouette
  score (0.2516) and exceptional stability (ARI = 0.9946).

• PCA and t-SNE visualizations revealed two clearly separated latent wine
  populations.

• Comparison against the excluded wine type variable demonstrated that the
  two-component solution reproduced the red-versus-white wine distinction
  with approximately 95.13% agreement.

• The final GMM solution was selected because it provided the strongest
  balance of separation, stability, interpretability, and reproducibility.

FINAL MODEL

Algorithm: Gaussian Mixture Model
Components: 2
Covariance Type: Diagonal
Silhouette Score: 0.2516
Davies-Bouldin Index: 1.7782
Average ARI: 0.9946
Agreement with Wine Type: 95.13%
"""
)

# =========================================================
# Agglomerative Clustering: Key Insights
# =========================================================

print(
    """
These Ward results are actually much better and much more believable than 
the earlier Complete/Average solutions.

KEY INSIGHTS

• Complete and Average linkage initially produced extremely strong
  validation metrics, with silhouette scores approaching 0.79 and
  Davies-Bouldin indices near 0.15.

• Inspection of cluster sizes revealed these solutions were highly
  imbalanced, often assigning more than 99% of observations to a
  single cluster while isolating only one or a few wines.

• These solutions were therefore interpreted as outlier-detection
  behavior rather than meaningful wine segmentation.

• A cluster-balance filter requiring each cluster to contain at
  least 5% of the dataset was introduced to identify more practical
  clustering solutions.

• After applying this constraint, Ward linkage produced the strongest
  valid clustering solutions.

• Ward linkage with two clusters achieved the highest silhouette score
  among the balanced solutions (0.2607) while maintaining reasonable
  cluster proportions of 72.2% and 27.8%.

• Cluster profiles showed two chemically distinct wine populations.

• Cluster 0 contained wines with:
  - higher residual sugar
  - higher sulfur dioxide concentrations
  - lower volatile acidity
  - slightly higher quality scores

• Cluster 1 contained wines with:
  - higher fixed acidity
  - higher volatile acidity
  - higher chlorides
  - higher sulphates
  - lower residual sugar

• PCA revealed moderate separation between the two Ward clusters.

• t-SNE showed substantially stronger separation, with the two groups
  forming distinct regions in the projected feature space.

• UMAP also displayed a clear two-group structure with limited overlap,
  supporting the existence of two dominant latent wine populations.

• Unlike K-Means, which identified four wine profiles, Agglomerative
  Clustering favored a broader two-group segmentation of the dataset.

• Ward linkage with two clusters was selected as the final
  Agglomerative model because it achieved the highest silhouette
  score among the balanced solutions (0.2607), maintained
  reasonable cluster sizes, and reproduced the strongest natural
  separation within the dataset.

• Comparison against the excluded wine type variable showed that
  the two-cluster Ward solution achieved approximately 96.94%
  agreement with the known red and white wine categories,
  despite wine type not being used during clustering.

• Dendrogram analysis revealed additional hierarchical structure
  beyond the final two-cluster solution.

• Ward linkage with three clusters produced the lowest
  Davies-Bouldin Index (1.5702) and aligned closely with the
  three major branches visible in the dendrogram.

• The three-cluster solution preserved the predominantly red-wine
  group while separating white wines into two distinct profiles:
  a drier, higher-alcohol, higher-quality subgroup and a sweeter,
  higher-sulfur, lower-alcohol subgroup.

• Ward linkage with four clusters provided the most detailed
  wine-profile segmentation.

• The four-cluster solution further divided the red-wine
  population into two chemically distinct profiles while
  preserving the two white-wine profiles identified in the
  three-cluster solution.

• Although Ward linkage with three and four clusters revealed
  meaningful substructure, neither solution exceeded the
  two-cluster model in overall cluster separation.

• Therefore, Ward linkage with two clusters was retained as the
  final Agglomerative model, while the three- and four-cluster
  solutions were interpreted as hierarchical refinements that
  provide additional insight into the underlying wine
  populations.

FINAL MODEL

Algorithm: Agglomerative Clustering
Linkage Method: Ward
Clusters: 2

Silhouette Score: 0.2607
Davies-Bouldin Index: 1.7164

Cluster Sizes:
Cluster 0 = 3,842 wines (72.2%)
Cluster 1 = 1,478 wines (27.8%)

Additional Hierarchical Findings:
Ward k=3 → Best Davies-Bouldin Index (1.5702)
Ward k=4 → Richest Wine Profile Segmentation
"""
)

# =========================================================
# DBSCAN Key Insights
# =========================================================

print(
    """
DBSCAN Key Insights
KEY INSIGHTS

• Multiple combinations of eps and min_samples were evaluated.

• The best-performing DBSCAN configuration used:

  eps = 1.3
  min_samples = 10

• The final model identified two density-based clusters and
  classified 1,424 observations (26.77% of the dataset) as noise.

• Cluster 0 represented a more acidic and structured wine profile
  characterized by higher fixed acidity, volatile acidity,
  chlorides, and sulphates.

• Cluster 1 represented the largest dense wine population and
  exhibited lower volatile acidity, lower chlorides, higher
  alcohol content, and the highest average quality score.

• Unlike K-Means, GMM, and Agglomerative Clustering, DBSCAN did not
  force all observations into clusters.

• PCA showed moderate overlap between groups.

• t-SNE revealed two recognizable dense wine populations with a
  substantial transition region between them.

• UMAP showed two clearly separated dense regions surrounded by
  boundary observations and sparse pockets.

• Many of the observations classified as noise were not extreme
  outliers but instead occupied transitional regions between the
  major wine populations.

• DBSCAN functioned more effectively as a detector of sparse and
  transitional observations than as a primary clustering method
  for this dataset.

FINAL MODEL

Algorithm: DBSCAN
eps = 1.3
min_samples = 10

Clusters Found: 2
Noise Points: 1,424
Noise Percentage: 26.77%

Silhouette Score: 0.1505
Davies-Bouldin Index: 3.3824
"""
)

# =========================================================
# Final Clustering Comparison
# =========================================================

print(
    """
FINAL COMPARISON

Agglomerative Clustering (Ward k=2) achieved the highest
Silhouette Score and strongest recovery of the dominant
wine-population structure.

K-Means (k=4) produced the richest and most interpretable
wine-profile segmentation.

Gaussian Mixture Models provided a strong probabilistic
representation and closely reproduced the wine-type structure.

DBSCAN was most useful for identifying sparse and transitional
observations rather than maximizing cluster separation.

Goal	WinnerBest overall cluster separation	Agglomerative (Ward k=2)
Best wine profile discovery	K-Means (k=4)
Best probabilistic model	GMM (2-diag)
Best outlier detection	DBSCAN

And for a business recommendation:

K-Means (k=4) is recommended for customer-facing segmentation and interpretation
because it produced four stable and chemically distinct wine profiles. 
Agglomerative Clustering (Ward k=2) most effectively recovered the dominant 
natural structure of the dataset, while GMM confirmed the existence of two 
underlying wine populations. DBSCAN was most useful for identifying sparse 
and transitional observations.


FINAL REPORT:
    
- Best Overall Clustering Model: Agglomerative Clustering (Ward k=2)

Agglomerative Clustering provided the strongest recovery of the underlying 
structure of the Wine Quality dataset, achieving the highest silhouette score 
and approximately 96.94% agreement with the known wine type categories.

- Best Business Segmentation Model: K-Means (k=4)

K-Means generated four highly stable and interpretable wine profiles, providing 
the most actionable segmentation for business analysis and decision-making.

- Best Probabilistic Model: Gaussian Mixture Model (2 Components, Diagonal Covariance)

GMM confirmed the existence of two dominant wine populations while providing a 
probabilistic framework for cluster membership.

- Best Outlier Detection Model: DBSCAN

DBSCAN proved most useful for identifying sparse and transitional wines that 
did not belong strongly to the major wine populations.


QUESTIONS AND ANSWERS

Q: "Which clustering model should we deploy?"
A: My recommendation would be: K-Means (k=4)
This is because it provides the most useful segmentation for business decisions.

Q: "Which model best captures the true underlying structure of the wine data?"
A: My recommendation would be: Agglomerative Clustering (Ward k=2).
This is because it produced the strongest overall clustering evidence.
"""
)