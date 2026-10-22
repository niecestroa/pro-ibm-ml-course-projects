"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October 22, 2026
Status:     Completed

Dimensionality Reduction Insights
"""

# =========================================================
# Warning and Note
# =========================================================

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

# =========================================================
# File 1: WQRW_DR_01_PCA.py
# =========================================================

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
# File 2: WQRW_DR_02_PCA_Kernels.py
# PCA Kernels Key Insights
# =========================================================

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
# File 3: WQRW_DR_03_TSNE.py
# =========================================================

# =========================================================
# Overall t-SNE Key Insights
# =========================================================

"""
KEY INSIGHTS

• t-SNE was the strongest visualization technique evaluated
  so far.

• Red and white wine populations formed highly distinct
  regions of the embedding.

• Quality ratings showed little evidence of forming separate
  groups.

• The local structure identified by t-SNE strongly supports
  previous findings from PCA, Kernel PCA, and Ward
  Clustering.

• Multiple unsupervised methods independently identified
  wine type as the primary source of variation within the
  dataset.

• t-SNE demonstrated that the dominant wine populations are
  naturally separable using wine chemistry measurements
  alone.

OVERALL INTERPRETATION

The Wine Quality dataset contains strong underlying structure
driven primarily by differences between red and white wines.
t-SNE provided the clearest representation of this structure,
revealing highly separated wine populations while confirming
that quality ratings do not form distinct low-dimensional
clusters.

FINAL TAKEAWAY

PCA explained the structure, Kernel PCA revealed nonlinear
relationships, and t-SNE provided the most effective
visualization of the underlying wine populations.
"""

# =========================================================
# Part 1: t-SNE 
# Results Interpretation
# =========================================================

"""
Wine Type Projection

• t-SNE produced a clear separation between red and white
  wine populations.

• Red wines formed a compact and well-defined region on the
  left side of the embedding.

• White wines occupied a larger nonlinear structure on the
  right side of the embedding.

• Only limited overlap occurred between the two wine types.

• The resulting separation was substantially stronger than
  both PCA and Kernel PCA.

Quality Projection

• Quality classes remained highly mixed throughout the
  embedding.

• No clearly separated quality-specific groups emerged.

• Higher-quality and lower-quality wines appeared within
  most regions of the embedding.

• These results indicate that quality is not the primary
  source of variation within the dataset.

Conclusion

t-SNE successfully preserved local structure and revealed
strong separation between wine populations. The resulting
embedding reinforces previous findings that wine type is the
dominant organizational feature of the dataset.
"""

# =========================================================
# Part 1: t-SNE 
# Key Insights
# =========================================================

"""
KEY INSIGHTS

• t-SNE produced the strongest wine-type separation observed
  so far.

• Red wines formed a compact cluster largely isolated from
  white wines.

• White wines occupied a larger and more complex nonlinear
  manifold.

• Wine quality remained highly overlapping throughout the
  embedding.

• The dominant structure revealed by t-SNE was wine type
  rather than quality.

• The embedding suggests that local neighborhood structure
  strongly reflects differences between red and white wine
  chemistry.

FINAL TAKEAWAY

t-SNE revealed exceptionally strong wine-type separation and
provided the clearest visualization of the underlying wine
populations observed in the project so far.
"""

# =========================================================
# Part 2: Method Comparisons 
# Results Interpretation
# =========================================================

"""
Results Interpretation

PCA vs t-SNE

• PCA successfully reduced dimensionality while preserving
  variance and maintaining interpretability.

• PCA clearly separated red and white wines but retained a
  moderate amount of overlap between the populations.

• t-SNE produced substantially stronger separation and
  revealed additional local structure within the data.

• The t-SNE embedding displayed more distinct boundaries
  between wine populations.

Kernel PCA vs t-SNE

• Kernel PCA revealed nonlinear structure that was not
  visible within the PCA embedding.

• Kernel PCA improved wine-type separation but still showed
  meaningful overlap between groups.

• t-SNE further enhanced separation and generated the
  clearest visual distinction between red and white wines.

• The improvement suggests that local neighborhood
  preservation is highly effective for representing wine
  chemistry structure.

Conclusion

All three methods identified wine type as the dominant source
of variation. However, t-SNE provided the strongest visual
representation of that structure.
"""

# =========================================================
# Part 2: Method Comparisons 
# Key Insights
# =========================================================

"""
KEY INSIGHTS

• PCA provided the highest interpretability through variance
  analysis and component loadings.

• Kernel PCA improved nonlinear representation and produced
  better class separation than PCA.

• t-SNE generated the strongest visual separation of wine
  populations.

• t-SNE revealed local structure not visible in either PCA
  or Kernel PCA.

• Wine type remained the dominant pattern across all three
  dimensionality-reduction techniques.

• Quality remained highly overlapping regardless of the
  embedding method applied.

FINAL TAKEAWAY

PCA best explained the data, Kernel PCA revealed nonlinear
relationships, and t-SNE provided the strongest visualization
of wine population structure.
"""

# =========================================================
# File 4: WQRW_DR_04_UMAP.py
# UMAP Key Insights
# =========================================================

# =========================================================
# Overall UMAP Key Insights
# =========================================================

"""
Overall UMAP Key Insights
KEY INSIGHTS

• UMAP produced one of the strongest visual representations
  of the Wine Quality dataset.

• Red and white wines formed highly distinct and compact
  clusters.

• Internal structure remained visible within the larger
  white-wine population.

• Quality ratings did not form separate groups and remained
  widely distributed throughout the embedding.

• UMAP successfully preserved both local neighborhood
  structure and larger global organization.

• Multiple dimensionality-reduction techniques independently
  confirmed that wine type is the dominant source of
  variation within the dataset.

OVERALL INTERPRETATION

The Wine Quality dataset contains strong underlying
structure driven by wine type rather than quality.
UMAP provided a highly organized low-dimensional
representation of this structure and delivered one of the
best overall visualizations observed in the project.

FINAL TAKEAWAY

PCA was the most interpretable method, t-SNE produced the
strongest local separation, and UMAP provided the best
combination of separation, organization, and visualization
quality among all dimensionality-reduction techniques
evaluated.
"""

# =========================================================
# Part 1: UMAP 
# Interpertation
# =========================================================

"""
Results Interpretation

Wine Type Projection

• UMAP produced exceptionally strong separation between red
  and white wine populations.

• Red wines formed a compact and well-defined cluster on the
  left side of the embedding.

• White wines formed a larger structured manifold on the
  right side of the embedding.

• Only a very small overlap region existed between the two
  wine types.

• The resulting separation was comparable to or stronger than
  the separation observed using t-SNE.

• UMAP preserved distinct substructure within both wine
  populations while maintaining clear global organization.

Quality Projection

• Quality classes remained heavily mixed throughout the
  embedding.

• No isolated quality-specific clusters emerged.

• Higher-quality and lower-quality wines appeared throughout
  both major wine populations.

• These results suggest that quality does not represent the
  dominant source of variation within the dataset.

Conclusion

UMAP successfully captured the dominant chemical structure of
the Wine Quality dataset and produced one of the clearest
representations of the underlying wine populations. The
embedding strongly supports previous findings that wine type
drives most of the observed variation.
"""

# =========================================================
# Part 1: UMAP 
# Key Insights
# =========================================================

"""
KEY INSIGHTS

• UMAP produced extremely strong separation between red and
  white wine populations.

• Red wines formed a compact and highly concentrated region.

• White wines formed a larger manifold with visible internal
  structure.

• Wine quality remained broadly distributed across the
  embedding.

• The dominant pattern recovered by UMAP was wine type
  rather than wine quality.

• UMAP preserved both local neighborhood structure and
  larger global organization.

FINAL TAKEAWAY

UMAP provided one of the clearest visualizations of wine
population structure observed in the project and strongly
confirmed that wine type is the dominant source of variation.
"""

# =========================================================
# Part 2: Method Comparison
# Interpretation
# =========================================================

"""
Part 2 Results Interpretation

1. PCA vs Kernel PCA vs t-SNE vs UMAP

PCA

Produced clear but overlapping separation between red and white wines.
Retained the highest interpretability through explained variance and component loadings.
Best method for understanding why structure exists.

Kernel PCA

Improved separation relative to PCA.
Revealed nonlinear structure in the wine chemistry data.
Reduced overlap but still maintained a continuous manifold.

t-SNE

Produced extremely strong local separation.
Created clear boundaries between red and white wine populations.
Revealed local neighborhood structure more effectively than PCA and Kernel PCA.

UMAP

Produced separation comparable to t-SNE.
Preserved clearer overall organization of the wine populations.
Generated the cleanest and most compact representation of red and white wines.
Maintained visible internal structure within the white-wine population.

t-SNE vs UMAP

t-SNE

Strengths
• Excellent local cluster separation
• Clear class boundaries
• Strong subgroup discovery

Limitations
• Global distances are less meaningful
• Overall shape can be difficult to interpret

UMAP

Strengths
• Excellent separation
• Better global structure preservation
• More compact and organized embedding
• Easier to interpret visually

Limitations
• Less directly interpretable than PCA

Conclusion

The progression across methods was:

PCA
    ↓
Kernel PCA
    ↓
t-SNE
    ↓
UMAP

in terms of visual separation and structure discovery.

PCA provided the strongest interpretability.

Kernel PCA revealed nonlinear relationships.

t-SNE generated the strongest local separation.

UMAP provided comparable separation while preserving a cleaner global organization of the dataset.
"""

# =========================================================
# Part 2: Method Comparison
# Key Insights
# =========================================================

"""
Part 2 Key Insights

KEY INSIGHTS

• PCA provided the most interpretable representation of
  wine chemistry through explained variance and feature
  loadings.

• Kernel PCA revealed nonlinear structure and improved
  separation between wine populations.

• t-SNE produced outstanding local separation and displayed
  clearly defined wine populations.

• UMAP matched the strong separation produced by t-SNE while
  maintaining a more organized global structure.

• All dimensionality-reduction techniques consistently
  identified wine type as the dominant source of variation.

• Red and white wines occupied distinct regions regardless
  of the embedding method applied.

• Quality remained highly overlapping across all methods,
  suggesting that quality contributes substantially less
  structure than wine type.

FINAL TAKEAWAY

PCA explained the structure, Kernel PCA revealed nonlinear
relationships, t-SNE highlighted local cluster separation,
and UMAP provided the strongest overall balance between
separation, structure preservation, and visualization.
"""

# =========================================================
# File 5: WQRW_DR_05_NMF.py
# NMF Key Insights
# =========================================================

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