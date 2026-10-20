# -*- coding: utf-8 -*-
"""
Last Edit on October 20, 2026
"""

# =========================================================
# Unsupervised ML Wine EDA Key Insights
# =========================================================

'''
1. Dataset Quality and Structure

- Duplicate records = Yes
- The dataset contains no missing values and very few duplicate observations, 
making it suitable for clustering without heavy preprocessing.

2. Distribution of Wine Types

- White wines account for roughly 75% of observations while red wines account 
for 25%, indicating class imbalance by wine type.

3. Quality Distribution (Not Used for Clustering)

- Although quality is excluded from clustering inputs, the script explores it.
-Quality becomes the interpretation variable.
-Most wines have quality scores between 5 and 7, suggesting the dataset is 
heavily concentrated around average-quality wines.
- This becomes useful later: "Cluster 2 contains mostly wines with quality 
scores of 7 and above."

4. Feature Distributions

The histograms reveal the shape of each chemistry variable.

Findings

- Many wine chemistry variables are not normally distributed.
- Most wine chemistry features are positively skewed and may
require transformation before clustering.

5. Red vs White Wine Differences

- This is one of the most valuable sections.

White wines generally:
- Higher residual sugar
- Higher sulfur dioxide
- Slightly lower pH

Red wines generally:
- Higher volatile acidity
- Higher sulphates
- Higher fixed acidity
- Insight

- Wine type naturally separates many variables and may influence cluster formation.

This often predicts that clustering may discover: 
Red-wine-dominated clusters & 
White-wine-dominated clusters
without explicitly including wine type.

6. Pairwise Relationships

Key insight

- You can visually see whether natural groups already exist.
- White and red wines occupy different regions.
- Relationships appear nonlinear.
- No obvious cluster boundaries exist.
- Density tends to decrease as alcohol increases, suggesting a strong inverse 
relationship.

7. Correlation Analysis

One of the most important sections.

The heatmap identifies relationships among predictors.

Typical wine dataset findings
Strong Negative Correlation
alcohol vs density

Usually around:

A. r ≈ -0.70 to -0.80

Meaning:

Wines with higher alcohol generally have lower density.

Strong Positive Correlation
free sulfur dioxide
vs
total sulfur dioxide

Often:

B. r > 0.70

Meaning:

These variables capture similar information.

Moderate Relationships
residual sugar ↔ density

volatile acidity ↔ citric acid

Insight

Several features exhibit strong correlation and may introduce redundancy.

8. Highly Correlated Feature Pairs

This section specifically identifies:

|r| >= 0.80

Highly correlated features:

Contribute duplicate information.
Can dominate distance calculations.
May distort clustering.

Potential findings:

Feature 1	Feature 2Density	Residual Sugar
Alcohol	Density
Free SO₂	Total SO₂
Insight

Some variables may be candidates for dimensionality reduction through PCA.

9. Outlier Analysis

Largest outlier counts often appear in:

Residual sugar
Chlorides
Sulphates
Free sulfur dioxide
Insight

Several chemistry measurements contain extreme observations that may strongly influence centroid-based algorithms such as K-Means.

This helps determine:

Keep outliers?
Winsorize?
Transform?
Use robust clustering methods?

10. Data Integrity Checks

Nothing of note worth investigating deeper without expert

11. Skewness Assessment

Typical high-skew variables
Residual sugar
Chlorides
Total sulfur dioxide
Free sulfur dioxide
Insight

Several variables show strong positive skew (skewness > 1), indicating the need for log transformation before clustering.

12. Log Transformation Evaluation

The script compares:

original
vs
log1p

for skewed variables.

Log transformation substantially reduces skewness and produces more symmetric distributions.

This is often beneficial because distance-based clustering performs better when features are not extremely skewed.

13. Multicollinearity (VIF)

This section evaluates redundancy among predictors.

Typical interpretation:

VIF	Meaning< 5	Low concern
5-10	Moderate
>10	High
Potential findings

Features frequently showing elevated VIF:

Density
Alcohol
Residual Sugar
Insight

Density appears highly redundant with other chemistry measurements and 
may be contributing duplicate information.

14. Readiness for PCA

Although PCA is imported and not yet run, the EDA prepares for it.

The correlation and VIF analyses answer:

"Can PCA reduce dimensionality without losing much information?"

Likely answer:

Yes. Several chemistry variables exhibit substantial correlation and are likely to collapse into a smaller number of latent components.

15. Readiness for Clustering

The biggest business/data science conclusion from this EDA is:

Strengths

✅ Numerical variables available

✅ Few/no missing values

✅ Meaningful chemistry measurements

✅ Correlated variables suitable for PCA

✅ Wine types show measurable separation

Challenges

⚠ Outliers present

⚠ Strong skewness in multiple features

⚠ Multicollinearity among density, alcohol, sulfur measures, and sugar

⚠ Feature scaling required before K-Means

'''

'''
Executive Summary

- The dataset is generally clean and clustering-ready.
- White and red wines have distinct chemical profiles.
- Alcohol, density, sulfur dioxide, and residual sugar drive much of the 
variation in the dataset.
- Several variables are highly correlated, making PCA a valuable next step.
- Multiple features contain strong positive skew and outliers that should be 
transformed or standardized.
- Quality should remain excluded from clustering and used afterward to 
interpret cluster meaning and business value.
- The next logical step is Standard Scaling → PCA → K-Means/Hierarchical 
Clustering and evaluating whether clusters align with wine type and quality.

- This script is essentially a "cluster readiness assessment", and it does an 
excellent job of determining whether the wine chemistry variables contain 
meaningful structure before unsupervised modeling begins.
'''