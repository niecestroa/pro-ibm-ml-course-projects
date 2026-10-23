# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October __, 2026
Status:     Ongoing

"""

# =========================================================
# Isolation Forest (IF)
# =========================================================

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
# Local Outlier Factor (LOF) 
# =========================================================

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

