# -*- coding: utf-8 -*-
"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 1 ,2026
Last Edit:  November 2, 2026
Progress:   Completed

Analysis 1: Baseline Neural Network Performance
"""

# ==========================================
# Analysis 1: Baseline Neural Network Performance
# Key Insights
# ==========================================

print("""
# Key Insights
<br>

### 1. The MLP Outperformed the Perceptron
<br>

The Multi-Layer Perceptron (MLP) achieved lower prediction errors and a higher
Adjusted R² than the Perceptron.

<br>

| Model | MAE | RMSE | Adjusted R² |
|---------|---------:|---------:|---------:|
| Perceptron | 0.6223 | 0.7975 | 0.1705 |
| MLP | 0.5726 | 0.7465 | 0.2732 |

<br>

The MLP reduced MAE by approximately 8.0%, reduced RMSE by 6.4%, and improved
Adjusted R² by 60.2%, demonstrating a substantially stronger predictive capability
than the single-layer Perceptron.

<br>

---

<br>

### 2. Wine Quality Is Not Purely Linear
<br>

The superior performance of the MLP suggests that the relationship between
physicochemical wine measurements and wine quality contains nonlinear patterns
that cannot be fully captured by a simple linear Perceptron.

<br>

The hidden layer within the MLP allows the network to learn more complex feature
interactions, resulting in improved prediction accuracy.

<br>

---

<br>

### 3. Both Models Converged Successfully
<br>

Training and validation losses decreased consistently throughout training before
stabilizing, indicating successful optimization and learning.

<br>

Most performance gains occurred during the early epochs, suggesting efficient
convergence for both neural network architectures.

<br>

---

<br>

### 4. Limited Evidence of Overfitting
<br>

Training and validation loss curves remained relatively close throughout the
training process.

<br>

The absence of a large divergence between training and validation error suggests
that both models generalized reasonably well to previously unseen data.

<br>

---

<br>

### 5. The MLP Produced More Accurate Predictions
<br>

Actual-versus-predicted visualizations showed that the MLP generated predictions
that were more closely aligned with observed wine quality scores than the
Perceptron.

<br>

This improvement is reflected in both the lower MAE and lower RMSE values
achieved by the MLP.

<br>

---

<br>

### 6. Prediction Accuracy Was Highest for Common Quality Scores
<br>

Both models performed best on wines whose quality ratings were concentrated
around the middle of the distribution.

<br>

Predictions tended to cluster near the most frequently occurring quality scores,
where the models had the greatest amount of training data available.

<br>

---

<br>

### 7. Extreme Quality Ratings Remained Difficult to Predict
<br>

Prediction accuracy decreased for rare quality levels because relatively few
examples were available during training.

<br>

The limited representation of extreme quality scores reduced the models' ability
to learn reliable patterns associated with these observations.

<br>

---

<br>

### 8. Evidence of Regression Toward the Mean
<br>

Both models showed a tendency to overestimate lower-quality wines and
underestimate higher-quality wines.

<br>

Predictions were generally concentrated around the center of the quality-score
distribution, a common characteristic in regression problems with imbalanced
target values.

<br>

---

<br>

### Conclusion
<br>

The baseline experiment demonstrated that a Multi-Layer Perceptron provides
meaningful performance improvements over a simple Perceptron for wine quality
prediction.

<br>

The MLP achieved lower prediction errors, explained more variance in wine
quality scores, and exhibited stable training behavior with limited evidence of
overfitting.

<br>

Based on these findings, the MLP was selected as the baseline neural network
architecture for all subsequent analyses and model comparisons.
""")