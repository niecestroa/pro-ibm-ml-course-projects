# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:07:14 2026

@author: aniec
"""

# ==========================================
# Analysis 02 - Feature Scaling Comparison
# ==========================================

"""
## Conclusion

The objective of this analysis was to determine which feature scaling
method produced the strongest neural network performance for wine
quality prediction.

Three approaches were evaluated:

- No Scaling
- StandardScaler
- MinMaxScaler

Feature scaling consistently improved model performance compared with
the unscaled baseline. Both StandardScaler and MinMaxScaler reduced
prediction error while increasing the amount of variance explained by
the model.

Among all methods tested, MinMaxScaler achieved the best overall
performance, producing the lowest MAE and RMSE values along with the
highest Adjusted R² score.

Although prediction accuracy improved across all quality levels,
extreme wine quality ratings remained more difficult to predict due
to the highly imbalanced target distribution.

Based on these findings, MinMaxScaler was selected as the preferred
feature scaling method and will be used in all remaining analyses.

### Final Ranking

1. MinMaxScaler
2. StandardScaler
3. No Scaling

### Winning Results

| Metric | Value |
|----------|--------:|
| MAE | 0.5519 |
| RMSE | 0.7094 |
| Adjusted R² | 0.3435 |

### Selected Configuration Moving Forward

- Scaling Method: MinMaxScaler
- Network Architecture: Dense(16, ReLU) → Output
- Optimizer: Adam
- Epochs: 100
- Validation Split: 20%

With the preferred scaling method identified, the next phase of the
project will evaluate the impact of activation functions on neural
network performance.
"""

# ==========================================
# Summary 
# ==========================================

"""
## Summary

The objective of this analysis was to evaluate the impact of feature
scaling on neural network performance for wine quality prediction.

Three preprocessing approaches were tested using the same neural
network architecture, optimizer, train-test split, and training
configuration:

- No Scaling
- StandardScaler
- MinMaxScaler

The results demonstrated that feature scaling improved model
performance relative to the unscaled baseline. Both StandardScaler
and MinMaxScaler reduced prediction error while increasing the
amount of variance explained by the model.

MinMaxScaler achieved the strongest overall performance, producing
the lowest MAE and RMSE values and the highest Adjusted R² score
among all methods evaluated.

Training and validation losses remained closely aligned across all
experiments, indicating stable convergence and minimal evidence of
overfitting.

Based on these findings, MinMaxScaler was selected as the preferred
feature scaling method for all subsequent analyses.

### Final Results

| Scaling Method | MAE | RMSE | Adjusted R² |
|----------------|------|------|-------------|
| No Scaling | 0.5656 | 0.7334 | 0.2983 |
| StandardScaler | 0.5546 | 0.7170 | 0.3294 |
| MinMaxScaler | 0.5519 | 0.7094 | 0.3435 |
"""

# ==========================================
# Key Insights
# ==========================================

"""
## Key Insights

### 1. Feature Scaling Improved Model Performance

Both StandardScaler and MinMaxScaler outperformed the No Scaling
baseline across all evaluation metrics, demonstrating the value
of feature scaling for neural network training.

### 2. MinMaxScaler Produced the Best Results

MinMaxScaler achieved the lowest prediction error and highest
Adjusted R² score among all scaling methods tested.

### 3. StandardScaler Also Improved Performance

StandardScaler reduced MAE and RMSE while increasing Adjusted R²,
indicating that standardization helped the network learn more
effectively than the unscaled baseline.

### 4. Model Training Remained Stable

Training and validation loss curves remained closely aligned for
all scaling methods, indicating consistent learning behavior and
good generalization.

### 5. Overfitting Was Minimal

None of the scaling approaches produced substantial divergence
between training and validation losses, suggesting that the models
did not significantly overfit the training data.

### 6. Prediction Accuracy Improved Across Most Observations

Feature scaling produced tighter clustering around the ideal
prediction line and improved overall agreement between actual and
predicted wine quality scores.

### 7. Extreme Quality Ratings Remain Challenging

Despite performance gains from scaling, rare quality scores
continued to be more difficult to predict because of the highly
imbalanced target distribution.

### 8. MinMaxScaler Was Selected for Future Analyses

Based on superior MAE, RMSE, and Adjusted R² results, MinMaxScaler
was selected as the preferred scaling method for all remaining
experiments in the project.
"""

# ==========================================
# Analysis:   02 - Feature Scaling Comparison
# Method:     No Scaling
# ==========================================

"""
# Key Insights

### 1. The Baseline MLP Produced Stable Performance

The MLP achieved:

- MAE: 0.5656
- RMSE: 0.7334
- Adjusted R²: 0.2983

These results established the benchmark for evaluating the impact of feature 
scaling.

---

### 2. Training Converged Quickly

Training and validation losses decreased rapidly during the first several 
epochs before stabilizing.

Most learning occurred early in training, suggesting efficient optimization 
by the Adam optimizer.

---

### 3. Minimal Evidence of Overfitting

Training and validation loss curves remained closely aligned throughout training.

The absence of a widening gap between the curves suggests good generalization 
and limited overfitting.

---

### 4. Predictions Were Most Accurate for Common Quality Scores

The model produced its strongest predictions for wines with quality scores 
between 5 and 7, which represented the majority of observations within the dataset.

---

### 5. Extreme Quality Ratings Remained Difficult to Predict

The model consistently overestimated low-quality wines and underestimated 
high-quality wines.

This pattern was likely influenced by the imbalanced target distribution, 
where very few wines received quality scores of 3, 8, or 9.

---

### 6. Prediction Compression Was Observed

Predicted quality values were concentrated within a relatively narrow range.

The model rarely predicted the most extreme quality scores, causing 
predictions to regress toward the center of the distribution.

---

### 7. No Scaling Established the Baseline for the Scaling Study

The unscaled model provides a reference point against which StandardScaler 
and MinMaxScaler can be evaluated.

Any future improvements can therefore be attributed directly to feature 
scaling rather than changes in architecture or training methodology.
"""

# ==========================================
# Analysis:   02 - Feature Scaling Comparison
# Method:     StandardScaler
# ==========================================

"""
# Key Insights

### 1. StandardScaler Improved Predictive Performance

StandardScaler outperformed the No Scaling baseline across all evaluation metrics.

| Metric | No Scaling | StandardScaler |
|----------|---------:|---------:|
| MAE | 0.5656 | 0.5546 |
| RMSE | 0.7334 | 0.7170 |
| Adjusted R² | 0.2983 | 0.3294 |

---

### 2. Feature Standardization Improved Learning Efficiency

Standardizing feature values reduced the scale differences between variables, 
allowing the neural network to learn more effectively and achieve lower training 
and validation losses.

---

### 3. Model Convergence Remained Stable

Training and validation loss curves remained closely aligned throughout training, 
indicating good generalization and minimal evidence of overfitting.

---

### 4. Prediction Accuracy Improved

Actual-versus-predicted results demonstrated tighter clustering around the ideal 
prediction line, suggesting improved predictive accuracy relative to the 
No Scaling baseline.

---

### 5. Dataset Imbalance Continues to Limit Extreme Predictions

Despite the performance improvement, the model still struggled to accurately 
predict rare quality scores due to the highly imbalanced target distribution.

---

### 6. StandardScaler Became the Leading Scaling Method

Based on lower error metrics and improved explanatory power, StandardScaler 
currently ranks ahead of No Scaling and serves as the benchmark to beat in the 
remaining scaling experiments.
"""

# ==========================================
# Analysis:   02 - Feature Scaling Comparison
# Method:     MinMaxScaler
# ==========================================

"""
## Key Insights

### 1. MinMaxScaler Produced the Best Overall Performance

MinMaxScaler achieved the lowest MAE and RMSE values while producing the
highest Adjusted R² score among all scaling methods tested.

### 2. Feature Scaling Improved Neural Network Performance

Both StandardScaler and MinMaxScaler outperformed the No Scaling
baseline, demonstrating the importance of feature scaling for neural
network training on structured tabular data.

### 3. MinMaxScaler Outperformed StandardScaler

Although StandardScaler significantly improved predictive performance,
MinMaxScaler achieved additional reductions in prediction error and
provided the strongest overall results.

### 4. Model Convergence Remained Stable

Training and validation losses decreased rapidly during the early epochs
before stabilizing. The learning curves remained closely aligned
throughout training.

### 5. Evidence of Overfitting Remained Minimal

The small gap between training and validation losses suggests that the
model generalized well and did not exhibit substantial overfitting.

### 6. Prediction Accuracy Improved Across Most Quality Levels

Actual-versus-predicted results showed tighter clustering around the
ideal prediction line when compared with both previous scaling methods.

### 7. Extreme Quality Ratings Remain Challenging

Despite overall improvements in performance, the model continued to
struggle with very low and very high quality wines because these ratings
were underrepresented within the dataset.

### 8. MinMaxScaler Was Selected for Future Analyses

Based on lower prediction error, higher explanatory power, and stable
training behavior, MinMaxScaler was selected as the preferred scaling
method for all subsequent experiments.
"""