# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:08:58 2026

@author: aniec
"""

# ==========================================
# Analysis:   03 - Activation Function Comparison
# ==========================================

# ==========================================
# Conclusion
# ==========================================

"""
## Conclusion

The objective of this analysis was to determine which activation
function produced the strongest neural network performance for wine
quality prediction.

Three activation functions were evaluated:

- ReLU
- Sigmoid
- Tanh

The experimental results indicate that activation function selection
has a meaningful impact on model performance. ReLU consistently
outperformed both Tanh and Sigmoid across all evaluation metrics.

Although Tanh provided moderate performance improvements over Sigmoid,
neither alternative was able to surpass the predictive accuracy
achieved by ReLU.

All three activation functions demonstrated stable convergence and good
generalization performance. However, ReLU generated the lowest
prediction error and highest explanatory power while maintaining
efficient learning behavior.

Based on these findings, ReLU was selected as the preferred activation
function for all remaining analyses.

### Final Ranking

1. ReLU
2. Tanh
3. Sigmoid

### Winning Results

| Metric | Value |
|----------|--------:|
| MAE | 0.5520 |
| RMSE | 0.7120 |
| Adjusted R² | 0.3388 |

### Selected Configuration Moving Forward

- Scaling Method: MinMaxScaler
- Activation Function: ReLU
- Hidden Neurons: 16
- Optimizer: Adam
- Epochs: 100
- Validation Split: 20%

With the preferred activation function identified, the next phase of
the project will evaluate the impact of network architecture on wine
quality prediction performance.
"""

# ==========================================
# Summary
# ==========================================

"""
## Summary

The objective of this analysis was to evaluate the impact of activation
functions on neural network performance for wine quality prediction.

Three activation functions were tested using the same neural network
architecture, scaling method, optimizer, train-test split, and training
configuration:

- ReLU
- Sigmoid
- Tanh

The results demonstrated measurable differences in predictive
performance between activation functions.

ReLU produced the strongest overall results, achieving the lowest MAE
and RMSE values while generating the highest Adjusted R² score.
Tanh outperformed Sigmoid but did not match the performance of ReLU.

All three activation functions exhibited stable convergence and minimal
evidence of overfitting. However, ReLU consistently learned more
effective representations of the wine quality data and generated more
accurate predictions.

Overall, ReLU was identified as the best activation function for the
wine quality prediction task and was selected for subsequent analyses.

### Final Results

| Activation Function | MAE | RMSE | Adjusted R² |
|--------------------|------|------|-------------|
| ReLU | 0.5520 | 0.7120 | 0.3388 |
| Tanh | 0.5627 | 0.7297 | 0.3054 |
| Sigmoid | 0.5645 | 0.7334 | 0.2984 |
"""

# ==========================================
# Key Inisghts
# ==========================================

"""
## Key Insights

### 1. ReLU Produced the Strongest Overall Performance

ReLU achieved the lowest prediction error and highest Adjusted R²
score among all activation functions evaluated.

### 2. Activation Function Choice Influenced Model Performance

Although architecture, scaling, optimizer, and training settings
remained constant, predictive performance varied noticeably across
activation functions.

### 3. Tanh Outperformed Sigmoid

Tanh achieved lower MAE and RMSE values while explaining more variance
than Sigmoid, placing second overall in the comparison.

### 4. Stable Learning Was Observed Across All Methods

All activation functions exhibited rapid convergence and closely aligned
training and validation loss curves.

### 5. Overfitting Remained Limited

Training and validation losses remained similar throughout training,
indicating effective generalization and minimal overfitting.

### 6. ReLU Learned More Efficiently

ReLU produced stronger predictive performance while maintaining stable
training behavior, demonstrating its effectiveness for structured
tabular data.

### 7. Prediction Compression Persisted Across All Models

All activation functions tended to predict values near the center of
the quality distribution, causing low-quality wines to be
overestimated and high-quality wines to be underestimated.

### 8. Dataset Imbalance Continued to Affect Extreme Predictions

Rare quality scores remained the most difficult observations to predict
accurately regardless of activation function choice.

### 9. Activation Function Ranking Emerged

Based on all evaluation metrics, activation functions ranked as:

1. ReLU
2. Tanh
3. Sigmoid
"""

# ==========================================
# Method:     ReLU
# Key Inisghts
# ==========================================

"""
## Key Insights

### 1. ReLU Produced Strong Baseline Performance

The ReLU activation function achieved:

- MAE: 0.5520
- RMSE: 0.7120
- Adjusted R²: 0.3388

These results establish the baseline for the Activation Function
Comparison study.

### 2. Learning Converged Rapidly

Training and validation losses decreased sharply within the first
few epochs before stabilizing.

Most learning occurred early in the training process, indicating
efficient optimization and rapid convergence.

### 3. Model Training Was Stable

Training and validation loss curves remained closely aligned
throughout training.

The lack of separation between the curves suggests stable learning
behavior and strong generalization performance.

### 4. Limited Evidence of Overfitting

The model maintained similar training and validation losses after
convergence, indicating minimal overfitting despite training for
100 epochs.

### 5. Prediction Accuracy Was Strongest for Mid-Range Quality Scores

The model produced its most accurate predictions for wines with
quality scores between 5 and 7, which represented the majority of
observations within the dataset.

### 6. Prediction Compression Remained Present

Low-quality wines tended to be overestimated while high-quality
wines tended to be underestimated.

Predictions remained concentrated near the center of the quality
distribution.

### 7. Dataset Imbalance Continues to Affect Extreme Values

Rare quality ratings such as 3 and 9 remained difficult to predict
accurately because they were underrepresented within the training
data.

### 8. ReLU Becomes the Benchmark Activation Function

ReLU currently serves as the benchmark activation function against
which Sigmoid and Tanh will be evaluated in subsequent analyses.
"""

# ==========================================
# Method:     Sigmoid
# Key Inisghts
# ==========================================

"""
## Key Insights

### 1. Sigmoid Underperformed ReLU

Sigmoid produced higher prediction errors and a lower Adjusted R²
score than the ReLU baseline.

### 2. Learning Was Stable

Training and validation losses remained closely aligned throughout
training, indicating stable optimization and good generalization.

### 3. Overfitting Was Minimal

The similarity between training and validation curves suggests that
the model did not suffer from substantial overfitting.

### 4. Convergence Occurred Quickly

Most learning occurred during the first several epochs, after which
loss values stabilized with little additional improvement.

### 5. Prediction Compression Was Pronounced

The model exhibited a strong tendency to predict values near the
center of the quality distribution, reducing sensitivity to extreme
quality ratings.

### 6. Extreme Ratings Remained Difficult to Predict

Low-frequency quality scores continued to be challenging because of
the highly imbalanced target distribution.

### 7. ReLU Remains the Leading Activation Function

Based on current results, ReLU maintains lower MAE and RMSE values
while achieving higher explanatory power than Sigmoid.

### 8. Sigmoid Serves as an Important Benchmark

Although performance was lower than ReLU, the Sigmoid results provide
a useful reference point for evaluating the final activation function,
Tanh.
"""

# ==========================================
# Method:     Tanh
# Key Inisghts
# ==========================================

"""
## Key Insights

### 1. Tanh Outperformed Sigmoid

Tanh achieved lower MAE and RMSE values while producing a higher
Adjusted R² score than Sigmoid.

### 2. ReLU Remained the Strongest Activation Function

Although Tanh improved upon Sigmoid, ReLU continued to achieve the
best overall predictive performance among the activation functions
tested.

### 3. Training Behavior Was Stable

Training and validation losses remained closely aligned throughout
training, indicating stable optimization and good generalization.

### 4. Convergence Occurred Quickly

Most model learning occurred during the first several epochs before
loss values stabilized for the remainder of training.

### 5. Overfitting Remained Minimal

The small difference between training and validation losses suggests
that the model generalized well and did not overfit the training
data.

### 6. Prediction Compression Persisted

Predictions remained concentrated near the most common quality scores,
causing low-quality wines to be overestimated and high-quality wines
to be underestimated.

### 7. Dataset Imbalance Continues to Affect Rare Ratings

Extreme quality scores remained difficult to predict accurately due
to the limited number of observations within those categories.

### 8. Activation Function Performance Ranking Emerged

Based on the current results, activation functions rank as follows:

1. ReLU
2. Tanh
3. Sigmoid

This ranking suggests that ReLU is best suited for the structured
wine-quality prediction task.
"""
