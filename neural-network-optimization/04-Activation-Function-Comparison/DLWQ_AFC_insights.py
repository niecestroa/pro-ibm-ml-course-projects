# -*- coding: utf-8 -*-
"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 2, 2026
Last Edit:  November 2, 2026
Progress:   Complete

Analysis:   03 - Activation Function Comparison
"""

# ==========================================
# Analysis 3 Conclusion
# ==========================================

print("""
# Analysis 3 Conclusion:
<br>

The objective of this analysis was to evaluate the impact of activation
function selection on neural network performance for wine quality prediction.

<br>

Three activation functions were evaluated: ReLU, Sigmoid, and Tanh.

<br>

| Activation | MAE | RMSE | Adjusted R² |
|------------|---------:|---------:|---------:|
| Sigmoid | 0.5392 | 0.6975 | 0.3654 |
| Tanh | 0.5492 | 0.7109 | 0.3408 |
| ReLU | 0.5508 | 0.7152 | 0.3329 |

<br>

Among the evaluated activation functions, Sigmoid achieved the strongest
overall performance, producing the lowest MAE and RMSE values while achieving
the highest Adjusted R² score.

<br>

Sigmoid also produced the lowest validation loss, indicating strong
generalization performance on unseen data.

<br>

Although Tanh and ReLU produced competitive results, neither activation
function was able to surpass Sigmoid across the primary evaluation metrics.

<br>

Learning curves demonstrated stable convergence for all three activation
functions, suggesting that each activation function was capable of learning
meaningful relationships within the wine quality dataset.

<br>

The results indicate that Sigmoid provides the most effective balance of
prediction accuracy, model fit, and generalization performance for this
problem.

<br>

Based on these findings, Sigmoid was selected as the preferred activation
function for subsequent analyses and model optimization experiments.
""")

# ==========================================
# Analysis 3 Interpretation Summary
# ==========================================

print("""
# Analysis 3 Interpretation Summary:
<br>

The objective of this analysis was to determine which activation function
produced the strongest predictive performance when applied to the baseline MLP
architecture using the best-performing scaling method identified in Analysis 2.

<br>

Three activation functions were evaluated: ReLU, Sigmoid, and Tanh.

<br>

| Activation | MAE | RMSE | Adjusted R² |
|------------|---------:|---------:|---------:|
| Sigmoid | 0.5392 | 0.6975 | 0.3654 |
| Tanh | 0.5492 | 0.7109 | 0.3408 |
| ReLU | 0.5508 | 0.7152 | 0.3329 |

<br>

Sigmoid achieved the strongest overall performance, producing the lowest
prediction errors and the highest Adjusted R² value.

<br>

Compared with ReLU, Sigmoid reduced RMSE from 0.7152 to 0.6975 and improved
Adjusted R² from 0.3329 to 0.3654.

<br>

These results suggest that smoother nonlinear transformations provided by the
Sigmoid activation function are well suited to modeling the relationships
within the wine quality dataset.

<br>

Based on these findings, Sigmoid was selected as the preferred activation
function for future analyses.
""")

# ==========================================
# Analysis 3 Key Insights
# ==========================================

print("""
# Key Insights
<br>

### 1. Sigmoid Achieved the Best Overall Performance
<br>

Sigmoid outperformed both ReLU and Tanh across all primary evaluation metrics.

<br>

| Activation | MAE | RMSE | Adjusted R² |
|------------|---------:|---------:|---------:|
| Sigmoid | 0.5392 | 0.6975 | 0.3654 |
| Tanh | 0.5492 | 0.7109 | 0.3408 |
| ReLU | 0.5508 | 0.7152 | 0.3329 |

<br>

Sigmoid produced the lowest prediction error and explained the greatest
proportion of variance in wine quality scores.

<br>

---

<br>

### 2. Activation Function Selection Significantly Influenced Performance
<br>

The results demonstrate that activation function choice can have a meaningful
impact on neural network predictive performance.

<br>

Although the network architecture remained unchanged, performance varied across
activation functions.

<br>

---

<br>

### 3. Sigmoid Produced the Lowest Prediction Error
<br>

Sigmoid achieved:

<br>

- Lowest MAE (0.5392)
- Lowest RMSE (0.6975)
- Highest Adjusted R² (0.3654)

<br>

These results indicate that Sigmoid generated the most accurate predictions
among the evaluated activation functions.

<br>

---

<br>

### 4. Sigmoid Demonstrated Strong Generalization
<br>

Sigmoid achieved the lowest final validation loss (0.4766).

<br>

| Activation | Final Training Loss | Final Validation Loss |
|------------|-------------------:|----------------------:|
| Sigmoid | 0.4841 | 0.4766 |
| Tanh | 0.4732 | 0.4792 |
| ReLU | 0.4714 | 0.4783 |

<br>

The low validation loss suggests effective learning and strong generalization
to unseen observations.

<br>

---

<br>

### 5. Tanh Ranked Second Overall
<br>

Tanh consistently outperformed ReLU and achieved the second-best values for
RMSE and Adjusted R².

<br>

This indicates that Tanh is a viable alternative activation function on this
dataset.

<br>

---

<br>

### 6. ReLU Produced the Weakest Results
<br>

ReLU generated the highest MAE and RMSE values and produced the lowest
Adjusted R² score among the evaluated activation functions.

<br>

Although ReLU remains widely used in deep learning applications, it did not
provide the strongest performance for this specific wine quality prediction
problem.

<br>

---

<br>

### 7. All Models Converged Successfully
<br>

Training and validation loss curves decreased steadily and stabilized over
time for all activation functions.

<br>

This indicates successful optimization and stable learning behavior across all
models.

<br>

---

<br>

### Conclusion
<br>

The activation function comparison demonstrated that Sigmoid was the most
effective activation function for wine quality prediction when used with the
best-performing preprocessing pipeline identified in earlier analyses.

<br>

Sigmoid achieved the lowest prediction errors, the highest Adjusted R² value,
and the strongest validation performance among all evaluated activation
functions.

<br>

Based on these findings, Sigmoid was selected as the preferred activation
function for all subsequent analyses.
""")
