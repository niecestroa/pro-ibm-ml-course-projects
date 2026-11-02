"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 2, 2026
Last Edit:  November 2, 2026
Progress:   Complete

Analysis:   02 - Feature Scaling Comparison
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
| ReLU | 0.5612 | 0.7257 | 0.3131 |
| Tanh | 0.5632 | 0.7308 | 0.3034 |
| Sigmoid | 0.5654 | 0.7323 | 0.3005 |

<br>

Among the evaluated activation functions, ReLU achieved the strongest overall
performance, producing the lowest MAE and RMSE values while achieving the
highest Adjusted R² score.

<br>

ReLU also achieved the lowest final training loss and validation loss,
indicating more effective optimization during training and stronger overall
model fit.

<br>

Although Tanh and Sigmoid produced competitive results, neither activation
function was able to surpass ReLU across any major evaluation metric.

<br>

Learning curves demonstrated stable convergence for all three activation
functions, with only minor differences in final performance. This suggests
that the chosen network architecture is relatively robust to activation
function selection, while still benefiting from the efficiency of ReLU.

<br>

The results indicate that ReLU provides the most effective balance of learning
efficiency, prediction accuracy, and explanatory power for wine quality
prediction.

<br>

Based on these findings, ReLU was selected as the preferred activation
function for subsequent analyses and model optimization experiments.
""")

# ==========================================
# Analysis 3: Activation Function Comparison
# Interpretation Summary
# ==========================================

print("""
# Analysis 3 Conclusion:
<br>
The objective of this analysis was to evaluate how different activation
functions influence neural network performance for wine quality prediction.
<br>
Three commonly used activation functions were evaluated: ReLU, Sigmoid,
and Tanh.
<br>
Among the tested activation functions, ReLU achieved the strongest overall
performance, producing the lowest MAE and RMSE values while achieving the
highest Adjusted R² score.
<br>
| Activation | MAE | RMSE | Adjusted R² |
|------------|---------:|---------:|---------:|
| ReLU | 0.5612 | 0.7257 | 0.3131 |
| Tanh | 0.5632 | 0.7308 | 0.3034 |
| Sigmoid | 0.5654 | 0.7323 | 0.3005 |
<br>
The results indicate that ReLU provides more effective learning and predictive
performance on the wine quality dataset than either Sigmoid or Tanh.
<br>
Learning curves showed stable convergence across all three activation
functions, with ReLU achieving the lowest final training loss and validation
loss values.
<br>
Based on these findings, ReLU was selected as the preferred activation
function for subsequent analyses and model development.
""")

# ==========================================
# Analysis 3: Activation Function Comparison
# Key Insights
# ==========================================

print("""
# Key Insights
<br>

### 1. ReLU Achieved the Best Overall Performance
<br>

ReLU outperformed both Sigmoid and Tanh across all evaluation metrics.

<br>

| Activation | MAE | RMSE | Adjusted R² |
|------------|---------:|---------:|---------:|
| ReLU | 0.5612 | 0.7257 | 0.3131 |
| Tanh | 0.5632 | 0.7308 | 0.3034 |
| Sigmoid | 0.5654 | 0.7323 | 0.3005 |

<br>

ReLU produced the lowest prediction errors and explained the greatest amount
of variance in wine quality scores.

<br>

---

<br>

### 2. Differences Between Activation Functions Were Modest
<br>

Although ReLU ranked first, the differences among the three activation
functions were relatively small.

<br>

All activation functions produced similar MAE and RMSE values, indicating that
the network architecture was reasonably robust regardless of the activation
function selected.

<br>

---

<br>

### 3. ReLU Generated the Lowest Prediction Error
<br>

ReLU achieved:

<br>

- Lowest MAE (0.5612)
- Lowest RMSE (0.7257)
- Highest Adjusted R² (0.3131)

<br>

These results indicate that ReLU provided the most accurate overall
predictions among the tested activation functions.

<br>

---

<br>

### 4. ReLU Demonstrated the Most Effective Learning
<br>

Final training and validation losses were lowest for the ReLU model.

<br>

| Activation | Final Training Loss | Final Validation Loss |
|------------|-------------------:|----------------------:|
| ReLU | 0.5325 | 0.5008 |
| Tanh | 0.5388 | 0.5024 |
| Sigmoid | 0.5442 | 0.5108 |

<br>

Lower loss values suggest more effective optimization and better overall model
fit during training.

<br>

---

<br>

### 5. Sigmoid Produced the Weakest Overall Results
<br>

Sigmoid generated the highest MAE and RMSE values while achieving the lowest
Adjusted R² score.

<br>

Although the differences were not substantial, Sigmoid consistently ranked
behind both ReLU and Tanh across all evaluation metrics.

<br>

---

<br>

### 6. Tanh Performed Between ReLU and Sigmoid
<br>

Tanh achieved performance levels very close to those of ReLU and slightly
better than Sigmoid.

<br>

This suggests that Tanh is a viable alternative, although it did not provide
enough improvement to surpass ReLU on this dataset.

<br>

---

<br>

### 7. All Models Showed Stable Convergence
<br>

Training and validation loss curves decreased smoothly and stabilized over
time for all three activation functions.

<br>

The relatively small gap between training and validation losses indicates that
the models generalized reasonably well without substantial overfitting.

<br>

---

<br>

### 8. Activation Function Choice Influenced Predictive Performance
<br>

The results demonstrate that activation functions affect how effectively a
neural network learns nonlinear relationships within data.

<br>

While all three activation functions successfully learned the wine quality
patterns, ReLU provided the strongest combination of prediction accuracy,
model fit, and training efficiency.

<br>

---

<br>

### Conclusion
<br>

The activation function comparison demonstrated that ReLU was the most
effective activation function for wine quality prediction.

<br>

ReLU achieved the lowest prediction errors, the highest Adjusted R² value,
and the most favorable training performance among the evaluated activation
functions.

<br>

Although Tanh and Sigmoid produced competitive results, ReLU consistently
ranked first across every evaluation metric and was therefore selected for
subsequent analyses and model optimization experiments.
""")