"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Analysis:   04 - Architecture Comparison
Method:     Final Comparison
"""

# ==========================================
# Conclusion
# ==========================================

"""
## Conclusion

The objective of this analysis was to determine whether increasing
neural network complexity improves wine quality prediction
performance.

Four architectures of progressively increasing depth were evaluated
while maintaining the strongest configuration identified in previous
analyses: StandardScaler preprocessing and Sigmoid activation
functions.

The results demonstrated that increasing network depth improved
predictive performance from Model A through Model C. Additional hidden
layers enabled the network to learn more informative nonlinear
relationships within the wine quality dataset, resulting in lower
prediction error and improved explanatory power.

Model C achieved the strongest overall performance, producing the
lowest MAE (0.5354), lowest RMSE (0.6947), and highest Adjusted R²
(0.3705) among all architectures evaluated.

Although Model D introduced substantially greater complexity, it failed
to surpass Model C. This finding suggests that increasing model depth
beyond the Model C architecture provides limited benefit while adding
additional computational complexity.

The results indicate that moderate network depth offers the most
effective balance between learning capacity, predictive accuracy, and
generalization for this prediction task.

### Final Architecture Ranking

1. Model C (64 → 32 → 16)
2. Model B (32 → 16)
3. Model A (16)
4. Model D (128 → 64 → 32 → 16)

### Winning Architecture

Model C

64 → 32 → 16

### Winning Performance

- MAE: 0.5354
- RMSE: 0.6947
- Adjusted R²: 0.3705

### Selected Configuration Moving Forward

- Scaling Method: StandardScaler
- Activation Function: Sigmoid
- Architecture: 64 → 32 → 16
- Optimizer: Adam
- Epochs: 100
- Validation Split: 20%

The findings demonstrate that combining StandardScaler preprocessing,
Sigmoid activation functions, and a moderately deep neural network
architecture produces the strongest predictive performance for wine
quality prediction while avoiding unnecessary model complexity.
"""

# ==========================================
# Interpretation
# ==========================================

"""
## Interpretation

The architecture comparison evaluated whether increasing neural
network depth improves wine quality prediction performance when using
the strongest configuration identified in previous analyses:
StandardScaler preprocessing and Sigmoid activation functions.

Results showed that network depth positively influenced predictive
performance up to a certain point. Performance improved consistently
from Model A through Model C, indicating that additional hidden layers
enabled the network to capture increasingly complex relationships
within the wine quality dataset.

Model C produced the strongest overall results, achieving the lowest
MAE (0.5354), lowest RMSE (0.6947), and highest Adjusted R² (0.3705)
among all architectures evaluated.

Although Model D introduced an additional hidden layer and
substantially more parameters, it failed to improve upon Model C.
Performance gains observed from earlier increases in network depth
began to diminish, suggesting that additional complexity beyond Model
C offered limited practical benefit.

Training and validation losses remained closely aligned across all
architectures, indicating stable learning behavior and good
generalization. Significant overfitting was not observed, even in the
deepest network evaluated.

Overall, the results indicate that the combination of StandardScaler,
Sigmoid activation, and a moderately deep architecture (64 → 32 → 16)
provides the most effective balance between predictive accuracy,
generalization, and model complexity for wine quality prediction.
"""

# ==========================================
# Key Insights
# ==========================================

"""
## Key Insights

### 1. Model C Produced the Best Overall Results

Model C achieved the lowest MAE, lowest RMSE, and highest Adjusted R²
among all architectures evaluated.

### 2. Increasing Depth Improved Predictive Performance

Performance improved steadily from Model A through Model C,
demonstrating that additional hidden layers enabled the network to
learn more informative feature relationships.

### 3. Model B Demonstrated Meaningful Improvement Over Model A

Adding a second hidden layer improved RMSE and Adjusted R², indicating
that additional model capacity enhanced predictive performance.

### 4. Model C Represented the Optimal Architecture

The 64 → 32 → 16 architecture provided the strongest balance between
model complexity and predictive accuracy.

### 5. Model D Demonstrated Diminishing Returns

Despite being the most complex architecture evaluated, Model D failed
to outperform Model C and produced weaker RMSE and Adjusted R² values.

### 6. Additional Complexity Did Not Guarantee Better Performance

Increasing the number of layers beyond Model C increased model
complexity without generating meaningful improvements in predictive
accuracy.

### 7. Learning Remained Stable Across All Architectures

Training and validation loss curves remained closely aligned
throughout training, indicating stable optimization behavior.

### 8. Significant Overfitting Was Not Observed

None of the evaluated architectures exhibited substantial divergence
between training and validation performance.

### 9. Prediction Compression Remained Present

All architectures continued to overestimate lower wine quality scores
and underestimate higher quality scores, reflecting the challenging
nature of the dataset.

### 10. A Clear Architecture Ranking Emerged

Based on overall predictive performance:

1. Model C (64 → 32 → 16)
2. Model B (32 → 16)
3. Model A (16)
4. Model D (128 → 64 → 32 → 16)
"""