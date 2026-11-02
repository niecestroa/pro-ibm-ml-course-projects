"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 2, 2026
Last Edit:  November 2, 2026
Progress:   Complete
"""

# ==========================================
# Analysis: 06 - Optimizer Benchmarking
# ==========================================

# ==========================================
# Conclusion
# ==========================================

"""
## Conclusion

The objective of this analysis was to determine which optimization
algorithm produces the strongest predictive performance for wine
quality prediction while maintaining the best-performing
configuration identified in previous analyses.

The benchmark configuration included:

- StandardScaler
- Sigmoid Activation
- Architecture: 64 → 32 → 16
- 125 Epochs

Three optimizers were evaluated:

- SGD
- RMSProp
- Adam

The results demonstrated that optimizer selection has a meaningful
impact on predictive performance.

RMSProp achieved the strongest overall results, producing the lowest
MAE (0.5383), lowest RMSE (0.6933), and highest Adjusted R² (0.3731)
among all optimizers evaluated.

Adam produced highly competitive performance and finished a close
second. However, RMSProp maintained a small advantage across all
evaluation metrics.

SGD produced substantially weaker results and failed to match the
performance achieved by the adaptive optimization methods.

These findings indicate that adaptive learning-rate optimization is
better suited for this wine quality prediction task than traditional
stochastic gradient descent.

### Final Optimizer Ranking

1. RMSProp
2. Adam
3. SGD

### Winning Optimizer

RMSProp

### Winning Performance

- MAE: 0.5383
- RMSE: 0.6933
- Adjusted R²: 0.3731

### Selected Configuration Moving Forward

- Scaling Method: StandardScaler
- Activation Function: Sigmoid
- Architecture: 64 → 32 → 16
- Epochs: 125
- Optimizer: RMSProp

The findings demonstrate that RMSProp provides the most effective
optimization strategy for this project and should be retained for all
subsequent analyses unless a future experiment specifically evaluates
alternative optimization approaches.
"""

# ==========================================
# Interpretation
# ==========================================

"""
## Interpretation

The optimizer benchmarking analysis evaluated how different
optimization algorithms influence neural network performance while
holding all previously optimized components constant.

The retained configuration included:

- StandardScaler
- Sigmoid Activation
- Architecture: 64 → 32 → 16
- 125 Epochs

Results demonstrated clear differences in optimizer effectiveness.

RMSProp achieved the strongest overall performance, producing the
lowest MAE (0.5383), lowest RMSE (0.6933), and highest Adjusted R²
(0.3731) among all optimizers evaluated.

Adam produced nearly identical MAE performance but generated slightly
higher RMSE and lower Adjusted R² values, indicating marginally weaker
overall predictive performance.

SGD performed substantially worse than both adaptive optimizers,
producing the highest prediction errors and lowest explanatory power.
These results suggest that SGD converged less effectively for this
problem despite using the same architecture and training duration.

The similarity between RMSProp and Adam suggests that adaptive learning
rate methods are highly effective for structured wine quality
prediction tasks. However, RMSProp demonstrated a small but consistent
advantage across all evaluation metrics.

Overall, the results indicate that RMSProp provides the most effective
optimization strategy for this project, producing the strongest
combination of predictive accuracy, convergence quality, and model
generalization.
"""

# ==========================================
# Key Insights
# ==========================================

"""
## Key Insights

### 1. RMSProp Produced the Best Overall Performance

RMSProp achieved the lowest MAE, lowest RMSE, and highest Adjusted R²
among all optimizers evaluated.

### 2. Adaptive Optimizers Clearly Outperformed SGD

Both RMSProp and Adam substantially outperformed SGD across all
evaluation metrics, demonstrating the effectiveness of adaptive
learning rates.

### 3. RMSProp Narrowly Surpassed Adam

Although performance differences were small, RMSProp achieved slightly
better MAE, RMSE, and Adjusted R² values than Adam.

### 4. SGD Produced The Weakest Results

SGD generated the highest prediction errors and lowest Adjusted R²,
indicating slower or less effective convergence.

### 5. Optimizer Choice Meaningfully Affected Performance

All model architecture, scaling, activation functions, and epoch
counts remained fixed, demonstrating that optimization strategy alone
can influence final predictive accuracy.

### 6. RMSProp Achieved The Lowest RMSE

RMSE is the primary evaluation metric for this project. RMSProp
achieved the best overall predictive accuracy with an RMSE of 0.6933.

### 7. Adam Remained Highly Competitive

Adam finished a close second and produced only marginally weaker
results than RMSProp.

### 8. Model Generalization Remained Strong

Training and validation losses remained well aligned, indicating that
all optimizers produced stable learning behavior without substantial
overfitting.

### 9. RMSProp Achieved The Highest Explanatory Power

RMSProp generated the highest Adjusted R² value (0.3731), explaining
more variance in wine quality scores than competing optimizers.

### 10. A Clear Optimizer Ranking Emerged

Based on overall predictive performance:

1. RMSProp
2. Adam
3. SGD
"""
