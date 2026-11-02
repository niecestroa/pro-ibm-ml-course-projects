"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 2, 2026
Last Edit:  November 2, 2026
Progress:   Complete

Analysis:   07 - Regularization and Generalization
"""

# ==========================================
# Analysis: 07 - Regularization and Generalization
# ==========================================

# ==========================================
# Overall Conclusion
# ==========================================

"""
## Conclusion

The objective of this analysis was to determine which regularization
strategy produces the most reliable neural network for wine quality
prediction.

Five approaches were evaluated:

- Baseline
- L1 Regularization
- L2 Regularization
- Dropout
- Early Stopping

All experiments used identical preprocessing, architecture,
activation functions, optimization settings, and training duration.
This ensured that observed differences were attributable to the
regularization strategy itself.

The benchmark produced the following ranking:

1. L2 Regularization
2. Baseline
3. Early Stopping
4. Dropout
5. L1 Regularization

L2 Regularization achieved the strongest overall performance:

- MAE: 0.5453
- RMSE: 0.7035
- Adjusted R²: 0.3546

Although the improvement over the Baseline model was modest, L2
consistently delivered the best combination of predictive accuracy,
generalization performance, and robustness.

The Baseline model remained highly competitive, indicating that the
network architecture was already well optimized before the addition of
regularization techniques.

Early Stopping reduced training time substantially but did not improve
predictive accuracy. Dropout weakened performance, while L1
Regularization proved overly restrictive and produced the weakest
results in the study.

### Final Regularization Ranking

1. L2 Regularization
2. Baseline
3. Early Stopping
4. Dropout
5. L1 Regularization

### Winning Configuration

- MinMaxScaler
- ReLU Activation
- Architecture: 64 → 32 → 16
- RMSProp Optimizer
- 75 Epochs
- L2 Regularization
- Validation Split: 20%

### Winning Performance

- MAE: 0.5453
- RMSE: 0.7035
- Adjusted R²: 0.3546

The findings indicate that modest weight regularization using L2
provides the most effective strategy for improving model reliability
while preserving predictive accuracy. Based on all evaluation
metrics, L2 Regularization is selected as the preferred approach for
subsequent analysis and final model development.
"""

# ==========================================
# Overall Summary
# ==========================================

"""
## Overall Summary

The objective of the Regularization and Generalization study was to
determine whether regularization techniques could reduce overfitting
and improve model reliability while maintaining strong predictive
performance.

The following winning configuration from previous analyses was held
constant throughout the study:

- MinMaxScaler
- ReLU Activation
- Architecture: 64 → 32 → 16
- RMSProp Optimizer
- 75 Epochs
- Validation Split: 20%

Five regularization strategies were evaluated:

- Baseline
- L1 Regularization
- L2 Regularization
- Dropout
- Early Stopping

Results demonstrated that regularization influenced predictive
performance, although the impact varied considerably between methods.

L2 Regularization achieved the strongest overall results:

- MAE: 0.5453
- RMSE: 0.7035
- Adjusted R²: 0.3546

The Baseline model remained highly competitive and finished a close
second. Early Stopping and Dropout reduced predictive performance,
while L1 Regularization produced the weakest overall results.

Overall, the study indicates that modest weight regularization using
L2 provides the best balance between predictive accuracy,
generalization performance, and model robustness.
"""

# ==========================================
# Overall Key Insights 
# ==========================================

"""
## Overall Key Insights

### 1. L2 Regularization Produced the Best Overall Performance

L2 achieved the lowest RMSE and highest Adjusted R² observed in the
study.

### 2. The Baseline Model Was Already Strong

The Baseline configuration remained highly competitive and was only
slightly outperformed by L2 Regularization.

### 3. Severe Overfitting Was Not Present

Generalization gaps remained relatively small across most models,
suggesting that the existing architecture already generalized
reasonably well.

### 4. L2 Improved Generalization Without Sacrificing Accuracy

The method successfully controlled weight growth while maintaining
strong predictive performance.

### 5. L1 Regularization Was Too Aggressive

L1 produced the highest prediction error and lowest explained
variance, indicating that important model information may have been
removed.

### 6. Dropout Reduced Predictive Performance

Dropout introduced additional regularization but did not improve
generalization or predictive accuracy.

### 7. Early Stopping Improved Efficiency

Training time decreased substantially; however, model accuracy
declined relative to the leading methods.

### 8. More Regularization Is Not Always Better

Several regularization methods reduced predictive performance despite
their intended purpose of improving generalization.

### 9. Weight Penalties Were More Effective Than Structural Changes

L2 outperformed Dropout and Early Stopping, suggesting that mild
weight control was more beneficial than aggressive training
intervention.

### 10. L2 Established a New Project Benchmark

L2 Regularization produced the strongest overall model and becomes
the preferred configuration moving forward.
"""

# ==========================================
# Analysis: 07 - Regularization and Generalization
# Method: Baseline
# ==========================================

"""
## Interpretation

The Baseline model was evaluated to establish a reference point for
comparing regularization strategies and assessing generalization
performance.

The model achieved:

- MAE: 0.5486
- RMSE: 0.7047
- Adjusted R²: 0.3522

The model demonstrated strong predictive performance and remained
highly competitive throughout the study. The relatively small
generalization gap suggests that severe overfitting was not present.

Training and validation performance remained closely aligned,
indicating that the selected architecture, optimizer, and epoch
configuration already produced a stable learning process.

Overall, the Baseline model established a strong benchmark that proved
difficult for regularization methods to significantly improve upon.
"""

"""
## Summary

The objective of this analysis was to establish a baseline model for
evaluating regularization techniques.

Performance results were:

- MAE: 0.5486
- RMSE: 0.7047
- Adjusted R²: 0.3522
- Generalization Gap: 0.0060

The model demonstrated strong predictive performance and minimal
evidence of overfitting.

Overall, the Baseline model served as a highly competitive reference
configuration for subsequent comparisons.
"""

"""
## Key Insights

### 1. Baseline Performance Was Strong

The model achieved competitive performance without regularization.

### 2. Overfitting Was Limited

The generalization gap remained very small.

### 3. Model Stability Was High

Training and validation performance remained closely aligned.

### 4. The Network Generalized Well

Performance remained strong on unseen test data.

### 5. Improvement Opportunities Were Limited

The strong baseline made meaningful gains difficult to achieve.

### Current Regularization Ranking

1. L2
2. Baseline
3. EarlyStopping
4. Dropout
5. L1
"""

# ==========================================
# Analysis: 07 - Regularization and Generalization
# Method: L1 Regularization
# ==========================================

"""
## Interpretation

The L1 regularized model was evaluated to determine whether sparsity
constraints could improve model generalization.

The model achieved:

- MAE: 0.5631
- RMSE: 0.7308
- Adjusted R²: 0.3034

Performance declined noticeably relative to the Baseline model across
all primary evaluation metrics.

The results suggest that L1 regularization penalized model complexity
too aggressively for this dataset, reducing the network's ability to
capture meaningful predictive relationships.

Overall, L1 regularization reduced model effectiveness and failed to
improve generalization performance.
"""

"""
## Summary

The objective of this analysis was to evaluate the impact of L1
regularization on predictive performance and generalization.

Performance results were:

- MAE: 0.5631
- RMSE: 0.7308
- Adjusted R²: 0.3034
- Generalization Gap: -0.0391

The model produced the weakest predictive performance observed in the
study.

Overall, L1 regularization was overly restrictive and negatively
affected model quality.
"""

"""
## Key Insights

### 1. L1 Produced the Weakest Performance

The highest prediction error was observed under L1 regularization.

### 2. Model Capacity Was Reduced

The regularization penalty likely constrained useful network weights.

### 3. Generalization Did Not Improve

Lower model complexity did not translate into stronger performance.

### 4. Predictive Accuracy Declined

All primary metrics were weaker than the Baseline model.

### 5. L1 Is Not Recommended

The results indicate that L1 regularization is not well suited for
this modeling task.

### Current Regularization Ranking

1. L2
2. Baseline
3. EarlyStopping
4. Dropout
5. L1
"""

# ==========================================
# Analysis: 07 - Regularization and Generalization
# Method: L2 Regularization
# ==========================================

"""
## Interpretation

The L2 regularized model was evaluated to determine whether weight
penalization could improve predictive performance while maintaining
strong generalization behavior.

The model achieved:

- MAE: 0.5453
- RMSE: 0.7035
- Adjusted R²: 0.3546

These results represent the strongest overall performance observed in
the regularization study.

L2 regularization successfully controlled model complexity without
restricting learning capacity. The model achieved lower prediction
error and higher explanatory power than the Baseline configuration.

Overall, L2 regularization provided the best balance between
regularization strength, predictive accuracy, and generalization.
"""

"""
## Summary

The objective of this analysis was to evaluate L2 regularization as a
method for improving model reliability and generalization.

Performance results were:

- MAE: 0.5453
- RMSE: 0.7035
- Adjusted R²: 0.3546
- Generalization Gap: -0.0203

L2 regularization achieved the strongest overall predictive
performance in the study and slightly outperformed the Baseline model.

Overall, L2 regularization emerged as the preferred regularization
strategy.
"""

"""
## Key Insights

### 1. L2 Produced the Best Overall Performance

The lowest RMSE and highest Adjusted R² were achieved using L2.

### 2. Regularization Improved Performance

L2 slightly improved predictive accuracy relative to the Baseline.

### 3. Model Complexity Was Controlled Effectively

The regularization penalty prevented excessive weight growth.

### 4. Generalization Remained Strong

Performance remained stable on unseen data.

### 5. L2 Became the New Benchmark

The strongest model observed in Analysis 7 used L2 regularization.

### Current Regularization Ranking

1. L2
2. Baseline
3. EarlyStopping
4. Dropout
5. L1
"""

# ==========================================
# Analysis: 07 - Regularization and Generalization
# Method: Dropout
# ==========================================

"""
## Interpretation

The Dropout model was evaluated to determine whether randomly
deactivating neurons during training could improve generalization.

The model achieved:

- MAE: 0.5492
- RMSE: 0.7165
- Adjusted R²: 0.3304

Although the model remained stable during training, predictive
performance declined relative to both L2 regularization and the
Baseline model.

The results suggest that Dropout may have removed too much useful
information from a relatively small tabular prediction problem.

Overall, Dropout did not improve model performance and reduced
predictive accuracy.
"""

"""
## Summary

The objective of this analysis was to evaluate the effectiveness of
Dropout regularization.

Performance results were:

- MAE: 0.5492
- RMSE: 0.7165
- Adjusted R²: 0.3304
- Generalization Gap: -0.0254

The model remained stable but failed to improve predictive
performance.

Overall, Dropout produced weaker results than both L2 regularization
and the Baseline configuration.
"""

"""
## Key Insights

### 1. Dropout Reduced Predictive Accuracy

Prediction error increased relative to stronger models.

### 2. Generalization Benefits Were Limited

The technique provided little practical improvement.

### 3. The Dataset May Not Require Aggressive Regularization

Baseline performance was already strong.

### 4. Convergence Remained Stable

The learning process remained reliable throughout training.

### 5. Dropout Was Not Competitive

Other methods achieved stronger overall results.

### Current Regularization Ranking

1. L2
2. Baseline
3. EarlyStopping
4. Dropout
5. L1
"""

# ==========================================
# Analysis: 07 - Regularization and Generalization
# Method: Early Stopping
# ==========================================

"""
## Interpretation

The Early Stopping model was evaluated to determine whether stopping
training when validation performance ceased improving could enhance
generalization performance.

The model achieved:

- MAE: 0.5535
- RMSE: 0.7113
- Adjusted R²: 0.3400

Although Early Stopping reduced training time considerably, it did not
produce superior predictive performance.

The model likely terminated training before fully benefiting from the
75-epoch training schedule identified in the Epoch Optimization study.

Overall, Early Stopping improved efficiency but did not improve model
accuracy.
"""

"""
## Summary

The objective of this analysis was to evaluate Early Stopping as a
generalization strategy.

Performance results were:

- MAE: 0.5535
- RMSE: 0.7113
- Adjusted R²: 0.3400
- Generalization Gap: 0.0776

The method substantially reduced training time but did not outperform
either L2 regularization or the Baseline model.

Overall, Early Stopping provided computational benefits but failed to
improve predictive performance.
"""

"""
## Key Insights

### 1. Training Time Was Reduced

Early Stopping finished substantially faster than other methods.

### 2. Predictive Performance Declined

Accuracy remained below the leading configurations.

### 3. The Model May Have Stopped Too Early

The network likely benefited from additional training.

### 4. Generalization Did Not Improve

Performance remained weaker than L2 and Baseline.

### 5. Efficiency Improved More Than Accuracy

The primary advantage of Early Stopping was reduced computation time.

### Current Regularization Ranking

1. L2
2. Baseline
3. EarlyStopping
4. Dropout
5. L1
"""