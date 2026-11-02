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
algorithm is most effective for neural network-based wine quality
prediction.

Three optimization algorithms were evaluated:

- SGD
- RMSProp
- Adam

All experiments used the same modeling framework, including feature
scaling, network architecture, activation functions, training
duration, and validation strategy. As a result, observed performance
differences can be directly attributed to optimizer selection.

The benchmark produced a clear ranking:

1. RMSProp
2. Adam
3. SGD

RMSProp achieved the best overall predictive performance:

- MAE: 0.5472
- RMSE: 0.7052
- Adjusted R²: 0.3513

Adam remained highly competitive and produced results close to
RMSProp; however, it did not achieve superior predictive accuracy.

SGD produced considerably weaker performance and was unable to match
the results achieved by the adaptive optimization methods.

The findings indicate that adaptive learning-rate optimization is
highly beneficial for this structured tabular prediction problem.
RMSProp successfully balanced convergence speed, model stability, and
generalization performance, resulting in the strongest overall model.

### Final Optimizer Ranking

1. RMSProp
2. Adam
3. SGD

### Winning Configuration

- MinMaxScaler
- ReLU Activation
- Architecture: 64 → 32 → 16
- RMSProp Optimizer
- 75 Epochs
- Validation Split: 20%

### Winning Performance

- MAE: 0.5472
- RMSE: 0.7052
- Adjusted R²: 0.3513

Based on all evaluation metrics and benchmarking results, RMSProp was
selected as the optimal optimizer for wine quality prediction and will
be used in subsequent analyses.
"""

# ==========================================
# Summary
# ==========================================

"""
## Overall Summary

The objective of the Optimizer Benchmarking study was to determine
which deep learning optimization algorithm produces the strongest
predictive performance for wine quality prediction.

The following model configuration was held constant throughout the
study:

- MinMaxScaler
- ReLU Activation
- Architecture: 64 → 32 → 16
- Epochs: 75
- Validation Split: 20%

Three commonly used optimizers were evaluated:

- SGD
- RMSProp
- Adam

Performance results demonstrated substantial differences between the
optimization methods.

RMSProp achieved the strongest overall results:

- MAE: 0.5472
- RMSE: 0.7052
- Adjusted R²: 0.3513

Adam produced competitive performance but failed to surpass RMSProp,
while SGD produced substantially weaker predictive accuracy.

The findings indicate that optimizer selection has a meaningful impact
on neural network performance even when all other model settings are
held constant.

Overall, RMSProp emerged as the most effective optimizer for wine
quality prediction and became the preferred optimization algorithm for
subsequent analyses.
"""

# ==========================================
# Key Insights
# ==========================================

"""
## Overall Key Insights

### 1. Optimizer Selection Matters

Model performance changed noticeably despite all experiments using the
same dataset, scaling method, architecture, and training duration.

### 2. RMSProp Produced the Strongest Results

RMSProp achieved the lowest RMSE and highest Adjusted R² scores of the
benchmark.

### 3. Adaptive Optimizers Outperformed SGD

Both RMSProp and Adam substantially outperformed SGD across all
evaluation metrics.

### 4. RMSProp Slightly Outperformed Adam

Although Adam remained highly competitive, RMSProp produced lower
prediction error and better explanatory power.

### 5. SGD Was Less Effective

SGD struggled to reach the same performance levels achieved by the
adaptive optimization algorithms.

### 6. Convergence Behavior Improved with Adaptive Learning Rates

The results suggest that automatic learning-rate adaptation provided
meaningful benefits during optimization.

### 7. Predictive Accuracy Was Maximized with RMSProp

RMSProp consistently delivered the strongest balance of accuracy and
generalization.

### 8. RMSProp Established a New Project Benchmark

The optimizer achieved the strongest performance observed during
Analysis 6 and becomes the optimizer carried forward into future
experiments.

### 9. Training Stability Remained Strong

All optimizers successfully trained the network, but RMSProp and Adam
converged more effectively than SGD.

### 10. Optimizer Choice Is a Critical Hyperparameter

The benchmark demonstrated that optimizer selection can influence
model quality as much as other architectural decisions.
"""

# ==========================================
# Analysis: 06 - Optimizer Benchmarking
# Method: SGD
# ==========================================

"""
## Interpretation

The SGD model was evaluated to establish a benchmark for traditional
gradient-based optimization and to determine how effectively SGD
could optimize a neural network for wine quality prediction.

The model achieved:

- MAE: 0.5935
- RMSE: 0.7564
- Adjusted R²: 0.2538

These results represent the weakest overall performance observed in
the optimizer benchmarking study.

Although SGD successfully trained the network, convergence occurred
more slowly and predictive accuracy remained lower than both RMSProp
and Adam.

The learning curve likely demonstrates slower convergence and less
efficient optimization compared with adaptive methods. Validation
loss remained substantially higher than that achieved by the other
optimizers.

Overall, the results indicate that SGD was less effective for this
dataset and neural network architecture.
"""

"""
## Summary

The objective of this analysis was to evaluate the performance of
Stochastic Gradient Descent (SGD) for wine quality prediction.

Performance results were:

- MAE: 0.5935
- RMSE: 0.7564
- Adjusted R²: 0.2538

SGD produced the highest prediction error and lowest explained
variance of all optimizers evaluated.

While training remained stable, convergence was less effective than
the adaptive optimization methods.

Overall, SGD was the weakest optimizer evaluated in this study.
"""

"""
## Key Insights

### 1. SGD Produced the Weakest Performance

All primary metrics were worse than RMSProp and Adam.

### 2. Convergence Was Less Effective

The optimizer struggled to achieve the same level of performance as
adaptive optimization methods.

### 3. Generalization Performance Was Lower

Validation performance remained weaker than competing optimizers.

### 4. Optimization Efficiency Was Limited

The learning process appeared less capable of finding strong model
parameters.

### 5. SGD Is Not Recommended

The results suggest that SGD is not the optimal optimizer for this
wine quality prediction task.

### Current Optimizer Ranking

1. RMSProp
2. Adam
3. SGD
"""

# ==========================================
# Analysis: 06 - Optimizer Benchmarking
# Method: RMSProp
# ==========================================

"""
## Interpretation

The RMSProp model was evaluated to determine whether adaptive learning
rates could improve optimization and predictive performance.

The model achieved:

- MAE: 0.5472
- RMSE: 0.7052
- Adjusted R²: 0.3513

These results represent the strongest overall performance observed in
the optimizer benchmarking study.

RMSProp converged efficiently and achieved the lowest prediction
error among all evaluated optimizers. Training and validation losses
remained stable throughout the learning process.

The learning curve suggests effective optimization behavior and strong
generalization performance. RMSProp successfully balanced convergence
speed and predictive accuracy.

Overall, RMSProp emerged as the most effective optimizer for wine
quality prediction.
"""

"""
## Summary

The objective of this analysis was to evaluate RMSProp as an adaptive
optimization algorithm for wine quality prediction.

Performance results were:

- MAE: 0.5472
- RMSE: 0.7052
- Adjusted R²: 0.3513

RMSProp achieved the strongest results across all primary evaluation
metrics and demonstrated stable optimization behavior.

The optimizer successfully balanced efficient learning and strong
generalization performance.

Overall, RMSProp was the best-performing optimizer evaluated in this
study.
"""

"""
## Key Insights

### 1. RMSProp Produced the Best Overall Performance

The optimizer achieved the lowest RMSE and highest Adjusted R².

### 2. Adaptive Learning Rates Were Beneficial

RMSProp effectively adjusted model updates during training.

### 3. Training Stability Was Strong

Learning remained stable throughout optimization.

### 4. Generalization Performance Was Strong

Validation performance remained competitive and consistent.

### 5. RMSProp Became the New Benchmark

The optimizer established the strongest performance achieved so far.

### Current Optimizer Ranking

1. RMSProp
2. Adam
3. SGD
"""

# ==========================================
# Analysis: 06 - Optimizer Benchmarking
# Method: Adam
# ==========================================

"""
## Interpretation

The Adam optimizer was evaluated because it served as the optimization
method used in previous analyses and represented the current project
benchmark.

The model achieved:

- MAE: 0.5547
- RMSE: 0.7103
- Adjusted R²: 0.3420

Adam produced strong predictive performance and substantially
outperformed SGD. However, it did not exceed the results achieved by
RMSProp.

The learning curve likely demonstrated efficient convergence and
stable optimization behavior. Despite these strengths, RMSProp
achieved slightly lower prediction error and stronger explanatory
power.

Overall, Adam remained a highly effective optimizer but was not the
top-performing method in this study.
"""

"""
## Summary

The objective of this analysis was to evaluate Adam relative to other
commonly used neural network optimizers.

Performance results were:

- MAE: 0.5547
- RMSE: 0.7103
- Adjusted R²: 0.3420

Adam delivered strong predictive performance and clearly outperformed
SGD. However, RMSProp achieved slightly better results across all
primary metrics.

Overall, Adam remained a competitive optimizer but did not emerge as
the best-performing method.
"""

"""
## Key Insights

### 1. Adam Performed Well

Predictive performance remained strong across all metrics.

### 2. RMSProp Slightly Outperformed Adam

RMSProp achieved lower prediction error and higher explained variance.

### 3. Adam Remained Superior to SGD

All primary metrics were substantially stronger than SGD.

### 4. Training Stability Was Strong

Optimization behavior remained stable throughout training.

### 5. Adam Remains a Competitive Choice

Despite not winning the benchmark, Adam continues to be a highly
effective optimizer for this task.

### Current Optimizer Ranking

1. RMSProp
2. Adam
3. SGD
"""

