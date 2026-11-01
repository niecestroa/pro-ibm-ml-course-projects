# Analysis 1: Baseline Neural Network Performance

## Key Insights

### 1. The MLP Outperformed the Perceptron

The Multi-Layer Perceptron (MLP) achieved lower prediction errors and a higher Adjusted R² than the Perceptron.

| Model | MAE | RMSE | Adjusted R² |
|---------|---------:|---------:|---------:|
| Perceptron | 0.5894 | 0.7686 | 0.2295 |
| MLP | 0.5660 | 0.7365 | 0.2924 |

**Insight**

The results indicate that adding a hidden layer improves predictive performance for wine quality prediction.

---

### 2. Wine Quality Is Not Purely Linear

The superior performance of the MLP suggests that the relationship between physicochemical wine measurements and wine quality contains nonlinear patterns that cannot be fully captured by a simple linear Perceptron.

**Insight**

Hidden-layer architectures are better able to model complex relationships between wine characteristics and perceived quality.

---

### 3. Both Models Converged Quickly

Training and validation losses decreased rapidly during the first several epochs before stabilizing.

Most learning occurred early in the training process, indicating efficient convergence and stable optimization.

**Insight**

Both neural network architectures were able to learn meaningful patterns without requiring extensive training time.

---

### 4. Limited Evidence of Overfitting

Training and validation loss curves remained closely aligned throughout training.

The absence of a widening gap between the two curves suggests that neither model experienced significant overfitting during the baseline experiment.

**Insight**

The models demonstrated good generalization to unseen data and maintained stable learning behavior throughout training.

---

### 5. Prediction Accuracy Was Strongest Near the Center of the Distribution

Both models produced their best predictions for wines with quality scores between 5 and 7, which represented the majority of observations in the dataset.

Prediction accuracy declined at the extreme quality levels.

**Insight**

Prediction performance was strongest for the most frequently occurring quality scores and weaker for rare quality categories.

---

### 6. Extreme Quality Ratings Were Difficult to Predict

The dataset contained very few examples of high- and low-quality wines.

| Quality Score | Observations |
|--------------|-------------:|
| 3 | 30 |
| 4 | 206 |
| 5 | 1,752 |
| 6 | 2,323 |
| 7 | 856 |
| 8 | 148 |
| 9 | 5 |

**Insight**

The scarcity of quality scores 3, 8, and 9 limited the models' ability to learn patterns associated with these rare observations.

---

### 7. Both Models Exhibited Regression Toward the Mean

Actual-versus-predicted plots showed a tendency to overestimate lower-quality wines and underestimate higher-quality wines.

Predictions were concentrated around quality scores of approximately 5 to 7, reflecting the distribution of the training data.

**Insight**

Both models were influenced by the concentration of observations in the middle quality range, resulting in reduced accuracy at distribution extremes.

---

## Overall Findings

### Strengths

- MLP outperformed the Perceptron across all evaluation metrics.
- Both models converged quickly and trained efficiently.
- Minimal evidence of overfitting was observed.
- Models generalized reasonably well to unseen data.
- Neural networks successfully captured meaningful relationships within the wine chemistry data.

### Limitations

- Prediction accuracy decreased for rare quality scores.
- Both models showed regression toward the mean.
- Extreme wine quality ratings were difficult to model due to severe class imbalance.
- Additional model complexity may be required to capture more nuanced relationships.

---

## Conclusion

The baseline experiment demonstrated that a simple Multi-Layer Perceptron (MLP) provides measurable improvements over a Perceptron for wine quality prediction while maintaining stable learning behavior and good generalization. The MLP achieved lower MAE and RMSE values along with a higher Adjusted R², indicating a stronger overall fit to the data.

Given its superior predictive performance and stable training characteristics, the MLP was selected as the baseline neural network architecture for all subsequent analyses.
