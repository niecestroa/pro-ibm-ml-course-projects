# Analysis 02: Feature Scaling Comparison

## Objective

Evaluate the impact of feature scaling on neural network performance for wine quality prediction using a consistent network architecture and training configuration.

### Scaling Methods Evaluated

1. No Scaling
2. StandardScaler
3. MinMaxScaler

---

## Results Summary

| Scaling Method | MAE | RMSE | Adjusted R² |
|----------------|------:|------:|------:|
| No Scaling | 0.5656 | 0.7334 | 0.2983 |
| StandardScaler | 0.5546 | 0.7170 | 0.3294 |
| MinMaxScaler | 0.5519 | 0.7094 | 0.3435 |

---

## Key Findings

### 1. Feature Scaling Improved Model Performance

Both StandardScaler and MinMaxScaler improved predictive performance compared with the unscaled baseline.

Observed improvements included:

- Lower MAE
- Lower RMSE
- Higher Adjusted R²
- Better prediction accuracy across most observations

### 2. MinMaxScaler Produced the Best Results

MinMaxScaler achieved the strongest overall performance among all methods evaluated.

| Metric | Value |
|----------|------:|
| MAE | 0.5519 |
| RMSE | 0.7094 |
| Adjusted R² | 0.3435 |

**Finding:** Scaling features into a bounded range improved neural network learning and predictive accuracy.

### 3. StandardScaler Also Improved Performance

StandardScaler consistently outperformed the unscaled baseline.

| Metric | No Scaling | StandardScaler |
|----------|---------:|---------:|
| MAE | 0.5656 | 0.5546 |
| RMSE | 0.7334 | 0.7170 |
| Adjusted R² | 0.2983 | 0.3294 |

**Finding:** Standardizing predictor variables improved network optimization and predictive performance.

### 4. Training Stability Remained Strong

Across all scaling methods:

- Training and validation losses declined rapidly.
- Convergence occurred early in training.
- Learning curves remained stable throughout training.

**Finding:** Feature scaling improved performance without negatively affecting training stability.

### 5. Limited Evidence of Overfitting

Training and validation loss curves remained closely aligned for all experiments.

**Finding:** None of the scaling approaches introduced substantial overfitting and all models demonstrated good generalization.

### 6. Prediction Accuracy Improved

Scaled models produced:

- Tighter clustering around the ideal prediction line.
- Reduced prediction error.
- Improved agreement between actual and predicted wine quality scores.

**Finding:** Scaling allowed the neural network to learn more effectively from the physicochemical wine features.

### 7. Extreme Quality Ratings Remained Difficult to Predict

Despite overall improvements:

- Low-frequency quality scores remained challenging.
- Predictions continued to regress toward the middle of the distribution.
- Rare quality ratings (3, 8, and 9) produced larger prediction errors.

**Finding:** Dataset imbalance remains a primary limitation independent of scaling method.

---

## Method-Specific Findings

### No Scaling

#### Findings

- Established baseline performance.
- Converged quickly and trained successfully.
- Predictions were strongest for quality scores between 5 and 7.
- Regression toward the mean was observed.
- Prediction range was compressed around common quality values.

#### Metrics

| Metric | Value |
|---------|------:|
| MAE | 0.5656 |
| RMSE | 0.7334 |
| Adjusted R² | 0.2983 |

---

### StandardScaler

#### Findings

- Improved all evaluation metrics relative to the baseline.
- Reduced scale differences between variables.
- Improved learning efficiency.
- Produced more accurate predictions.
- Maintained stable convergence and good generalization.

#### Metrics

| Metric | Value |
|---------|------:|
| MAE | 0.5546 |
| RMSE | 0.7170 |
| Adjusted R² | 0.3294 |

---

### MinMaxScaler

#### Findings

- Produced the strongest predictive performance.
- Achieved the lowest prediction errors.
- Generated the highest Adjusted R² score.
- Maintained stable convergence.
- Showed minimal evidence of overfitting.
- Improved prediction accuracy across most quality levels.

#### Metrics

| Metric | Value |
|---------|------:|
| MAE | 0.5519 |
| RMSE | 0.7094 |
| Adjusted R² | 0.3435 |

---

## Final Ranking

1. **MinMaxScaler**
2. **StandardScaler**
3. **No Scaling**

---

## Selected Configuration

| Parameter | Selection |
|------------|------------|
| Scaling Method | MinMaxScaler |
| Hidden Layer | Dense(16, ReLU) |
| Optimizer | Adam |
| Epochs | 100 |
| Validation Split | 20% |

---

## Conclusion

Feature scaling significantly improved neural network performance for wine quality prediction. Both StandardScaler and MinMaxScaler reduced prediction error and increased explanatory power relative to the unscaled baseline. MinMaxScaler achieved the lowest MAE and RMSE values and the highest Adjusted R² score, making it the preferred preprocessing approach for all subsequent analyses.

Although scaled models improved predictive accuracy, rare wine quality ratings remained difficult to predict due to the highly imbalanced target distribution. Based on overall performance, MinMaxScaler was selected as the standard preprocessing method moving forward.
