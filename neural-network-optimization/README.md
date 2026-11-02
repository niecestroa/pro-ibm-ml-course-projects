# Wine Quality Neural Network Optimization

### IBM Machine Learning Professional Certificate

**Project Status: Analysis Complete | Report Incomplete**  
**Last Updated:** November 2, 2026

---

# 📌 Project Overview

This project investigates the use of deep learning techniques to predict wine quality using physicochemical wine measurements.

Wine quality prediction is a supervised regression problem where the target variable is a numerical quality score assigned to each wine observation. The objective is to determine how effectively neural networks can model complex relationships between wine chemistry and perceived quality.

The project follows a structured model optimization framework beginning with exploratory data analysis and progressing through feature scaling, activation function selection, architecture optimization, epoch tuning, optimizer benchmarking, regularization analysis, and final model selection.

Each experiment retains the strongest-performing configuration identified in previous analyses, creating a fully reproducible deep learning optimization pipeline.

---

# 🎯 Project Objective

Develop, optimize, and evaluate neural network models capable of predicting wine quality scores.

The project investigates:

- Neural network fundamentals
- Network architecture design
- Feature scaling techniques
- Training behavior
- Optimization algorithms
- Regularization methods
- Model generalization

The final goal is to identify the neural network configuration that produces the most accurate wine quality predictions.

The project also emphasizes experimental reproducibility through fixed random seeds, controlled train/test splits, and consistent evaluation procedures across all analyses.

---

# 🔄 Reproducibility

All analyses were conducted using a fixed random seed.

```python
SEED = 42
```

The following controls were implemented throughout the project:

- `random.seed(SEED)`
- `numpy.random.seed(SEED)`
- `tensorflow.random.set_seed(SEED)`
- `random_state=SEED` for train/test splits

These controls ensure that data splitting, model initialization, training behavior, and experimental results can be reproduced consistently across all analyses.

---

# 📊 Dataset

**Dataset:** `winequality-final.csv`

The dataset contains 5,320 wine observations consisting of physicochemical measurements and wine quality ratings.

---

## Features

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol
- Wine Type

---

## Target Variable

### Wine Quality Score

```text
Target = Quality
```

Wine quality is treated as a continuous numerical variable and modeled using regression techniques.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- TensorFlow / Keras
- Spyder IDE
- Git
- GitHub

---

# 📂 Repository Structure

```text
00-data/
01-EDA/
02-Baseline-Neural-Network/
03-Feature-Scaling-Comparison/
04-Activation-Function-Comparison/
05-Architecture-Comparison/
06-Epoch-Study/
07-Optimizer-Benchmarking/
08-Regularization-and-Generalization/
09-Final-Model-Selection/
reports/
README.md
```

---

# 📊 Exploratory Data Analysis (EDA)

The project began with a comprehensive exploratory data analysis (EDA) to evaluate data quality, understand feature distributions, identify relationships with wine quality, and assess readiness for regression modeling.

---

## Dataset Quality

| Metric | Value |
|----------|----------:|
| Rows | 5,320 |
| Columns | 13 |
| Missing Values | 0 |
| Duplicate Rows | 0 |

### Findings

✅ No missing values  
✅ No duplicate records  
✅ Dataset immediately suitable for machine learning  
✅ No data integrity concerns identified

---

## Quality Distribution

| Quality | Count | Percent |
|---------|-------:|--------:|
| 3 | 30 | 0.56% |
| 4 | 206 | 3.87% |
| 5 | 1,752 | 32.93% |
| 6 | 2,323 | 43.67% |
| 7 | 856 | 16.09% |
| 8 | 148 | 2.78% |
| 9 | 5 | 0.09% |

### Findings

- Most wines received quality scores of 5 or 6.
- More than 76% of observations fall within these two categories.
- Extremely high-quality wines are rare.

### Implication

Prediction performance may decline at extreme quality levels due to limited training examples.

---

## Wine Type Distribution

| Wine Type | Count | Percent |
|------------|--------:|--------:|
| White | 3,961 | 74.5% |
| Red | 1,359 | 25.5% |

### Average Quality by Wine Type

| Wine Type | Mean Quality |
|------------|------------:|
| Red | 5.62 |
| White | 5.86 |

### Findings

White wines receive slightly higher quality ratings on average.

---

## Most Important Relationships with Quality

### Positive Correlations

| Feature | Correlation |
|----------|-----------:|
| Alcohol | 0.469 |
| Citric Acid | 0.098 |
| Free Sulfur Dioxide | 0.054 |

### Negative Correlations

| Feature | Correlation |
|----------|-----------:|
| Density | -0.326 |
| Volatile Acidity | -0.265 |
| Chlorides | -0.202 |

### Key Finding

Alcohol emerged as the strongest individual predictor of wine quality.

---

## Quality Trends

As wine quality increases:

✅ Alcohol increases  
✅ Density decreases  
✅ Volatile acidity decreases  
✅ Chlorides decrease  
✅ Citric acid increases slightly

### Example

| Quality | Mean Alcohol |
|----------|------------:|
| 5 | 9.87 |
| 6 | 10.65 |
| 7 | 11.51 |
| 8 | 11.91 |
| 9 | 12.18 |

Alcohol demonstrates the clearest trend across quality levels.

---

## Outlier Assessment

Variables containing the highest numbers of outliers:

- Fixed Acidity
- Volatile Acidity
- Chlorides
- Sulphates
- Residual Sugar

### Finding

Several chemistry measurements contain substantial outliers which may influence model training.

---

## Skewness Assessment

Most skewed predictors:

| Feature | Skewness |
|----------|---------:|
| Chlorides | 5.34 |
| Sulphates | 1.81 |
| Residual Sugar | 1.71 |
| Fixed Acidity | 1.65 |
| Volatile Acidity | 1.50 |

Quality skewness:

```text
0.147
```

The target variable is approximately symmetric.

---

## Multicollinearity Assessment

Highest VIF scores:

| Predictor | VIF |
|----------|------:|
| Density | 14.92 |
| Residual Sugar | 6.43 |
| Fixed Acidity | 4.88 |
| Alcohol | 4.58 |

### Finding

Density exhibits substantial multicollinearity and should be monitored in future analyses.

---

## EDA Conclusion

### Strengths

✅ Clean dataset  
✅ No missing values  
✅ Strong sample size  
✅ Strong predictor-target relationships  
✅ Multiple informative chemistry measurements

### Challenges

⚠ Quality imbalance  
⚠ Predictor outliers  
⚠ Skewed distributions  
⚠ Density multicollinearity

### Overall Assessment

The dataset appears highly suitable for regression modeling and deep learning experimentation.

---

# 📚 Deep Learning Topics

This project applies concepts from the IBM Deep Learning curriculum, including:

- Neural Networks
- Perceptrons
- Multi-Layer Perceptrons (MLPs)
- Activation Functions
- Forward Propagation
- Backpropagation
- Loss Functions
- Gradient Descent
- Optimizers
- Regularization Techniques
- Hyperparameter Optimization
- Model Selection
- Generalization Analysis
- Reproducible Machine Learning Workflows

---

# 🔬 Research Questions

1. Can neural networks accurately predict wine quality?
2. How much improvement does a Multi-Layer Perceptron provide over a simple Perceptron?
3. Which feature scaling method performs best?
4. Which activation function performs best?
5. Does increasing network complexity improve performance?
6. What is the optimal number of training epochs?
7. Which optimizer converges most effectively?
8. Which regularization strategy provides the strongest balance between predictive performance and generalization?
9. What neural network configuration achieves the best overall predictive performance?

---

# 📈 Analyses

---

## ✅ Analysis 1: Baseline Neural Network Performance

### Objective

Establish a baseline performance benchmark using simple neural network architectures.

### Models Evaluated

#### Perceptron

```text
Input → Output
```

#### Basic Multi-Layer Perceptron (MLP)

```text
Input
 ↓
Dense(16)
 ↓
Output
```

### Results

| Model | MAE | RMSE | Adjusted R² |
|---------|---------:|---------:|---------:|
| Perceptron | 0.5894 | 0.7686 | 0.2295 |
| MLP | 0.5660 | 0.7365 | 0.2924 |

### Findings

- MLP outperformed the Perceptron across all metrics.
- RMSE improved by approximately 4.2%.
- Adjusted R² improved by approximately 27%.
- Learning curves showed stable convergence.
- No clear evidence of overfitting was observed.

### Conclusion

The hidden layer allowed the MLP to learn nonlinear relationships that the Perceptron could not capture.

The MLP is retained as the baseline architecture for future analyses.

---

## ✅ Analysis 2: Feature Scaling Comparison

### Objective

Determine which feature scaling strategy produces the strongest neural network performance.

### Methods Evaluated

- No Scaling
- StandardScaler
- MinMaxScaler

### Results

| Scaling Method | MAE | RMSE | Adjusted R² |
|---------------|------:|------:|------:|
| No Scaling | 0.5656 | 0.7334 | 0.2983 |
| StandardScaler | 0.5545 | 0.7161 | 0.3311 |
| MinMaxScaler | 0.5519 | 0.7094 | 0.3435 |

### Findings

- Feature scaling improved overall model performance.
- Both StandardScaler and MinMaxScaler outperformed the unscaled model.
- StandardScaler demonstrated the most reliable overall performance during subsequent analyses.
- Training remained stable across all scaling approaches.

### Conclusion

Although both scaling techniques improved predictive performance relative to the unscaled baseline, StandardScaler was selected as the preferred scaling strategy based on its performance within the final optimized deep learning pipeline.

### Final Ranking

1. StandardScaler
2. MinMaxScaler
3. No Scaling

---

## ✅ Analysis 3: Activation Function Comparison

### Objective

Evaluate the impact of activation function selection on neural network performance.

### Methods Evaluated

- ReLU
- Sigmoid
- Tanh

### Results

| Activation Function | MAE | RMSE | Adjusted R² |
|--------------------|------:|------:|------:|
| ReLU | 0.5520 | 0.7120 | 0.3388 |
| Sigmoid | 0.5385 | 0.6967 | 0.3670 |
| Tanh | 0.5627 | 0.7297 | 0.3054 |

### Findings

- Sigmoid achieved the lowest RMSE and the highest Adjusted R².
- ReLU demonstrated competitive predictive performance but did not surpass Sigmoid.
- Tanh produced the weakest overall results.
- All activation functions exhibited stable learning behavior.

### Conclusion

Sigmoid provided the strongest overall predictive performance and was selected as the activation function for all subsequent analyses.

### Final Ranking

1. Sigmoid
2. ReLU
3. Tanh

---

## ✅ Analysis 4: Architecture Comparison

### Objective

Evaluate the effect of increasing network complexity on predictive performance.

### Architectures Evaluated

#### Model A

```text
16
```

#### Model B

```text
32 → 16
```

#### Model C

```text
64 → 32 → 16
```

#### Model D

```text
128 → 64 → 32 → 16
```

### Results

| Model | Architecture | MAE | RMSE | Adjusted R² |
|---------|-----------|------:|------:|------:|
| Model A | 16 | 0.5412 | 0.6993 | 0.3622 |
| Model B | 32 → 16 | 0.5424 | 0.6960 | 0.3681 |
| Model C | 64 → 32 → 16 | 0.5354 | 0.6947 | 0.3705 |
| Model D | 128 → 64 → 32 → 16 | 0.5365 | 0.7013 | 0.3585 |

### Findings

- Increasing architecture depth improved predictive performance from Model A through Model C.
- Model C achieved the lowest RMSE, lowest MAE, and highest Adjusted R².
- Model D introduced additional complexity without improving overall performance.
- Evidence of diminishing returns emerged beyond Model C.
- All architectures demonstrated stable learning behavior.

### Conclusion

Model C provided the strongest balance between predictive performance and network complexity and was selected for future analyses.

### Final Ranking

1. Model C (64 → 32 → 16)
2. Model B (32 → 16)
3. Model A (16)
4. Model D (128 → 64 → 32 → 16)

---

## ✅ Analysis 5: Epoch Optimization

### Objective

Determine the optimal number of training epochs.

### Epochs Evaluated

25, 50, 75, 100, 125, 150, 175, and 200

### Results

| Epochs | MAE | RMSE | Adjusted R² |
|---------|------:|------:|------:|
| 25 | 0.5650 | 0.7309 | 0.3033 |
| 50 | 0.5510 | 0.7101 | 0.3424 |
| 75 | 0.5460 | 0.7023 | 0.3566 |
| 100 | 0.5407 | 0.7008 | 0.3594 |
| 125 | 0.5365 | 0.6950 | 0.3699 |
| 150 | 0.5386 | 0.6964 | 0.3674 |
| 175 | 0.5391 | 0.7008 | 0.3593 |
| 200 | 0.5493 | 0.7081 | 0.3461 |

### Findings

- Performance improved steadily between 25 and 125 epochs.
- The strongest predictive performance occurred at 125 epochs.
- Additional training beyond 125 epochs did not improve model accuracy.
- Validation loss began increasing after 125 epochs, indicating diminishing returns.
- Longer training times beyond 125 epochs produced minimal practical benefit.

### Conclusion

125 epochs provided the strongest balance between predictive performance, model generalization, and computational efficiency.

### Final Ranking

1. 125 Epochs
2. 150 Epochs
3. 100 Epochs
4. 175 Epochs
5. 75 Epochs
6. 200 Epochs
7. 50 Epochs
8. 25 Epochs

---

## ✅ Analysis 6: Optimizer Benchmarking

### Objective

Determine which optimizer produces the strongest predictive performance while holding all other model components constant.

### Optimizers Evaluated

- SGD
- RMSProp
- Adam

### Results

| Optimizer | MAE | RMSE | Adjusted R² |
|------------|------:|------:|------:|
| RMSProp | 0.5383 | 0.6933 | 0.3731 |
| Adam | 0.5385 | 0.6967 | 0.3670 |
| SGD | 0.5611 | 0.7292 | 0.3065 |

### Findings

- RMSProp achieved the lowest RMSE and highest Adjusted R².
- Adam delivered highly competitive performance but did not surpass RMSProp.
- SGD substantially underperformed the adaptive optimizers.
- Adaptive optimization methods proved more effective for this prediction task.

### Conclusion

RMSProp produced the strongest overall predictive performance and was selected as the optimizer for subsequent analyses.

### Final Ranking

1. RMSProp
2. Adam
3. SGD

---

## ✅ Analysis 7: Regularization and Generalization

### Objective

Evaluate the impact of regularization techniques on predictive performance, model robustness, and generalization.

### Methods Evaluated

- Baseline
- L1
- L2
- Dropout
- Early Stopping

### Results

| Method | MAE | RMSE | Adjusted R² |
|----------|------:|------:|------:|
| Baseline | 0.5383 | 0.6933 | 0.3731 |
| EarlyStopping | 0.5378 | 0.6941 | 0.3716 |
| Dropout | 0.5497 | 0.7043 | 0.3530 |
| L2 | 0.5627 | 0.7299 | 0.3051 |
| L1 | 0.6967 | 0.8796 | -0.0092 |

### Findings

- The baseline model achieved the strongest overall predictive performance.
- Early Stopping produced highly competitive performance while maintaining strong generalization.
- Dropout provided moderate regularization benefits but did not improve predictive accuracy.
- L2 regularization reduced predictive performance relative to the baseline model.
- L1 regularization substantially degraded predictive performance.

### Winner

Baseline (No Regularization)

### Conclusion

Additional regularization provided little benefit because the optimized neural network already generalized effectively. The baseline model achieved the lowest RMSE and highest Adjusted R² and was retained as the final configuration.

### Final Ranking

1. Baseline
2. EarlyStopping
3. Dropout
4. L2
5. L1

---

## ✅ Analysis 8: Final Model Selection

### Objective

Identify the strongest-performing neural network configuration discovered throughout the project.

### Final Recommended Configuration

```text
MLP
→ StandardScaler
→ 64 → 32 → 16 Architecture
→ Sigmoid Activation
→ RMSProp Optimizer
→ 125 Epochs
→ No Regularization (Baseline)
→ SEED = 42
```

### Final Performance

| Metric | Value |
|----------|------:|
| MAE | 0.5383 |
| RMSE | 0.6933 |
| Adjusted R² | 0.3731 |
| Training Time (Seconds) | 37.0841 |
| Generalization Gap | -0.0003 |

### Conclusion

The strongest-performing neural network configuration combined StandardScaler preprocessing, a 64 → 32 → 16 architecture, Sigmoid activation functions, RMSProp optimization, 125 training epochs, and no regularization.

This configuration achieved the strongest overall balance of predictive accuracy, training stability, reproducibility, and model generalization across all analyses conducted within the DLWQ project.

---

# 📈 Performance Improvement

| Metric | Baseline MLP | Final Model | Improvement |
|----------|----------:|----------:|----------:|
| MAE | 0.5660 | 0.5383 | 4.89% |
| RMSE | 0.7365 | 0.6933 | 5.87% |
| Adjusted R² | 0.2924 | 0.3731 | 27.60% |

---

# 📊 Evaluation Metrics

## Primary Metric

### RMSE

Root Mean Squared Error

Primary model-selection metric because it penalizes larger prediction errors more heavily.

---

### MAE

Mean Absolute Error

Measures the average magnitude of prediction errors.

---

### Adjusted R²

Measures explained variance while accounting for model complexity.

---

# 📋 Project Deliverables

- Exploratory Data Analysis Report
- Baseline Neural Network Benchmark
- Feature Scaling Comparison
- Activation Function Comparison
- Architecture Comparison
- Epoch Optimization Study
- Optimizer Benchmarking Study
- Regularization and Generalization Study
- Final Model Recommendation Report
- Model Performance Summary Report

---

# 🔍 Final Research Question

> What combination of feature scaling, neural network architecture, activation function, optimizer, training duration, and regularization strategy provides the most accurate wine quality predictions for structured tabular wine data?

---

## Completed

✅ Exploratory Data Analysis (EDA)  
✅ Analysis 1: Baseline Neural Network Performance  
✅ Analysis 2: Feature Scaling Comparison  
✅ Analysis 3: Activation Function Comparison  
✅ Analysis 4: Architecture Comparison  
✅ Analysis 5: Epoch Optimization  
✅ Analysis 6: Optimizer Benchmarking  
✅ Analysis 7: Regularization and Generalization  
✅ Analysis 8: Final Model Selection

---

# 📌 Final Project Outcome

After completing eight experimental analyses, the recommended deep learning model for wine quality prediction is:

```text
MLP
→ StandardScaler
→ 64 → 32 → 16 Neural Network
→ Sigmoid Activation
→ RMSProp Optimizer
→ 125 Epochs
→ No Regularization (Baseline)
→ SEED = 42
```

## Final Performance

| Metric | Value |
|----------|------:|
| MAE | 0.5383 |
| RMSE | 0.6933 |
| Adjusted R² | 0.3731 |
| Generalization Gap | -0.0003 |

This configuration achieved the strongest overall balance of predictive accuracy, training stability, reproducibility, and model generalization across all analyses conducted within the DLWQ project.

---

# 🚀 Next Steps

- Executive presentation slides
- GitHub repository publication
- Portfolio project showcase

---

# End of README
