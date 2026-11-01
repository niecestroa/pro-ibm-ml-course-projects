# Wine Quality Regression with Deep Learning

### IBM Machine Learning Professional Certificate

**Project Status:** Active Development  
**Last Updated:** November 2026

---

# 📌 Project Overview

This project investigates the use of deep learning techniques to predict wine quality using physicochemical wine measurements.

Wine quality prediction is a supervised regression problem where the target variable is a numerical quality score assigned to each wine observation. The objective is to determine how effectively neural networks can model the complex relationships between wine chemistry and perceived quality.

The project progresses from exploratory data analysis through neural network experimentation, architecture design, optimization, regularization, and final model selection.

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

---

# 🔬 Research Questions

1. Can neural networks accurately predict wine quality?
2. How much improvement does a Multi-Layer Perceptron provide over a simple Perceptron?
3. Which feature scaling method performs best?
4. Which activation function performs best?
5. Does increasing network complexity improve performance?
6. Which optimizer converges most efficiently?
7. How can overfitting be minimized?
8. What neural network configuration achieves the best predictive performance?

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

## 🔄 Analysis 2: Feature Scaling Comparison

Status: Planned

Evaluate:

- No Scaling
- StandardScaler
- MinMaxScaler

Research Question:

Which scaling strategy produces the strongest neural network performance?

---

## 🔄 Analysis 3: Activation Function Comparison

Status: Planned

- ReLU
- Sigmoid
- Tanh

---

## 🔄 Analysis 4: Architecture Comparison

Status: Planned

- 16
- 32 → 16
- 64 → 32 → 16
- 128 → 64 → 32

---

## 🔄 Analysis 5: Learning Behavior Analysis

Status: Planned

Investigate convergence patterns, overfitting, and underfitting.

---

## 🔄 Analysis 6: Optimizer Benchmarking

Status: Planned

- SGD
- RMSProp
- Adam

---

## 🔄 Analysis 7: Regularization and Generalization

Status: Planned

- L1
- L2
- Dropout
- Early Stopping

---

## 🔄 Analysis 8: Final Model Selection

Status: Planned

Identify the optimal combination of:

- Scaling
- Architecture
- Activation
- Optimizer
- Regularization

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

# 📋 Expected Deliverables

- Exploratory Data Analysis Report
- Baseline Neural Network Benchmark
- Feature Scaling Study
- Activation Function Study
- Architecture Comparison Study
- Optimizer Benchmarking Study
- Regularization Study
- Final Optimized Neural Network
- Model Performance Summary Report

---

# 🔍 Final Research Question

> What combination of feature scaling, neural network architecture, activation function, optimizer, and regularization strategy provides the most accurate wine quality predictions for structured tabular wine data?

---

# 📅 Project Status

## Completed

✅ Exploratory Data Analysis (EDA)

✅ Analysis 1: Baseline Neural Network Performance

## In Progress

⬜ Analysis 2: Feature Scaling Comparison
⬜ Analysis 3: Activation Function Comparison
⬜ Analysis 4: Architecture Comparison
⬜ Analysis 5: Learning Behavior Analysis
⬜ Analysis 6: Optimizer Benchmarking
⬜ Analysis 7: Regularization and Generalization
⬜ Analysis 8: Final Model Selection

---

# End of README
