# Wine Quality Regression with Deep Learning

### IBM Machine Learning Professional Certificate

**Project Status: Complete**  
**Last Updated:** November 2, 2026

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
slides/
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

✅ No missing values <br>
✅ No duplicate records <br>
✅ Dataset immediately suitable for machine learning <br>
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

✅ Alcohol increases <br>
✅ Density decreases <br>
✅ Volatile acidity decreases <br>
✅ Chlorides decrease <br>
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

✅ Clean dataset <br>
✅ No missing values <br>
✅ Strong sample size <br>
✅ Strong predictor-target relationships <br>
✅ Multiple informative chemistry measurements

### Challenges

⚠ Quality imbalance <br>
⚠ Predictor outliers <br>
⚠ Skewed distributions <br>
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
| StandardScaler | 0.5546 | 0.7170 | 0.3294 |
| MinMaxScaler | 0.5519 | 0.7094 | 0.3435 |

### Findings

- Feature scaling improved overall model performance.
- MinMaxScaler produced the strongest predictive results.
- StandardScaler outperformed the unscaled model.
- Training remained stable across all scaling approaches.

### Conclusion

MinMaxScaler was selected as the preferred scaling strategy for all subsequent analyses.

### Final Ranking

1. MinMaxScaler
2. StandardScaler
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
| Tanh | 0.5627 | 0.7297 | 0.3054 |
| Sigmoid | 0.5645 | 0.7334 | 0.2984 |

### Findings

- ReLU produced the lowest prediction error.
- Tanh outperformed Sigmoid.
- ReLU achieved the highest Adjusted R².
- All activation functions demonstrated stable training behavior.

### Conclusion

ReLU was selected as the preferred activation function.

### Final Ranking

1. ReLU
2. Tanh
3. Sigmoid

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
| Model A | 16 | 0.5594 | 0.7226 | 0.3190 |
| Model B | 32 → 16 | 0.5502 | 0.7075 | 0.3471 |
| Model C | 64 → 32 → 16 | 0.5462 | 0.7023 | 0.3567 |
| Model D | 128 → 64 → 32 → 16 | 0.5424 | 0.7043 | 0.3530 |

### Findings

- Increasing architecture depth improved performance from Model A through Model C.
- Model C achieved the lowest RMSE and highest Adjusted R².
- Model D achieved the lowest MAE but did not outperform Model C overall.
- Evidence of diminishing returns emerged beyond Model C.
- All architectures exhibited stable learning behavior.

### Conclusion

Model C provided the best balance between complexity and predictive performance and was selected for future analyses.

### Final Ranking

1. Model C (64 → 32 → 16)
2. Model D (128 → 64 → 32 → 16)
3. Model B (32 → 16)
4. Model A (16)

---

## ✅ Analysis 5: Epoch Optimization

### Objective

Determine the optimal number of training epochs.

### Epochs Evaluated

25, 50, 75, 100, 125, 150, 175, and 200

### Winner

75 Epochs

### Conclusion

75 epochs produced the strongest overall balance between predictive performance and model generalization.

---

## ✅ Analysis 6: Optimizer Benchmarking

### Optimizers Evaluated

- SGD
- RMSProp
- Adam

### Winner

RMSProp

### Conclusion

RMSProp achieved the strongest overall predictive performance and convergence behavior.

---

## ✅ Analysis 7: Regularization and Generalization

### Methods Evaluated

- Baseline
- L1
- L2
- Dropout
- Early Stopping

### Results

| Method | MAE | RMSE | Adjusted R² |
|----------|------:|------:|------:|
| L2 | 0.5453 | 0.7035 | 0.3546 |
| Baseline | 0.5486 | 0.7047 | 0.3522 |
| EarlyStopping | 0.5535 | 0.7113 | 0.3400 |
| Dropout | 0.5492 | 0.7165 | 0.3304 |
| L1 | 0.5631 | 0.7308 | 0.3034 |

### Winner

L2 (0.001)

### Conclusion

L2 regularization produced the strongest combination of predictive performance and generalization capability.

---

## ✅ Analysis 8: Final Model Selection

### Objective

Identify the strongest-performing neural network configuration discovered throughout the project.

### Final Recommended Configuration

```text
MinMaxScaler
64 → 32 → 16 Architecture
ReLU Activation
RMSProp Optimizer
75 Epochs
L2 (0.001) Regularization
```

### Final Performance

| Metric | Value |
|----------|------:|
| MAE | 0.5453 |
| RMSE | 0.7035 |
| Adjusted R² | 0.3546 |
| Training Time (Seconds) | 22.6407 |
| Generalization Gap | -0.0203 |

### Conclusion

The strongest-performing neural network configuration combined MinMaxScaler preprocessing, a 64 → 32 → 16 architecture, ReLU activation functions, RMSProp optimization, 75 training epochs, and L2 regularization.  

This configuration achieved the strongest overall balance of predictive accuracy, training stability, and model generalization.

---

# 📈 Performance Improvement

| Metric | Baseline MLP | Final Model | Improvement |
|----------|----------:|----------:|----------:|
| MAE | 0.5660 | 0.5453 | 3.66% |
| RMSE | 0.7365 | 0.7035 | 4.48% |
| Adjusted R² | 0.2924 | 0.3546 | 21.27% |

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

> What combination of feature scaling, neural network architecture, activation function, optimizer, and regularization strategy provides the most accurate wine quality predictions for structured tabular wine data?

---

## Completed

✅ Exploratory Data Analysis (EDA) <br>
✅ Analysis 1: Baseline Neural Network Performance <br>
✅ Analysis 2: Feature Scaling Comparison <br>
✅ Analysis 3: Activation Function Comparison <br>
✅ Analysis 4: Architecture Comparison <br>
✅ Analysis 5: Epoch Optimization <br>
✅ Analysis 6: Optimizer Benchmarking <br>
✅ Analysis 7: Regularization and Generalization <br>
✅ Analysis 8: Final Model Selection

# 📌 Final Project Outcome

After completing eight experimental analyses, the recommended deep learning model for wine quality prediction is:

```text
MinMaxScaler
→ 64 → 32 → 16 Neural Network
→ ReLU Activation
→ RMSProp Optimizer
→ 75 Epochs
→ L2 (0.001) Regularization
```

## Final Performance

| Metric | Value |
|----------|------:|
| MAE | 0.5453 |
| RMSE | 0.7035 |
| Adjusted R² | 0.3546 |
| Generalization Gap | -0.0203 |

This configuration achieved the strongest overall balance of predictive accuracy, convergence stability, and generalization performance across all deep learning experiments conducted within the DLWQ project.

---

# 🚀 Next Steps

- Final written project report
- Executive presentation slides
- GitHub repository publication
- Portfolio project showcase
  

# End of README
