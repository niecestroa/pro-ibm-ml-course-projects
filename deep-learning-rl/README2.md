# Wine Quality Regression with Deep Learning

### IBM Machine Learning Professional Certificate

**Project Status:** Planning Phase  
**Last Updated:** November 2026

---

# 📌 Project Overview

This project explores the use of deep learning techniques to predict wine quality using physicochemical wine measurements.

Wine quality prediction is a regression problem where the target variable is a numerical quality score assigned to each wine observation. The objective is to determine how effectively neural networks can model the complex relationships between wine chemistry and perceived quality.

Throughout this project, multiple neural network architectures and training strategies will be evaluated to identify the most effective approach for structured tabular data.

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

The dataset contains approximately 5,320 wine observations consisting of physicochemical measurements and wine quality ratings.

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

This is treated as a continuous numerical variable and modeled using regression techniques.

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

This project seeks to answer the following questions:

1. Can neural networks accurately predict wine quality?
2. How much improvement does a Multi-Layer Perceptron provide over a simple Perceptron?
3. Which feature scaling method performs best?
4. Which activation function performs best?
5. Does increasing network complexity improve performance?
6. Which optimizer converges most efficiently?
7. How can overfitting be minimized?
8. What neural network configuration achieves the best predictive performance?

---

# 📈 Planned Analyses

---

## Analysis 1: Baseline Neural Network Performance

### Objective

Establish a baseline performance benchmark using a simple neural network.

### Models

#### Perceptron

```text
Input → Output
```

#### Basic MLP

```text
Input
  ↓
Dense Layer
  ↓
Output
```

### Evaluation

- MAE
- RMSE
- Adjusted R²

### Research Question

Does an MLP outperform a simple Perceptron for wine quality prediction?

---

## Analysis 2: Feature Scaling Comparison

### Objective

Evaluate how feature scaling influences neural network training and predictive performance.

### Scaling Methods

#### No Scaling

```text
Raw Features
```

#### StandardScaler

```text
Mean = 0
Standard Deviation = 1
```

#### MinMaxScaler

```text
Range: 0 → 1
```

### Evaluation

- MAE
- RMSE
- Adjusted R²
- Training Stability
- Convergence Speed

### Research Question

Which feature scaling method produces the best neural network performance for wine quality prediction?

---

## Analysis 3: Activation Function Comparison

### Objective

Evaluate how activation functions influence learning and predictive performance.

### Activation Functions

- ReLU
- Sigmoid
- Tanh

### Evaluation

- RMSE
- MAE
- Adjusted R²
- Training Behavior

### Research Question

Which activation function performs best for structured wine-quality data?

---

## Analysis 4: Architecture Comparison

### Objective

Investigate the impact of network depth and complexity.

### Architectures

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

### Evaluation

- RMSE
- MAE
- Adjusted R²
- Training Time

### Research Question

Does increasing network complexity improve predictive performance?

---

## Analysis 5: Learning Behavior Analysis

### Objective

Investigate how neural networks learn over time.

### Analyses

- Training Loss
- Validation Loss
- Learning Curves
- Epoch Studies

### Evaluation

- Convergence Patterns
- Underfitting Detection
- Overfitting Detection

### Research Question

How does model learning evolve during training?

---

## Analysis 6: Optimizer Benchmarking

### Objective

Compare common deep learning optimization algorithms.

### Optimizers

- Stochastic Gradient Descent (SGD)
- RMSProp
- Adam

### Evaluation

- Convergence Speed
- Final Performance
- Training Stability

### Research Question

Which optimizer is most effective for wine quality prediction?

---

## Analysis 7: Regularization and Generalization

### Objective

Reduce overfitting and improve generalization performance.

### Regularization Methods

- L1 Regularization
- L2 Regularization
- Dropout
- Early Stopping

### Evaluation

- Validation Performance
- Generalization Gap
- Model Robustness

### Research Question

Which regularization strategy produces the most reliable model?

---

## Analysis 8: Final Model Selection

### Objective

Identify the strongest-performing neural network configuration.

### Components Evaluated

- Scaling Method
- Architecture
- Activation Function
- Optimizer
- Regularization Strategy

### Deliverable

Recommended regression model for wine quality prediction.

---

# 📊 Evaluation Metrics

## Root Mean Squared Error (RMSE)

Measures prediction error while penalizing larger mistakes more heavily than MAE.

Lower values indicate better performance.

**Primary Performance Metric**

---

## Mean Absolute Error (MAE)

Measures the average magnitude of prediction errors.

Lower values indicate better performance.

---

## Adjusted R²

Measures the percentage of variance explained by the model while accounting for model complexity.

Unlike R², Adjusted R² penalizes unnecessary model parameters and provides a more reliable comparison between neural network configurations.

Higher values indicate better performance.

# 📋 Expected Deliverables

- Baseline neural network model
- Feature scaling comparison
- Activation function comparison
- Architecture comparison study
- Learning curve analysis
- Optimizer benchmarking results
- Regularization experiments
- Final optimized model
- Performance summary report

---

# 🔍 Final Research Question

> **What combination of feature scaling, neural network architecture, activation function, optimizer, and regularization strategy provides the most accurate wine quality predictions for structured tabular wine data?**

---

# 📅 Project Status

## Completed

✅ Exploratory Data Analysis (EDA)

## Planned

⬜ Baseline Neural Network  
⬜ Feature Scaling Comparison  
⬜ Activation Function Comparison  
⬜ Architecture Study  
⬜ Learning Curve Analysis  
⬜ Optimizer Benchmarking  
⬜ Regularization Study  
⬜ Final Model Selection

---

# End of README
