# 05-02 – Wine Type Classification with Deep Learning

### IBM Machine Learning Professional Certificate

**Project Status:** Planning Phase  
**Last Updated:** November 2026

---

# 📌 Project Overview

This project explores the use of deep learning techniques to classify wines as either red or white using physicochemical measurements.

Wine type classification is a binary classification problem where the objective is to predict whether an observation belongs to the red wine or white wine category. Neural networks will be evaluated to determine how effectively they learn the chemical patterns that distinguish wine types.

Multiple model architectures, training strategies, and regularization methods will be compared to identify the most effective classifier.

---

# 🎯 Project Objective

Develop, optimize, and evaluate neural network models capable of classifying wine type.

The project investigates:

- Neural network fundamentals
- Classification-focused architectures
- Training behavior
- Optimization algorithms
- Regularization techniques
- Model generalization

The final goal is to identify a neural network configuration capable of accurately distinguishing red and white wines.

---

# 📊 Dataset

**Dataset:** `winequality-final.csv`

The dataset contains approximately 5,320 wine observations and includes physicochemical measurements describing both red and white wines.

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
- Quality

---

## Target Variable

### Wine Type

```text
Target = Wine Type
```

Classification Labels:

```text
0 = Red Wine
1 = White Wine
```

This is treated as a binary classification problem.

---

# 📚 Deep Learning Topics

This project applies concepts from the IBM Deep Learning curriculum, including:

- Neural Networks
- Binary Classification
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

1. Can neural networks accurately classify wine type?
2. How much improvement does an MLP provide over a Perceptron?
3. Which activation function produces the best classification results?
4. Which optimizer learns most efficiently?
5. How can overfitting be reduced?
6. What neural network configuration achieves the highest classification performance?

---

# 📈 Planned Analyses

---

## Analysis 1: Baseline Classification Performance

### Objective

Establish baseline classification performance using simple neural networks.

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

- Accuracy
- Precision
- Recall
- F1 Score

### Research Question

Does an MLP outperform a simple Perceptron for wine type classification?

---

## Analysis 2: Activation Function Comparison

### Objective

Evaluate how activation functions influence classification performance.

### Activation Functions

- ReLU
- Sigmoid
- Tanh

### Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- Training Stability

### Research Question

Which activation function is most effective for wine classification?

---

## Analysis 3: Architecture Comparison

### Objective

Investigate the effects of network depth and complexity on classification accuracy.

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
128 → 64 → 32
```

### Evaluation

- Accuracy
- F1 Score
- ROC-AUC
- Training Time

### Research Question

Does increasing network complexity improve classification performance?

---

## Analysis 4: Learning Behavior Analysis

### Objective

Examine how classification models learn over time.

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

How do classification models evolve during training?

---

## Analysis 5: Optimizer Benchmarking

### Objective

Compare optimization algorithms used for classification tasks.

### Optimizers

- Stochastic Gradient Descent (SGD)
- RMSProp
- Adam

### Evaluation

- Convergence Speed
- Classification Accuracy
- Training Stability

### Research Question

Which optimizer produces the strongest classification results?

---

## Analysis 6: Regularization and Generalization

### Objective

Reduce overfitting and improve model reliability.

### Regularization Methods

- L1 Regularization
- L2 Regularization
- Dropout
- Early Stopping

### Evaluation

- Validation Accuracy
- Generalization Gap
- Model Robustness

### Research Question

Which regularization strategy produces the most reliable classifier?

---

## Analysis 7: Classification Error Analysis

### Objective

Understand where classification errors occur.

### Analyses

#### Confusion Matrix

Evaluate:

```text
True Positives
True Negatives
False Positives
False Negatives
```

#### Classification Report

Analyze:

- Precision
- Recall
- F1 Score

### Research Question

What types of classification mistakes occur most frequently?

---

## Analysis 8: Final Model Selection

### Objective

Identify the strongest-performing wine type classifier.

### Components Evaluated

- Architecture
- Activation Function
- Optimizer
- Regularization Strategy

### Deliverable

Recommended classification model.

---

# 📊 Evaluation Metrics

## Accuracy

Measures overall classification correctness.

Higher values indicate better performance.

---

## Precision

Measures prediction accuracy for positive classifications.

Higher values indicate fewer false positives.

---

## Recall

Measures the ability to identify positive observations.

Higher values indicate fewer false negatives.

---

## F1 Score

Balances precision and recall.

Higher values indicate better overall classification quality.

---

## ROC-AUC

Measures class separability.

Higher values indicate stronger classification capability.

---

# 📋 Expected Deliverables

- Baseline classification model
- Perceptron vs MLP comparison
- Architecture comparison study
- Activation function comparison
- Learning curve analysis
- Optimizer benchmarking results
- Regularization experiments
- Confusion matrix analysis
- Final optimized classifier
- Performance summary report

---

# 🔍 Final Research Question

> **What neural network architecture and training strategy provide the most accurate classification of red and white wines using physicochemical measurements?**

---

# 📅 Project Status

## Completed

✅ Exploratory Data Analysis (EDA)

## Planned

⬜ Baseline Classification Model  
⬜ Activation Function Comparison  
⬜ Architecture Study  
⬜ Learning Curve Analysis  
⬜ Optimizer Benchmarking  
⬜ Regularization Study  
⬜ Error Analysis  
⬜ Final Model Selection  

---

# End of README
