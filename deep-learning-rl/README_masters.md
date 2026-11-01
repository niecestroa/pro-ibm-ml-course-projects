# 05 – Deep Learning with Wine Data

### IBM Machine Learning Professional Certificate

**Status:** Planning Phase  
**Last Updated:** November 2026

---

# 📌 Project Overview

This repository contains a series of deep learning projects developed using the Wine Quality dataset as part of the IBM Machine Learning Professional Certificate.

The objective is to explore how neural networks perform on structured tabular data while investigating core deep learning concepts including neural network design, optimization, regularization, representation learning, and generative modeling.

Rather than focusing on a single prediction task, this repository examines multiple deep learning applications using the same dataset:

1. Wine Quality Regression
2. Wine Type Classification
3. Representation Learning and Generative Modeling

Together, these projects provide a comprehensive case study of deep learning techniques on structured numerical data.

---

# 📊 Dataset

**Dataset:** `winequality-final.csv`

The dataset contains approximately 5,320 wine observations and includes physicochemical measurements describing both red and white wines.

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

## Available Targets

### Continuous Target
- Wine Quality Score

### Binary Target
- Wine Type (Red vs White)

### Unsupervised Learning
- Feature relationships and latent structure

---

# 🎯 Repository Objectives

The purpose of this repository is to investigate:

- How well neural networks perform on structured tabular data
- The impact of neural network architecture choices
- The influence of activation functions and optimizers
- Methods for reducing overfitting
- The effectiveness of learned latent representations
- The ability of generative models to create realistic synthetic observations

---

# 📚 Repository Projects

---

## 🍷 Project 1: Wine Quality Regression

### Objective

Predict wine quality scores using deep learning models.

### Topics Covered

- Neural Networks
- Perceptrons
- Multi-Layer Perceptrons (MLPs)
- Activation Functions
- Forward Propagation
- Backpropagation
- Optimizers
- Regularization

### Key Analyses

- Perceptron vs MLP comparison
- Architecture comparison
- Activation function comparison
- Learning curve analysis
- Optimizer benchmarking
- Regularization studies
- Final model selection

### Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Research Question

**What neural network configuration provides the most accurate wine quality predictions?**

---

## 🍷 Project 2: Wine Type Classification

### Objective

Classify wines as red or white using deep learning techniques.

### Topics Covered

- Neural Networks
- Binary Classification
- Activation Functions
- Optimizers
- Regularization
- Classification Evaluation

### Key Analyses

- Perceptron vs MLP comparison
- Architecture comparison
- Activation function comparison
- Learning curve analysis
- Optimizer benchmarking
- Regularization studies
- Error analysis

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Research Question

**How accurately can neural networks distinguish red and white wines using physicochemical measurements?**

---

## 🍷 Project 3: Representation Learning and Generative Modeling

### Objective

Explore how deep learning can learn compressed feature representations and generate synthetic wine observations.

### Topics Covered

#### Autoencoders
- Encoder-Decoder Architectures
- Latent Space Learning
- Feature Compression

#### Variational Autoencoders (VAEs)
- Generative Modeling
- Latent Distributions
- Synthetic Data Generation

### Key Analyses

#### Representation Learning

- Autoencoder construction
- Reconstruction error analysis
- Latent-space visualization
- Autoencoder vs PCA comparison

#### Autoencoder-Assisted Prediction

- Latent feature extraction
- Regression performance comparison
- Classification performance comparison

#### Generative Modeling

- Synthetic wine generation
- Distribution comparison
- Correlation preservation analysis
- Latent-space exploration

### Evaluation Metrics

- Reconstruction Loss
- Compression Ratio
- Distribution Similarity Metrics
- Correlation Preservation Measures

### Research Questions

**Can autoencoders learn meaningful representations of wine data?**

**Can variational autoencoders generate realistic synthetic wine observations?**

---

# 📈 Expected Outcomes

Through these projects, the repository will investigate:

## Neural Network Behavior

- Impact of network depth
- Impact of activation functions
- Learning dynamics
- Model complexity trade-offs

## Optimization

- Convergence speed
- Training stability
- Optimizer performance differences

## Generalization

- Overfitting behavior
- Regularization effectiveness
- Robustness of learned models

## Representation Learning

- Nonlinear dimensionality reduction
- Latent feature discovery
- Information compression

## Generative Deep Learning

- Synthetic data generation
- Distribution preservation
- Latent space exploration

---

# 🔍 Key Research Theme

The overarching goal of this repository is to answer the following question:

> **How effective are deep learning methods for structured tabular wine data, and what unique capabilities do they provide beyond traditional machine learning approaches?**

---

# 📦 Repository Deliverables

- Regression neural network models
- Classification neural network models
- Architecture comparison studies
- Optimizer benchmarking analyses
- Regularization experiments
- Autoencoder implementations
- Latent-space visualizations
- Variational autoencoder experiments
- Synthetic data generation analyses
- Final project summaries and conclusions

---

# 📅 Project Status

## Completed

✅ Exploratory Data Analysis (EDA)

## In Progress

⬜ Project 1 – Wine Quality Regression  
⬜ Project 2 – Wine Type Classification  
⬜ Project 3 – Representation Learning & Generative Modeling

---

# Repository Structure

```text
05-Deep-Learning-Wine-Projects
│
├── 05-01-Wine-Quality-Regression
│
├── 05-02-Wine-Type-Classification
│
├── 05-03-Representation-Learning-Generative-Modeling
│
└── README.md
```

---

# End of README
