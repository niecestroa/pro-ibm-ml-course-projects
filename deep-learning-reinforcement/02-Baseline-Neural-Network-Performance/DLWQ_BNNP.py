"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 1 ,2026
Last Edit:  November 2, 2026
Progress:   Completed

Analysis 1: Baseline Neural Network Performance

Description:
This analysis establishes the initial deep learning benchmark for predicting
wine quality using physicochemical measurements and wine type information.
Two simple neural network architectures are evaluated: a single-layer
Perceptron and a basic Multi-Layer Perceptron (MLP) containing one hidden
layer.

Purpose:
The objective is to determine whether introducing nonlinear learning through
a hidden layer improves predictive performance compared with a purely linear
neural network architecture.

Methodology:
- Load and prepare the Wine Quality dataset.
- Encode the categorical wine type feature.
- Split the data into stratified training and testing sets.
- Train a baseline Perceptron model.
- Train a baseline Multi-Layer Perceptron (MLP).
- Evaluate models using MAE, RMSE, and Adjusted R².
- Compare convergence behavior using learning curves.
- Examine predictive performance using actual-versus-predicted plots.

Research Question:
Does a simple Multi-Layer Perceptron outperform a Perceptron when predicting
wine quality?

Expected Outcome:
Because wine quality is influenced by complex interactions among chemical
measurements, the MLP is expected to outperform the Perceptron by learning
nonlinear relationships that cannot be captured by a single-layer model.

Deliverables:
- Model comparison table
- Learning curve visualizations
- Actual versus predicted plots
- Training and testing distributions
- Baseline performance benchmark for future experiments

Success Criteria:
The model with the lowest RMSE and MAE and the highest Adjusted R² will be
selected as the baseline architecture for subsequent deep learning analyses.
"""

# ==========================================
# Analysis 1: Baseline Neural Network Performance
# Wine Quality Regression with Deep Learning
# ==========================================

# ------------------------------------------
# Imports
# ------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ------------------------------------------
# Reproducibility
# ------------------------------------------

import os
import random
import tensorflow as tf

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)

# ------------------------------------------
# Load Data
# ------------------------------------------

file_path = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\00-data\winequality-final.csv"
)

output_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\02-Baseline-Neural-Network-Performance\tables_figures"
)

output_dir.mkdir(parents=True, exist_ok=True)

dpwq = pd.read_csv(file_path)

dpwq.columns = dpwq.columns.str.strip()

print("----- DATA SHAPE (MASTER) -----")
print(f"Rows: {dpwq.shape[0]:,}")
print(f"Columns: {dpwq.shape[1]}")

print("\n----- FIRST FIVE ROWS -----")
print(dpwq.head())

# ------------------------------------------
# Features and Target
# ------------------------------------------

X = dpwq.drop("quality", axis=1)
y = dpwq["quality"]

# ------------------------------------------
# Encode Wine Type
# ------------------------------------------

X = pd.get_dummies(
    X,
    columns=["type"],
    drop_first=True
)

X["type_white"] = X["type_white"].astype(int)

print("\n----- FEATURE MATRIX AFTER ENCODING -----")
print(X.shape)

print("\n----- DATA TYPES AFTER ENCODING -----")
print(X.dtypes)

# ------------------------------------------
# Train/Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\n----- TRAIN/TEST SPLIT -----")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\n----- TRAIN DISTRIBUTION -----")
print(y_train.value_counts().sort_index())

print("\n----- TEST DISTRIBUTION -----")
print(y_test.value_counts().sort_index())

# ------------------------------------------
# Baseline Model 1: Perceptron
# ------------------------------------------

# verbose 1 is to see loss function work => very messy though

perceptron = Sequential([
    Dense(
        1,
        input_shape=(X_train.shape[1],)
    )
])

perceptron.compile(
    optimizer="adam",
    loss="mse"
)

history_perceptron = perceptron.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=100,
    verbose=0
)

#versbose 0 is for clean output, no show output as comes
# verbose 1 is to see loss function work => very messy though

# ------------------------------------------
# Baseline Model 2: Basic MLP
# ------------------------------------------

from tensorflow.keras import Input

mlp = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(1)
])

mlp.compile(
    optimizer="adam",
    loss="mse"
)

history_mlp = mlp.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=100,
    verbose=0
)

# ------------------------------------------
# Predictions
# ------------------------------------------

y_pred_perceptron = perceptron.predict(X_test).flatten()
y_pred_mlp = mlp.predict(X_test).flatten()

# ------------------------------------------
# Evaluation Functions
# ------------------------------------------

def adjusted_r2(r2, n, p):
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

def evaluate_model(y_true, y_pred, p):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    adj_r2 = adjusted_r2(
        r2,
        len(y_true),
        p
    )

    return mae, rmse, adj_r2

# ------------------------------------------
# Evaluate Models
# ------------------------------------------

p = X.shape[1]

mae_perc, rmse_perc, adjr2_perc = evaluate_model(
    y_test,
    y_pred_perceptron,
    p
)

mae_mlp, rmse_mlp, adjr2_mlp = evaluate_model(
    y_test,
    y_pred_mlp,
    p
)

# ------------------------------------------
# Results Table
# ------------------------------------------

results = pd.DataFrame({
    "Model": ["Perceptron", "MLP"],
    "MAE": [mae_perc, mae_mlp],
    "RMSE": [rmse_perc, rmse_mlp],
    "Adjusted_R2": [adjr2_perc, adjr2_mlp]
})

print("\n----- MODEL COMPARISON -----")
print(results.round(4))

# ------------------------------------------
# Best Model
# ------------------------------------------

best_model = results.loc[
    results["RMSE"].idxmin(),
    "Model"
]

print("\n----- BEST BASELINE MODEL -----")
print(best_model)

# ------------------------------------------
# Save Results
# ------------------------------------------

results.to_csv(
    output_dir / "baseline_model_comparison.csv",
    index=False
)

# ==========================================
# Learning Curve Plots
# ==========================================

# ------------------------------------------
# Perceptron Learning Curve
# ------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    history_perceptron.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_perceptron.history["val_loss"],
    label="Validation Loss"
)

plt.title("Perceptron Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.savefig(
    output_dir / "perceptron_learning_curve.png",
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------
# MLP Learning Curve
# ------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    history_mlp.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_mlp.history["val_loss"],
    label="Validation Loss"
)

plt.title("MLP Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.savefig(
    output_dir / "mlp_learning_curve.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Learning Curve Insights
# ==========================================

"""
## Learning Curve Analysis

Both the Perceptron and Multi-Layer Perceptron (MLP) exhibited rapid learning 
during the first few epochs, with training and validation losses decreasing 
substantially before stabilizing.

The Perceptron experienced an extremely large initial loss before converging 
quickly. Although training and validation loss decreased rapidly, performance 
plateaued early, suggesting that the linear architecture had limited capacity 
to model the underlying relationships within the wine quality data.

The MLP also demonstrated rapid convergence but achieved a substantially lower 
and more stable loss. Training and validation curves remained closely aligned 
throughout training, indicating good generalization and minimal evidence of 
overfitting.

For both models, the majority of performance improvements occurred during the 
first 10–15 epochs. After this point, additional training produced only 
marginal reductions in loss, suggesting that model convergence was largely 
achieved early in the training process.

The close agreement between training and validation losses across both models 
indicates that neither network suffered from significant overfitting during 
the baseline experiment. However, the MLP consistently maintained lower loss 
values than the Perceptron, supporting the quantitative results obtained from 
MAE, RMSE, and Adjusted R² metrics.

Overall, the learning curves suggest that the addition of a hidden layer 
enabled the MLP to learn more complex nonlinear relationships between 
physicochemical measurements and wine quality, resulting in improved predictive 
performance compared with the Perceptron.

Perceptron
- Extremely high starting loss (~2300).
- Very rapid reduction in the first few epochs.
- Plateau reached quickly.
- Limited improvement after convergence.
- Suggests the model is too simple.

MLP
- Much lower starting loss (~10).
- Fast convergence.
-Training and validation curves almost overlap.
= Very stable after convergence.
- No obvious signs of overfitting.

## Important Finding

The most interesting observation is actually:

Training Loss ≈ Validation Loss

for both models.

That means Analysis 1 shows:

✅ Convergence
✅ Stable learning
✅ No meaningful overfitting
✅ MLP outperforming Perceptron

which is exactly the story to tell in a baseline deep-learning experiment.

Conclusion: The MLP had a lower RMSE.
"""

# ==========================================
# Actual vs Predicted Plots
# ==========================================

# ------------------------------------------
# Actual vs Predicted - Perceptron
# ------------------------------------------

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    y_pred_perceptron,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title("Perceptron: Actual vs Predicted")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")

plt.savefig(
    output_dir / "perceptron_actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------
# Actual vs Predicted - MLP
# ------------------------------------------

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    y_pred_mlp,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title("MLP: Actual vs Predicted")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")

plt.savefig(
    output_dir / "mlp_actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Actual vs Predicted Insights
# ==========================================

"""
## Insights 

Actual versus predicted plots were generated for both the Perceptron and 
Multi-Layer Perceptron (MLP) models to visually evaluate predictive performance.

In both models, predictions generally followed the overall trend of increasing 
wine quality. However, the Perceptron exhibited greater dispersion around the 
ideal prediction line and struggled to accurately separate quality levels 
across the dataset.

The MLP demonstrated tighter clustering around the diagonal reference line, 
indicating improved agreement between actual and predicted wine quality scores. 
This observation is consistent with the lower MAE and RMSE values obtained 
by the MLP.

Both models exhibited a tendency to predict values near the center of the 
quality range. Low-quality wines were often overestimated, while high-quality 
wines were frequently underestimated. This behavior is expected because 
extreme quality scores were relatively rare within the dataset, causing the 
models to favor predictions near the more common quality levels of 5, 6, and 7.

The limited number of observations with quality ratings of 3, 8, and 9 likely 
reduced the models' ability to learn patterns associated with these extreme values. 
Consequently, prediction accuracy was strongest for the most frequently occurring 
quality classes.

Overall, the MLP produced predictions that more closely aligned with actual 
wine quality scores, providing further evidence that introducing a hidden layer 
improves predictive performance compared with a simple Perceptron.

## Looking at the plots:

Prediction Compression

Neither model predicts many values below about 4.5.
Neither model predicts many values above about 7.0.
Actual values range from 3 to 9.

This means both models are:

Overestimating low-quality wines
Underestimating high-quality wines

which is very common in regression.

Class Imbalance Effect

Remember your distribution:

Quality 3 = 30 wines
Quality 4 = 206 wines
Quality 5 = 1752 wines
Quality 6 = 2323 wines
Quality 7 = 856 wines
Quality 8 = 148 wines
Quality 9 = 5 wines

The model learned mostly from:

5
6
7

because those make up almost the entire dataset.

A nice transition sentence to Analysis 2 is:

Although the MLP outperformed the Perceptron, prediction dispersion remained 
substantial. The next phase of the project investigates whether feature scaling 
can improve neural network training and further reduce prediction error.
"""

# ==========================================
# Exports
# ==========================================

output_dir.mkdir(parents=True, exist_ok=True)

# Results Table
results.to_csv(
    output_dir / "baseline_model_comparison.csv",
    index=False
)

# Train Distribution
y_train.value_counts().sort_index().to_csv(
    output_dir / "train_distribution.csv"
)

# Test Distribution
y_test.value_counts().sort_index().to_csv(
    output_dir / "test_distribution.csv"
)

# Summary
with open(
    output_dir / "results_summary.txt",
    "w"
) as f:

    f.write("Analysis 1: Baseline Neural Network Performance\n\n")

    f.write("Model Comparison\n")
    f.write(results.round(4).to_string(index=False))

    f.write("\n\nBest Baseline Model:\n")
    f.write(best_model)

print("\nAnalysis 1: Baseline Neural Network Performance is Complete.")

print("""
# Analysis 1 Conclusion:
<br>
The objective of this analysis was to establish a baseline performance
benchmark using simple neural network architectures for wine quality prediction.
<br>
The Multi-Layer Perceptron (MLP) outperformed the Perceptron across all evaluation
metrics, achieving lower MAE and RMSE values while producing a higher Adjusted
R² score.
<br>
The MLP achieved an MAE of 0.5726, an RMSE of 0.7465, and an Adjusted R² of
0.2732, compared with the Perceptron's MAE of 0.6223, RMSE of 0.7975, and
Adjusted R² of 0.1705.
<br>
This represents approximately an 8.0% reduction in MAE, a 6.4% reduction in
RMSE, and a 60.2% improvement in Adjusted R².
<br>
Learning curves indicated stable convergence for both models with no significant
signs of overfitting. Actual-versus-predicted visualizations further demonstrated
that the MLP produced predictions more closely aligned with observed wine quality
scores.
<br>
Based on these findings, the MLP was selected as the baseline neural network
architecture for all subsequent analyses.
""")

# ==========================================
# Overall Insights
# ==========================================

print("""
Results
<br>

| Model | MAE | RMSE | Adjusted R² |
|---------|---------:|---------:|---------:|
| Perceptron | 0.6223 | 0.7975 | 0.1705 |
| MLP | 0.5726 | 0.7465 | 0.2732 |
| Improvement | -7.99% | -6.39% | +60.23% |

<br>

Research Question
<br>

Does an MLP outperform a simple Perceptron for wine quality prediction?
<br>

Answer: Yes.
<br>

The MLP outperformed the Perceptron across all evaluation metrics:
<br>

- Lower MAE
- Lower RMSE
- Higher Adjusted R²

<br>

The MLP reduced RMSE from 0.7975 to 0.7465, indicating more accurate
predictions overall and demonstrating an improved ability to model the
underlying relationships within the dataset.

<br>

## Results
<br>

The Multi-Layer Perceptron (MLP) outperformed the Perceptron across all
evaluation metrics.

<br>

| Model | MAE | RMSE | Adjusted R² |
|---------|---------:|---------:|---------:|
| Perceptron | 0.6223 | 0.7975 | 0.1705 |
| MLP | 0.5726 | 0.7465 | 0.2732 |

<br>

The MLP achieved lower prediction error and explained a substantially greater
proportion of variance in wine quality scores.

<br>

These results suggest that wine quality prediction is not purely a linear
problem. The hidden layer within the MLP enables the model to learn nonlinear
relationships between physicochemical measurements and wine quality, resulting
in improved predictive performance.

<br>

Based on these findings, the MLP was selected as the baseline neural network
architecture for subsequent experiments.

<br>

My Assessment
<br>

For a first deep-learning baseline:
<br>

RMSE = 0.7465
Adjusted R² = 0.2732

<br>

This represents a reasonable starting point for the wine quality dataset using
a relatively simple neural network architecture and unoptimized baseline
settings. Further improvements may be achieved through feature scaling,
activation-function tuning, architecture optimization, and hyperparameter
adjustments explored in subsequent analyses.
""")