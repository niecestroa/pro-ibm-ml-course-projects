"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 1, 2026
Last Edit:  November 1, 2026
Progress:   Complete

Analysis:   03 - Activation Function Comparison
Method:     ReLU

Description:
This script evaluates neural network performance using the ReLU
activation function.

The winning configuration from Analysis 2 is retained:
MinMaxScaler + Adam Optimizer + 16 Hidden Neurons.

Results from this analysis will be compared against Sigmoid
and Tanh activation functions.

Research Question:
Which activation function performs best for structured
wine-quality data?
"""

# ==========================================
# Analysis 3: Activation Function Comparison
# Method 1: ReLU
# ==========================================

# ------------------------------------------
# Imports
# ------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

# ------------------------------------------
# Load Data
# ------------------------------------------

file_path = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\00-data\winequality-final.csv"
)

output_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\04-Activation-Function-Comparison\tables_figures"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

dpwq = pd.read_csv(file_path)

dpwq.columns = dpwq.columns.str.strip()

print("----- DATA SHAPE (MASTER) -----")
print(f"Rows: {dpwq.shape[0]:,}")
print(f"Columns: {dpwq.shape[1]}")

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

# ------------------------------------------
# MinMax Scaling
# ------------------------------------------

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# ------------------------------------------
# ReLU Model
# ------------------------------------------

relu_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(
        16,
        activation="relu"
    ),
    Dense(1)
])

relu_model.compile(
    optimizer="adam",
    loss="mse"
)

# ------------------------------------------
# Train Model
# ------------------------------------------

history_relu = relu_model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=100,
    verbose=0
)

# ------------------------------------------
# Predictions
# ------------------------------------------

y_pred_relu = relu_model.predict(
    X_test_scaled
).flatten()

# ------------------------------------------
# Evaluation Functions
# ------------------------------------------

def adjusted_r2(r2, n, p):
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

def evaluate_model(y_true, y_pred, p):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    adj_r2 = adjusted_r2(
        r2,
        len(y_true),
        p
    )

    return mae, rmse, adj_r2

# ------------------------------------------
# Evaluation
# ------------------------------------------

mae_relu, rmse_relu, adjr2_relu = evaluate_model(
    y_test,
    y_pred_relu,
    X.shape[1]
)

# ------------------------------------------
# Results Table
# ------------------------------------------

results_activation = pd.DataFrame({
    "Activation_Function": ["ReLU"],
    "MAE": [mae_relu],
    "RMSE": [rmse_relu],
    "Adjusted_R2": [adjr2_relu],
    "Final_Training_Loss": [
        history_relu.history["loss"][-1]
    ],
    "Final_Validation_Loss": [
        history_relu.history["val_loss"][-1]
    ]
})

print("\n----- RELU RESULTS -----")
print(results_activation.round(4))

# ------------------------------------------
# Export Results
# ------------------------------------------

results_activation.to_csv(
    output_dir / "relu_results.csv",
    index=False
)

# ------------------------------------------
# Learning Curve
# ------------------------------------------

plt.figure(figsize=(10,6))

plt.plot(
    history_relu.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_relu.history["val_loss"],
    label="Validation Loss"
)

plt.title("ReLU Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Predicted Quality")

plt.legend()
plt.grid(True)

plt.savefig(
    output_dir / "relu_learning_curve.png",
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------
# Actual vs Predicted
# ------------------------------------------

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    y_pred_relu,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title("ReLU: Actual vs Predicted")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")

plt.savefig(
    output_dir / "relu_actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------
# Export Summary
# ------------------------------------------

with open(
    output_dir / "relu_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("Analysis 3: Activation Function Comparison\n")
    f.write("Method: ReLU\n\n")

    f.write(f"MAE: {mae_relu:.4f}\n")
    f.write(f"RMSE: {rmse_relu:.4f}\n")
    f.write(f"Adjusted R²: {adjr2_relu:.4f}\n")

# ------------------------------------------
# Export Training History
# ------------------------------------------

history_df = pd.DataFrame({
    "loss": history_relu.history["loss"],
    "val_loss": history_relu.history["val_loss"]
})

history_df.to_csv(
    output_dir / "relu_history.csv",
    index=False
)

# ------------------------------------------
# Final Output
# ------------------------------------------

print("\nAnalysis 3 - ReLU Complete")

print(f"MAE: {mae_relu:.4f}")
print(f"RMSE: {rmse_relu:.4f}")
print(f"Adjusted R²: {adjr2_relu:.4f}")

"""
## Interpretation

The ReLU activation function produced strong predictive performance and
served as the baseline activation function for this experiment.

The model achieved:

- MAE: 0.5520
- RMSE: 0.7120
- Adjusted R²: 0.3388

These results indicate that ReLU was effective at learning the
relationship between physicochemical wine measurements and wine quality
scores.

The learning curve showed rapid convergence during the first several
epochs before stabilizing. Training and validation losses remained
closely aligned throughout training, suggesting stable learning and
minimal evidence of overfitting.

The Actual vs Predicted plot demonstrated reasonable agreement between
predicted and observed quality scores. While some prediction compression
remained present, the model generally followed the overall trend of the
data and produced tighter clustering around the ideal prediction line
than previous baseline models.

Similar to earlier analyses, the model was most accurate for wines with
quality scores between 5 and 7 and struggled more with rare extreme
ratings. This behavior reflects the imbalanced distribution of quality
scores within the dataset.

Overall, ReLU demonstrated strong learning behavior, stable
optimization, and competitive predictive performance, establishing a
high benchmark for the remaining activation functions.
"""

"""
## Summary

The ReLU activation function produced strong predictive performance
when combined with MinMaxScaler and the selected MLP architecture.

The model achieved an MAE of 0.5520, RMSE of 0.7120, and Adjusted R²
of 0.3388.

Training and validation losses converged rapidly and remained closely
aligned throughout training, indicating effective learning and minimal
overfitting.

Predictions were most accurate for quality scores between 5 and 7,
while extreme quality ratings remained more difficult to predict due
to dataset imbalance.

The ReLU activation function establishes the baseline for the
Activation Function Comparison study.
"""