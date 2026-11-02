"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 1, 2026
Last Edit:  November 1, 2026
Progress:   Complete

Analysis:   04 - Architecture Comparison
Method:     Model A (16)

Description:
This script evaluates the baseline neural network architecture
containing a single hidden layer with 16 neurons.

The winning configuration from previous analyses is retained:

- MinMaxScaler
- ReLU Activation
- Adam Optimizer

Results will be compared against increasingly complex network
architectures.

Research Question:
Does increasing network complexity improve predictive performance?
"""

# ==========================================
# Imports
# ==========================================

import time
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

# ==========================================
# Load Data
# ==========================================

file_path = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\00-data\winequality-final.csv"
)

output_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\05-Architecture Comparison\tables_figures"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

dpwq = pd.read_csv(file_path)

dpwq.columns = dpwq.columns.str.strip()

print("----- DATA SHAPE (MASTER) -----")

print(f"Rows: {len(dpwq):,}")
print(f"Columns: {dpwq.shape[1]}")

# ==========================================
# Features and Target
# ==========================================

X = dpwq.drop(
    "quality",
    axis=1
)

y = dpwq["quality"]

# ==========================================
# Encode Wine Type
# ==========================================

X = pd.get_dummies(
    X,
    columns=["type"],
    drop_first=True
)

X["type_white"] = X["type_white"].astype(int)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ==========================================
# MinMax Scaling
# ==========================================

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# ==========================================
# Model A Architecture
# (16)
# ==========================================

model_a = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),

    Dense(
        16,
        activation="relu"
    ),

    Dense(1)
])

model_a.compile(
    optimizer="adam",
    loss="mse"
)

# ==========================================
# Train Model
# ==========================================

start_time = time.time()

history_a = model_a.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=100,
    verbose=0
)

training_time = (
    time.time() - start_time
)

# ==========================================
# Export Training History
# ==========================================

history_df = pd.DataFrame({
    "loss": history_a.history["loss"],
    "val_loss": history_a.history["val_loss"]
})

history_df.to_csv(
    output_dir / "model_a_history.csv",
    index=False
)

# ==========================================
# Predictions
# ==========================================

y_pred_a = model_a.predict(
    X_test_scaled
).flatten()

# ==========================================
# Evaluation Functions
# ==========================================

def adjusted_r2(r2, n, p):

    return (
        1 -
        ((1 - r2) * (n - 1))
        /
        (n - p - 1)
    )

def evaluate_model(
    y_true,
    y_pred,
    p
):

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

# ==========================================
# Evaluation
# ==========================================

mae_a, rmse_a, adjr2_a = evaluate_model(
    y_test,
    y_pred_a,
    X.shape[1]
)

# ==========================================
# Results Table
# ==========================================

results_architecture = pd.DataFrame({

    "Architecture": ["Model A"],

    "Structure": ["16"],

    "MAE": [mae_a],

    "RMSE": [rmse_a],

    "Adjusted_R2": [adjr2_a],

    "Training_Time_Seconds": [
        training_time
    ],

    "Final_Training_Loss": [
        history_a.history["loss"][-1]
    ],

    "Final_Validation_Loss": [
        history_a.history["val_loss"][-1]
    ]
})

print("\n----- MODEL A RESULTS -----")
print(results_architecture.round(4))

# ==========================================
# Export Results
# ==========================================

results_architecture.to_csv(
    output_dir / "model_a_results.csv",
    index=False
)

# ==========================================
# Learning Curve
# ==========================================

plt.figure(figsize=(10,6))

plt.plot(
    history_a.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_a.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Model A Learning Curve"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    output_dir /
    "model_a_learning_curve.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Actual vs Predicted
# ==========================================

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    y_pred_a,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title(
    "Model A: Actual vs Predicted"
)

plt.xlabel(
    "Actual Quality"
)

plt.ylabel(
    "Predicted Quality"
)

plt.savefig(
    output_dir /
    "model_a_actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Export Summary
# ==========================================

with open(
    output_dir /
    "model_a_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Analysis 4: Architecture Comparison\n"
    )

    f.write(
        "Method: Model A (16)\n\n"
    )

    f.write(
        f"MAE: {mae_a:.4f}\n"
    )

    f.write(
        f"RMSE: {rmse_a:.4f}\n"
    )

    f.write(
        f"Adjusted R²: {adjr2_a:.4f}\n"
    )

    f.write(
        f"Training Time: {training_time:.2f} seconds\n"
    )

# ==========================================
# Final Output
# ==========================================

print("\nAnalysis 4 - Model A Complete")

print(f"MAE: {mae_a:.4f}")
print(f"RMSE: {rmse_a:.4f}")
print(f"Adjusted R²: {adjr2_a:.4f}")
print(f"Training Time: {training_time:.2f} seconds")

"""
## Interpretation

Model A served as the baseline architecture for the Architecture
Comparison study and consisted of a single hidden layer containing
16 neurons.

The model achieved:

- MAE: 0.5594
- RMSE: 0.7226
- Adjusted R²: 0.3190

The learning curve showed rapid convergence during the first few
epochs followed by a long period of stable training. Training and
validation losses remained closely aligned throughout the training
process, indicating stable optimization and minimal evidence of
overfitting.

The Actual vs Predicted plot demonstrated that the model successfully
captured overall quality trends but continued to exhibit prediction
compression. Low-quality wines tended to be overestimated while
high-quality wines were frequently underestimated. Most predictions
remained concentrated near the center of the quality distribution.

The relatively small gap between training and validation loss suggests
that the model generalized well to unseen data despite its simple
architecture.

Overall, Model A provided stable training behavior and established
a strong baseline against which more complex architectures can be
evaluated.
"""

"""
## Summary

The objective of this analysis was to establish a baseline network
architecture for the Architecture Comparison study.

Model A utilized a single hidden layer containing 16 neurons while
retaining the previously selected configuration:

- MinMaxScaler
- ReLU Activation
- Adam Optimizer
- 100 Epochs

The model achieved:

- MAE: 0.5594
- RMSE: 0.7226
- Adjusted R²: 0.3190

Training and validation losses decreased rapidly before stabilizing
at similar values, indicating effective learning and strong
generalization performance.

Actual-versus-predicted visualizations showed reasonable agreement
between observed and predicted wine quality scores, although
prediction compression remained present for extreme quality ratings.

Overall, Model A provides a reliable baseline for determining whether
increased network depth and complexity improve predictive performance.
"""