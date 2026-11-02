"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 1, 2026
Last Edit:  November 1, 2026
Progress:   Complete

Analysis:   04 - Architecture Comparison
Method:     Model B (32 -> 16)

Description:
This script evaluates a neural network architecture containing two hidden 
layers with 32 and 16 neurons, respectively.

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
# Model B Architecture
# (32 -> 16)
# ==========================================

model_b = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        16,
        activation="relu"
    ),

    Dense(1)
])

model_b.compile(
    optimizer="adam",
    loss="mse"
)

# ==========================================
# Train Model
# ==========================================

start_time = time.time()

history_b = model_b.fit(
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
    "loss": history_b.history["loss"],
    "val_loss": history_b.history["val_loss"]
})

history_df.to_csv(
    output_dir / "model_b_history.csv",
    index=False
)

# ==========================================
# Predictions
# ==========================================

y_pred_b = model_b.predict(
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

mae_b, rmse_b, adjr2_b = evaluate_model(
    y_test,
    y_pred_b,
    X.shape[1]
)

# ==========================================
# Results Table
# ==========================================

results_architecture = pd.DataFrame({

    "Architecture": ["Model B"],

    "Structure": ["32 -> 16"],

    "MAE": [mae_b],

    "RMSE": [rmse_b],

    "Adjusted_R2": [adjr2_b],

    "Training_Time_Seconds": [
        training_time
    ],

    "Final_Training_Loss": [
        history_b.history["loss"][-1]
    ],

    "Final_Validation_Loss": [
        history_b.history["val_loss"][-1]
    ]
})

print("\n----- MODEL B RESULTS -----")
print(results_architecture.round(4))

# ==========================================
# Export Results
# ==========================================

results_architecture.to_csv(
    output_dir / "model_b_results.csv",
    index=False
)

# ==========================================
# Learning Curve
# ==========================================

plt.figure(figsize=(10,6))

plt.plot(
    history_b.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_b.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Model B Learning Curve"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    output_dir /
    "model_b_learning_curve.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Actual vs Predicted
# ==========================================

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    y_pred_b,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title(
    "Model B: Actual vs Predicted"
)

plt.xlabel(
    "Actual Quality"
)

plt.ylabel(
    "Predicted Quality"
)

plt.savefig(
    output_dir /
    "model_b_actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Export Summary
# ==========================================

with open(
    output_dir /
    "model_b_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Analysis 4: Architecture Comparison\n"
    )

    f.write(
        "Method: Model B (32 -> 16)\n\n"
    )

    f.write(
        f"MAE: {mae_b:.4f}\n"
    )

    f.write(
        f"RMSE: {rmse_b:.4f}\n"
    )

    f.write(
        f"Adjusted R²: {adjr2_b:.4f}\n"
    )

    f.write(
        f"Training Time: {training_time:.2f} seconds\n"
    )

# ==========================================
# Final Output
# ==========================================

print("\nAnalysis 4 - Model B Complete")

print(f"MAE: {mae_b:.4f}")
print(f"RMSE: {rmse_b:.4f}")
print(f"Adjusted R²: {adjr2_b:.4f}")
print(f"Training Time: {training_time:.2f} seconds")

"""
## Interpretation

Model B increased network complexity by introducing a second hidden
layer, creating a 32 → 16 architecture while retaining the winning
configuration from previous analyses.

The model achieved:

- MAE: 0.5502
- RMSE: 0.7075
- Adjusted R²: 0.3471

These results improved upon the baseline Model A architecture across
all evaluation metrics, indicating that the additional hidden layer
enabled the network to learn more informative feature relationships.

The learning curve showed rapid convergence during the initial epochs
before stabilizing at lower loss values than those observed for Model A.
Training and validation losses remained closely aligned throughout
training, suggesting effective learning and minimal overfitting.

The Actual vs Predicted plot demonstrated stronger agreement between
predicted and observed wine quality scores compared with Model A.
Predictions remained concentrated near the center of the quality
distribution, but the overall spread and alignment with the ideal
prediction line improved.

Although prediction compression remained present for extreme quality
ratings, Model B reduced prediction error and explained more variance
than the simpler baseline architecture.

Overall, the results suggest that increasing network complexity from
Model A to Model B improved predictive performance without introducing
substantial overfitting.
"""

"""
## Summary

The objective of this analysis was to evaluate whether increasing
network complexity improves wine quality prediction performance.

Model B utilized a two-hidden-layer architecture consisting of:

32 → 16

while retaining the previously selected configuration:

- MinMaxScaler
- ReLU Activation
- Adam Optimizer
- 100 Epochs

The model achieved:

- MAE: 0.5502
- RMSE: 0.7075
- Adjusted R²: 0.3471

Compared with the baseline Model A architecture, Model B produced lower
prediction error and higher explanatory power. Training and validation
losses converged rapidly and remained closely aligned throughout
training, indicating stable learning behavior and strong
generalization.

Actual-versus-predicted visualizations demonstrated improved prediction
accuracy while continuing to exhibit some difficulty with rare quality
ratings.

Overall, Model B outperformed the baseline architecture and became the
leading model in the Architecture Comparison study.
"""

