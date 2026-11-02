"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 1, 2026
Last Edit:  November 1, 2026
Progress:   Complete

Analysis:   04 - Architecture Comparison
Method:     Model C (64 -> 32 -> 16)

Description:
This script evaluates a neural network architecture containing three hidden 
layers with 64 and 32 and 16 neurons, respectively.

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
# Model C Architecture
# (64 -> 32 -> 16)
# ==========================================

model_c = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),

    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),

    Dense(1)
])

model_c.compile(
    optimizer="adam",
    loss="mse"
)

# ==========================================
# Train Model
# ==========================================

start_time = time.time()

history_c = model_c.fit(
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
    "loss": history_c.history["loss"],
    "val_loss": history_c.history["val_loss"]
})

history_df.to_csv(
    output_dir / "model_c_history.csv",
    index=False
)

# ==========================================
# Predictions
# ==========================================

y_pred_c = model_c.predict(
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

mae_c, rmse_c, adjr2_c = evaluate_model(
    y_test,
    y_pred_c,
    X.shape[1]
)

# ==========================================
# Results Table
# ==========================================

results_architecture = pd.DataFrame({

    "Architecture": ["Model C"],

    "Structure": ["64 -> 32 -> 16"],

    "MAE": [mae_c],

    "RMSE": [rmse_c],

    "Adjusted_R2": [adjr2_c],

    "Training_Time_Seconds": [
        training_time
    ],

    "Final_Training_Loss": [
        history_c.history["loss"][-1]
    ],

    "Final_Validation_Loss": [
        history_c.history["val_loss"][-1]
    ]
})

print("\n----- MODEL C RESULTS -----")
print(results_architecture.round(4))

# ==========================================
# Export Results
# ==========================================

results_architecture.to_csv(
    output_dir / "model_c_results.csv",
    index=False
)

# ==========================================
# Learning Curve
# ==========================================

plt.figure(figsize=(10,6))

plt.plot(
    history_c.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_c.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Model C Learning Curve"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    output_dir /
    "model_c_learning_curve.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Actual vs Predicted
# ==========================================

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    y_pred_c,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title(
    "Model C: Actual vs Predicted"
)

plt.xlabel(
    "Actual Quality"
)

plt.ylabel(
    "Predicted Quality"
)

plt.savefig(
    output_dir /
    "model_c_actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Export Summary
# ==========================================

with open(
    output_dir /
    "model_c_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Analysis 4: Architecture Comparison\n"
    )

    f.write(
        "Method: Model C (64 -> 32 -> 16)\n\n"
    )

    f.write(
        f"MAE: {mae_c:.4f}\n"
    )

    f.write(
        f"RMSE: {rmse_c:.4f}\n"
    )

    f.write(
        f"Adjusted R²: {adjr2_c:.4f}\n"
    )

    f.write(
        f"Training Time: {training_time:.2f} seconds\n"
    )

# ==========================================
# Final Output
# ==========================================

print("\nAnalysis 4 - Model C Complete")

print(f"MAE: {mae_c:.4f}")
print(f"RMSE: {rmse_c:.4f}")
print(f"Adjusted R²: {adjr2_c:.4f}")
print(f"Training Time: {training_time:.2f} seconds")

"""
## Interpretation

Model C further increased network complexity through a three-hidden-layer
architecture consisting of:

64 → 32 → 16

The model achieved:

- MAE: 0.5462
- RMSE: 0.7023
- Adjusted R²: 0.3567

These results improved upon both Model A and Model B across all primary
evaluation metrics, indicating that the additional depth enabled the
network to capture more complex relationships within the wine quality
dataset.

The learning curve demonstrated rapid convergence during the early
epochs followed by stable training throughout the remainder of the
learning process. Although training loss decreased further than in
previous models, validation loss remained stable and closely aligned,
suggesting that the deeper architecture continued to generalize well
without substantial overfitting.

The Actual vs Predicted plot showed improved agreement between
predicted and observed wine quality scores. Predictions exhibited
slightly greater dispersion and alignment with the ideal prediction
line, indicating stronger predictive performance than earlier
architectures.

Prediction compression remained present, particularly at the extreme
quality ratings, but the magnitude of prediction error continued to
decrease as network complexity increased.

Overall, Model C demonstrated the strongest performance observed thus
far and suggests that additional depth remains beneficial for this
wine-quality prediction task.
"""

"""
## Summary

The objective of this analysis was to evaluate whether a deeper neural
network architecture could improve wine quality prediction performance.

Model C utilized a three-hidden-layer architecture:

64 → 32 → 16

while maintaining the previously selected configuration:

- MinMaxScaler
- ReLU Activation
- Adam Optimizer
- 100 Epochs

The model achieved:

- MAE: 0.5462
- RMSE: 0.7023
- Adjusted R²: 0.3567

Compared with Models A and B, Model C produced lower prediction error
and explained a greater proportion of variance in wine quality scores.

Training and validation losses converged rapidly and remained closely
aligned throughout training, indicating stable optimization and strong
generalization performance.

Actual-versus-predicted visualizations showed improved agreement
between observed and predicted quality values while continuing to
struggle with rare extreme quality ratings.

Overall, Model C became the strongest-performing architecture tested
to this point and established a new benchmark for Model D.
"""
