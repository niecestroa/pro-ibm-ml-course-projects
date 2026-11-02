"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 1, 2026
Last Edit:  November 1, 2026
Progress:   Complete

Analysis:   04 - Architecture Comparison
Method:     Model D (128 -> 64 -> 32 -> 16)

Description:
This script evaluates a neural network architecture containing four hidden 
layers with 128 and 64 and 32 and 16 neurons, respectively.

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
# Model D Architecture
# (128 -> 64 -> 32 -> 16)
# ==========================================

model_d = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),

    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),

    Dense(1)
])

model_d.compile(
    optimizer="adam",
    loss="mse"
)

# ==========================================
# Train Model
# ==========================================

start_time = time.time()

history_d = model_d.fit(
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
    "loss": history_d.history["loss"],
    "val_loss": history_d.history["val_loss"]
})

history_df.to_csv(
    output_dir / "model_d_history.csv",
    index=False
)

# ==========================================
# Predictions
# ==========================================

y_pred_d = model_d.predict(
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

mae_d, rmse_d, adjr2_d = evaluate_model(
    y_test,
    y_pred_d,
    X.shape[1]
)

# ==========================================
# Results Table
# ==========================================

results_architecture = pd.DataFrame({

    "Architecture": ["Model D"],

    "Structure": ["128 -> 64 -> 32 -> 16"],

    "MAE": [mae_d],

    "RMSE": [rmse_d],

    "Adjusted_R2": [adjr2_d],

    "Training_Time_Seconds": [
        training_time
    ],

    "Final_Training_Loss": [
        history_d.history["loss"][-1]
    ],

    "Final_Validation_Loss": [
        history_d.history["val_loss"][-1]
    ]
})

print("\n----- MODEL D RESULTS -----")
print(results_architecture.round(4))

# ==========================================
# Export Results
# ==========================================

results_architecture.to_csv(
    output_dir / "model_d_results.csv",
    index=False
)

# ==========================================
# Learning Curve
# ==========================================

plt.figure(figsize=(10,6))

plt.plot(
    history_d.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_d.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Model D Learning Curve"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    output_dir /
    "model_d_learning_curve.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Actual vs Predicted
# ==========================================

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    y_pred_d,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title(
    "Model D: Actual vs Predicted"
)

plt.xlabel(
    "Actual Quality"
)

plt.ylabel(
    "Predicted Quality"
)

plt.savefig(
    output_dir /
    "model_d_actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Export Summary
# ==========================================

with open(
    output_dir /
    "model_d_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Analysis 4: Architecture Comparison\n"
    )

    f.write(
        "Method: Model D (128 -> 64 -> 32 -> 16)\n\n"
    )

    f.write(
        f"MAE: {mae_d:.4f}\n"
    )

    f.write(
        f"RMSE: {rmse_d:.4f}\n"
    )

    f.write(
        f"Adjusted R²: {adjr2_d:.4f}\n"
    )

    f.write(
        f"Training Time: {training_time:.2f} seconds\n"
    )

# ==========================================
# Final Output
# ==========================================

print("\nAnalysis 4 - Model D Complete")

print(f"MAE: {mae_d:.4f}")
print(f"RMSE: {rmse_d:.4f}")
print(f"Adjusted R²: {adjr2_d:.4f}")
print(f"Training Time: {training_time:.2f} seconds")

"""
## Interpretation

Model D represented the most complex architecture evaluated in this
study and utilized a four-hidden-layer configuration:

128 → 64 → 32 → 16

The model achieved:

- MAE: 0.5424
- RMSE: 0.7043
- Adjusted R²: 0.3530

Although Model D achieved the lowest MAE observed in the architecture
comparison, improvements in RMSE and Adjusted R² were not sufficient
to surpass the performance achieved by Model C.

The learning curve showed rapid convergence during the early epochs
followed by stable optimization throughout training. Training and
validation losses remained closely aligned, indicating effective
generalization and little evidence of substantial overfitting despite
the increased model complexity.

The Actual vs Predicted plot demonstrated prediction behavior similar
to Model C. Predictions generally followed the underlying trend in
the data, with greater accuracy observed for quality scores between
5 and 7.

However, prediction compression remained evident, as lower-quality
wines continued to be overestimated while higher-quality wines were
underestimated. Additional network complexity did not completely
eliminate this behavior.

Overall, Model D demonstrated strong predictive performance but
provided only marginal improvement over Model C despite a substantial
increase in architectural complexity.
"""

"""
## Summary

The objective of this analysis was to determine whether a highly
complex neural network architecture could further improve wine quality
prediction performance.

Model D utilized a four-hidden-layer architecture:

128 → 64 → 32 → 16

while retaining the previously selected configuration:

- MinMaxScaler
- ReLU Activation
- Adam Optimizer
- 100 Epochs

The model achieved:

- MAE: 0.5424
- RMSE: 0.7043
- Adjusted R²: 0.3530

Compared with Models A and B, Model D improved predictive
performance and reduced prediction error. However, Model D narrowly
underperformed Model C on RMSE and Adjusted R² despite possessing
substantially greater complexity.

Training and validation losses converged rapidly and remained closely
aligned throughout training, indicating stable learning behavior and
minimal overfitting.

Overall, the results suggest that increasing network complexity beyond
Model C offers limited practical benefit for this dataset. Model D
demonstrates diminishing returns and does not clearly justify its
additional computational complexity.
"""