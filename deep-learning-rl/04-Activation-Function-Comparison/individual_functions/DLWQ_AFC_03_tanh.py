"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 1, 2026
Last Edit:  November 1, 2026
Progress:   Complete

Analysis:   03 - Activation Function Comparison
Method:     Tanh

Description:
This script evaluates neural network performance using the Tanh
activation function.

The winning configuration from Analysis 2 is retained:
MinMaxScaler + Adam Optimizer + 16 Hidden Neurons.

Results from this analysis will be compared against ReLU
and Sigmoid activation functions.

Research Question:
Which activation function performs best for structured
wine-quality data?
"""

# ==========================================
# Analysis 3: Activation Function Comparison
# Method 3: Tanh
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
# Tanh Model
# ------------------------------------------

tanh_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(
        16,
        activation="tanh"
    ),
    Dense(1)
])

tanh_model.compile(
    optimizer="adam",
    loss="mse"
)

# ------------------------------------------
# Train Model
# ------------------------------------------

history_tanh = tanh_model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=100,
    verbose=0
)

# ------------------------------------------
# Predictions
# ------------------------------------------

y_pred_tanh = tanh_model.predict(
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

mae_tanh, rmse_tanh, adjr2_tanh = evaluate_model(
    y_test,
    y_pred_tanh,
    X.shape[1]
)

# ------------------------------------------
# Results Table
# ------------------------------------------

results_activation = pd.DataFrame({
    "Activation_Function": ["Tanh"],
    "MAE": [mae_tanh],
    "RMSE": [rmse_tanh],
    "Adjusted_R2": [adjr2_tanh],
    "Final_Training_Loss": [
        history_tanh.history["loss"][-1]
    ],
    "Final_Validation_Loss": [
        history_tanh.history["val_loss"][-1]
    ]
})

print("\n----- TANH RESULTS -----")
print(results_activation.round(4))

# ------------------------------------------
# Export Results
# ------------------------------------------

results_activation.to_csv(
    output_dir / "tanh_results.csv",
    index=False
)

# ------------------------------------------
# Learning Curve
# ------------------------------------------

plt.figure(figsize=(10,6))

plt.plot(
    history_tanh.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_tanh.history["val_loss"],
    label="Validation Loss"
)

plt.title("Tanh Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    output_dir / "tanh_learning_curve.png",
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------
# Actual vs Predicted
# ------------------------------------------

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    y_pred_tanh,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title("Tanh: Actual vs Predicted")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")

plt.savefig(
    output_dir / "tanh_actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------
# Export Summary
# ------------------------------------------

with open(
    output_dir / "tanh_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("Analysis 3: Activation Function Comparison\n")
    f.write("Method: Tanh\n\n")

    f.write(f"MAE: {mae_tanh:.4f}\n")
    f.write(f"RMSE: {rmse_tanh:.4f}\n")
    f.write(f"Adjusted R²: {adjr2_tanh:.4f}\n")

# ------------------------------------------
# Export Training History
# ------------------------------------------

history_df = pd.DataFrame({
    "loss": history_tanh.history["loss"],
    "val_loss": history_tanh.history["val_loss"]
})

history_df.to_csv(
    output_dir / "tanh_history.csv",
    index=False
)

# ------------------------------------------
# Final Output
# ------------------------------------------

print("\nAnalysis 3 - Tanh Complete")

print(f"MAE: {mae_tanh:.4f}")
print(f"RMSE: {rmse_tanh:.4f}")
print(f"Adjusted R²: {adjr2_tanh:.4f}")

"""
## Interpretation

The Tanh activation function produced moderate predictive performance,
outperforming Sigmoid but failing to surpass the ReLU baseline.

The model achieved:

- MAE: 0.5627
- RMSE: 0.7297
- Adjusted R²: 0.3054

The learning curve showed rapid convergence during the first several
epochs before stabilizing. Training and validation losses remained
closely aligned throughout training, indicating stable learning and
minimal evidence of overfitting.

The Actual vs Predicted plot demonstrated reasonable agreement between
predicted and observed wine quality scores. However, prediction
compression remained evident, with low-quality wines generally
overestimated and high-quality wines underestimated.

Compared with Sigmoid, Tanh achieved lower prediction error and a
higher Adjusted R² score. However, ReLU maintained superior
performance across all evaluation metrics and produced predictions
that were more closely aligned with actual quality ratings.

Overall, the Tanh activation function provided stable training
behavior and acceptable predictive performance but did not outperform
the ReLU activation function.
"""

"""
## Summary

The objective of this analysis was to evaluate the Tanh activation
function and compare its performance against ReLU and Sigmoid using
the winning configuration from previous experiments.

The model achieved:

- MAE: 0.5627
- RMSE: 0.7297
- Adjusted R²: 0.3054

Training and validation losses decreased rapidly during the initial
epochs and remained closely aligned throughout training, indicating
stable convergence and good generalization.

Actual-versus-predicted visualizations showed that the model captured
overall wine quality trends while continuing to exhibit prediction
compression near the center of the quality distribution.

Although Tanh improved upon Sigmoid, it did not achieve the same level
of predictive accuracy as ReLU.

Overall, Tanh represents a viable activation function for this
problem, but ReLU remains the strongest performer based on the current
results.
"""