"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 1, 2026
Last Edit:  November 1, 2026
Progress:   Complete

Analysis:   03 - Activation Function Comparison
Method:     Sigmoid

Description:
This script evaluates neural network performance using the Sigmoid
activation function.

The winning configuration from Analysis 2 is retained:
MinMaxScaler + Adam Optimizer + 16 Hidden Neurons.

Results from this analysis will be compared against ReLU
and Tanh activation functions.

Research Question:
Which activation function performs best for structured
wine-quality data?
"""

# ==========================================
# Analysis 3: Activation Function Comparison
# Method 2: Sigmoid
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
# Sigmoid Model
# ------------------------------------------

sigmoid_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(
        16,
        activation="sigmoid"
    ),
    Dense(1)
])

sigmoid_model.compile(
    optimizer="adam",
    loss="mse"
)

# ------------------------------------------
# Train Model
# ------------------------------------------

history_sigmoid = sigmoid_model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=100,
    verbose=0
)

# ------------------------------------------
# Predictions
# ------------------------------------------

y_pred_sigmoid = sigmoid_model.predict(
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

mae_sigmoid, rmse_sigmoid, adjr2_sigmoid = evaluate_model(
    y_test,
    y_pred_sigmoid,
    X.shape[1]
)

# ------------------------------------------
# Results Table
# ------------------------------------------

results_activation = pd.DataFrame({
    "Activation_Function": ["Sigmoid"],
    "MAE": [mae_sigmoid],
    "RMSE": [rmse_sigmoid],
    "Adjusted_R2": [adjr2_sigmoid],
    "Final_Training_Loss": [
        history_sigmoid.history["loss"][-1]
    ],
    "Final_Validation_Loss": [
        history_sigmoid.history["val_loss"][-1]
    ]
})

print("\n----- SIGMOID RESULTS -----")
print(results_activation.round(4))

# --------------------------------------
# Export Results
# ------------------------------------------

results_activation.to_csv(
    output_dir / "sigmoid_results.csv",
    index=False
)

# ------------------------------------------
# Learning Curve
# ------------------------------------------

plt.figure(figsize=(10,6))

plt.plot(
    history_sigmoid.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_sigmoid.history["val_loss"],
    label="Validation Loss"
)

plt.title("Sigmoid Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    output_dir / "sigmoid_learning_curve.png",
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------
# Actual vs Predicted
# ------------------------------------------

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    y_pred_sigmoid,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title("Sigmoid: Actual vs Predicted")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")

plt.savefig(
    output_dir / "sigmoid_actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------
# Export Summary
# ------------------------------------------

with open(
    output_dir / "sigmoid_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("Analysis 3: Activation Function Comparison\n")
    f.write("Method: Sigmoid\n\n")

    f.write(f"MAE: {mae_sigmoid:.4f}\n")
    f.write(f"RMSE: {rmse_sigmoid:.4f}\n")
    f.write(f"Adjusted R²: {adjr2_sigmoid:.4f}\n")

# ------------------------------------------
# Export Training History
# ------------------------------------------

history_df = pd.DataFrame({
    "loss": history_sigmoid.history["loss"],
    "val_loss": history_sigmoid.history["val_loss"]
})

history_df.to_csv(
    output_dir / "sigmoid_history.csv",
    index=False
)

# ------------------------------------------
# Final Output
# ------------------------------------------

print("\nAnalysis 3 - Sigmoid Complete")

print(f"MAE: {mae_sigmoid:.4f}")
print(f"RMSE: {rmse_sigmoid:.4f}")
print(f"Adjusted R²: {adjr2_sigmoid:.4f}")

"""
## Interpretation

The Sigmoid activation function produced weaker predictive performance
than the ReLU baseline.

The model achieved:

- MAE: 0.5645
- RMSE: 0.7334
- Adjusted R²: 0.2984

These results are notably lower than those obtained using ReLU,
suggesting that Sigmoid was less effective at learning relationships
within the wine quality dataset.

The learning curve shows rapid convergence during the first several
epochs followed by an extended plateau. Training and validation losses
remain closely aligned, indicating stable learning and little evidence
of overfitting.

The Actual vs Predicted plot reveals substantial prediction
compression. Most predictions are concentrated near the center of the
quality distribution, causing low-quality wines to be overestimated and
high-quality wines to be underestimated.

This behavior is consistent with the known limitations of Sigmoid
activations, which can suffer from saturation effects and weaker
gradient propagation compared with ReLU.

Overall, the results suggest that Sigmoid is capable of learning the
wine quality problem but does so less effectively than ReLU.
"""

"""
## Summary

The objective of this analysis was to evaluate the Sigmoid activation
function within the selected neural network architecture and compare
its performance against alternative activation functions.

Using MinMax-scaled inputs and the same training configuration as the
ReLU experiment, the model achieved:

- MAE: 0.5645
- RMSE: 0.7334
- Adjusted R²: 0.2984

Training and validation losses decreased quickly during the early
epochs before stabilizing. Although convergence remained stable, final
loss values were higher than those achieved by the ReLU model.

Actual-versus-predicted visualizations showed that the model
successfully captured overall quality trends but produced predictions
that were more heavily concentrated around average quality values.

Overall, Sigmoid delivered acceptable performance but failed to match
the predictive accuracy achieved by the ReLU baseline.
"""