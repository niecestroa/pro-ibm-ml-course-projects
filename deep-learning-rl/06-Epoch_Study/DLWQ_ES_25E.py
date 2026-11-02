"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 2, 2026
Last Edit:  November 2, 2026
Progress:   Complete

Analysis:   05 - Epoch Optimization
Method:     25 Epochs

Description:
This script evaluates the impact of limiting training
to 25 epochs using the winning neural network
configuration from previous analyses.

The winning configuration from previous analyses is retained:

- MinMaxScaler
- ReLU Activation
- Adam Optimizer
- Architecture: 64 -> 32 -> 16

Research Question:
How many training epochs are required to achieve
optimal predictive performance without unnecessary
training or overfitting?
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
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\06-Epoch_Study\tables_figures"
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
# Neural Network Architecture
# (64 -> 32 -> 16)
# ==========================================

model_25e = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),

    Dense(64, activation="relu"),

    Dense(32, activation="relu"),

    Dense(16, activation="relu"),

    Dense(1)
])

model_25e.compile(
    optimizer="adam",
    loss="mse"
)

# ==========================================
# Train Model
# ==========================================

start_time = time.time()

history_25e = model_25e.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=25,
    verbose=0
)

training_time = (
    time.time() - start_time
)

# ==========================================
# Export Training History
# ==========================================

history_df = pd.DataFrame({
    "loss": history_25e.history["loss"],
    "val_loss": history_25e.history["val_loss"]
})

history_df.to_csv(
    output_dir / "25_epochs_history.csv",
    index=False
)

# ==========================================
# Predictions
# ==========================================

y_pred_25e = model_25e.predict(
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

mae_25e, rmse_25e, adjr2_25e = evaluate_model(
    y_test,
    y_pred_25e,
    X.shape[1]
)

# ==========================================
# Results Table
# ==========================================

results_epochs = pd.DataFrame({

    "Epochs": [25],

    "MAE": [mae_25e],

    "RMSE": [rmse_25e],

    "Adjusted_R2": [adjr2_25e],

    "Training_Time_Seconds": [
        training_time
    ],

    "Final_Training_Loss": [
        history_25e.history["loss"][-1]
    ],

    "Final_Validation_Loss": [
        history_25e.history["val_loss"][-1]
    ]
})

print("\n----- MODEL 25 EPOCHS RESULTS -----")
print(results_epochs.round(4))

# ==========================================
# Export Results
# ==========================================

results_epochs.to_csv(
    output_dir / "25_epochs_results.csv",
    index=False
)

# ==========================================
# Learning Curve
# ==========================================

plt.figure(figsize=(10,6))

plt.plot(
    history_25e.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_25e.history["val_loss"],
    label="Validation Loss"
)


plt.title(
    "25 Epoch Learning Curve"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    output_dir /
    "25_epochs_learning_curve.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Actual vs Predicted
# ==========================================

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    y_pred_25e,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title(
    "25 Epochs: Actual vs Predicted"
)

plt.xlabel(
    "Actual Quality"
)

plt.ylabel(
    "Predicted Quality"
)

plt.savefig(
    output_dir /
    "25_epochs_actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()

# ==========================================
# Export Summary
# ==========================================

with open(
    output_dir /
    "25_epochs_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Analysis 5: Epoch Optimization\n"
    )

    f.write(
        "Method: 25 Epochs\n\n"
    )

    f.write(
        f"MAE: {mae_25e:.4f}\n"
    )

    f.write(
        f"RMSE: {rmse_25e:.4f}\n"
    )

    f.write(
        f"Adjusted R²: {adjr2_25e:.4f}\n"
    )

    f.write(
        f"Training Time: {training_time:.2f} seconds\n"
    )

# ==========================================
# Final Output
# ==========================================

print("\n----- 25 EPOCH RESULTS -----")
print(results_epochs.round(4))

print("\nAnalysis 5 - 25 Epochs Complete")

print(f"MAE: {mae_25e:.4f}")
print(f"RMSE: {rmse_25e:.4f}")
print(f"Adjusted R²: {adjr2_25e:.4f}")
print(f"Training Time: {training_time:.2f} seconds")

"""
## Interpretation

The 25-epoch model was evaluated to determine whether a reduced
training duration could achieve comparable predictive performance
while decreasing computational cost.

The model achieved:

- MAE: 0.5545
- RMSE: 0.7161
- Adjusted R²: 0.3311

Compared with the current benchmark architecture trained for
100 epochs, the 25-epoch model produced higher prediction error
and explained less variance in wine quality scores.

The learning curve indicates rapid improvement during the first few
epochs followed by continued loss reduction throughout training.
Although the losses stabilized substantially by the end of training,
both training and validation loss curves suggest that learning had
not completely plateaued.

Training and validation losses remained closely aligned throughout
the training process, indicating strong generalization and little
evidence of overfitting. However, the model appears to be slightly
undertrained relative to the 100-epoch benchmark.

The Actual vs Predicted visualization demonstrated reasonable
agreement between predicted and observed wine quality scores, but
prediction accuracy remained lower than models trained for longer
durations.

Overall, the 25-epoch configuration provided efficient training and
stable generalization but did not achieve the predictive performance
observed in models trained for more epochs.
"""

"""
## Summary

The objective of this analysis was to evaluate the impact of limiting
training to 25 epochs using the previously selected optimal neural
network configuration.

The model retained:

- MinMaxScaler
- ReLU Activation
- Adam Optimizer
- Architecture: 64 → 32 → 16

Performance results were:

- MAE: 0.5545
- RMSE: 0.7161
- Adjusted R²: 0.3311
- Training Time: 13.68 Seconds

The learning curve demonstrated rapid convergence during the initial
epochs and stable learning behavior throughout training. Training and
validation losses remained closely aligned, indicating effective
generalization and minimal evidence of overfitting.

Despite its computational efficiency, the 25-epoch model did not
match the predictive accuracy achieved by the current 100-epoch
benchmark model. The results suggest that additional training
epochs remain beneficial.

Overall, 25 epochs appear sufficient for initial convergence but
insufficient for achieving optimal predictive performance.
"""
