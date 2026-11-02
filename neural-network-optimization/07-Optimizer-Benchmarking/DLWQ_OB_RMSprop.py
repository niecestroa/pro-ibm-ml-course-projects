"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 2, 2026
Last Edit:  November 2, 2026
Progress:   Complete

Analysis:   06 - Optimizer Benchmarking
Method:     RMSProp

Description:
This script evaluates the winning optimizer identified
during Analysis 6 and establishes the standalone
optimizer benchmark for future analyses.

Winning Configuration:

- StandardScaler
- Sigmoid Activation
- Architecture: 64 -> 32 -> 16
- RMSProp Optimizer
- 125 Epochs

Research Question:
Can RMSProp provide the strongest optimization
performance for wine quality prediction?
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

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

from tensorflow.keras.optimizers import RMSprop

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

# ==========================================
# Load Data
# ==========================================

file_path = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\00-data\winequality-final.csv"
)

output_dir = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\07-Optimizer-Benchmarking\tables_figures"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

dpwq = pd.read_csv(
    file_path
)

dpwq.columns = (
    dpwq.columns
    .str.strip()
)

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

X["type_white"] = (
    X["type_white"]
    .astype(int)
)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=SEED,
    stratify=y
)

# ==========================================
# StandardScaler Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# ==========================================
# Helper Functions
# ==========================================

def adjusted_r2(r2, n, p):

    return (
        1 -
        (((1 - r2) * (n - 1))
         / (n - p - 1))
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

    return (
        mae,
        rmse,
        adj_r2
    )

# ==========================================
# RMSProp Model
# ==========================================

model = Sequential([
    
    tf.keras.backend.clear_session(),

    Input(
        shape=(
            X_train_scaled.shape[1],
        )
    ),

    Dense(
        64,
        activation="sigmoid"
    ),

    Dense(
        32,
        activation="sigmoid"
    ),

    Dense(
        16,
        activation="sigmoid"
    ),

    Dense(1)

])

model.compile(
    optimizer=RMSprop(),
    loss="mse"
)

# ==========================================
# Train Model
# ==========================================

start_time = time.time()

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=125,
    verbose=0
)

training_time = (
    time.time() - start_time
)

# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(
    X_test_scaled
).flatten()

# ==========================================
# Evaluation
# ==========================================

mae, rmse, adjr2 = evaluate_model(
    y_test,
    y_pred,
    X.shape[1]
)

# ==========================================
# Results Table
# ==========================================

results_df = pd.DataFrame({

    "Optimizer": ["RMSProp"],

    "MAE": [mae],

    "RMSE": [rmse],

    "Adjusted_R2": [adjr2],

    "Training_Time_Seconds": [
        training_time
    ],

    "Final_Training_Loss": [
        history.history["loss"][-1]
    ],

    "Final_Validation_Loss": [
        history.history["val_loss"][-1]
    ]

})

results_df.to_csv(
    output_dir /
    "rmsprop_results.csv",
    index=False
)

# ==========================================
# History
# ==========================================

history_df = pd.DataFrame({

    "loss":
        history.history["loss"],

    "val_loss":
        history.history["val_loss"]

})

history_df.to_csv(
    output_dir /
    "rmsprop_history.csv",
    index=False
)

# ==========================================
# Learning Curve
# ==========================================

plt.figure(figsize=(10,6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "RMSProp Learning Curve"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    output_dir /
    "rmsprop_learning_curve.png",
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# Actual vs Predicted
# ==========================================

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title(
    "RMSProp: Actual vs Predicted"
)

plt.xlabel(
    "Actual Quality"
)

plt.ylabel(
    "Predicted Quality"
)

plt.savefig(
    output_dir /
    "rmsprop_actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()
plt.close()

# ==========================================
# Export Summary
# ==========================================

with open(
    output_dir /
    "rmsprop_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Analysis 6 - RMSProp\n\n"
    )

    f.write(
        f"MAE: {mae:.4f}\n"
    )

    f.write(
        f"RMSE: {rmse:.4f}\n"
    )

    f.write(
        f"Adjusted R²: {adjr2:.4f}\n"
    )

    f.write(
        f"Training Time: {training_time:.2f} Seconds\n"
    )

# ==========================================
# Final Output
# ==========================================

print(
    "\n----- RMSPROP RESULTS -----"
)

print(
    f"MAE: {mae:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"Adjusted R²: {adjr2:.4f}"
)

print(
    f"Training Time: {training_time:.2f} Seconds"
)

print(
    "\nAnalysis 6 - RMSProp Complete"
)