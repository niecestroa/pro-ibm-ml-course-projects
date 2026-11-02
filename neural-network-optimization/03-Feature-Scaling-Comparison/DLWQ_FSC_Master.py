"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 2, 2026
Last Edit:  November 2, 2026
Progress:   Complete

Analysis:   02 - Feature Scaling Comparison
Method:     Master Script

Description:
Evaluates multiple feature scaling strategies using
the baseline neural network architecture and compares
their impact on wine quality prediction performance.

Scaling Methods Evaluated:

- No Scaling
- StandardScaler
- MinMaxScaler

Research Question:
Which feature scaling method produces the strongest
neural network performance?
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

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

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
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\03-Feature-Scaling-Comparison\tables_figures"
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
    random_state=42,
    stratify=y
)

# ==========================================
# Helper Functions
# ==========================================

def adjusted_r2(
    r2,
    n,
    p
):

    return (
        1 -
        (
            ((1 - r2) * (n - 1))
            /
            (n - p - 1)
        )
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
# Scaling Methods
# ==========================================

scalers = {

    "No_Scaling": None,

    "StandardScaler":
        StandardScaler(),

    "MinMaxScaler":
        MinMaxScaler()
}

all_results = []

# ==========================================
# Run Study
# ==========================================

for scaler_name, scaler in scalers.items():

    print(
        f"\n----- RUNNING {scaler_name} -----"
    )

    # ==========================
    # Scaling
    # ==========================

    if scaler is None:

        X_train_scaled = (
            X_train.values
        )

        X_test_scaled = (
            X_test.values
        )

    else:

        X_train_scaled = (
            scaler.fit_transform(
                X_train
            )
        )

        X_test_scaled = (
            scaler.transform(
                X_test
            )
        )

    # ==========================
    # Model
    # ==========================

    model = Sequential([

        Input(
            shape=(
                X_train_scaled.shape[1],
            )
        ),

        Dense(
            16,
            activation="relu"
        ),

        Dense(1)

    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    # ==========================
    # Train
    # ==========================

    start_time = time.time()

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.20,
        epochs=100,
        verbose=0
    )

    training_time = (
        time.time()
        - start_time
    )

    # ==========================
    # Predictions
    # ==========================

    y_pred = model.predict(
        X_test_scaled
    ).flatten()

    # ==========================
    # Evaluation
    # ==========================

    mae, rmse, adjr2 = (
        evaluate_model(
            y_test,
            y_pred,
            X.shape[1]
        )
    )

    # ==========================
    # Results Table
    # ==========================

    results_df = pd.DataFrame({

        "Scaling_Method":
            [scaler_name],

        "MAE":
            [mae],

        "RMSE":
            [rmse],

        "Adjusted_R2":
            [adjr2],

        "Training_Time_Seconds":
            [training_time],

        "Final_Training_Loss":
            [history.history["loss"][-1]],

        "Final_Validation_Loss":
            [history.history["val_loss"][-1]]

    })

    results_df.to_csv(
        output_dir /
        f"{scaler_name.lower()}_results.csv",
        index=False
    )

    # ==========================
    # History
    # ==========================

    history_df = pd.DataFrame({

        "loss":
            history.history["loss"],

        "val_loss":
            history.history["val_loss"]

    })

    history_df.to_csv(
        output_dir /
        f"{scaler_name.lower()}_history.csv",
        index=False
    )

    # ==========================
    # Learning Curve
    # ==========================

    plt.figure(
        figsize=(10,6)
    )

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        f"{scaler_name} Learning Curve"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        output_dir /
        f"{scaler_name.lower()}_learning_curve.png",
        bbox_inches="tight"
    )

    plt.close()

    # ==========================
    # Actual vs Predicted
    # ==========================

    plt.figure(
        figsize=(8,8)
    )

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.5
    )

    plt.plot(
        [y_test.min(),
         y_test.max()],

        [y_test.min(),
         y_test.max()],

        color="red"
    )

    plt.title(
        f"{scaler_name}: Actual vs Predicted"
    )

    plt.xlabel(
        "Actual Quality"
    )

    plt.ylabel(
        "Predicted Quality"
    )

    plt.savefig(
        output_dir /
        f"{scaler_name.lower()}_actual_vs_predicted.png",
        bbox_inches="tight"
    )

    plt.close()

    # ==========================
    # Save Results
    # ==========================

    all_results.append({

        "Scaling_Method":
            scaler_name,

        "MAE":
            mae,

        "RMSE":
            rmse,

        "Adjusted_R2":
            adjr2,

        "Training_Time_Seconds":
            training_time,

        "Final_Training_Loss":
            history.history["loss"][-1],

        "Final_Validation_Loss":
            history.history["val_loss"][-1]
    })

    print(
        f"\n----- {scaler_name} RESULTS -----"
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

# ==========================================
# Comparison Table
# ==========================================

comparison = pd.DataFrame(
    all_results
)

comparison = comparison.sort_values(
    "RMSE"
)

comparison.to_csv(
    output_dir /
    "feature_scaling_comparison.csv",
    index=False
)

print(
    "\n----- FINAL SCALING COMPARISON -----"
)

print(
    comparison.round(4)
)

print(
    "\nAnalysis 2 Master Script Complete"
)

