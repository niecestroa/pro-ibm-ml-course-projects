"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 2, 2026
Last Edit:  November 2, 2026
Progress:   Complete

Analysis:   07 - Regularization and Generalization
Method:     Master Script

Description:
Evaluates multiple regularization techniques and
compares their impact on predictive performance,
generalization, and model robustness.

Selected Configuration Retained:

- StandardScaler
- Sigmoid Activation
- Architecture: 64 → 32 → 16
- RMSProp Optimizer
- 125 Epochs

Methods Evaluated:

- Baseline
- L1 Regularization
- L2 Regularization
- Dropout
- Early Stopping

Research Question:
Which regularization strategy produces the
most reliable model?
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
from tensorflow.keras.layers import (
    Dense,
    Dropout
)

from tensorflow.keras import Input

from tensorflow.keras.regularizers import (
    l1,
    l2
)

from tensorflow.keras.callbacks import (
    EarlyStopping
)

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
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\08-Regularization-and-Generalization\tables_figures"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

dpwq = pd.read_csv(file_path)

dpwq.columns = dpwq.columns.str.strip()

# ==========================================
# Features / Target
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
# Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=SEED,
    stratify=y
)

# ==========================================
# Scaling
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
# Methods
# ==========================================

methods = [

    "Baseline",
    "L1",
    "L2",
    "Dropout",
    "EarlyStopping"

]

all_results = []

# ==========================================
# Run Study
# ==========================================

for method in methods:

    tf.keras.backend.clear_session()
    
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    print(
        f"\n----- RUNNING {method} -----"
    )

    model = Sequential()

    model.add(
        Input(
            shape=(
                X_train_scaled.shape[1],
            )
        )
    )

    if method == "Baseline":

        model.add(
            Dense(
                64,
                activation="sigmoid"
            )
        )

        model.add(
            Dense(
                32,
                activation="sigmoid"
            )
        )

        model.add(
            Dense(
                16,
                activation="sigmoid"
            )
        )

    elif method == "L1":

        model.add(
            Dense(
                64,
                activation="sigmoid",
                kernel_regularizer=l1(0.001)
            )
        )

        model.add(
            Dense(
                32,
                activation="sigmoid",
                kernel_regularizer=l1(0.001)
            )
        )

        model.add(
            Dense(
                16,
                activation="sigmoid",
                kernel_regularizer=l1(0.001)
            )
        )

    elif method == "L2":

        model.add(
            Dense(
                64,
                activation="sigmoid",
                kernel_regularizer=l2(0.001)
            )
        )

        model.add(
            Dense(
                32,
                activation="sigmoid",
                kernel_regularizer=l2(0.001)
            )
        )

        model.add(
            Dense(
                16,
                activation="sigmoid",
                kernel_regularizer=l2(0.001)
            )
        )

    elif method == "Dropout":

        model.add(
            Dense(
                64,
                activation="sigmoid"
            )
        )

        model.add(
            Dropout(0.20)
        )

        model.add(
            Dense(
                32,
                activation="sigmoid"
            )
        )

        model.add(
            Dropout(0.20)
        )

        model.add(
            Dense(
                16,
                activation="sigmoid"
            )
        )

    elif method == "EarlyStopping":

        model.add(
            Dense(
                64,
                activation="sigmoid"
            )
        )

        model.add(
            Dense(
                32,
                activation="sigmoid"
            )
        )

        model.add(
            Dense(
                16,
                activation="sigmoid"
            )
        )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer=RMSprop(),
        loss="mse"
    )

    callbacks = []

    if method == "EarlyStopping":

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )
        ]

    start_time = time.time()

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.20,
        epochs=125,
        callbacks=callbacks,
        verbose=0
    )

    training_time = (
        time.time() - start_time
    )

    y_pred = model.predict(
        X_test_scaled
    ).flatten()

    mae, rmse, adjr2 = (
        evaluate_model(
            y_test,
            y_pred,
            X.shape[1]
        )
    )

    generalization_gap = (
        history.history["val_loss"][-1]
        -
        history.history["loss"][-1]
    )

    results_df = pd.DataFrame({

        "Method": [method],

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
        ],

        "Generalization_Gap": [
            generalization_gap
        ]

    })

    results_df.to_csv(
        output_dir /
        f"{method.lower()}_results.csv",
        index=False
    )

    history_df = pd.DataFrame({

        "loss":
            history.history["loss"],

        "val_loss":
            history.history["val_loss"]

    })
    
    history_df.to_csv(
        output_dir /f"{method.lower()}_history.csv",
        index=False
    )
    
    # ==========================================
    # Learning Curve
    # ==========================================
    
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
        f"{method} Learning Curve"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        output_dir /
        f"{method.lower()}_learning_curve.png",
        bbox_inches="tight"
    )

    plt.show()
    plt.close()
    
    # ==========================================
    # Actual vs Predicted
    # ==========================================

    plt.figure(
        figsize=(8,8)
    )

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
        f"{method}: Actual vs Predicted"
    )

    plt.xlabel(
        "Actual Quality"
    )

    plt.ylabel(
        "Predicted Quality"
    )

    plt.savefig(
        output_dir /
        f"{method.lower()}_actual_vs_predicted.png",
        bbox_inches="tight"
    )

    plt.show()
    plt.close()
    
    all_results.append({

        "Method": method,

        "MAE": mae,

        "RMSE": rmse,

        "Adjusted_R2": adjr2,

        "Training_Time_Seconds":
            training_time,

        "Final_Training_Loss":
            history.history["loss"][-1],

        "Final_Validation_Loss":
            history.history["val_loss"][-1],

        "Generalization_Gap":
            generalization_gap

    })
    
    print(
        f"\n----- {method} RESULTS -----"
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
        f"Generalization Gap: "
        f"{generalization_gap:.4f}"
    )

    print(
        f"Results collected: "
        f"{len(all_results)}"
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
    "regularization_comparison.csv",
    index=False
)

print(
    "\n----- FINAL REGULARIZATION COMPARISON -----"
)

print(
    comparison
    .round(4)
    .to_string(index=False)
)

print(
    "\nAnalysis 7 Master Script Complete"
)
