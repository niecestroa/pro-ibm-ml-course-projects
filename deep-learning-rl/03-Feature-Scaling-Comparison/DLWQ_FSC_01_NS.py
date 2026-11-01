"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 1, 2026
Last Edit:  November 1, 2026
Progress:   Completed

Analysis:   02 - Feature Scaling Comparison
Method:     No Scaling

Description:
This script evaluates baseline neural network performance using
unscaled feature values and serves as the benchmark model for the
Feature Scaling Comparison study.

The cleaned wine quality dataset is loaded, categorical wine type
information is encoded, and a stratified train-test split is performed
to preserve the distribution of wine quality ratings.

A Multi-Layer Perceptron (MLP) consisting of a single hidden layer
with 16 ReLU neurons is trained using the Adam optimizer and Mean
Squared Error (MSE) loss.

Model performance is evaluated using:

    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Adjusted R²

Additional outputs include:

    - Model comparison table
    - Learning curve visualization
    - Actual vs. Predicted visualization
    - Exported performance metrics
    - Experiment summary

Purpose:
Establish a no-scaling baseline against which StandardScaler and
MinMaxScaler results will be compared in subsequent analyses.

Research Question:
Which feature scaling method produces the best neural network
performance for wine quality prediction?
"""

# ==========================================
# Analysis 2: Feature Scaling Comparison
# Method 1: No Scaling
# ==========================================

# ------------------------------------------
# Imports
# ------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
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
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\03-Feature-Scaling-Comparison\tables_figures"
)

output_dir.mkdir(parents=True, exist_ok=True)

dpwq = pd.read_csv(file_path)

dpwq.columns = dpwq.columns.str.strip()

print("----- DATA SHAPE (MASTER) -----")
print(f"Rows: {dpwq.shape[0]:,}")
print(f"Columns: {dpwq.shape[1]}")

print("\n----- FIRST FIVE ROWS -----")
print(dpwq.head())

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

print("\n----- FEATURE MATRIX AFTER ENCODING -----")
print(X.shape)

print("\n----- DATA TYPES AFTER ENCODING -----")
print(X.dtypes)

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

print("\n----- TRAIN/TEST SPLIT -----")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\n----- TRAIN DISTRIBUTION -----")
print(y_train.value_counts().sort_index())

print("\n----- TEST DISTRIBUTION -----")
print(y_test.value_counts().sort_index())

# ------------------------------------------
# No Scaling Data
# ------------------------------------------

X_train_ns = X_train.copy()
X_test_ns = X_test.copy()

# ------------------------------------------
# MLP Model
# ------------------------------------------

mlp_ns = Sequential([
    Input(shape=(X_train_ns.shape[1],)),
    Dense(16, activation="relu"),
    Dense(1)
])

mlp_ns.compile(
    optimizer="adam",
    loss="mse"
)

# ------------------------------------------
# Train Model
# ------------------------------------------

history_ns = mlp_ns.fit(
    X_train_ns,
    y_train,
    validation_split=0.20,
    epochs=100,
    verbose=0
)

# ------------------------------------------
# Predictions
# ------------------------------------------

y_pred_ns = mlp_ns.predict(X_test_ns).flatten()

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

mae_ns, rmse_ns, adjr2_ns = evaluate_model(
    y_test,
    y_pred_ns,
    X.shape[1]
)

# ------------------------------------------
# Results Table
# ------------------------------------------

results_scaling = pd.DataFrame({
    "Scaling_Method": ["No Scaling"],
    "MAE": [mae_ns],
    "RMSE": [rmse_ns],
    "Adjusted_R2": [adjr2_ns],
    "Final_Training_Loss": [
        history_ns.history["loss"][-1]
    ],
    "Final_Validation_Loss": [
        history_ns.history["val_loss"][-1]
    ]
})

print("\n----- NO SCALING RESULTS -----")
print(results_scaling.round(4))

# ------------------------------------------
# Export Results
# ------------------------------------------

results_scaling.to_csv(
    output_dir / "no_scaling_results.csv",
    index=False
)

# ------------------------------------------
# Learning Curve
# ------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    history_ns.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_ns.history["val_loss"],
    label="Validation Loss"
)

plt.title("No Scaling Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    output_dir / "no_scaling_learning_curve.png",
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------
# Actual vs Predicted
# ------------------------------------------

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    y_pred_ns,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title("No Scaling: Actual vs Predicted")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")

plt.savefig(
    output_dir / "no_scaling_actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------
# Export Summary
# ------------------------------------------

with open(
    output_dir / "no_scaling_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("Analysis 2: Feature Scaling Comparison\n")
    f.write("Method: No Scaling\n\n")

    f.write(f"MAE: {mae_ns:.4f}\n")
    f.write(f"RMSE: {rmse_ns:.4f}\n")
    f.write(f"Adjusted R²: {adjr2_ns:.4f}\n")

    f.write(
        f"\nTraining Loss: "
        f"{history_ns.history['loss'][-1]:.4f}\n"
    )

    f.write(
        f"Validation Loss: "
        f"{history_ns.history['val_loss'][-1]:.4f}\n"
    )

# ------------------------------------------
# Summary
# ------------------------------------------

print("\nAnalysis 2 - No Scaling Complete")

print(f"MAE: {mae_ns:.4f}")
print(f"RMSE: {rmse_ns:.4f}")
print(f"Adjusted R²: {adjr2_ns:.4f}")

"""
## Summary

The objective of this analysis was to evaluate baseline MLP performance using 
unscaled feature values and establish a benchmark for future scaling experiments.

The model achieved:

- MAE: 0.5656
- RMSE: 0.7334
- Adjusted R²: 0.2983

Observation:

The model converged within approximately the first 5–10 epochs and
showed little improvement thereafter. However, 100 epochs will be
maintained throughout the scaling study to ensure a fair comparison
between scaling methods.

The learning curve showed rapid convergence during the first few epochs, 
followed by a stable plateau for the remainder of training. Training and 
validation losses remained closely aligned, indicating stable learning and 
minimal evidence of overfitting.

Actual-versus-predicted visualizations showed that the model was most accurate 
for wines with quality scores between 5 and 7, which represented the majority 
of observations in the dataset. Predictions for extreme quality values were 
less accurate, reflecting the highly imbalanced distribution of quality ratings.

Overall, the baseline MLP demonstrated consistent predictive performance without 
feature scaling and established a benchmark against which StandardScaler and 
MinMaxScaler results will be evaluated. Any future performance improvements 
can therefore be attributed to the impact of feature scaling rather than 
changes in architecture or training methodology.

### Key Results

| Metric | Value |
|----------|--------:|
| MAE | 0.5656 |
| RMSE | 0.7334 |
| Adjusted R² | 0.2983 |
| Training Loss | 0.5906 |
| Validation Loss | 0.5144 |

### Conclusion

The baseline MLP trained on unscaled features provided stable performance and 
reasonable predictive accuracy. This model serves as the reference point for 
the Feature Scaling Comparison study, where StandardScaler and MinMaxScaler 
will be evaluated to determine whether scaled inputs improve neural network 
performance.
"""

"""
# Key Insights

### 1. The Baseline MLP Produced Stable Performance

The MLP achieved:

- MAE: 0.5656
- RMSE: 0.7334
- Adjusted R²: 0.2983

These results established the benchmark for evaluating the impact of feature 
scaling.

---

### 2. Training Converged Quickly

Training and validation losses decreased rapidly during the first several 
epochs before stabilizing.

Most learning occurred early in training, suggesting efficient optimization 
by the Adam optimizer.

---

### 3. Minimal Evidence of Overfitting

Training and validation loss curves remained closely aligned throughout training.

The absence of a widening gap between the curves suggests good generalization 
and limited overfitting.

---

### 4. Predictions Were Most Accurate for Common Quality Scores

The model produced its strongest predictions for wines with quality scores 
between 5 and 7, which represented the majority of observations within the dataset.

---

### 5. Extreme Quality Ratings Remained Difficult to Predict

The model consistently overestimated low-quality wines and underestimated 
high-quality wines.

This pattern was likely influenced by the imbalanced target distribution, 
where very few wines received quality scores of 3, 8, or 9.

---

### 6. Prediction Compression Was Observed

Predicted quality values were concentrated within a relatively narrow range.

The model rarely predicted the most extreme quality scores, causing 
predictions to regress toward the center of the distribution.

---

### 7. No Scaling Established the Baseline for the Scaling Study

The unscaled model provides a reference point against which StandardScaler 
and MinMaxScaler can be evaluated.

Any future improvements can therefore be attributed directly to feature 
scaling rather than changes in architecture or training methodology.
"""