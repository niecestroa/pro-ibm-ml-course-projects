"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 1, 2026
Last Edit:  November 1, 2026
Progress:   Complete

Analysis:   02 - Feature Scaling Comparison
Method:     MinMaxScaler

Description:
This script evaluates neural network performance using MinMaxScaler.

All numerical features are scaled to the range [0,1] before training.
The resulting model performance is compared against the No Scaling
and StandardScaler approaches to determine which feature scaling
strategy produces the strongest predictive performance.

Research Question:
Which feature scaling method produces the best neural network
performance for wine quality prediction?
"""

# ==========================================
# Analysis 2: Feature Scaling Comparison
# Method 3: MinMaxScaler
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
    r"C:\Users\aniec\Desktop\ibm-ml-project\deep-learning-rl\03-Feature-Scaling-Comparison\tables_figures"
)

output_dir.mkdir(parents=True, exist_ok=True)

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
# MinMaxScaler
# ------------------------------------------

scaler = MinMaxScaler()

X_train_mm = scaler.fit_transform(X_train)

X_test_mm = scaler.transform(X_test)

# ------------------------------------------
# MLP Model
# ------------------------------------------

mlp_mm = Sequential([
    Input(shape=(X_train_mm.shape[1],)),
    Dense(16, activation="relu"),
    Dense(1)
])

mlp_mm.compile(
    optimizer="adam",
    loss="mse"
)

# ------------------------------------------
# Train Model
# ------------------------------------------

history_mm = mlp_mm.fit(
    X_train_mm,
    y_train,
    validation_split=0.20,
    epochs=100,
    verbose=0
)

# ------------------------------------------
# Predictions
# ------------------------------------------

y_pred_mm = mlp_mm.predict(
    X_test_mm
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

mae_mm, rmse_mm, adjr2_mm = evaluate_model(
    y_test,
    y_pred_mm,
    X.shape[1]
)

# ------------------------------------------
# Results Table
# ------------------------------------------

results_scaling = pd.DataFrame({
    "Scaling_Method": ["MinMaxScaler"],
    "MAE": [mae_mm],
    "RMSE": [rmse_mm],
    "Adjusted_R2": [adjr2_mm],
    "Final_Training_Loss": [
        history_mm.history["loss"][-1]
    ],
    "Final_Validation_Loss": [
        history_mm.history["val_loss"][-1]
    ]
})

print("\n----- MINMAXSCALER RESULTS -----")
print(results_scaling.round(4))

# ------------------------------------------
# Export Results
# ------------------------------------------

results_scaling.to_csv(
    output_dir / "minmaxscaler_results.csv",
    index=False
)

# ------------------------------------------
# Learning Curve
# ------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    history_mm.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_mm.history["val_loss"],
    label="Validation Loss"
)

plt.title("MinMaxScaler Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    output_dir / "minmaxscaler_learning_curve.png",
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------
# Actual vs Predicted
# ------------------------------------------

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    y_pred_mm,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title("MinMaxScaler: Actual vs Predicted")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")

plt.savefig(
    output_dir / "minmaxscaler_actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------
# Export Summary
# ------------------------------------------

with open(
    output_dir / "minmaxscaler_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("Analysis 2: Feature Scaling Comparison\n")
    f.write("Method: MinMaxScaler\n\n")

    f.write(f"MAE: {mae_mm:.4f}\n")
    f.write(f"RMSE: {rmse_mm:.4f}\n")
    f.write(f"Adjusted R²: {adjr2_mm:.4f}\n")

    f.write(
        f"\nTraining Loss: "
        f"{history_mm.history['loss'][-1]:.4f}\n"
    )

    f.write(
        f"Validation Loss: "
        f"{history_mm.history['val_loss'][-1]:.4f}\n"
    )

# ------------------------------------------
# Summary
# ------------------------------------------

print("\nAnalysis 2 - MinMaxScaler Complete")

print(f"MAE: {mae_mm:.4f}")
print(f"RMSE: {rmse_mm:.4f}")
print(f"Adjusted R²: {adjr2_mm:.4f}")

"""
## Summary

The objective of this analysis was to evaluate the impact of MinMaxScaler
on neural network performance for wine quality prediction.

All input features were transformed to a range of 0 to 1 prior to model
training while maintaining the same neural network architecture,
optimizer, train-test split, and training configuration used in the
previous scaling experiments.

The model achieved:

- MAE: 0.5519
- RMSE: 0.7094
- Adjusted R²: 0.3435

Training and validation losses decreased rapidly during the early epochs
before stabilizing, indicating efficient learning and stable convergence.
The close alignment between training and validation losses suggests that
the model generalized well with minimal evidence of overfitting.

Actual-versus-predicted visualizations demonstrated improved prediction
accuracy relative to both the No Scaling and StandardScaler approaches.
Predictions remained most accurate for wines with quality scores between
5 and 7, while extreme quality ratings continued to be more difficult
to predict due to the imbalanced target distribution.

Overall, MinMaxScaler produced the strongest predictive performance
observed thus far and became the leading scaling method in the study.

### Key Results

| Metric | Value |
|----------|--------:|
| MAE | 0.5519 |
| RMSE | 0.7094 |
| Adjusted R² | 0.3435 |
| Training Loss | 0.5062 |
| Validation Loss | 0.4864 |
"""

"""
## Key Insights

### 1. MinMaxScaler Produced the Best Overall Performance

MinMaxScaler achieved the lowest MAE and RMSE values while producing the
highest Adjusted R² score among all scaling methods tested.

### 2. Feature Scaling Improved Neural Network Performance

Both StandardScaler and MinMaxScaler outperformed the No Scaling
baseline, demonstrating the importance of feature scaling for neural
network training on structured tabular data.

### 3. MinMaxScaler Outperformed StandardScaler

Although StandardScaler significantly improved predictive performance,
MinMaxScaler achieved additional reductions in prediction error and
provided the strongest overall results.

### 4. Model Convergence Remained Stable

Training and validation losses decreased rapidly during the early epochs
before stabilizing. The learning curves remained closely aligned
throughout training.

### 5. Evidence of Overfitting Remained Minimal

The small gap between training and validation losses suggests that the
model generalized well and did not exhibit substantial overfitting.

### 6. Prediction Accuracy Improved Across Most Quality Levels

Actual-versus-predicted results showed tighter clustering around the
ideal prediction line when compared with both previous scaling methods.

### 7. Extreme Quality Ratings Remain Challenging

Despite overall improvements in performance, the model continued to
struggle with very low and very high quality wines because these ratings
were underrepresented within the dataset.

### 8. MinMaxScaler Was Selected for Future Analyses

Based on lower prediction error, higher explanatory power, and stable
training behavior, MinMaxScaler was selected as the preferred scaling
method for all subsequent experiments.
"""
