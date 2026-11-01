"""
Author:     Aaron Niecestro
Project:    Deep Learning for Wine Quality (DLWQ)

Created:    November 1, 2026
Last Edit:  November 1, 2026
Progress:   Completed

Analysis:   02 - Feature Scaling Comparison
Method:     StandardScaler

Description:
This script evaluates neural network performance using standardized
feature values. Features are transformed to have a mean of zero and
a standard deviation of one before model training.

The resulting performance is compared against the No Scaling baseline
to determine whether feature standardization improves neural network
learning, convergence speed, and predictive accuracy.

Research Question:
Which feature scaling method produces the best neural network
performance for wine quality prediction?
"""

# ==========================================
# Analysis 2: Feature Scaling Comparison
# Method 2: StandardScaler
# ==========================================

# ------------------------------------------
# Imports
# ------------------------------------------

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
# StandardScaler
# ------------------------------------------

scaler = StandardScaler()

X_train_ss = scaler.fit_transform(X_train)

X_test_ss = scaler.transform(X_test)

# ------------------------------------------
# MLP Model
# ------------------------------------------

mlp_ss = Sequential([
    Input(shape=(X_train_ss.shape[1],)),
    Dense(16, activation="relu"),
    Dense(1)
])

mlp_ss.compile(
    optimizer="adam",
    loss="mse"
)

# ------------------------------------------
# Train Model
# ------------------------------------------

history_ss = mlp_ss.fit(
    X_train_ss,
    y_train,
    validation_split=0.20,
    epochs=100,
    verbose=0
)

# ------------------------------------------
# Predictions
# ------------------------------------------

y_pred_ss = mlp_ss.predict(
    X_test_ss
).flatten()

# ------------------------------------------
# Evaluation Functions
# ------------------------------------------

def adjusted_r2(r2, n, p):
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

def evaluate_model(y_true, y_pred, p):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    adj_r2 = adjusted_r2(
        r2,
        len(y_true),
        p
    )

    return mae, rmse, adj_r2

# ------------------------------------------
# Evaluation
# ------------------------------------------

mae_ss, rmse_ss, adjr2_ss = evaluate_model(
    y_test,
    y_pred_ss,
    X.shape[1]
)

# ------------------------------------------
# Results Table
# ------------------------------------------

results_scaling = pd.DataFrame({
    "Scaling_Method": ["StandardScaler"],
    "MAE": [mae_ss],
    "RMSE": [rmse_ss],
    "Adjusted_R2": [adjr2_ss],
    "Final_Training_Loss": [
        history_ss.history["loss"][-1]
    ],
    "Final_Validation_Loss": [
        history_ss.history["val_loss"][-1]
    ]
})

print("\n----- STANDARDSCALER RESULTS -----")
print(results_scaling.round(4))

# ------------------------------------------
# Export Results
# ------------------------------------------

results_scaling.to_csv(
    output_dir / "standardscaler_results.csv",
    index=False
)

# ------------------------------------------
# Learning Curve
# ------------------------------------------

plt.figure(figsize=(10,6))

plt.plot(
    history_ss.history["loss"],
    label="Training Loss"
)

plt.plot(
    history_ss.history["val_loss"],
    label="Validation Loss"
)

plt.title("StandardScaler Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    output_dir / "standardscaler_learning_curve.png",
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------
# Actual vs Predicted
# ------------------------------------------

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    y_pred_ss,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.title("StandardScaler: Actual vs Predicted")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")

plt.savefig(
    output_dir / "standardscaler_actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------
# Export Summary
# ------------------------------------------

with open(
    output_dir / "standardscaler_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("Analysis 2: Feature Scaling Comparison\n")
    f.write("Method: StandardScaler\n\n")

    f.write(f"MAE: {mae_ss:.4f}\n")
    f.write(f"RMSE: {rmse_ss:.4f}\n")
    f.write(f"Adjusted R²: {adjr2_ss:.4f}\n")

    f.write(
        f"\nTraining Loss: "
        f"{history_ss.history['loss'][-1]:.4f}\n"
    )

    f.write(
        f"Validation Loss: "
        f"{history_ss.history['val_loss'][-1]:.4f}\n"
    )

# ------------------------------------------
# Summary
# ------------------------------------------

print("\nAnalysis 2 - StandardScaler Complete")

print(f"MAE: {mae_ss:.4f}")
print(f"RMSE: {rmse_ss:.4f}")
print(f"Adjusted R²: {adjr2_ss:.4f}")

"""
## Summary

The objective of this analysis was to evaluate the effect of feature 
standardization on neural network performance.

All input features were transformed using StandardScaler, resulting in 
feature distributions with a mean of zero and a standard deviation of one. 
The same Multi-Layer Perceptron architecture, optimizer, training configuration, 
and evaluation methodology used in the No Scaling experiment were retained to 
ensure a fair comparison.

The StandardScaler model achieved:

- MAE: 0.5546
- RMSE: 0.7170
- Adjusted R²: 0.3294

Compared with the No Scaling baseline, StandardScaler reduced both MAE and RMSE 
while producing a noticeable improvement in Adjusted R². These results indicate 
that feature standardization improved the network's ability to learn 
relationships between wine characteristics and quality scores.

Training and validation loss curves decreased smoothly and remained closely 
aligned throughout training, indicating stable optimization and limited 
overfitting. Although convergence again occurred rapidly during the early 
training epochs, the standardized model consistently reached a lower loss 
level than the unscaled model.

Actual-versus-predicted visualizations showed tighter clustering around the 
ideal prediction line when compared with the No Scaling baseline. While 
extreme quality scores remained difficult to predict due to dataset imbalance, 
overall prediction accuracy improved across the majority of observations.

Overall, StandardScaler produced the strongest performance observed so far and 
became the leading candidate for the preferred scaling method.
"""

"""
# Key Insights

### 1. StandardScaler Improved Predictive Performance

StandardScaler outperformed the No Scaling baseline across all evaluation metrics.

| Metric | No Scaling | StandardScaler |
|----------|---------:|---------:|
| MAE | 0.5656 | 0.5546 |
| RMSE | 0.7334 | 0.7170 |
| Adjusted R² | 0.2983 | 0.3294 |

---

### 2. Feature Standardization Improved Learning Efficiency

Standardizing feature values reduced the scale differences between variables, 
allowing the neural network to learn more effectively and achieve lower training 
and validation losses.

---

### 3. Model Convergence Remained Stable

Training and validation loss curves remained closely aligned throughout training, 
indicating good generalization and minimal evidence of overfitting.

---

### 4. Prediction Accuracy Improved

Actual-versus-predicted results demonstrated tighter clustering around the ideal 
prediction line, suggesting improved predictive accuracy relative to the 
No Scaling baseline.

---

### 5. Dataset Imbalance Continues to Limit Extreme Predictions

Despite the performance improvement, the model still struggled to accurately 
predict rare quality scores due to the highly imbalanced target distribution.

---

### 6. StandardScaler Became the Leading Scaling Method

Based on lower error metrics and improved explanatory power, StandardScaler 
currently ranks ahead of No Scaling and serves as the benchmark to beat in the 
remaining scaling experiments.
"""
