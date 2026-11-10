"""
Author: Aaron Niecestro
Project: Course Recommender System Capstone

Created: November 4, 2026
Last Edit: November 4, 2026
Progress: Complete

Analysis: Embedding-Based Rating Prediction
Method: Regression Modeling Using User and Course
        Embedding Features

Description:

Predicts user-course rating scores using embedding
features extracted from user and course representations.

User and course embeddings are combined into a single
feature vector and used as inputs for supervised
regression models. Multiple regression algorithms are
evaluated and compared to determine which model most
accurately predicts user ratings.

Techniques Applied:

- Feature Engineering
- Embedding Vector Aggregation
- Train-Test Split Validation
- 5-Fold Cross Validation
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Regression Performance Evaluation
- Adjusted R² Evaluation

Datasets:

- ratings.csv
- user_embeddings.csv
- course_embeddings.csv

Research Questions:

- Can embedding features predict course ratings?
- Which regression model performs best?
- Are model results stable across multiple folds?
- Do regularized regression techniques improve
  prediction performance?
- Which model generalizes best to unseen data?

Outputs:

- regression_results.csv
- model_comparison.csv
- best_model_predictions.csv
- regression_model_comparison.png
- regression_model_comparison_adjusted_r2.png
- actual_vs_predicted.png
- residual_distribution.png
"""

# ============================================================
# Imports
# ============================================================

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# ============================================================
# Reproducibility
# ============================================================

SEED = 423

random.seed(SEED)
np.random.seed(SEED)

RANDOM_STATE = SEED

# ============================================================
# Paths
# ============================================================

DATA_DIR = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\capstone\00-Data"
)

OUTPUT_DIR = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\capstone\03-Supervised\supervised-results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# Load Data
# ============================================================

rating_df = pd.read_csv(
    DATA_DIR / "ratings.csv"
)

user_emb = pd.read_csv(
    DATA_DIR / "user_embeddings.csv"
)

item_emb = pd.read_csv(
    DATA_DIR / "course_embeddings.csv"
)

print("\nDatasets Loaded Successfully")
print("-" * 40)

print(f"Ratings Shape           : {rating_df.shape}")
print(f"User Embeddings Shape   : {user_emb.shape}")
print(f"Course Embeddings Shape : {item_emb.shape}")

print(
    f"Unique Users   : "
    f"{rating_df['user'].nunique():,}"
)

print(
    f"Unique Courses : "
    f"{rating_df['item'].nunique():,}"
)

print(
    f"Interactions   : "
    f"{len(rating_df):,}"
)

# ============================================================
# Create Regression Dataset
# ============================================================

user_emb_merged = pd.merge(
    rating_df,
    user_emb,
    how="left",
    on="user"
).fillna(0)

merged_df = pd.merge(
    user_emb_merged,
    item_emb,
    how="left",
    on="item"
).fillna(0)

u_features = [
    f"UFeature{i}"
    for i in range(16)
]

c_features = [
    f"CFeature{i}"
    for i in range(16)
]

user_embeddings = merged_df[u_features]
course_embeddings = merged_df[c_features]

ratings = merged_df["rating"]

regression_dataset = (
    user_embeddings +
    course_embeddings.values
)

regression_dataset.columns = [
    f"Feature{i}"
    for i in range(16)
]

regression_dataset["rating"] = ratings

print(
    f"\nRegression Dataset Shape: "
    f"{regression_dataset.shape}"
)

# ============================================================
# Prepare Features and Labels
# ============================================================

X = regression_dataset.iloc[:, :-1]

y = regression_dataset.iloc[:, -1]

print(
    f"Input Shape : {X.shape}"
)

print(
    f"Output Shape: {y.shape}"
)

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE
    )
)

print("\nTrain/Test Split Complete")
print("-" * 40)

print(
    f"Training Samples: "
    f"{len(X_train):,}"
)

print(
    f"Testing Samples : "
    f"{len(X_test):,}"
)

# ============================================================
# Train Regression Models
# ============================================================

models = {
    "Linear Regression":
        LinearRegression(),

    "Ridge":
        Ridge(alpha=1.0),

    "Lasso":
        Lasso(
            alpha=0.001,
            random_state=RANDOM_STATE
        ),

    "Elastic Net":
        ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            random_state=RANDOM_STATE
        )
}
    
# ============================================================
# Model Evaluation
# ============================================================

results = []

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )
    
    n = len(y_test)

    p = X_test.shape[1]

    r2 = (
        1 -
        (
            np.sum(
                (y_test - predictions) ** 2
            )
            /
            np.sum(
                (y_test - y_test.mean()) ** 2
            )
        )
    )

    adjusted_r2 = (
        1 -
        (
            (1 - r2)
            * (n - 1)
            / (n - p - 1)
        )
    )

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    cv_rmse = np.sqrt(
        -cv_scores
    )

    cv_rmse_mean = (
        cv_rmse.mean()
    )

    cv_rmse_std = (
        cv_rmse.std()
    )

    results.append([
        name,
        mae,
        rmse,
        adjusted_r2,
        cv_rmse_mean,
        cv_rmse_std
    ])

    print(
        f"{name:<20}"
        f"MAE={mae:.4f} | "
        f"RMSE={rmse:.4f} | "
        f"Adj R²={adjusted_r2:.4f} | "
        f"CV RMSE={cv_rmse_mean:.4f}"
    )

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "Adjusted_R2",
        "CV_RMSE",
        "CV_STD"
    ]
).sort_values(
    by="CV_RMSE"
)

best_model_name = (
    results_df.iloc[0]["Model"]
)

print(
    f"\nBest Model Selected: "
    f"{best_model_name}"
)

best_model = models[
    best_model_name
]

best_predictions = (
    best_model.predict(X_test)
)

predictions_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": best_predictions
})

print("\nModel Comparison")
print("-" * 40)

print(results_df)

print("\nCross-Validation Summary")
print("-" * 40)

print(
    results_df[
        [
            "Model",
            "CV_RMSE",
            "CV_STD"
        ]
    ]
)

# ============================================================
# Visualization 1
# Regression Model Comparison
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="CV_RMSE",
    y="Model",
    palette="viridis"
)

plt.title(
    "Cross-Validated Regression Model Comparison"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "regression_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Visualization 2
# Adjusted R² Comparison
# ============================================================

plt.figure(figsize=(10,6))

sns.barplot(
    data=results_df,
    x="Adjusted_R2",
    y="Model",
    palette="viridis"
)

plt.title(
    "Regression Model Comparison (Adjusted R²)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "regression_model_comparison_adjusted_r2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Visualization 3
# Actual vs Predicted Ratings
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--"
)

plt.title(
    "Actual vs Predicted Ratings"
)

plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Visualization 4
# Prediction Error Distribution
# ============================================================

residuals = (
    y_test -
    best_predictions
)

plt.figure(figsize=(8, 6))

sns.histplot(
    residuals,
    bins=20,
    kde=True
)

plt.title(
    "Residual Distribution"
)

plt.xlabel(
    "Prediction Error"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "residual_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Save Results
# ============================================================

results_df.to_csv(
    OUTPUT_DIR /
    "regression_results.csv",
    index=False
)

predictions_df.to_csv(
    OUTPUT_DIR /
    "best_model_predictions.csv",
    index=False
)

results_df.to_csv(
    OUTPUT_DIR /
    "model_comparison.csv",
    index=False
)

print("\nResults Saved")
print("-" * 40)

print("regression_results.csv")
print("model_comparison.csv")
print("best_model_predictions.csv")
print("regression_model_comparison.png")
print("regression_model_comparison_adjusted_r2.png")
print("actual_vs_predicted.png")
print("residual_distribution.png")

print(
    f"\nResults saved to:\n{OUTPUT_DIR}"
)

# ============================================================
# Final Summary
# ============================================================

best_model = (
    results_df.iloc[0]
)

print("\n" + "=" * 50)
print("REGRESSION MODEL SUMMARY")
print("=" * 50)

print(
    f"Best Model               : "
    f"{best_model_name}"
)

print(
    f"Best MAE                 : "
    f"{best_model['MAE']:.4f}"
)

print(
    f"Best RMSE                : "
    f"{best_model['RMSE']:.4f}"
)

print(
    f"Best CV RMSE             : "
    f"{best_model['CV_RMSE']:.4f}"
)

print(
    f"CV RMSE Std Dev          : "
    f"{best_model['CV_STD']:.4f}"
)

print(
    f"Best Adjusted R²         : "
    f"{best_model['Adjusted_R2']:.4f}"
)

print(
    f"Training Samples         : "
    f"{len(X_train):,}"
)

print(
    f"Testing Samples          : "
    f"{len(X_test):,}"
)

print("\nModel Ranking")
print("-" * 40)

for i, row in enumerate(
    results_df.itertuples(),
    start=1
):
    print(
        f"{i}. "
        f"{row.Model} "
        f"(CV RMSE={row.CV_RMSE:.4f})"
    )

print(
    "\nEmbedding-Based Regression "
    "Analysis Complete."
)

