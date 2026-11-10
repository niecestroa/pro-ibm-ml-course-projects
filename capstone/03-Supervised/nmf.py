"""
Author: Aaron Niecestro
Project: Course Recommender System Capstone

Created: November 4, 2026
Last Edit: November 4, 2026
Progress: Complete

Analysis: Non-Negative Matrix Factorization (NMF)

Method:
Latent Factor Collaborative Filtering
Using Matrix Decomposition

Description:

Builds a recommendation system using
Non-Negative Matrix Factorization (NMF).

The user-item interaction matrix is decomposed
into lower-dimensional latent user and item
feature matrices. Predicted ratings are generated
through matrix reconstruction and evaluated using
Root Mean Squared Error (RMSE).

Multiple latent factor dimensions are evaluated
to determine the optimal matrix factorization
configuration.

Techniques Applied:

- Collaborative Filtering
- Matrix Factorization
- Dimensionality Reduction
- Non-Negative Matrix Factorization
- Latent Feature Learning
- Hyperparameter Optimization
- Train-Test Split Validation
- RMSE Evaluation
- Recommendation System Analysis

Datasets:

- ratings.csv

Research Questions:

- Can matrix factorization accurately predict
  user-course ratings?

- What number of latent factors produces the
  best recommendation performance?

- Can latent feature learning reduce
  dimensionality while maintaining accuracy?

- How does recommendation accuracy change as
  the number of latent factors increases?

- What latent dimension minimizes RMSE?

Outputs:

- nmf_results.csv
- model_comparison.csv
- nmf_rmse_comparison.png
- latent_factor_comparison.png

Evaluation Metrics:

- RMSE
- Latent Factor Comparison
- Model Ranking
"""

# ============================================================
# Imports
# ============================================================

import random
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.decomposition import NMF

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    mean_squared_error
)

warnings.filterwarnings("ignore")

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

print("\nDataset Loaded Successfully")
print("-" * 40)

print(
    f"Dataset Shape : "
    f"{rating_df.shape}"
)

# ============================================================
# Dataset Summary
# ============================================================

print("\nDataset Summary")
print("-" * 40)

print(
    f"Total Ratings : "
    f"{len(rating_df):,}"
)

print(
    f"Unique Users : "
    f"{rating_df['user'].nunique():,}"
)

print(
    f"Unique Courses : "
    f"{rating_df['item'].nunique():,}"
)

print("\nRating Distribution")
print("-" * 40)

print(
    rating_df["rating"]
    .value_counts()
    .sort_index()
)

# ============================================================
# Train-Test Split
# ============================================================

train_df, test_df = train_test_split(
    rating_df,
    test_size=0.30,
    random_state=RANDOM_STATE
)

print("\nTrain/Test Split")
print("-" * 40)

print(
    f"Training Ratings : "
    f"{len(train_df):,}"
)

print(
    f"Testing Ratings  : "
    f"{len(test_df):,}"
)

# ============================================================
# Training User-Item Matrix
# ============================================================

train_matrix = (
    train_df.pivot_table(
        index="user",
        columns="item",
        values="rating",
        fill_value=0
    )
)

print("\nTraining Matrix")
print("-" * 40)

print(
    f"Matrix Shape : "
    f"{train_matrix.shape}"
)

# ============================================================
# NMF Hyperparameter Search
# ============================================================

latent_factors = [
    4,
    8,
    16,
    24,
    32,
    48,
    64
]

results = []

global_mean = (
    train_df["rating"]
    .mean()
)

print("\nGlobal Mean Rating")
print("-" * 40)

print(
    f"{global_mean:.4f}"
)

# ============================================================
# Baseline RMSE
# ============================================================

sample_test = test_df.sample(
    n=min(5000, len(test_df)),
    random_state=RANDOM_STATE
)

baseline_predictions = [
    global_mean
] * len(sample_test)

baseline_rmse = np.sqrt(
    mean_squared_error(
        sample_test["rating"],
        baseline_predictions
    )
)

print("\nBaseline RMSE")
print("-" * 40)

print(
    f"{baseline_rmse:.4f}"
)

# ============================================================
# NMF Evaluation
# ============================================================

print("\nNMF Evaluation")
print("-" * 40)

for n_components in latent_factors:

    nmf_model = NMF(
        n_components=n_components,
        init="random",
        random_state=RANDOM_STATE,
        max_iter=500
    )

    W = nmf_model.fit_transform(
        train_matrix
    )

    H = nmf_model.components_

    reconstructed_matrix = np.dot(
        W,
        H
    )

    reconstructed_df = pd.DataFrame(
        reconstructed_matrix,
        index=train_matrix.index,
        columns=train_matrix.columns
    )

    predictions = []
    actuals = []

    sample_test = test_df.sample(
        n=min(
            5000,
            len(test_df)
        ),
        random_state=RANDOM_STATE
    )

    for row in sample_test.itertuples():

        if (
            row.user in reconstructed_df.index
            and
            row.item in reconstructed_df.columns
        ):

            prediction = (
                reconstructed_df.loc[
                    row.user,
                    row.item
                ]
            )

        else:

            prediction = global_mean

        predictions.append(
            prediction
        )

        actuals.append(
            row.rating
        )

    rmse = np.sqrt(
        mean_squared_error(
            actuals,
            predictions
        )
    )

    results.append([
        n_components,
        rmse
    ])

    print(
        f"Latent Factors = "
        f"{n_components:<2} | "
        f"RMSE = {rmse:.4f}"
    )
    
    print(
        f"Min Prediction : "
        f"{reconstructed_df.min().min():.4f}"
    )

    print(
        f"Max Prediction : "
        f"{reconstructed_df.max().max():.4f}"
    )

# ============================================================
# Results DataFrame
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Latent_Factors",
        "RMSE"
    ]
)

results_df = (
    results_df
    .sort_values(
        by="RMSE",
        ascending=True
    )
)

print("\nNMF Results")
print("-" * 40)

print(
    results_df
)

# ============================================================
# Best Configuration
# ============================================================

best_result = (
    results_df.iloc[0]
)

best_factors = int(
    best_result["Latent_Factors"]
)

best_rmse = (
    best_result["RMSE"]
)

print("\nBest Configuration")
print("-" * 40)

print(
    f"Latent Factors : "
    f"{best_factors}"
)

print(
    f"RMSE           : "
    f"{best_rmse:.4f}"
)

# ============================================================
# Matrix Sparsity Analysis
# ============================================================

total_cells = (
    train_matrix.shape[0]
    * train_matrix.shape[1]
)

non_zero_cells = (
    train_matrix != 0
).sum().sum()

sparsity = (
    1 -
    (
        non_zero_cells /
        total_cells
    )
) * 100

print("\nMatrix Sparsity")
print("-" * 40)

print(
    f"Total Cells      : "
    f"{total_cells:,}"
)

print(
    f"Observed Ratings : "
    f"{non_zero_cells:,}"
)

print(
    f"Sparsity (%)     : "
    f"{sparsity:.2f}"
)

# ============================================================
# Results DataFrame
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Latent_Factors",
        "RMSE"
    ]
)

results_df = (
    results_df
    .sort_values(
        by="RMSE",
        ascending=True
    )
)

print("\nNMF Results")
print("-" * 40)

print(results_df)

# ============================================================
# Baseline Comparison
# ============================================================

rmse_difference = (
    best_rmse -
    baseline_rmse
)

performance_gap = (
    (best_rmse - baseline_rmse)
    /
    baseline_rmse
) * 100

print("\nBaseline Comparison")
print("-" * 40)

print(
    f"Baseline RMSE : "
    f"{baseline_rmse:.4f}"
)

print(
    f"NMF RMSE      : "
    f"{best_rmse:.4f}"
)

print(
    f"Difference    : "
    f"{rmse_difference:.4f}"
)

print(
    f"Gap (%)       : "
    f"{performance_gap:.2f}%"
)

# ============================================================
# Visualization 1
# RMSE vs Latent Factors
# ============================================================

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=results_df,
    x="Latent_Factors",
    y="RMSE",
    marker="o",
    color="#1f77b4"
)

plt.title(
    "NMF RMSE by Number of Latent Factors"
)

plt.xlabel(
    "Latent Factors"
)

plt.ylabel(
    "RMSE"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "nmf_rmse_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Visualization 2
# NMF Configuration Comparison
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="Latent_Factors",
    y="RMSE",
    color="#1f77b4"
)

plt.title(
    "NMF Performance by Latent Dimension"
)

plt.xlabel(
    "Latent Factors"
)

plt.ylabel(
    "RMSE"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "latent_factor_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Visualization 3
# NMF vs Baseline
# ============================================================

comparison_df = pd.DataFrame({
    "Model": [
        "Baseline",
        "Best NMF"
    ],
    "RMSE": [
        baseline_rmse,
        best_rmse
    ]
})

plt.figure(figsize=(8, 5))

sns.barplot(
    data=comparison_df,
    x="Model",
    y="RMSE",
    palette=[
        "#ff7f0e",
        "#1f77b4"
    ]
)

plt.title(
    "Baseline vs NMF"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "RMSE"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "nmf_baseline_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Final Summary Table
# ============================================================

summary_df = pd.DataFrame({
    "Metric": [
        "Best Latent Factors",
        "Best RMSE",
        "Baseline RMSE",
        "Matrix Sparsity (%)",
        "Users",
        "Courses",
        "Ratings"
    ],
    "Value": [
        int(best_factors),
        round(best_rmse, 4),
        round(baseline_rmse, 4),
        round(sparsity, 2),
        f"{rating_df['user'].nunique():,}",
        f"{rating_df['item'].nunique():,}",
        f"{len(rating_df):,}"
    ]
})

print("\nFinal Summary")
print("-" * 40)

print(
    summary_df.to_string(
        index=False
    )
)

print("\nFinding")
print("-" * 40)

print(
    "Standard NMF struggled with the "
    "95.8% sparse user-item matrix and "
    "performed substantially worse than "
    "the baseline predictor."
)

results_df.to_csv(
    OUTPUT_DIR /
    "nmf_results.csv",
    index=False
)

"""
Best Configuration
• Latent Factors: 4
• RMSE: 3.2763

Key Findings
• User-item matrix sparsity reached 95.8%.
• NMF performance deteriorated as latent factors increased.
• Standard NMF was unable to effectively model the sparse interaction matrix.
• Baseline RMSE (0.8187) substantially outperformed NMF RMSE (3.2763).
• Item-Based KNN with MSD similarity remained the strongest recommendation approach.
"""

"""
Findings

The user-item matrix exhibited 95.8% sparsity.
Standard Non-Negative Matrix Factorization struggled to reconstruct user preferences effectively.
Increasing latent factors increased prediction error.
The best NMF configuration used 4 latent factors and achieved an RMSE of 3.2763.
NMF underperformed the baseline predictor (RMSE = 0.8187).
Matrix factorization was less effective than item-based KNN collaborative filtering for this dataset.

Conclusion

NMF was sensitive to the extreme sparsity of the interaction matrix.
Additional latent factors did not improve recommendation quality.
Item-Based KNN with MSD similarity remained the strongest recommendation approach evaluated in the project.
"""