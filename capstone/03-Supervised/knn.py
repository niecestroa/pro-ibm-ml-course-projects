"""
Author: Aaron Niecestro
Project: Course Recommender System Capstone

Created: November 4, 2026
Last Edit: November 4, 2026
Progress: Complete

Analysis: K-Nearest Neighbors Collaborative Filtering
Method: Item-Based Collaborative Filtering
        Using Course Similarity Analysis

Description:

Builds collaborative filtering recommendation systems
using K-Nearest Neighbors (KNN) and user-course
interaction data.

User-item interaction matrices are constructed and
similarity measurements are calculated between users
and items. Recommendations and rating predictions are
generated using neighboring users or similar courses.

Multiple neighborhood sizes and similarity metrics
are evaluated to determine the optimal recommendation
configuration and minimize prediction error.

Techniques Applied:

- User-Item Matrix Construction
- Collaborative Filtering
- Item Similarity Analysis
- Cosine Similarity
- Mean Squared Difference (MSD)
- Pearson Correlation
- K-Nearest Neighbors (KNN)
- Hyperparameter Optimization
- Train-Test Split Validation
- RMSE Evaluation
- Recommendation System Performance Analysis

Datasets:

- ratings.csv

Research Questions:

- Can collaborative filtering accurately predict
  user-course ratings?

- What K value produces the best recommendation
  performance?

- Which item similarity metric produces the
  lowest prediction error?

- Which similarity metric produces the lowest
  prediction error?

- How does prediction accuracy change as K
  increases?

- What neighborhood size minimizes prediction
  error?

Outputs:

- knn_results.csv
- model_comparison.csv
- best_model_predictions.csv
- knn_rmse_comparison.png
- similarity_comparison.png

Evaluation Metrics:

- RMSE
- K Value Comparison
- Similarity Metric Comparison
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

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    mean_squared_error
)

from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances
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
    f"Training Ratings : {len(train_df):,}"
)

print(
    f"Testing Ratings  : {len(test_df):,}"
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
# Item Similarity Matrices
# ============================================================

item_matrix = train_matrix.T

# Cosine

item_cosine_df = pd.DataFrame(
    cosine_similarity(
        item_matrix
    ),
    index=item_matrix.index,
    columns=item_matrix.index
)

print(
    "\nItem Cosine Similarity Matrix Created"
)

# Pearson

item_pearson_df = pd.DataFrame(
    np.corrcoef(item_matrix),
    index=item_matrix.index,
    columns=item_matrix.index
)

item_pearson_df = (
    item_pearson_df
    .fillna(0)
)

print(
    "Item Pearson Similarity Matrix Created"
)

# MSD

item_distance = euclidean_distances(
    item_matrix
)

item_msd_df = pd.DataFrame(
    1 / (1 + item_distance),
    index=item_matrix.index,
    columns=item_matrix.index
)

print(
    "Item MSD Similarity Matrix Created"
)

# ============================================================
# Matrix Validation
# ============================================================

print("\nSimilarity Matrix Shapes")
print("-" * 40)

print(
    f"Item Cosine  : {item_cosine_df.shape}"
)

print(
    f"Item Pearson : {item_pearson_df.shape}"
)

print(
    f"Item MSD     : {item_msd_df.shape}"
)

# ============================================================
# Item-Based Prediction Function
# ============================================================

def predict_item_rating(
    user_id,
    item_id,
    similarity_df,
    train_matrix,
    global_mean,
    k
):

    if user_id not in train_matrix.index:
        return global_mean

    if item_id not in similarity_df.index:
        return global_mean

    user_ratings = train_matrix.loc[user_id]

    rated_items = user_ratings[
        user_ratings > 0
    ]

    if len(rated_items) == 0:
        return global_mean

    similarities = similarity_df.loc[
        item_id,
        rated_items.index
    ]

    neighbors = (
        similarities
        .sort_values(
            ascending=False
        )
        .head(k)
    )

    weighted_sum = np.sum(
        neighbors.values *
        rated_items.loc[
            neighbors.index
        ].values
    )

    similarity_sum = np.sum(
        np.abs(neighbors.values)
    )

    if similarity_sum == 0:
        return global_mean

    return (
        weighted_sum /
        similarity_sum
    )

# ============================================================
# Baseline Statistics
# ============================================================

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
# Results Storage
# ============================================================

results = []

# ============================================================
# Cosine Similarity Search
# ============================================================

print("\nCosine Similarity Search for K")
print("-" * 40)

for k in range(2, 41):

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

        prediction = predict_item_rating(
            user_id=row.user,
            item_id=row.item,
            similarity_df=item_cosine_df,
            train_matrix=train_matrix,
            global_mean=global_mean,
            k=k
        )

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
        "Item-Based",
        "Cosine",
        k,
        rmse
    ])

    print(
        f"Cosine | "
        f"K={k:<2} | "
        f"RMSE={rmse:.4f}"
    )

# ============================================================
# Pearson Similarity Search
# ============================================================

print("\nPearson Similarity Search for K")
print("-" * 40)

for k in range(2, 41):

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

        prediction = predict_item_rating(
            user_id=row.user,
            item_id=row.item,
            similarity_df=item_pearson_df,
            train_matrix=train_matrix,
            global_mean=global_mean,
            k=k
        )

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
        "Item-Based",
        "Pearson",
        k,
        rmse
    ])

    print(
        f"Pearson | "
        f"K={k:<2} | "
        f"RMSE={rmse:.4f}"
    )

# ============================================================
# MSD Similarity Search
# ============================================================

print("\nMSD Similarity Search for K")
print("-" * 40)

for k in range(2, 41):

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

        prediction = predict_item_rating(
            user_id=row.user,
            item_id=row.item,
            similarity_df=item_msd_df,
            train_matrix=train_matrix,
            global_mean=global_mean,
            k=k
        )

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
        "Item-Based",
        "MSD",
        k,
        rmse
    ])

    print(
        f"MSD | "
        f"K={k:<2} | "
        f"RMSE={rmse:.4f}"
    )

# ============================================================
# Results DataFrame
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Method",
        "Similarity",
        "K",
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

print("\nTop Configurations")
print("-" * 40)

print(
    results_df.head(10)
)

# ============================================================
# Best Configuration
# ============================================================

best_result = results_df.iloc[0]

best_method = best_result["Method"]
best_similarity = best_result["Similarity"]
best_k = best_result["K"]
best_rmse = best_result["RMSE"]

print("\nBest Configuration")
print("-" * 40)

print(
    f"Method      : {best_method}"
)

print(
    f"Similarity  : {best_similarity}"
)

print(
    f"K           : {best_k}"
)

print(
    f"RMSE        : {best_rmse:.4f}"
)

# ============================================================
# Save Results
# ============================================================

results_df.to_csv(
    OUTPUT_DIR /
    "knn_results.csv",
    index=False
)

results_df.to_csv(
    OUTPUT_DIR /
    "model_comparison.csv",
    index=False
)

print("\nResults Saved")
print("-" * 40)

print("knn_results.csv")
print("model_comparison.csv")

# ============================================================
# Visualization 1
# RMSE vs K by Similarity Metric
# ============================================================

SIMILARITY_COLORS = {
    "MSD": "#1f77b4",      # Blue
    "Cosine": "#ff7f0e",   # Orange
    "Pearson": "#2ca02c"   # Green
}

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=results_df,
    x="K",
    y="RMSE",
    hue="Similarity",
    palette=SIMILARITY_COLORS,
    marker="o"
)

plt.title(
    "RMSE by K Value and Similarity Metric"
)

plt.xlabel(
    "K Value"
)

plt.ylabel(
    "RMSE"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "knn_rmse_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Visualization 2
# Best RMSE by Similarity
# ============================================================

plt.figure(figsize=(8, 5))

best_similarity_df = (
    results_df
    .groupby("Similarity")["RMSE"]
    .min()
    .reset_index()
    .rename(
        columns={
            "RMSE": "Best_RMSE"
        }
    )
)

sns.barplot(
    data=best_similarity_df,
    x="Similarity",
    y="Best_RMSE",
    palette=SIMILARITY_COLORS
)

plt.title(
    "Best RMSE by Similarity Metric"
)

plt.xlabel(
    "Similarity Metric"
)

plt.ylabel(
    "Best RMSE"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "similarity_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Visualization 3
# Top 10 Configurations
# ============================================================

top10 = results_df.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top10,
    x="K",
    y="RMSE",
    color=SIMILARITY_COLORS["MSD"]
)

plt.title(
    "Top 10 MSD Configurations"
)

plt.xlabel("K Value")
plt.ylabel("RMSE")

plt.tight_layout()

plt.show()

# ============================================================
# Improvement Analysis
# ============================================================

best_cosine = (
    results_df[
        results_df["Similarity"] == "Cosine"
    ]["RMSE"].min()
)

best_pearson = (
    results_df[
        results_df["Similarity"] == "Pearson"
    ]["RMSE"].min()
)

best_msd = (
    results_df[
        results_df["Similarity"] == "MSD"
    ]["RMSE"].min()
)

msd_vs_cosine = (
    (best_cosine - best_msd)
    / best_cosine
) * 100

msd_vs_pearson = (
    (best_pearson - best_msd)
    / best_pearson
) * 100

print("\nImprovement Analysis")
print("-" * 40)

print(
    f"MSD vs Cosine Improvement : "
    f"{msd_vs_cosine:.2f}%"
)

print(
    f"MSD vs Pearson Improvement : "
    f"{msd_vs_pearson:.2f}%"
)

# ============================================================
# Final Summary Table
# ============================================================

summary_df = pd.DataFrame({
    "Metric": [
        "Best Method",
        "Best Similarity",
        "Best K",
        "Best RMSE",
        "MSD Improvement vs Cosine (%)",
        "MSD Improvement vs Pearson (%)",
        "Total Configurations Tested",
        "Number of Users",
        "Number of Courses",
        "Number of Ratings"
    ],
    "Value": [
        best_method,
        best_similarity,
        best_k,
        round(best_rmse, 4),
        round(msd_vs_cosine, 2),
        round(msd_vs_pearson, 2),
        len(results_df),
        rating_df["user"].nunique(),
        rating_df["item"].nunique(),
        len(rating_df)
    ]
})

# ============================================================
# Similarity Performance Summary
# ============================================================

similarity_summary = (
    results_df
    .groupby("Similarity")
    .agg(
        Best_K=("K", lambda x:
            results_df.loc[
                results_df.loc[x.index, "RMSE"].idxmin(),
                "K"
            ]
        ),
        Best_RMSE=("RMSE", "min")
    )
    .reset_index()
    .sort_values(
        "Best_RMSE"
    )
)

print("\nSimilarity Performance Summary")
print("-" * 40)

print(
    similarity_summary.to_string(
        index=False
    )
)

# ============================================================
# Best Result by Similarity Metric
# ============================================================

similarity_summary = (
    results_df
    .groupby("Similarity")
    .agg(
        Best_RMSE=("RMSE", "min")
    )
    .reset_index()
)

print("\nSimilarity Performance")
print("-" * 40)

print(
    similarity_summary
    .sort_values("Best_RMSE")
    .to_string(index=False)
)
