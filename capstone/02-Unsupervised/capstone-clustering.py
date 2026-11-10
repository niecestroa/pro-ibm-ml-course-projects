# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro
Project: Course Recommender System Capstone

Created: November 3, 2026
Last Edit: November 9, 2026
Progress: Complete

Analysis: 02 - Clustering-Based Recommender System
Method: K-Means Clustering + Principal Component Analysis (PCA)

Description:

Groups users according to their learning interests and
generates personalized course recommendations based on
popular courses within each cluster.

User profile features are standardized before applying
K-Means clustering. Silhouette analysis is used to
determine the optimal number of clusters, while PCA is
used to reduce dimensionality and improve visualization
and model efficiency.

Techniques Applied:

- Feature Standardization
- K-Means Clustering
- Silhouette Analysis
- Principal Component Analysis (PCA)
- Cluster-Based Recommendation Engine

Datasets:

- user_profile.csv
- ratings.csv
- course_genre.csv

Key Parameters:

- Silhouette-Based Cluster Selection
- PCA Variance Threshold = 90%
- Popularity Threshold = 10 Enrollments

Research Questions:

- Can users be grouped based on learning interests?
- What is the optimal number of clusters?
- How much dimensionality reduction can PCA achieve?
- Can cluster enrollments be used to generate meaningful
  course recommendations?

Outputs:

- User cluster assignments
- Silhouette analysis results
- PCA-transformed features
- Cluster recommendation results
- Recommendation evaluation metrics
"""

# ============================================================
# Clustering-Based Course Recommender System
# IBM Machine Learning Capstone
#
# Author: Aaron Niecestro
#
# Objective:
# - Cluster users based on learning interests
# - Reduce dimensions with PCA
# - Generate cluster-based recommendations
# - Analyze recommendation performance
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from pathlib import Path

# ============================================================
# Reproducibility
# ============================================================

import random

SEED = 423

random.seed(SEED)
np.random.seed(SEED)

RANDOM_STATE = SEED

# ============================================================
# Configuration
# ============================================================

MAX_CLUSTERS = 15
PCA_VARIANCE_THRESHOLD = 0.90
POPULARITY_THRESHOLD = 10

sns.set_style("whitegrid")

# ============================================================
# Paths
# ============================================================

DATA_DIR = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\capstone\00-Data"
)

OUTPUT_DIR = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\capstone\02-Unsupervised\unsupervised-results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# Runtime Tracking
# ============================================================

import time

SCRIPT_START = time.time()

# ============================================================
# Load Data
# ============================================================

print("\nLoading Data...")
print("-" * 40)

user_profile_df = pd.read_csv(
    DATA_DIR / "user_profile.csv"
)

ratings_df = pd.read_csv(
    DATA_DIR / "ratings.csv"
)

course_genres_df = pd.read_csv(
    DATA_DIR / "course_genre.csv"
)

print("Data Loaded Successfully")

print(
    f"User Profiles Shape : "
    f"{user_profile_df.shape}"
)

print(
    f"Ratings Shape       : "
    f"{ratings_df.shape}"
)

print(
    f"Course Genres Shape : "
    f"{course_genres_df.shape}"
)

# ============================================================
# Prepare Features
# ============================================================

print("\nPreparing Features...")
print("-" * 40)

feature_names = list(user_profile_df.columns[1:])

scaler = StandardScaler()

user_profile_df[feature_names] = scaler.fit_transform(
    user_profile_df[feature_names]
)

features = user_profile_df.drop(
    columns=["user"]
)

user_ids = user_profile_df[["user"]]

print(
    f"Total Users    : {len(user_profile_df):,}"
)

print(
    f"Total Features : {len(feature_names)}"
)

print("Feature Preparation Complete")

# ============================================================
# K-Means Optimization
# ============================================================

print("\nOptimizing K-Means Clusters...")
print("-" * 40)

k_values = range(2, MAX_CLUSTERS + 1)

inertias = []
silhouette_scores = []

for k in k_values:

    print(f"\nEvaluating K = {k}")

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = model.fit_predict(features)

    inertias.append(model.inertia_)

    sample_idx = (
        features.sample(
            min(5000, len(features)),
            random_state=RANDOM_STATE
        )
        .index
    )

    score = silhouette_score(
        features.loc[sample_idx],
        labels[sample_idx]
    )

    silhouette_scores.append(score)

    print(
        f"   Inertia = {model.inertia_:,.2f} | "
        f"Silhouette = {score:.4f}"
    )

best_k = k_values[
    np.argmax(silhouette_scores)
]

print("\n" + "-" * 40)
print(f"Optimal Clusters Found : {best_k}")
print(
    f"Best Silhouette Score  : "
    f"{max(silhouette_scores):.4f}"
)
print("K-Means Optimization Complete")

# ============================================================
# Elbow + Silhouette Plot
# ============================================================

print("\nGenerating K-Means Evaluation Plots...")
print("-" * 40)

fig, ax = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

# ------------------------------------------------------------
# Elbow Plot
# ------------------------------------------------------------

ax[0].plot(
    k_values,
    inertias,
    marker="o"
)

ax[0].axvline(
    x=best_k,
    color="red",
    linestyle="--",
    label=f"Best K = {best_k}"
)

ax[0].scatter(
    best_k,
    inertias[best_k - 2],
    color="red",
    s=100,
    zorder=5
)

ax[0].set_title(
    "Elbow Method"
)

ax[0].set_xlabel(
    "Number of Clusters (K)"
)

ax[0].set_ylabel(
    "Inertia"
)

ax[0].legend()

# ------------------------------------------------------------
# Silhouette Plot
# ------------------------------------------------------------

ax[1].plot(
    k_values,
    silhouette_scores,
    marker="o",
    color="green"
)

ax[1].axvline(
    x=best_k,
    color="red",
    linestyle="--",
    label=f"Best K = {best_k}"
)

ax[1].scatter(
    best_k,
    max(silhouette_scores),
    color="red",
    s=100,
    zorder=5
)

ax[1].set_title(
    "Silhouette Scores"
)

ax[1].set_xlabel(
    "Number of Clusters (K)"
)

ax[1].set_ylabel(
    "Silhouette Score"
)

ax[1].legend()

plt.tight_layout()
plt.show()

print("K-Means Evaluation Plots Complete")

# ============================================================
# Train Final K-Means Model
# ============================================================

print("\nTraining Final K-Means Model...")
print("-" * 40)

kmeans = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=10
)

cluster_labels = kmeans.fit_predict(
    features
)

cluster_df = pd.DataFrame({
    "user": user_ids["user"],
    "cluster": cluster_labels
})

print(
    f"Users Assigned to Clusters : "
    f"{len(cluster_df):,}"
)

print(
    f"Clusters Created           : "
    f"{cluster_df['cluster'].nunique()}"
)

cluster_sizes = (
    cluster_df["cluster"]
    .value_counts()
    .sort_index()
)

print("\nCluster Sizes")
print("-" * 40)

for cluster_id, count in cluster_sizes.items():

    print(
        f"Cluster {cluster_id:<2}: "
        f"{count:>6,} users"
    )

print("\nCluster Assignment Sample")
print(cluster_df.head())

print("Final K-Means Model Complete")

# ============================================================
# PCA Optimization
# ============================================================

print("\nRunning PCA Optimization...")
print("-" * 40)

variance_explained = []

for n in range(
    1,
    features.shape[1] + 1
):

    pca = PCA(
        n_components=n
    )

    pca.fit(features)

    variance_explained.append(
        np.sum(
            pca.explained_variance_ratio_
        )
    )

best_components = next(
    i + 1
    for i, var in enumerate(
        variance_explained
    )
    if var >= PCA_VARIANCE_THRESHOLD
)

print(
    f"Original Features          : "
    f"{features.shape[1]}"
)

print(
    f"Selected Components        : "
    f"{best_components}"
)

print(
    f"Cumulative Variance Kept   : "
    f"{variance_explained[best_components - 1]:.2%}"
)

print("PCA Optimization Complete")

# ============================================================
# PCA Variance Plot
# ============================================================

print("\nGenerating PCA Variance Plot...")
print("-" * 40)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(
        1,
        features.shape[1] + 1
    ),
    variance_explained,
    marker="o"
)

# 90% variance threshold

plt.axhline(
    y=PCA_VARIANCE_THRESHOLD,
    color="red",
    linestyle="--",
    label="90% Variance Threshold"
)

# Selected component count

plt.axvline(
    x=best_components,
    color="green",
    linestyle="--",
    label=f"Selected Components = {best_components}"
)

plt.scatter(
    best_components,
    variance_explained[best_components - 1],
    color="green",
    s=100,
    zorder=5
)

plt.xlabel(
    "Number of Components"
)

plt.ylabel(
    "Cumulative Explained Variance"
)

plt.title(
    "PCA Optimization"
)

plt.legend()

plt.tight_layout()

plt.show()

print(
    f"Variance Explained : "
    f"{variance_explained[best_components - 1]:.2%}"
)

print("PCA Variance Plot Complete")

# ============================================================
# PCA Transformation
# ============================================================

print("\nTransforming Features with PCA...")
print("-" * 40)

pca = PCA(
    n_components=best_components
)

pca_components = pca.fit_transform(
    features
)

pca_df = pd.DataFrame(
    pca_components
)

pca_df.insert(
    0,
    "user",
    user_ids["user"]
)

print(
    f"Original Dimensions  : "
    f"{features.shape[1]}"
)

print(
    f"PCA Components       : "
    f"{best_components}"
)

print(
    f"Dimensions Reduced   : "
    f"{features.shape[1] - best_components}"
)

print(
    f"Reduction Percentage : "
    f"{((features.shape[1] - best_components) / features.shape[1]) * 100:.2f}%"
)

print("PCA Transformation Complete")

# ============================================================
# PCA-Based K-Means
# ============================================================

print("\nRunning K-Means on PCA Components...")
print("-" * 40)

kmeans_pca = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=10
)

pca_cluster_labels = kmeans_pca.fit_predict(
    pca_components
)

pca_silhouette = silhouette_score(
    pca_components,
    pca_cluster_labels
)

print(
    f"PCA Silhouette Score : "
    f"{pca_silhouette:.4f}"
)

pca_cluster_df = pd.DataFrame({
    "user": user_ids["user"],
    "cluster": pca_cluster_labels
})

print(
    f"Users Assigned to Clusters : "
    f"{len(pca_cluster_df):,}"
)

print(
    f"Clusters Created           : "
    f"{pca_cluster_df['cluster'].nunique()}"
)

pca_cluster_sizes = (
    pca_cluster_df["cluster"]
    .value_counts()
    .sort_index()
)

print("\nPCA Cluster Sizes")
print("-" * 40)

for cluster_id, count in pca_cluster_sizes.items():

    print(
        f"Cluster {cluster_id:<2}: "
        f"{count:>6,} users"
    )

print("\nPCA Cluster Assignment Sample")
print(
    pca_cluster_df.head()
)

print("PCA-Based K-Means Complete")

# ============================================================
# Cluster-Based Recommendation Engine
# ============================================================

print("\nGenerating Recommendations...")
print("-" * 40)

test_users_df = ratings_df[
    ["user", "item"]
]

test_users_labelled = pd.merge(
    test_users_df,
    pca_cluster_df,
    on="user"
)

course_counts = (
    test_users_labelled
    .groupby(
        ["cluster", "item"]
    )
    .size()
    .reset_index(
        name="enrollments"
    )
)

recommended_users = []
recommended_courses = []
recommendation_scores = []

unique_users = (
    test_users_labelled["user"]
    .unique()
)

print(
    f"Users Evaluated : "
    f"{len(unique_users):,}"
)

for user_id in unique_users:

    user_subset = (
        test_users_labelled[
            test_users_labelled["user"]
            == user_id
        ]
    )

    enrolled_courses = set(
        user_subset["item"]
    )

    cluster_id = (
        user_subset["cluster"]
        .iloc[0]
    )

    popular_courses = (
        course_counts[
            (
                course_counts["cluster"]
                == cluster_id
            )
            &
            (
                course_counts[
                    "enrollments"
                ]
                >= POPULARITY_THRESHOLD
            )
        ]
    )

    for _, row in popular_courses.iterrows():

        course = row["item"]
        count = row["enrollments"]

        if course not in enrolled_courses:

            recommended_users.append(
                user_id
            )

            recommended_courses.append(
                course
            )

            recommendation_scores.append(
                count
            )

recommendation_df = pd.DataFrame({

    "USER":
    recommended_users,

    "COURSE_ID":
    recommended_courses,

    "SCORE":
    recommendation_scores

})

print(
    f"Recommendations Generated : "
    f"{len(recommendation_df):,}"
)

print(
    f"Unique Courses Recommended : "
    f"{recommendation_df['COURSE_ID'].nunique():,}"
)

print(
    f"Users Receiving Recommendations : "
    f"{recommendation_df['USER'].nunique():,}"
)

print("Recommendation Generation Complete")

# ============================================================
# Analytics
# ============================================================

print("\nRunning Recommendation Analytics...")
print("-" * 40)

avg_recommendations = (
    recommendation_df
    .groupby("USER")
    .size()
    .mean()
)

top_courses = (
    recommendation_df["COURSE_ID"]
    .value_counts()
    .head(10)
)

course_lookup = dict(
    zip(
        course_genres_df["COURSE_ID"],
        course_genres_df["TITLE"]
    )
)

print(
    f"Users Receiving Recommendations : "
    f"{recommendation_df['USER'].nunique():,}"
)

print(
    f"Unique Courses Recommended      : "
    f"{recommendation_df['COURSE_ID'].nunique():,}"
)

print(
    f"Total Recommendations Generated : "
    f"{len(recommendation_df):,}"
)

print(
    f"Average Recommendations/User    : "
    f"{avg_recommendations:.2f}"
)

print(
    f"Average Recommendation Score    : "
    f"{recommendation_df['SCORE'].mean():.2f}"
)

print("\nTop 10 Recommended Courses")
print("-" * 100)

for course_id, count in top_courses.items():

    course_title = course_lookup.get(
        course_id,
        "Unknown Course"
    )

    print(
        f"{course_id:<12} "
        f"{count:>8,}   "
        f"{course_title}"
    )

print("\nRecommendation Analytics Complete")

# ============================================================
# Visualization
# ============================================================

print("\nGenerating Visualizations...")
print("-" * 40)

plot_df = pd.DataFrame({
    "Course_ID": top_courses.index,
    "Count": top_courses.values
})

plot_df["Course"] = (
    plot_df["Course_ID"]
    .map(course_lookup)
)

colors = ["#1f77b4"] + ["#7daed3"] * 9

plt.figure(figsize=(12, 6))

ax = sns.barplot(
    data=plot_df,
    x="Count",
    y="Course",
    palette=colors
)

# Add data labels
for i, value in enumerate(plot_df["Count"]):
    ax.text(
        value + 25,
        i,
        f"{value:,}",
        va="center"
    )

# Give labels room on right side
plt.xlim(
    0,
    plot_df["Count"].max() * 1.06
)

plt.title(
    "Top 10 Recommended Courses",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel(
    "Recommendation Count"
)

plt.ylabel(
    "Course"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "top_recommended_courses.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"Courses Displayed        : "
    f"{len(top_courses)}"
)

print(
    f"Most Recommended Course  : "
    f"{top_courses.index[0]}"
)

print(
    f"Recommendation Count     : "
    f"{top_courses.iloc[0]:,}"
)

print("Visualization Complete")

# ============================================================
# Export Results
# ============================================================

print("\nSaving Results...")
print("-" * 40)

cluster_df.to_csv(
    OUTPUT_DIR / "user_clusters.csv",
    index=False
)

pca_df.to_csv(
    OUTPUT_DIR / "pca_features.csv",
    index=False
)

recommendation_df.to_csv(
    OUTPUT_DIR / "cluster_recommendations.csv",
    index=False
)

print(
    f"Cluster Assignments Saved : "
    f"{len(cluster_df):,} records"
)

print(
    f"PCA Features Saved        : "
    f"{len(pca_df):,} records"
)

print(
    f"Recommendations Saved     : "
    f"{len(recommendation_df):,} records"
)

print(
    f"\nResults saved to:\n{OUTPUT_DIR}"
)

print("Export Complete")

# ============================================================
# Final Summary
# ============================================================

total_runtime = time.time() - SCRIPT_START

print("\n==============================")
print("FINAL SUMMARY")
print("==============================")

print(
    f"Users Clustered             : "
    f"{len(user_profile_df):,}"
)

print(
    f"Optimal Clusters            : "
    f"{best_k}"
)

print(
    f"PCA Components Retained     : "
    f"{best_components}"
)

print(
    f"Recommendations Generated   : "
    f"{len(recommendation_df):,}"
)

print(
    f"Unique Courses Recommended  : "
    f"{recommendation_df['COURSE_ID'].nunique():,}"
)

print(
    f"Average Recommendations/User: "
    f"{avg_recommendations:.2f}"
)

print(
    f"Average Recommendation Score: "
    f"{recommendation_df['SCORE'].mean():.2f}"
)

print(
    f"Total Runtime               : "
    f"{total_runtime / 60:.2f} minutes"
)

print("\nPipeline Complete")

