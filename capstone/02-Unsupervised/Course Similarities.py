# -*- coding: utf-8 -*-

"""
Author: Aaron Niecestro
Project: Course Recommender System Capstone

Created: November 4, 2026
Last Edit: November 9, 2026
Progress: Complete

Analysis: Course Similarity-Based Recommender System
Method: Content-Based Filtering Using Precomputed
        Course Similarity Matrix

Description:

Generates personalized course recommendations using
precomputed course similarity scores derived from
Bag-of-Words course representations.

For each user, enrolled courses are compared against
unseen courses using a course similarity matrix.
Recommendations are generated when similarity scores
exceed a specified threshold and are ranked according
to similarity strength.

Techniques Applied:

- Bag-of-Words Feature Representation
- Course Similarity Matrix
- Content-Based Filtering
- Similarity Threshold Filtering
- Recommendation Frequency Analysis

Datasets:

- sim.csv
- course_processed.csv
- courses_bows.csv
- ratings.csv

Research Questions:

- Can course similarity be used to generate meaningful
  recommendations?
- Which courses are recommended most frequently?
- How many recommendations are produced per user?
- What similarity scores characterize recommended
  courses?
- Which courses demonstrate the strongest content-based
  similarity relationships?

Outputs:

- similarity_recommendations.csv
- top_recommended_courses.csv
- Recommendation analytics
- Recommendation visualizations
"""

# ============================================================
# Imports
# ============================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

SIMILARITY_THRESHOLD = 0.60

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
# Load Data
# ============================================================

sim_df = pd.read_csv(
    DATA_DIR / "sim.csv"
)

course_df = pd.read_csv(
    DATA_DIR / "course_processed.csv"
)

bow_df = pd.read_csv(
    DATA_DIR / "courses_bows.csv"
)

ratings_df = pd.read_csv(
    DATA_DIR / "ratings.csv"
)

sim_matrix = sim_df.to_numpy()

print("\nDatasets Loaded Successfully")
print("-" * 40)

print(f"Similarity Matrix : {sim_df.shape}")
print(f"Course Data       : {course_df.shape}")
print(f"BoW Features      : {bow_df.shape}")
print(f"Ratings Data      : {ratings_df.shape}")

# ============================================================
# Create Course ID ↔ Index Mappings
# ============================================================

def get_doc_dicts(bow_df):
    """
    Create dictionaries that map course IDs
    to similarity matrix indices and vice versa.
    """

    grouped_df = (
        bow_df
        .groupby(
            ["doc_index", "doc_id"]
        )
        .max()
        .reset_index(drop=False)
    )

    idx_id_dict = (
        grouped_df[["doc_id"]]
        .to_dict()["doc_id"]
    )

    id_idx_dict = {
        v: k
        for k, v in idx_id_dict.items()
    }

    return idx_id_dict, id_idx_dict


idx_id_dict, id_idx_dict = get_doc_dicts(
    bow_df
)

print("\nCourse Mappings Created")
print("-" * 40)

print(
    f"Mapped Courses: {len(id_idx_dict)}"
)

# ============================================================
# Recommendation Engine
# ============================================================

def generate_recommendations_for_one_user(
    enrolled_course_ids,
    unselected_course_ids,
    id_idx_dict,
    sim_matrix,
    threshold=SIMILARITY_THRESHOLD
):
    """
    Generate course recommendations for
    a single user based on course similarity.
    """

    recommendations = {}

    for enrolled_course in enrolled_course_ids:

        for candidate_course in unselected_course_ids:

            if (
                enrolled_course in id_idx_dict
                and candidate_course in id_idx_dict
            ):

                enrolled_idx = (
                    id_idx_dict[
                        enrolled_course
                    ]
                )

                candidate_idx = (
                    id_idx_dict[
                        candidate_course
                    ]
                )

                similarity = sim_matrix[
                    enrolled_idx
                ][
                    candidate_idx
                ]

                if similarity > threshold:

                    if (
                        candidate_course
                        not in recommendations
                    ):

                        recommendations[
                            candidate_course
                        ] = similarity

                    else:

                        recommendations[
                            candidate_course
                        ] = max(
                            recommendations[
                                candidate_course
                            ],
                            similarity
                        )

    recommendations = dict(
        sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    return recommendations

# ============================================================
# Generate Recommendations For All Users
# ============================================================

def generate_recommendations_for_all(
    ratings_df,
    course_df,
    id_idx_dict,
    sim_matrix,
    threshold=SIMILARITY_THRESHOLD
):

    users = []
    courses = []
    scores = []

    all_courses = set(
        course_df["COURSE_ID"]
    )

    user_ids = (
        ratings_df["user"]
        .unique()
    )

    for user_id in user_ids:

        enrolled_courses = (
            ratings_df[
                ratings_df["user"]
                == user_id
            ]["item"]
            .tolist()
        )

        unselected_courses = (
            all_courses.difference(
                enrolled_courses
            )
        )

        recommendations = (
            generate_recommendations_for_one_user(
                enrolled_courses,
                unselected_courses,
                id_idx_dict,
                sim_matrix,
                threshold
            )
        )

        for (
            course,
            score
        ) in recommendations.items():

            users.append(user_id)
            courses.append(course)
            scores.append(score)

    return users, courses, scores

# ============================================================
# Generate Recommendation Results
# ============================================================

users, courses, scores = (
    generate_recommendations_for_all(
        ratings_df,
        course_df,
        id_idx_dict,
        sim_matrix
    )
)

res_df = pd.DataFrame(
    {
        "USER": users,
        "COURSE_ID": courses,
        "SCORE": scores
    }
)

print("\nRecommendation Sample")
print("-" * 40)

print(
    res_df.head()
)

print(
    f"\nTotal Recommendations: "
    f"{len(res_df):,}"
)

# ============================================================
# Recommendation Analytics
# ============================================================

total_users = (
    res_df["USER"]
    .nunique()
)

total_recommendations = (
    len(res_df)
)

avg_recommendations = (
    res_df
    .groupby("USER")
    .size()
    .mean()
)

print("\nRecommendation Analytics")
print("-" * 40)

print(
    f"Users Evaluated: "
    f"{total_users}"
)

print(
    f"Total Recommendations: "
    f"{total_recommendations}"
)

print(
    f"Average Recommendations "
    f"Per User: "
    f"{avg_recommendations:.2f}"
)

# ============================================================
# Top Recommended Courses
# ============================================================

top_courses = (
    res_df["COURSE_ID"]
    .value_counts()
    .head(10)
)

print("\nTop 10 Recommended Courses")
print("-" * 40)

print(top_courses)

# ============================================================
# Course Information Lookup
# ============================================================

course_lookup = dict(
    zip(
        course_df["COURSE_ID"],
        course_df["TITLE"]
    )
)

top_course_df = pd.DataFrame(
    {
        "COURSE_ID": top_courses.index,
        "COUNT": top_courses.values
    }
)

top_course_df["TITLE"] = (
    top_course_df["COURSE_ID"]
    .map(course_lookup)
)

print("\nTop Recommended Course Details")
print("-" * 40)

print(top_course_df)

# ============================================================
# Recommendation Score Statistics
# ============================================================

score_stats = (
    res_df["SCORE"]
    .describe()
)

print("\nRecommendation Score Statistics")
print("-" * 40)

print(score_stats)

# ============================================================
# Visualization 1
# Top Recommended Courses
# ============================================================

colors = ["#0B3C6D"] + ["#5B9BD5"] * 9

plt.figure(figsize=(12, 6))

ax = sns.barplot(
    data=top_course_df,
    x="COUNT",
    y="TITLE",
    palette=colors
)

# Add value labels
for i, value in enumerate(top_course_df["COUNT"]):
    ax.text(
        value + 100,
        i,
        f"{value:,}",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

plt.xlim(
    0,
    top_course_df["COUNT"].max() * 1.08
)

plt.title(
    "Top 10 Most Frequently Recommended Courses",
    fontsize=14,
)

plt.xlabel(
    "Recommendation Count",
    fontsize=11
)

plt.ylabel(
    "Course",
    fontsize=11
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "top_recommended_courses.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Visualization 2
# Similarity Score Distribution
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    res_df["SCORE"],
    bins=20,
    kde=True,
    color="green"
)

plt.title(
    "Similarity Score Distribution"
)

plt.xlabel("Similarity Score")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "similarity_score_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Visualization 3
# Recommendations Per User
# ============================================================

recommendations_per_user = (
    res_df
    .groupby("USER")
    .size()
)

plt.figure(figsize=(10, 6))

sns.histplot(
    recommendations_per_user,
    bins=20,
    color="orange"
)

plt.title(
    "Recommendations Per User"
)

plt.xlabel(
    "Number of Recommendations"
)

plt.ylabel("Users")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "recommendations_per_user.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Visualization 4
# Highest Similarity Recommendations
# ============================================================

top_scores = (
    res_df
    .sort_values(
        by="SCORE",
        ascending=False
    )
    .head(20)
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=top_scores,
    x="SCORE",
    y="COURSE_ID",
    palette="viridis"
)

plt.title(
    "Top 20 Highest Similarity Recommendations"
)

plt.xlabel("Similarity Score")
plt.ylabel("Course ID")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "highest_similarity_recommendations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Save Results
# ============================================================

res_df.to_csv(
    OUTPUT_DIR /
    "similarity_recommendations.csv",
    index=False
)

top_course_df.to_csv(
    OUTPUT_DIR /
    "top_recommended_courses.csv",
    index=False
)

score_stats.to_csv(
    OUTPUT_DIR /
    "recommendation_score_statistics.csv"
)

print(
    f"\nResults saved to: "
    f"{OUTPUT_DIR}"
)

# ============================================================
# Final Summary
# ============================================================

print("\n" + "=" * 50)
print("COURSE SIMILARITY RECOMMENDER SUMMARY")
print("=" * 50)

print(
    f"Users Evaluated            : {total_users}"
)

print(
    f"Total Recommendations      : {total_recommendations}"
)

print(
    f"Average Recommendations    : "
    f"{avg_recommendations:.2f}"
)

print(
    f"Average Similarity Score   : "
    f"{res_df['SCORE'].mean():.4f}"
)

print(
    f"Maximum Similarity Score   : "
    f"{res_df['SCORE'].max():.4f}"
)

print(
    f"Minimum Similarity Score   : "
    f"{res_df['SCORE'].min():.4f}"
)

print(
    "\nCourse Similarity Recommendation "
    "Analysis Complete."
)
