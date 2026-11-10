# -*- coding: utf-8 -*-

"""
Author: Aaron Niecestro
Project: Course Recommender System Capstone

Created: November 4, 2026
Last Edit: November 9, 2026
Progress: Complete

Analysis: User Profile-Based Recommender System
Method: Content-Based Filtering Using User Profiles
        and Course Genres

Description:

Builds personalized course recommendations using
user profile vectors and course genre features.

User interests are represented as weighted genre
vectors derived from historical course enrollments.
Recommendation scores are calculated using the dot
product between user profile vectors and course
genre vectors. Courses are then ranked according
to predicted user interest and filtered using a
recommendation score threshold.

Techniques Applied:

- User Profile Construction
- Genre-Based Feature Engineering
- Dot Product Similarity
- Recommendation Score Generation
- Content-Based Filtering

Datasets:

- user_profile.csv
- course_genre.csv
- ratings.csv

Research Questions:

- Can user interests be represented using genre
  preference vectors?
- Can user profile vectors generate meaningful
  personalized recommendations?
- Which courses receive the highest recommendation
  scores?
- How many recommendations are generated per user?
- What courses are most frequently recommended
  across users?

Outputs:

- profile_rs_results.csv
- User recommendation scores
- Recommended course lists
- Recommendation frequency statistics
"""

# ============================================================
# Imports
# ============================================================

import numpy as np
import pandas as pd
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

course_genres_df = pd.read_csv(
    DATA_DIR / "course_genre.csv"
)

profile_df = pd.read_csv(
    DATA_DIR / "user_profile.csv"
)

ratings_df = pd.read_csv(
    DATA_DIR / "ratings.csv"
)

print("\nDatasets Loaded Successfully")
print("-" * 40)

print(
    f"Course Genres Shape : {course_genres_df.shape}"
)

print(
    f"User Profiles Shape : {profile_df.shape}"
)

print(
    f"Ratings Shape       : {ratings_df.shape}"
)

# ============================================================
# Prepare User and Course Data
# ============================================================

# Unique users in the ratings dataset
test_users = (
    ratings_df
    .groupby("user")
    .max()
    .reset_index(drop=False)
)

test_user_ids = test_users["user"].tolist()

print("\nRecommendation Dataset Summary")
print("-" * 40)

print(f"Total Users   : {len(test_user_ids)}")

all_courses = set(
    course_genres_df["COURSE_ID"].values
)

print(f"Total Courses : {len(all_courses)}")

# ============================================================
# Recommendation Parameters
# ============================================================

score_threshold = 10.0

print(
    f"Score Threshold: "
    f"{score_threshold}"
)


# ============================================================
# Generate Recommendation Scores
# ============================================================

def generate_recommendation_scores():

    users = []
    courses = []
    scores = []

    for user_id in test_user_ids:

        # User profile vector
        user_profile = (
            profile_df[
                profile_df["user"] == user_id
            ]
        )

        user_vector = (
            user_profile
            .iloc[0, 1:]
            .values
        )

        # Courses already enrolled
        enrolled_courses = set(
            ratings_df[
                ratings_df["user"] == user_id
            ]["item"]
        )

        # Unknown courses
        unknown_courses = (
            all_courses -
            enrolled_courses
        )

        # Candidate courses
        unknown_course_df = (
            course_genres_df[
                course_genres_df["COURSE_ID"]
                .isin(unknown_courses)
            ]
        )

        unknown_course_ids = (
            unknown_course_df["COURSE_ID"]
            .values
        )

        course_matrix = (
            unknown_course_df
            .iloc[:, 2:]
            .values
        )

        # Recommendation scores
        recommendation_scores = np.dot(
            course_matrix,
            user_vector
        )

        # Store recommendations above threshold
        for i, score in enumerate(
            recommendation_scores
        ):

            if score >= score_threshold:

                users.append(user_id)
                courses.append(
                    unknown_course_ids[i]
                )
                scores.append(score)

    return users, courses, scores

# ============================================================
# Generate Recommendations
# ============================================================

users, courses, scores = (
    generate_recommendation_scores()
)

recommendations_df = pd.DataFrame(
    {
        "USER": users,
        "COURSE_ID": courses,
        "SCORE": scores
    }
)

print("\nRecommendation Results")
print("-" * 40)

print(
    recommendations_df.head()
)

print(
    f"\nTotal Recommendations: "
    f"{len(recommendations_df)}"
)

# ============================================================
# Recommendation Score Statistics
# ============================================================

score_stats = (
    recommendations_df["SCORE"]
    .describe()
)

print("\nRecommendation Score Statistics")
print("-" * 40)

print(score_stats)

# ============================================================
# Recommendation Analytics
# ============================================================

avg_recommendations = (
    recommendations_df
    .groupby("USER")
    .size()
    .mean()
)

print(
    f"\nAverage Recommendations "
    f"per User: "
    f"{avg_recommendations:.2f}"
)

top_courses = (
    recommendations_df
    .groupby("COURSE_ID")
    .size()
    .reset_index(
        name="Recommendation_Count"
    )
    .sort_values(
        by="Recommendation_Count",
        ascending=False
    )
    .head(10)
)

# Add course titles
top_courses = top_courses.merge(
    course_genres_df[
        ["COURSE_ID", "TITLE"]
    ],
    on="COURSE_ID",
    how="left"
)

print("\nTop 10 Recommended Courses")
print("-" * 40)

print(
    top_courses[
        ["TITLE", "Recommendation_Count"]
    ]
)

# ============================================================
# Save Results
# ============================================================

recommendations_df.to_csv(
    OUTPUT_DIR /
    "profile_rs_results.csv",
    index=False
)

top_courses.to_csv(
    OUTPUT_DIR /
    "top_10_recommended_courses.csv",
    index=False
)

print(
    f"\nResults saved to: "
    f"{OUTPUT_DIR}"
)

print(
    "\nUser Profile "
    "Recommendation Analysis Complete."
)

# ============================================================
# Final Summary
# ============================================================

print("\n" + "=" * 50)
print("USER PROFILE RECOMMENDER SUMMARY")
print("=" * 50)

print(
    f"Users Evaluated            : "
    f"{len(test_user_ids)}"
)

print(
    f"Total Recommendations      : "
    f"{len(recommendations_df)}"
)

print(
    f"Average Recommendations    : "
    f"{avg_recommendations:.2f}"
)

print(
    f"Average Recommendation Score: "
    f"{recommendations_df['SCORE'].mean():.2f}"
)

print(
    "\nUser Profile Recommendation "
    "Analysis Complete."
)
