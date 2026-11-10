"""
Author: Aaron Niecestro
Project: Course Recommender System Capstone

Created: November 3, 2026
Last Edit: November 3, 2026
Progress: Complete

Analysis: Exploratory Data Analysis (EDA)
Section: Course Similarity Analysis Using Bag-of-Words

Description:

Implements a content-based recommendation approach by
calculating similarity between courses using Bag-of-Words
(BoW) feature vectors.

The course "Machine Learning with Python"
(ML0101ENv3) is used as the target course and compared
against all other courses in the dataset.

Cosine similarity is used to measure similarity between
course feature vectors. Courses with similarity scores
above a specified threshold are retained, ranked, and
exported for review.

Datasets:

- courses_bows.csv

Outputs:

- Ranked list of similar courses
- Course similarity scores
- course_similarity_results.csv
"""

# =============================================================================
# Course Similarity using Bag-of-Words Features
# IBM ML321 - Module 2
# =============================================================================

import os
import pandas as pd
from scipy.spatial.distance import cosine

import numpy as np
import random

# ============================================================
# Reproducibility
# ============================================================

SEED = 423

random.seed(SEED)
np.random.seed(SEED)

RANDOM_STATE = SEED

# =============================================================================
# Configuration
# =============================================================================

TARGET_COURSE = "ML0101ENv3"
SIMILARITY_THRESHOLD = 0.50

# =============================================================================
# Output Directory
# =============================================================================

OUTPUT_DIR = r"C:\Users\aniec\Desktop\ibm-ml-project\capstone\01-EDA\EDA-results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Load Datasets
# =============================================================================

bows_url = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-ML321EN-SkillsNetwork/labs/datasets/courses_bows.csv"
)

course_url = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-ML321EN-SkillsNetwork/labs/datasets/course_processed.csv"
)

bows_df = pd.read_csv(bows_url)
bows_df = bows_df[["doc_id", "token", "bow"]]

course_df = pd.read_csv(course_url)

print("BoW Shape:", bows_df.shape)
print("Course Shape:", course_df.shape)

# =============================================================================
# Review Target Course
# =============================================================================

print("\nTarget Course:")
print(
    course_df[
        course_df["COURSE_ID"] == TARGET_COURSE
    ][["COURSE_ID", "TITLE"]]
)

# =============================================================================
# Helper Function
# =============================================================================

def pivot_two_bows(basedoc, comparedoc):
    """
    Convert two vertical BoW vectors into a common
    horizontal representation.
    """

    base = basedoc.copy()
    base["type"] = "base"

    compare = comparedoc.copy()
    compare["type"] = "compare"

    merged = pd.concat([base, compare])

    merged = (
        merged
        .pivot(
            index=["doc_id", "type"],
            columns="token",
            values="bow"
        )
        .fillna(0)
        .reset_index()
    )

    merged.columns = (
        ["doc_id", "type"]
        + list(merged.columns[2:])
    )

    return merged


# =============================================================================
# Find Similar Courses
# =============================================================================

target_bow = bows_df[
    bows_df["doc_id"] == TARGET_COURSE
]

results = []

all_courses = sorted(
    bows_df["doc_id"].unique()
)

for course_id in all_courses:

    if course_id == TARGET_COURSE:
        continue

    comparison_bow = bows_df[
        bows_df["doc_id"] == course_id
    ]

    try:

        vectors = pivot_two_bows(
            target_bow,
            comparison_bow
        )

        similarity = 1 - cosine(
            vectors.iloc[0, 2:],
            vectors.iloc[1, 2:]
        )

        if similarity >= SIMILARITY_THRESHOLD:

            course_info = course_df[
                course_df["COURSE_ID"] == course_id
            ]

            if len(course_info) > 0:

                title = course_info.iloc[0]["TITLE"]
                description = course_info.iloc[0]["DESCRIPTION"]

                results.append({
                    "COURSE_ID": course_id,
                    "TITLE": title,
                    "SIMILARITY": round(similarity, 4),
                    "DESCRIPTION": description
                })

    except Exception:
        continue

# =============================================================================
# Results DataFrame
# =============================================================================

similar_courses_df = pd.DataFrame(results)

similar_courses_df = (
    similar_courses_df
    .sort_values(
        by="SIMILARITY",
        ascending=False
    )
    .reset_index(drop=True)
)

# =============================================================================
# Display Results
# =============================================================================

print("\nTop Similar Courses")
print("=" * 80)

print(
    similar_courses_df[
        ["COURSE_ID", "TITLE", "SIMILARITY"]
    ].head(20)
)

# =============================================================================
# Save Results
# =============================================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "course_similarity_results.csv"
)

similar_courses_df.to_csv(
    output_file,
    index=False
)

print("\nResults Saved")
print(output_file)

# =============================================================================
# Summary Statistics
# =============================================================================

print("\nSummary")
print("-" * 40)
print("Target Course:", TARGET_COURSE)
print("Threshold:", SIMILARITY_THRESHOLD)
print("Matching Courses:", len(similar_courses_df))

if len(similar_courses_df) > 0:

    print("\nMost Similar Course:")

    print(
        similar_courses_df[
            ["COURSE_ID", "TITLE", "SIMILARITY"]
        ].iloc[0]
    )
