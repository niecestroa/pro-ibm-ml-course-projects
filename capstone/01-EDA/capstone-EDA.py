# -*- coding: utf-8 -*-

"""
Author: Aaron Niecestro
Project: Course Recommender System Capstone

Created: November 3, 2026
Last Edit: November 3, 2026
Progress: Complete

Analysis: Exploratory Data Analysis (EDA)
Section: Course Catalog and Enrollment Exploration

Description:
Performs exploratory analysis on course catalog and
enrollment datasets to understand course availability,
genre distributions, user behavior, and course popularity.

Objectives:

- Understand dataset structure
- Analyze course genre distributions
- Examine learner enrollment patterns
- Identify popular courses
- Generate visual summaries

Datasets:

- course_genre.csv
- ratings.csv

Outputs:

- genre_counts.csv
- top_20_courses.csv
- EDA visualizations

Research Questions:

- How many courses are available?
- How many users are represented?
- Which courses are most frequently enrolled?
- How are course genres distributed?
- What patterns exist within user enrollment behavior?
"""

# ============================================================
# IBM ML321 Capstone Project
# Exploratory Data Analysis (EDA)
# Complete Solution for All TODO Sections
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# ============================================================
# Reproducibility
# ============================================================

import random

SEED = 423

random.seed(SEED)
np.random.seed(SEED)

RANDOM_STATE = SEED

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\capstone\00-Data"
)

OUTPUT_DIR = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\capstone\01-EDA\EDA-results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------

course_df = pd.read_csv(
    DATA_DIR / "course_genre.csv"
)

ratings_df = pd.read_csv(
    DATA_DIR / "ratings.csv"
)

print("\nDatasets Loaded Successfully")
print("-" * 40)

print(f"Course Data Shape : {course_df.shape}")
print(f"Ratings Shape     : {ratings_df.shape}")

# ------------------------------------------------------------
# Word Cloud of Course Titles
# ------------------------------------------------------------

titles = " ".join(title for title in course_df['TITLE'].astype(str))

stopwords = set(STOPWORDS)
stopwords.update([
    "getting started",
    "using",
    "enabling",
    "template",
    "university",
    "end",
    "introduction",
    "basic"
])

wordcloud = WordCloud(
    stopwords=stopwords,
    background_color="white",
    width=800,
    height=400
)

wordcloud.generate(titles)

plt.figure(figsize=(16,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Course Titles")
plt.show()

# ============================================================
# TODO 1
# Find all Machine Learning Courses
# ============================================================

ml_courses = course_df[course_df['MachineLearning'] == 1]

print("\n==============================")
print("MACHINE LEARNING COURSES")
print("==============================")
print(ml_courses[['COURSE_ID', 'TITLE']])

# ============================================================
# TODO 2
# Find Machine Learning + Big Data Courses
# ============================================================

scalable_ml_courses = course_df[
    (course_df['MachineLearning'] == 1) &
    (course_df['BigData'] == 1)
]

print("\n==============================")
print("MACHINE LEARNING + BIG DATA COURSES")
print("==============================")
print(scalable_ml_courses[['COURSE_ID', 'TITLE']])

# ============================================================
# TODO 3
# Calculate Course Counts Per Genre
# ============================================================

genres = course_df.columns[2:]

genre_sums = course_df[genres].sum(axis=0)

genre_counts = pd.DataFrame(
    genre_sums,
    columns=["Count"]
).sort_values(
    by="Count",
    ascending=False
)

print("\n==============================")
print("COURSE COUNTS PER GENRE")
print("==============================")
print(genre_counts)

# ============================================================
# TODO 4
# Plot Genre Counts Bar Chart
# ============================================================

plt.figure(figsize=(14,6))

plot = sns.barplot(
    x=genre_counts.index,
    y="Count",
    data=genre_counts
)

plot.set_xticklabels(
    plot.get_xticklabels(),
    rotation=90
)

plt.title("Course Counts per Genre")
plt.xlabel("Genre")
plt.ylabel("Number of Courses")
plt.tight_layout()
plt.show()

# ============================================================
# TODO 5
# Aggregate Rating Count Per User
# ============================================================

user_ratings_count = (
    ratings_df
    .groupby('user')
    .size()
    .reset_index(name='Enrollments')
)

print("\n==============================")
print("USER ENROLLMENT COUNTS")
print("==============================")
print(user_ratings_count.head())

print(
    f"\nTotal Users: {len(user_ratings_count)}"
)

# ============================================================
# TODO 6
# Enrollment Statistics
# ============================================================

print("\n==============================")
print("ENROLLMENT SUMMARY STATISTICS")
print("==============================")
print(user_ratings_count['Enrollments'].describe())

# ============================================================
# TODO 7
# Enrollment Distribution Histogram
# ============================================================

plt.figure(figsize=(10,6))

user_ratings_count['Enrollments'].hist(
    bins=30,
    edgecolor='black'
)

plt.title("Course Enrollment Distribution")
plt.xlabel("Number of Courses Enrolled")
plt.ylabel("Number of Users")

plt.tight_layout()
plt.show()

# ============================================================
# TODO 8
# Top 20 Most Popular Courses
# ============================================================

top_courses = (
    ratings_df
    .groupby('item')
    .size()
    .reset_index(name='Ratings')
    .sort_values(
        by='Ratings',
        ascending=False
    )
    .head(20)
)

print("\n==============================")
print("TOP 20 MOST POPULAR COURSES")
print("==============================")
print(top_courses)

# ============================================================
# TODO 9
# Merge Course Names
# ============================================================

top_courses = pd.merge(
    top_courses,
    course_df[['COURSE_ID', 'TITLE']],
    how='left',
    left_on='item',
    right_on='COURSE_ID'
)

top_courses = top_courses[['TITLE', 'Ratings']]

print("\n==============================")
print("TOP 20 COURSES WITH TITLES")
print("==============================")
print(top_courses)

# ============================================================
# TODO 10
# Percentage of Top-20 Enrollments
# ============================================================

total = ratings_df.shape[0]

top = top_courses['Ratings'].sum()

percentage = round((top * 100) / total, 2)

print("\n==============================")
print("TOP-20 COURSE ENROLLMENT SHARE")
print("==============================")
print(f"Percentage of the top course enrollments: {percentage}%")

# ============================================================
# Optional: Save Results for Capstone Slides
# ============================================================

import os

output_dir = r"C:\Users\aniec\Desktop\ibm-ml-project\capstone\01-EDA\EDA-results"

# Create folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

genre_counts.to_csv(
    os.path.join(output_dir, "genre_counts.csv")
)

top_courses.to_csv(
    os.path.join(output_dir, "top_20_courses.csv"),
    index=False
)

print(f"Results saved to: {output_dir}")

print("\nCapstone EDA Analysis Complete.")
