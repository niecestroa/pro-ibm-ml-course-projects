"""
Author: Aaron Niecestro
Project: Course Recommender System Capstone

Created: November 3, 2026
Last Edit: November 3, 2026
Progress: Complete

Analysis: Exploratory Data Analysis (EDA)
Section: Bag-of-Words (BoW) Feature Extraction

Description:

Extracts Bag-of-Words (BoW) features from course titles
and descriptions to create a numerical representation of
course content for text analysis and recommendation
modeling.

The workflow tokenizes course text, removes stop words,
retains noun-based terms, generates BoW vectors, and
builds a structured feature dataset containing token
frequencies for each course.

Outputs:

- course_bow.csv
- Token frequency statistics
- Vocabulary size
- Course-level BoW feature vectors

Purpose:

The resulting BoW dataset serves as the foundation for
content-based recommendation models and course similarity
analysis.
"""

# =============================================================================
# Extract Bag of Words (BoW) Features from Course Textual Content
# =============================================================================

# Install required packages (run once)
# !pip install nltk==3.6.7
# !pip install gensim
# !pip install scipy==1.10
# !pip install pandas
# !pip install matplotlib

import gensim
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# ============================================================
# Reproducibility
# ============================================================

import random
import numpy as np

SEED = 423

random.seed(SEED)
np.random.seed(SEED)

RANDOM_STATE = SEED

# =============================================================================
# Load Course Dataset
# =============================================================================

course_url = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-ML321EN-SkillsNetwork/labs/datasets/course_processed.csv"
)

course_content_df = pd.read_csv(course_url)

print(course_content_df.head())

# =============================================================================
# Combine Title and Description
# =============================================================================

course_content_df["course_texts"] = (
    course_content_df[["TITLE", "DESCRIPTION"]]
    .agg(" ".join, axis=1)
)

course_content_df = course_content_df.reset_index(drop=True)
course_content_df["index"] = course_content_df.index

print(course_content_df[["COURSE_ID", "course_texts"]].head())

# =============================================================================
# Tokenization Function
# =============================================================================

def tokenize_course(course, keep_only_nouns=True):

    stop_words = set(stopwords.words("english"))

    word_tokens = word_tokenize(course)

    # Remove stop words and numeric values
    word_tokens = [
        w for w in word_tokens
        if (not w.lower() in stop_words)
        and (not w.isnumeric())
    ]

    if keep_only_nouns:

        filter_list = [
            'WDT', 'WP', 'WRB', 'FW',
            'IN', 'JJR', 'JJS', 'MD',
            'PDT', 'POS', 'PRP', 'RB',
            'RBR', 'RBS', 'RP'
        ]

        tags = nltk.pos_tag(word_tokens)

        noun_tags = [
            'NN',
            'NNS',
            'NNP',
            'NNPS'
        ]

        word_tokens = [
            word
            for word, pos in tags
            if pos in noun_tags
        ]

    return word_tokens

# =============================================================================
# Tokenize All Courses (TODO #1)
# =============================================================================

tokenized_courses = []

for text in course_content_df["course_texts"]:
    tokens = tokenize_course(
        text,
        keep_only_nouns=True
    )
    tokenized_courses.append(tokens)

print("\nSample Tokenized Course:")
print(tokenized_courses[0])

# =============================================================================
# Create Dictionary (TODO #2)
# =============================================================================

tokens_dict = corpora.Dictionary(tokenized_courses)

print("\nTotal Unique Tokens:")
print(len(tokens_dict))

print("\nSample Dictionary Entries:")
for i in range(10):
    if i in tokens_dict:
        print(i, ":", tokens_dict[i])

# =============================================================================
# Generate BoW Features (TODO #3)
# =============================================================================

bow_docs = []

for course in tokenized_courses:
    bow = tokens_dict.doc2bow(course)
    bow_docs.append(bow)

print("\nFirst Course BoW:")
print(bow_docs[0])

# =============================================================================
# Build BoW Dataset (TODO #4)
# =============================================================================

doc_indices = []
doc_ids = []
tokens = []
bow_values = []

for doc_index, doc_bow in enumerate(bow_docs):

    course_id = course_content_df.iloc[
        doc_index
    ]["COURSE_ID"]

    for token_index, token_bow in doc_bow:

        token = tokens_dict[token_index]

        doc_indices.append(doc_index)
        doc_ids.append(course_id)
        tokens.append(token)
        bow_values.append(token_bow)

bow_dicts = {
    "doc_index": doc_indices,
    "doc_id": doc_ids,
    "token": tokens,
    "bow": bow_values
}

course_bow_df = pd.DataFrame(bow_dicts)

# =============================================================================
# Inspect Results
# =============================================================================

print("\nBoW Dataset Shape:")
print(course_bow_df.shape)

print("\nBoW Dataset Sample:")
print(course_bow_df.head(20))

# =============================================================================
# Save Dataset
# =============================================================================

import os

output_dir = r"C:\Users\aniec\Desktop\ibm-ml-project\capstone\01-EDA\EDA-results"

os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    "course_bow.csv"
)

course_bow_df.to_csv(
    output_file,
    index=False
)

print("\ncourse_bow.csv saved successfully!")
print(f"Location: {output_file}")

# =============================================================================
# Optional Statistics
# =============================================================================

print("\nUnique Courses:")
print(course_bow_df["doc_id"].nunique())

print("\nUnique Tokens:")
print(course_bow_df["token"].nunique())

print("\nTop 20 Most Common Tokens:")

top_tokens = (
    course_bow_df
    .groupby("token")["bow"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
)

print(top_tokens)