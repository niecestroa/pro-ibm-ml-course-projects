# -*- coding: utf-8 -*-
"""
Author: Aaron Niecestro

Created:    October 19, 2026
Editted:    October 19, 2026
Status:     Completed

Description

This block of code performs the **initial data‑quality audit** on the combined 
wine dataset. It loads the merged red/white wine data, prints structural 
information, and checks for basic issues before any preprocessing. 

Specifically, it does the following:

- **Loads the combined dataset** (6497 rows, 13 columns).
- **Shows the first few rows** so you can visually confirm the merge worked.
- **Prints dataset structure (`info()`)** to verify:
  - column names  
  - data types  
  - non‑null counts  
- **Generates descriptive statistics (`describe()`)** for all numeric columns:
  - mean, std, min, max, quartiles  
- **Checks for missing values** in every column.
- **Counts duplicate rows**, which reveals **1177 duplicates** in the raw UCI wine data.

This step ensures the dataset is complete, correctly formatted, and ready for 
cleaning (e.g., removing duplicates) before scaling, PCA, or clustering.

"""

# ---
# Loading Packages
# ---

import pandas as pd  # data handling
import os  # path management

# ---
# Loading and Combining Data
# ---

raw_path = r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\00-data\raw-data"  # raw data folder
clean_path = r"C:\Users\aniec\Desktop\ibm-ml-project\unsupervised-ml\00-data\clean-data"  # clean data folder

red_path = os.path.join(raw_path, "winequality-red.csv")  # red wine file path
white_path = os.path.join(raw_path, "winequality-white.csv")  # white wine file path

red = pd.read_csv(red_path, sep=";")  # load red wine data
white = pd.read_csv(white_path, sep=";")  # load white wine data

red["type"] = "red"  # label red wine
white["type"] = "white"  # label white wine

wine = pd.concat([red, white], ignore_index=True)  # combine datasets

combined_path = os.path.join(clean_path, "winequality-combined.csv")  # combined dataset path
wine.to_csv(combined_path, index=False)  # save combined dataset
print(f"Combined dataset saved to: {combined_path}")  # confirmation message

# ---
# Basic Summaries + Missing Data Checks
# ---

print(wine.shape)  # dataset shape
print(wine.head())  # preview first rows
print(wine.info())  # column types + non-null counts
print(wine.describe())  # descriptive statistics
print(wine.isnull().sum())  # missing values per column
print(wine.isnull().sum().sum())  # total missing values
print(wine.duplicated().sum())  # number of duplicate rows

# ---
# Remove Duplicates + Re‑summaries
# ---

wine_clean = wine.drop_duplicates()  # remove duplicate rows
print(wine_clean.shape)  # shape after removing duplicates
print(wine_clean.duplicated().sum())  # confirm duplicates removed
print(wine_clean.describe())  # updated descriptive statistics
print(wine_clean.isnull().sum())  # missing values check

# ---
# Save Final Cleaned Dataset
# ---

final_path = os.path.join(clean_path, "winequality-final.csv")  # final cleaned dataset path
wine_clean.to_csv(final_path, index=False)  # save cleaned dataset
print(f"Final cleaned dataset saved to: {final_path}")  # confirmation message
