# =========================================================
# DS Salaries – Kaggle Download + Merge Pipeline (R Version)
# Author: Aaron Niecestro
# Created: 2026-09-10
# Last Edit: 2026-09-10
# =========================================================

# =========================================================
# DS Salaries – Kaggle Download + Merge Pipeline (R Version)
# Author: Aaron Niecestro
# =========================================================

library(data.table)
library(dplyr)
library(stringr)
library(fs)

# ---------------------------------------------------------
# 1. DATASET FOLDER PATHS (KaggleHub equivalent)
# ---------------------------------------------------------
# NOTE:
# R does not have kagglehub. You must download datasets manually
# or via Kaggle CLI. This script assumes the folders already exist.

paths <- c(
  "saurabhbadole_latest-data-science-job-salaries-2024",
  "arnabchaki_data-science-salaries-2023",
  "yusufdelikkaya_datascience-salaries-2024",
  "josiagiven_data-science-salaries-and-fields",
  "sazidthe1_data-science-salaries",
  "zain280_data-science-salaries",
  "lainguyn123_data-science-salary-landscape"
)

cat("Downloaded dataset folders:\n")
print(paths)

# ---------------------------------------------------------
# 2. LOAD ALL CSV FILES
# ---------------------------------------------------------

dfs <- list()

for (folder in paths) {
  
  csv_files <- dir(folder, pattern = "\\.csv$", full.names = TRUE)
  
  if (length(csv_files) == 0) {
    cat("No CSV found in", folder, "\n")
    next
  }
  
  for (csv in csv_files) {
    cat("Loading:", csv, "\n")
    
    df <- fread(csv)
    
    # Normalize column names
    names(df) <- tolower(trimws(names(df)))
    
    # Remove duplicate columns
    df <- df[, !duplicated(names(df)), with = FALSE]
    
    dfs[[length(dfs) + 1]] <- df
  }
}

cat("\nLoaded", length(dfs), "datasets.\n")

# ---------------------------------------------------------
# 3. REQUIRED SCHEMA
# ---------------------------------------------------------

required_cols <- c(
  "work_year",
  "experience_level",
  "employment_type",
  "job_title",
  "salary",
  "salary_currency",
  "salary_in_usd",
  "employee_residence",
  "remote_ratio",
  "company_location",
  "company_size"
)

# ---------------------------------------------------------
# 4. REINDEX EACH DATASET TO REQUIRED SCHEMA
# ---------------------------------------------------------

final_dfs <- list()

for (df in dfs) {
  
  # Remove duplicate columns again
  df <- df[, !duplicated(names(df)), with = FALSE]
  
  # Reindex safely (missing columns become NA)
  sub <- df[, required_cols, with = FALSE]
  
  final_dfs[[length(final_dfs) + 1]] <- sub
}

# ---------------------------------------------------------
# 5. MERGE ALL DATASETS
# ---------------------------------------------------------

df_master <- rbindlist(final_dfs, use.names = TRUE, fill = TRUE)

cat("\nMerged dataset size BEFORE deduplication:\n")
print(dim(df_master))

# ---------------------------------------------------------
# 6. REMOVE DUPLICATE ROWS
# ---------------------------------------------------------

df_master <- unique(df_master)

cat("\nMerged dataset size AFTER deduplication:\n")
print(dim(df_master))

# ---------------------------------------------------------
# 7. SAVE FINAL MERGED DATASET
# ---------------------------------------------------------

fwrite(df_master, "master_data_science_salaries.csv")
cat("\nSaved merged dataset as master_data_science_salaries.csv\n")

# ---------------------------------------------------------
# 8. VIEW COMPLETED MASTER DATASET
# ---------------------------------------------------------

df <- fread("master_data_science_salaries.csv")

cat("\n----- FIRST 10 ROWS -----\n")
print(head(df, 10))

cat("\n----- DATASET SHAPE -----\n")
print(dim(df))

cat("\n----- COLUMN NAMES -----\n")
print(names(df))

cat("\n----- STRUCTURE -----\n")
print(str(df))

cat("\nUnique work_year values:\n")
print(unique(df$work_year))

cat("\nMissing work_year count:\n")
print(sum(is.na(df$work_year)))

# ---------------------------------------------------------
# 9. REMOVE NA work_year AND RECHECK
# ---------------------------------------------------------

df <- df[!is.na(work_year)]

cat("\n----- FIRST 10 ROWS (AFTER NA REMOVAL) -----\n")
print(head(df, 10))

cat("\n----- DATASET SHAPE (AFTER NA REMOVAL) -----\n")
print(dim(df))

cat("\n----- COLUMN NAMES -----\n")
print(names(df))

cat("\n----- STRUCTURE -----\n")
print(str(df))

cat("\nUnique work_year values:\n")
print(unique(df$work_year))

cat("\nMissing work_year count:\n")
print(sum(is.na(df$work_year)))
