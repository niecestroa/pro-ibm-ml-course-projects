# =========================================================
# DS Salaries – Cleaning + Pre‑EDA Pipeline (R Version)
# Author: Aaron Niecestro
# Created: 2026-09-10
# Last Edit: 2026-09-10
# =========================================================

library(data.table)
library(dplyr)
library(stringr)
library(ggplot2)
library(forcats)
library(skimr)

# ---------------------------------------------------------
# 0. Load Data
# ---------------------------------------------------------

file_path <- "C:/Users/aniec/Desktop/ibm-ml-project/00-data/kaggle-data/dss2025_final.csv"
dss2025 <- fread(file_path)

# ---------------------------------------------------------
# 1. Understanding the Data
# ---------------------------------------------------------

cat("----- SHAPE -----\n")
print(dim(dss2025))

cat("\n----- COLUMN NAMES -----\n")
print(names(dss2025))

cat("\n----- DATA TYPES -----\n")
print(str(dss2025))

cat("\n----- PREVIEW -----\n")
print(head(dss2025))

# ---------------------------------------------------------
# 2. Standardize Experience Levels
# ---------------------------------------------------------

experience_map <- c(
  "EN" = "Entry-Level", "Entry" = "Entry-Level", "Entry Level" = "Entry-Level", "Entry-Level" = "Entry-Level",
  "MI" = "Middle-Level", "Mid" = "Middle-Level", "Middle" = "Middle-Level", "Mid Level" = "Middle-Level",
  "Middle Level" = "Middle-Level", "Middle-Level" = "Middle-Level",
  "SE" = "Senior-Level", "Senior" = "Senior-Level", "Senior Level" = "Senior-Level", "Senior-Level" = "Senior-Level",
  "EX" = "Executive-Level", "Executive" = "Executive-Level", "Executive Level" = "Executive-Level",
  "Executive-Level" = "Executive-Level"
)

dss2025$experience_level <- recode(dss2025$experience_level, !!!experience_map)

dss2025$experience_level <- factor(
  dss2025$experience_level,
  levels = c("Entry-Level", "Middle-Level", "Senior-Level", "Executive-Level"),
  ordered = TRUE
)

# ---------------------------------------------------------
# 3. Create Categorical Versions of Numeric Variables
# ---------------------------------------------------------

# Work year categorical
dss2025$work_year_cat <- factor(as.character(dss2025$work_year))

# Remote ratio categorical
remote_map <- c("0" = "On-Site", "50" = "Hybrid", "100" = "Remote")

dss2025$remote_work_cat <- recode(as.character(dss2025$remote_ratio), !!!remote_map)

dss2025$remote_work_cat <- factor(
  dss2025$remote_work_cat,
  levels = c("On-Site", "Hybrid", "Remote"),
  ordered = TRUE
)

# ---------------------------------------------------------
# 4. Salary Categories (Mean, Median, 3‑Band)
# ---------------------------------------------------------

mean_salary <- mean(dss2025$salary_in_usd, na.rm = TRUE)
median_salary <- median(dss2025$salary_in_usd, na.rm = TRUE)

dss2025$salary_mean_cat <- ifelse(
  dss2025$salary_in_usd >= mean_salary,
  "Above-Average",
  "Below-Average"
)

dss2025$salary_median_cat <- ifelse(
  dss2025$salary_in_usd >= median_salary,
  "Above-Median",
  "Below-Median"
)

q1 <- quantile(dss2025$salary_in_usd, 0.33, na.rm = TRUE)
q2 <- quantile(dss2025$salary_in_usd, 0.66, na.rm = TRUE)

dss2025$salary_3cat <- case_when(
  dss2025$salary_in_usd < q1 ~ "Low-Salary",
  dss2025$salary_in_usd < q2 ~ "Middle-Salary",
  TRUE ~ "High-Salary"
)

# ---------------------------------------------------------
# 5. Numeric Columns
# ---------------------------------------------------------

dss2025$data_age <- 2026 - dss2025$work_year

numeric_cols <- c("work_year", "salary", "salary_in_usd", "remote_ratio", "data_age")

# ---------------------------------------------------------
# 6. Categorical Columns
# ---------------------------------------------------------

categorical_cols <- c(
  "experience_level", "employment_type", "job_title", "salary_currency",
  "employee_residence", "company_location", "company_size",
  "work_year_cat", "remote_work_cat",
  "salary_mean_cat", "salary_median_cat", "salary_3cat"
)

# Convert numeric columns
dss2025[, (numeric_cols) := lapply(.SD, as.numeric), .SDcols = numeric_cols]

# Convert categorical columns
dss2025[, (categorical_cols) := lapply(.SD, as.factor), .SDcols = categorical_cols]

# ---------------------------------------------------------
# 7. Verify Results
# ---------------------------------------------------------

cat("\n----- EXPERIENCE LEVEL COUNTS -----\n")
print(table(dss2025$experience_level))

cat("\n----- DATA TYPES -----\n")
print(str(dss2025))

# ---------------------------------------------------------
# 8. Unique Values
# ---------------------------------------------------------

cat("\n----- UNIQUE VALUES -----\n")
for (col in names(dss2025)) {
  cat("\n", col, "\n")
  cat("Unique Count:", length(unique(dss2025[[col]])), "\n")
}

# ---------------------------------------------------------
# 9. Summary Statistics
# ---------------------------------------------------------

cat("\n----- SUMMARY (R-style) -----\n")
print(skim(dss2025))

cat("\n----- NUMERIC SUMMARY -----\n")
print(summary(dss2025[, ..numeric_cols]))

cat("\n----- CATEGORICAL SUMMARY -----\n")
print(summary(dss2025[, ..categorical_cols]))

# ---------------------------------------------------------
# 10. Missing Values
# ---------------------------------------------------------

missing_df <- data.frame(
  Missing_Count = colSums(is.na(dss2025)),
  Missing_Percent = colSums(is.na(dss2025)) / nrow(dss2025) * 100
)

cat("\n----- MISSING VALUES -----\n")
print(missing_df)

# Missing heatmap
ggplot(as.data.frame(is.na(dss2025)), aes()) +
  geom_tile(aes(x = rep(1:ncol(dss2025), each = nrow(dss2025)),
                y = rep(1:nrow(dss2025), ncol(dss2025)),
                fill = as.vector(is.na(dss2025)))) +
  scale_fill_manual(values = c("FALSE" = "white", "TRUE" = "red")) +
  theme_void() +
  ggtitle("Missing Values Heatmap")

# ---------------------------------------------------------
# 11. Duplicate Check
# ---------------------------------------------------------

duplicates <- sum(duplicated(dss2025))
cat("\n----- DUPLICATES -----\n")
cat("Duplicate Rows:", duplicates, "\n")

# ---------------------------------------------------------
# 12. Clean Dataset
# ---------------------------------------------------------

dss2025 <- unique(dss2025)

dss2025 <- dss2025[!is.na(salary_in_usd)]

dss2025 <- dss2025[, ]

# ---------------------------------------------------------
# 13. Final Checks
# ---------------------------------------------------------

cat("\n----- FINAL SHAPE -----\n")
print(dim(dss2025))

cat("\n----- FINAL DATA TYPES -----\n")
print(str(dss2025))

cat("\n----- FINAL MISSING VALUES -----\n")
print(colSums(is.na(dss2025)))

cat("\n----- FINAL SUMMARY -----\n")
print(summary(dss2025))

# ---------------------------------------------------------
# 14. Save Final Dataset
# ---------------------------------------------------------

final_file <- "C:/Users/aniec/Desktop/ibm-ml-project/00-data/kaggle-data/dss2025_final.csv"
fwrite(dss2025, final_file)

cat("\nFinal dataset saved to:\n", final_file, "\n")
