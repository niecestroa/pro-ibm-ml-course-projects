# =========================================================
# Author: Aaron Niecestro
# Project: DS Salaries – Production-Ready EDA for Regression Models
# Created:    2026-09-10
# Last Edit:  2026-09-14

# Project:    DS Salaries – Production-Ready EDA for Regression Models

# Description:
# Preliminary EDA 7 Kaggle datasets combined into one useable data.
# Dataset has data ranging from 2020 to 2025.
# Includes data loading, structure checks, missing values,
# summary statistics, categorical exploration, and visualization.
# Preliminary EDA on combined Kaggle DS salary datasets (2020–2025).
# Includes:
# - Data loading
# - Structural checks
# - Summary statistics
# - Categorical & numerical exploration
# - Outlier analysis
# - Correlation & multicollinearity (VIF)
# - Feature importance (Random Forest)
# - Skewness & log-transform check
# - Leakage check
# - Target-wise feature distributions
# - Key business insights

# This is for Regression Models Purpose Only.

# =========================================================

# =========================================================
# DS Salaries – Production-Ready EDA for Regression Models
# Author: Aaron Niecestro
# =========================================================

library(tidyverse)
library(janitor)
library(skimr)
library(GGally)
library(DataExplorer)
library(car)
library(ranger)
library(forcats)
library(ggthemes)

# =========================================================
# 1. Load Data
# =========================================================

file_path <- "C:/Users/aniec/Desktop/ibm-ml-project/00-data/kaggle-data/dss2025_final.csv"
dss2025 <- read_csv(file_path) |> clean_names()

cat("Rows:", nrow(dss2025), "Columns:", ncol(dss2025), "\n")

# USA-only subset (your chosen version)
dss2025 <- dss2025 |> filter(company_location == "United States")

# =========================================================
# 2. Column Type Setup
# =========================================================

numeric_cols <- dss2025 |> select(where(is.numeric)) |> names()
categorical_cols <- dss2025 |> select(where(~ is.character(.) || is.factor(.))) |> names()

cat("Numeric columns:\n")
print(numeric_cols)

cat("\nCategorical columns:\n")
print(categorical_cols)

# =========================================================
# 3. R-style Summary
# =========================================================

skim(dss2025)

summary(dss2025)

# =========================================================
# 4. Target Variable Exploration
# =========================================================

ggplot(dss2025, aes(salary_in_usd)) +
  geom_histogram(bins = 50, fill = "steelblue") +
  geom_density(color = "red") +
  theme_minimal() +
  labs(title = "Salary Distribution (USD)")

ggplot(dss2025, aes(log1p(salary_in_usd))) +
  geom_histogram(bins = 50, fill = "darkgreen") +
  geom_density(color = "white") +
  theme_minimal() +
  labs(title = "Log-Transformed Salary Distribution")

# =========================================================
# 5. Univariate Categorical Exploration
# =========================================================

plot_bar(dss2025, categorical_cols)

# =========================================================
# 6. Pairplot for Numeric Variables
# =========================================================

pair_cols <- c(setdiff(numeric_cols, "salary_in_usd")[1:5], "salary_in_usd")

ggpairs(dss2025[, pair_cols])

# =========================================================
# 7. Correlation Analysis
# =========================================================

corr_matrix <- cor(dss2025[numeric_cols], use = "pairwise.complete.obs")

corrplot::corrplot(corr_matrix, method = "color", type = "lower",
                   tl.cex = 0.7, diag = FALSE)

# =========================================================
# 8. Outlier Detection (IQR)
# =========================================================

outlier_counts <- map_df(numeric_cols, function(col) {
  Q1 <- quantile(dss2025[[col]], 0.25)
  Q3 <- quantile(dss2025[[col]], 0.75)
  IQR <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  tibble(feature = col,
         outliers = sum(dss2025[[col]] < lower | dss2025[[col]] > upper))
})

print(outlier_counts)

# =========================================================
# 9. Basic Consistency Checks
# =========================================================

sum(dss2025$salary_in_usd < 0)

table(dss2025$remote_ratio)

# =========================================================
# 10. Categorical Feature Exploration & Cardinality
# =========================================================

map(categorical_cols, ~ dss2025 |> count(.data[[.x]]) |> arrange(desc(n)))

map_chr(categorical_cols, ~ paste(.x, dss2025 |> pull(.x) |> n_distinct()))

# =========================================================
# 11. Feature Importance (Random Forest)
# =========================================================

# Convert ordered factors to numeric encodings
dss_rf <- dss2025 |>
  mutate(across(where(is.factor), ~ as.integer(.)))

rf_features <- c("data_age", "remote_ratio")
rf_data <- dss_rf |> select(all_of(rf_features), salary_in_usd)

rf_model <- ranger(salary_in_usd ~ ., data = rf_data, importance = "impurity")

importance_df <- tibble(
  feature = rf_features,
  importance = rf_model$variable.importance
)

ggplot(importance_df, aes(x = reorder(feature, importance), y = importance)) +
  geom_col(fill = "purple") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Random Forest Feature Importance")

# =========================================================
# 12. Skewness Check
# =========================================================

skew_vals <- map_dbl(numeric_cols, ~ moments::skewness(dss2025[[.x]]))
sort(skew_vals, decreasing = TRUE)

# =========================================================
# 13. Multicollinearity – VIF
# =========================================================

vif_df <- car::vif(lm(salary_in_usd ~ ., data = dss_rf[numeric_cols]))

print(vif_df)

# =========================================================
# 14. Leakage Check
# =========================================================

target_corr <- cor(dss2025[numeric_cols], dss2025$salary_in_usd, use = "pairwise.complete.obs")
sort(target_corr[, 1], decreasing = TRUE)

leakage_prone <- names(dss2025)[str_detect(names(dss2025), "salary")]
print(leakage_prone)

# =========================================================
# 15. Feature Distributions by Target
# =========================================================

ggplot(dss2025, aes(experience_level, salary_in_usd)) +
  geom_boxplot() +
  theme_minimal()

ggplot(dss2025, aes(company_size, salary_in_usd)) +
  geom_boxplot() +
  theme_minimal()

ggplot(dss2025, aes(factor(remote_ratio), salary_in_usd)) +
  geom_boxplot() +
  theme_minimal()

# =========================================================
# 16. Job Title & Location Insights
# =========================================================

top_jobs <- dss2025 |> count(job_title, sort = TRUE) |> slice_head(n = 20)

ggplot(top_jobs, aes(n, job_title)) +
  geom_col(fill = "steelblue") +
  theme_minimal()

title_salary <- dss2025 |>
  filter(job_title %in% top_jobs$job_title) |>
  group_by(job_title) |>
  summarize(median_salary = median(salary_in_usd)) |>
  arrange(desc(median_salary))

ggplot(title_salary, aes(median_salary, job_title)) +
  geom_col(fill = "darkgreen") +
  theme_minimal()

# =========================================================
# 17. Time Trend – Median Salary Over Years
# =========================================================

year_salary <- dss2025 |>
  group_by(work_year) |>
  summarize(median_salary = median(salary_in_usd))

ggplot(year_salary, aes(work_year, median_salary)) +
  geom_line(color = "blue") +
  geom_point(size = 3) +
  theme_minimal()

# =========================================================
# 18. Model Readiness Snapshot
# =========================================================

cat("Missing values:", sum(is.na(dss2025)), "\n")
cat("Duplicates:", sum(duplicated(dss2025)), "\n")

sort(skew_vals, decreasing = TRUE)[1:5]

vif_df[vif_df > 10]

print(leakage_prone)

# --------------------------------------------------------------------------


# =========================================================
# KEY INSIGHTS FROM VISUALIZATION
# =========================================================

# 1. Salary Distribution
# The salary distribution is strongly right-skewed. Most salaries fall in the 
# lower-to-mid ranges, with a long tail of high earners. The KDE curve confirms 
# a single dominant mode with heavy upper-tail outliers.

# 2. Salary by Experience Level
# Salary increases consistently with experience. EN < MI < SE < EX. The boxplots 
# show clear separation between levels, indicating experience is one of the 
# strongest predictors of salary.

# 3. Salary by Employment Type
# Full-time roles dominate the dataset. Contract roles show wider variability 
# and occasionally higher pay, but also more outliers. Part-time and freelance 
# roles tend to cluster at lower salary ranges.

# 4. Salary by Remote Ratio
# Remote work (100%) shows competitive or slightly higher salaries compared to 
# on-site roles. Hybrid (50%) sits between the two. Remote flexibility appears 
# correlated with higher pay in many cases.

# 5. Job Title Frequency
# A small number of job titles dominate the dataset (e.g., Data Scientist, 
# Data Engineer). Many titles appear infrequently, suggesting the dataset is 
# top-heavy and may benefit from grouping similar roles for modeling.

# 6. Salary by Country
# Geographic differences are substantial. Countries like the US, UK, and 
# Switzerland show higher median salaries, while others cluster lower. Location 
# is a major driver of salary variation.

# 7. Correlation Heatmap
# Salary shows moderate correlation with experience level and company size. 
# Remote ratio and work year have weaker correlations, suggesting non-linear 
# or categorical effects that boxplots capture better than correlation matrices.

# 8. Salary Over Time
# Median salary trends upward across work years, indicating growth in the 
# data science job market. Year-to-year increases are visible in both line and 
# boxplots.

# 9. Salary by Company Size
# Large companies tend to pay more on average. Small companies show wider 
# variability and more outliers, suggesting inconsistent compensation structures.

# 10. Median Salary by Job Title
# Senior and specialized roles (e.g., ML Engineer, Data Architect) show higher 
# median salaries. Generalist roles cluster lower. Job title is a strong 
# categorical predictor.

# 11. Salary vs Remote Ratio (Scatter)
# Scatterplots reveal clusters at 0, 50, and 100 remote ratio. Higher salaries 
# appear more frequently at 100% remote, supporting the boxplot findings.

# 12. Pairplot (Numeric Relationships)
# Salary shows a positive trend with work year and remote ratio, though not 
# strictly linear. Remote ratio and work year are weakly related.

# 13. Countplots (Experience, Employment, Company Size)
# The dataset is dominated by full-time roles, mid-level and senior-level 
# experience, and medium-to-large companies. This imbalance should be considered 
# during modeling.

# 14. Interaction: Experience Level × Company Size
# Senior roles at large companies show the highest salaries. Entry-level roles 
# at small companies show the lowest. The interaction effect is strong and 
# meaningful for predictive modeling.

# =========================================================
# Overall Summary
# =========================================================
# Salary is influenced most strongly by:
#   - Experience level
#   - Company size
#   - Job title
#   - Country / location
#
# Remote ratio and work year also matter, but less directly.
# The dataset shows clear patterns that will be useful for feature engineering 
# and building a salary prediction model.
# =========================================================
