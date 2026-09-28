# =========================================================

# Classification EDA for DS Salaries Dataset

# Author: Aaron Niecestro
# Created: September 9, 2026
# Last Edit: September 10, 2026

# Targets:
# - salary_mean_cat (2 classes)
# - salary_median_cat (2 classes)
# - salary_3cat (3 classes using IQR)

# This script performs:
# - Summary statistics
# - Profiling
# - Target distribution & imbalance checks
# - Categorical & numerical exploration
# - Outlier detection
# - Correlation & multicollinearity (VIF)
# - Leakage checks
# - Feature importance (Random Forest)

# This is for Classification Models Purpose Only.
# The classification models are the following:
# 1. 2 categories of mean salary from salary in USD
# 2. 2 categories of median salary from salary in USD
# 3. 3 categories of salary from salary in USD using IQR

# =========================================================

# =========================================================
# Classification EDA for DS Salaries Dataset
# Author: Aaron Niecestro
# Created:    2026-09-09
# Last Edit:  2026-09-14
# =========================================================

# =========================================================
# Classification EDA for DS Salaries Dataset (USA-only)
# =========================================================

library(tidyverse)
library(janitor)
library(skimr)
library(GGally)
library(DataExplorer)
library(car)
library(ranger)
library(ggthemes)

# ---------------------------------------------------------
# 1. Load Data (USA-only)
# ---------------------------------------------------------

file_path <- "C:/Users/aniec/Desktop/ibm-ml-project/00-data/kaggle-data/dss2025_final.csv"

dss2025 <- read_csv(file_path) |>
  clean_names() |>
  filter(company_location == "United States")

# ---------------------------------------------------------
# 2. Numeric Encodings
# ---------------------------------------------------------

exp_map <- c("EN" = 0, "MI" = 1, "SE" = 2, "EX" = 3)
size_map <- c("S" = 0, "M" = 1, "L" = 2)

dss2025 <- dss2025 |>
  mutate(
    experience_level_num = exp_map[as.character(experience_level)],
    company_size_num     = size_map[as.character(company_size)]
  )

# ---------------------------------------------------------
# 3. Identify Columns
# ---------------------------------------------------------

numeric_cols <- dss2025 |> select(where(is.numeric)) |> names()
categorical_cols <- dss2025 |> select(where(~ is.character(.) || is.factor(.))) |> names()

cat("Numeric Columns:\n"); print(numeric_cols)
cat("Categorical Columns:\n"); print(categorical_cols)

# ---------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------

skim(dss2025)
summary(dss2025)

# ---------------------------------------------------------
# 5. Target Variable Distribution
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# 6. Pairplot (Safe)
# ---------------------------------------------------------

pairplot_cols <- head(numeric_cols, 6)

if (length(pairplot_cols) > 1) {
  ggpairs(dss2025[, pairplot_cols])
} else {
  message("Not enough numeric columns for pairplot.")
}

# ---------------------------------------------------------
# 7. Correlation Matrix
# ---------------------------------------------------------

corr <- cor(dss2025[, numeric_cols], use = "pairwise.complete.obs")

corrplot::corrplot(corr, method = "color", type = "lower",
                   tl.cex = 0.7, diag = FALSE)

# ---------------------------------------------------------
# 8. Outlier Detection (IQR)
# ---------------------------------------------------------

outlier_counts <- purrr::map_df(numeric_cols, function(col) {
  if (dss2025 |> pull(col) |> n_distinct() < 2) {
    tibble(feature = col, outliers = NA_integer_)
  } else {
    x  <- dss2025[[col]]
    Q1 <- quantile(x, 0.25, na.rm = TRUE)
    Q3 <- quantile(x, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower <- Q1 - 1.5 * IQR
    upper <- Q3 + 1.5 * IQR
    tibble(feature = col,
           outliers = sum(x < lower | x > upper, na.rm = TRUE))
  }
})

print(outlier_counts)

# ---------------------------------------------------------
# 9. Class Imbalance
# ---------------------------------------------------------

targets <- c("salary_mean_cat", "salary_median_cat", "salary_3cat")

check_imbalance <- function(df, col) {
  tab <- df |> count(.data[[col]]) |>
    mutate(percentage = n / sum(n) * 100)
  cat("\nClass distribution for", col, ":\n")
  print(tab)
}

plot_imbalance <- function(df, col) {
  ggplot(df, aes(x = .data[[col]])) +
    geom_bar(fill = "skyblue") +
    theme_minimal() +
    labs(title = paste("Class Imbalance:", col), x = col, y = "Count")
}

purrr::walk(targets, ~ check_imbalance(dss2025, .x))
purrr::walk(targets, ~ plot_imbalance(dss2025, .x))

# ---------------------------------------------------------
# 10. Feature Importance (Random Forest)
# ---------------------------------------------------------

numeric_features <- c("data_age", "remote_ratio",
                      "experience_level_num", "company_size_num")
numeric_features <- intersect(numeric_features, names(dss2025))

X <- dss2025 |> select(all_of(numeric_features))
y <- dss2025$salary_mean_cat

rf_data <- X |> mutate(target = as.factor(y))

rf_model <- ranger(target ~ ., data = rf_data, importance = "impurity")

importance_df <- tibble(
  feature    = numeric_features,
  importance = rf_model$variable.importance
)

ggplot(importance_df,
       aes(x = reorder(feature, importance), y = importance)) +
  geom_col(fill = "purple") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Feature Importance (Classification RF)",
       x = "Feature", y = "Importance")

# ---------------------------------------------------------
# 11. VIF (Safe)
# ---------------------------------------------------------

vif_cols <- numeric_cols[dss2025[, numeric_cols] |> summarise(across(everything(), n_distinct)) |> unlist() > 1]

vif_data <- dss2025 |> select(all_of(vif_cols)) |> drop_na()

vif_values <- car::vif(lm(salary_in_usd ~ ., data = vif_data))

vif_df <- tibble(feature = names(vif_values), VIF = as.numeric(vif_values)) |>
  arrange(desc(VIF))

print(vif_df)

# ---------------------------------------------------------
# 12. Leakage Check
# ---------------------------------------------------------

corr_target <- cor(dss2025[, numeric_cols], dss2025$salary_in_usd,
                   use = "pairwise.complete.obs")[, 1] |>
  sort(decreasing = TRUE)

cat("\nCorrelation with salary_in_usd:\n")
print(corr_target)

leakage_prone <- names(dss2025)[stringr::str_detect(names(dss2025), "salary")]
cat("\nLeakage-prone features:\n")
print(leakage_prone)

# ---------------------------------------------------------
# 13. Salary by Categorical Features
# ---------------------------------------------------------

ggplot(dss2025, aes(employment_type, salary_in_usd)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Salary by Employment Type")

ggplot(dss2025, aes(company_size, salary_in_usd)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Salary by Company Size")

ggplot(dss2025, aes(factor(remote_ratio), salary_in_usd)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Salary by Remote Ratio", x = "Remote Ratio")

# ---------------------------------------------------------
# 14. Job Title Insights
# ---------------------------------------------------------

top_jobs <- dss2025 |>
  count(job_title, sort = TRUE) |>
  slice_head(n = 20)

ggplot(top_jobs, aes(x = n, y = job_title)) +
  geom_col(fill = "steelblue") +
  theme_minimal() +
  labs(title = "Top 20 Job Titles", x = "Count", y = "Job Title")

title_salary <- dss2025 |>
  filter(job_title %in% top_jobs$job_title) |>
  group_by(job_title) |>
  summarise(median_salary = median(salary_in_usd, na.rm = TRUE)) |>
  arrange(desc(median_salary))

ggplot(title_salary, aes(x = median_salary, y = job_title)) +
  geom_col(fill = "darkgreen") +
  theme_minimal() +
  labs(title = "Median Salary by Job Title",
       x = "Median Salary (USD)", y = "Job Title")

# ---------------------------------------------------------
# 15. Time Trend
# ---------------------------------------------------------

year_salary <- dss2025 |>
  group_by(work_year) |>
  summarise(median_salary = median(salary_in_usd, na.rm = TRUE))

ggplot(year_salary, aes(x = work_year, y = median_salary)) +
  geom_line(color = "blue") +
  geom_point(size = 3) +
  theme_minimal() +
  labs(title = "Median Salary Over Time",
       x = "Work Year", y = "Median Salary (USD)")

# ---------------------------------------------------------
# Final checklist (as comments)
# ---------------------------------------------------------
# 1. Missing values handled?
# 2. Duplicates removed?
# 3. Correct data types?
# 4. Outliers checked and noted?
# 5. Decision on outliers?
# 6. Consistent data?
# 7. Valid target distributions?
# 8. Leakage checked?
# 9. All plots reviewed?

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
