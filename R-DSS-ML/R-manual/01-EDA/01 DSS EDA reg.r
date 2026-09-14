# =========================================================
# Author: Aaron Niecestro
# Project: DS Salaries – Production-Ready EDA for Regression Models
# Created: 2026-09-10
# Last Edit: 2026-09-10

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
# 0. Libraries
# =========================================================

library(data.table)       # Fast data loading
library(dplyr)            # Data manipulation
library(ggplot2)          # Visualization
library(skimr)            # Summary statistics
library(GGally)           # Pairplots
library(corrplot)         # Correlation heatmap
library(randomForest)     # Feature importance
library(car)              # VIF
library(janitor)          # Clean names
library(forcats)          # Factor handling

# =========================================================
# 1. Load Data
# =========================================================

file_path <- "C:/Users/aniec/Desktop/ibm-ml-project/00-data/kaggle-data/dss2025_final.csv"
dss2025 <- fread(file_path)

cat("----- DATA SHAPE -----\n")
cat("Rows:", nrow(dss2025), "Columns:", ncol(dss2025), "\n\n")

# =========================================================
# 2. Column Type Setup
# =========================================================

numeric_cols <- names(dss2025)[sapply(dss2025, is.numeric)]
categorical_cols <- names(dss2025)[sapply(dss2025, is.character)]

cat("----- NUMERIC COLUMNS -----\n")
print(numeric_cols)

cat("\n----- CATEGORICAL COLUMNS -----\n")
print(categorical_cols)

# =========================================================
# 3. R-style Summary
# =========================================================

summary_r <- function(df) {
  cat("----- R-style Summary -----\n")
  cat("Rows:", nrow(df), "Columns:", ncol(df), "\n\n")
  
  for (col in names(df)) {
    cat("---", col, "---\n")
    
    if (is.numeric(df[[col]])) {
      desc <- summary(df[[col]])
      print(desc)
    } else {
      desc <- summary(as.factor(df[[col]]))
      print(desc)
    }
    cat("\n")
  }
}

summary_r(dss2025)

cat("\n----- SUMMARY TABLE -----\n")
print(skim(dss2025))

# =========================================================
# 4. Target Variable Exploration
# =========================================================

ggplot(dss2025, aes(salary_in_usd)) +
  geom_histogram(bins = 50, fill = "steelblue") +
  geom_density(color = "red") +
  ggtitle("Salary in USD Distribution")

ggplot(dss2025, aes(log1p(salary_in_usd))) +
  geom_histogram(bins = 50, fill = "purple") +
  geom_density(color = "black") +
  ggtitle("Log-Transformed Salary Distribution")

# =========================================================
# 5. Univariate Categorical Exploration
# =========================================================

ggplot(dss2025, aes(experience_level)) +
  geom_bar(fill = "skyblue") +
  ggtitle("Experience Level Distribution")

ggplot(dss2025, aes(factor(work_year))) +
  geom_bar(fill = "orange") +
  ggtitle("Work Year Distribution")

# =========================================================
# 6. Pairplot for Numeric Variables
# =========================================================

pair_cols <- head(setdiff(numeric_cols, "salary_in_usd"), 6)
pair_cols <- c(pair_cols, "salary_in_usd")

ggpairs(dss2025[, ..pair_cols])

# =========================================================
# 7. Correlation Analysis
# =========================================================

corr_matrix <- cor(dss2025[, ..numeric_cols], use = "complete.obs")

corrplot(corr_matrix, method = "color", tl.cex = 0.7)

# =========================================================
# 8. Outlier Detection & Visualization
# =========================================================

cat("\n----- OUTLIER COUNTS (IQR Method) -----\n")

for (col in numeric_cols) {
  Q1 <- quantile(dss2025[[col]], 0.25)
  Q3 <- quantile(dss2025[[col]], 0.75)
  IQR <- Q3 - Q1
  
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  
  outliers <- dss2025[[col]] < lower | dss2025[[col]] > upper
  
  cat(col, ":", sum(outliers), "outliers\n")
  
  ggplot(dss2025, aes(x = "", y = .data[[col]])) +
    geom_boxplot(fill = "lightgreen") +
    ggtitle(paste("Boxplot of", col)) +
    theme(axis.title.x = element_blank())
}

# =========================================================
# 9. Basic Consistency Checks
# =========================================================

cat("\nNegative Salary Values:", sum(dss2025$salary_in_usd < 0), "\n")

cat("\nInvalid Remote Ratios:\n")
print(table(dss2025$remote_ratio[!dss2025$remote_ratio %in% c(0, 50, 100)]))

# =========================================================
# 10. Categorical Feature Exploration & Cardinality
# =========================================================

for (col in categorical_cols) {
  ggplot(dss2025, aes(.data[[col]])) +
    geom_bar(fill = "steelblue") +
    ggtitle(paste("Distribution of", col)) +
    coord_flip()
}

cat("\n----- CARDINALITY -----\n")
for (col in categorical_cols) {
  cat(col, ":", n_distinct(dss2025[[col]]), "\n")
}

# =========================================================
# 11. Feature Importance (Random Forest)
# =========================================================

numeric_features <- c("data_age", "remote_ratio", "experience_level_num", "company_size_num")
numeric_features <- numeric_features[numeric_features %in% names(dss2025)]

rf_model <- randomForest(
  x = dss2025[, ..numeric_features],
  y = dss2025$salary_in_usd,
  importance = TRUE
)

importance_df <- data.frame(
  Feature = numeric_features,
  Importance = rf_model$importance[, "IncNodePurity"]
)

ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "darkred") +
  coord_flip() +
  ggtitle("Feature Importance (Random Forest Regression)")

# =========================================================
# 12. Skewness Check
# =========================================================

cat("\n----- SKEWNESS -----\n")
print(sapply(dss2025[, ..numeric_cols], skewness))

# =========================================================
# 13. Multicollinearity – VIF
# =========================================================

cat("\n----- VIF -----\n")

vif_df <- data.frame(
  Feature = numeric_cols,
  VIF = vif(lm(salary_in_usd ~ ., data = dss2025[, ..numeric_cols]))
)

print(vif_df)

# =========================================================
# 14. Data Leakage Check
# =========================================================

corr_target <- corr_matrix[, "salary_in_usd"]
print(sort(corr_target, decreasing = TRUE))

leakage_prone <- names(dss2025)[grepl("salary", names(dss2025), ignore.case = TRUE)]
cat("\nLeakage-prone features:\n")
print(leakage_prone)

# =========================================================
# 15. Feature Distributions by Target
# =========================================================

ggplot(dss2025, aes(employment_type, salary_in_usd)) +
  geom_boxplot(fill = "purple") +
  ggtitle("Salary by Employment Type")

ggplot(dss2025, aes(company_size, salary_in_usd)) +
  geom_boxplot(fill = "green") +
  ggtitle("Salary by Company Size")

ggplot(dss2025, aes(factor(remote_ratio), salary_in_usd)) +
  geom_boxplot(fill = "orange") +
  ggtitle("Salary by Remote Ratio")

# =========================================================
# 16. Job Title & Location Insights
# =========================================================

top_jobs <- dss2025 %>% count(job_title, sort = TRUE) %>% head(20)

ggplot(top_jobs, aes(n, job_title)) +
  geom_col(fill = "steelblue") +
  ggtitle("Top 20 Most Common Job Titles")

top_titles <- top_jobs$job_title

title_salary <- dss2025 %>%
  filter(job_title %in% top_titles) %>%
  group_by(job_title) %>%
  summarize(median_salary = median(salary_in_usd)) %>%
  arrange(desc(median_salary))

ggplot(title_salary, aes(median_salary, job_title)) +
  geom_col(fill = "darkgreen") +
  ggtitle("Median Salary by Job Title")

# =========================================================
# 17. Time Trend – Median Salary Over Years
# =========================================================

year_salary <- dss2025 %>%
  group_by(work_year) %>%
  summarize(median_salary = median(salary_in_usd))

ggplot(year_salary, aes(work_year, median_salary)) +
  geom_line(color = "blue") +
  geom_point(size = 3) +
  ggtitle("Median Salary Over Time")

# =========================================================
# 18. Model Readiness Snapshot
# =========================================================

cat("\n----- MODEL READINESS SNAPSHOT -----\n")
cat("Missing values handled previously\n")
cat("Duplicates handled previously\n")

cat("\nMost skewed features:\n")
print(sort(sapply(dss2025[, ..numeric_cols], skewness), decreasing = TRUE)[1:5])

cat("\nHigh VIF features:\n")
print(vif_df[vif_df$VIF > 10, ])

cat("\nLeakage-prone features:\n")
print(leakage_prone)
