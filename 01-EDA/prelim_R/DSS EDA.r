# =========================================================
# Created:    2026-07-19
# Last Edit:  2026-07-20
# Author:     Aaron Niecestro
#
# Description:
# Preliminary EDA for the Data Science Salaries 2023 dataset.
# Includes data loading, structure checks, missing values,
# summary statistics, categorical exploration, and visualization.
# =========================================================

# =========================================================
# 1. Load Libraries
# =========================================================
library(tidyverse)
library(skimr)
library(DataExplorer)
library(janitor)
library(ggplot2)
library(forcats)
library(readr)
library(GGally)
library(ggcorrplot)
library(reshape2)

# =========================================================
# 2. Load Data
# =========================================================
dss23 <- read_csv("ds_salaries.csv")

head(dss23)
glimpse(dss23)

dss23 <- dss23 %>%
  mutate(
    experience_label = fct_recode(
      experience_level,
      "Entry"     = "EN",
      "Mid"       = "MI",
      "Senior"    = "SE",
      "Executive" = "EX"
    ),
    employment_label = fct_recode(
      employment_type,
      "Full-Time" = "FT",
      "Part-Time" = "PT",
      "Contract"  = "CT",
      "Freelance" = "FL"
    )
  )


head(dss23)
glimpse(dss23)

# =========================================================
# 3. Basic Structure
# =========================================================
cat("----- SHAPE -----\n")
print(dim(dss23))

cat("\n----- COLUMN NAMES -----\n")
print(colnames(dss23))

cat("\n----- DATA TYPES -----\n")
print(sapply(dss23, class))

# =========================================================
# 4. Missing Values
# =========================================================
cat("\n----- MISSING VALUES (COUNT) -----\n")
print(colSums(is.na(dss23)))

cat("\n----- MISSING VALUES (PERCENT) -----\n")
print(colMeans(is.na(dss23)) * 100)

# =========================================================
# 5. Simplified R-style Summary Function
# =========================================================
summary_r <- function(df) {
  cat("----- R-style Summary -----\n")
  cat("Rows:", nrow(df), "Columns:", ncol(df), "\n\n")
  print(summary(df))
}

summary_r(dss23)

# =========================================================
# 6. Profiling Report (R Equivalent)
# =========================================================
create_report(dss23, output_file = "ds_salaries_profile.html")

# =========================================================
# 7. Unique Values for Categorical Columns
# =========================================================
cat_cols <- names(Filter(is.character, dss23))

cat("\n----- UNIQUE VALUES (CATEGORICAL) -----\n")
for (col in cat_cols) {
  cat("\n", col, ":\n")
  print(unique(dss23[[col]])[1:20])
}

# =========================================================
# 8. Numeric Summary
# =========================================================
cat("\n----- NUMERIC SUMMARY -----\n")
print(summary(select_if(dss23, is.numeric)))

# =========================================================
# 9. Duplicated Rows
# =========================================================
cat("\n----- DUPLICATED ROWS -----\n")
print(sum(duplicated(dss23)))

# =========================================================
# 10. Value Counts
# =========================================================
cat("\n----- VALUE COUNTS (TOP 10 PER COLUMN) -----\n")
for (col in cat_cols) {
  cat("\n", col, ":\n")
  print(head(table(dss23[[col]]), 10))
}

# =========================================================
# 11. Correlation Matrix
# =========================================================
numeric_cols <- select_if(dss23, is.numeric)
cor_matrix <- cor(numeric_cols, use = "complete.obs")
print(cor_matrix)

# =========================================================
# MOVING FROM SUMMARY TO VISUALIZATION
# =========================================================
# The summary section helped us understand the dataset's structure:
#   - data types
#   - missing values
#   - numeric ranges and quartiles
#   - category frequencies
#
# Now we move into the visualization stage, where graphs help reveal:
#   - distribution shapes
#   - group differences
#   - category patterns
#   - correlations
#   - geographic salary trends
#
# Visuals allow us to see patterns that summary statistics alone cannot.
# =========================================================

# =========================================================
# 12. Visualization (ggplot2)
# =========================================================

# 0. Variable Histograms and Boxplots
# Identify numeric columns
num_cols <- dss23 %>% select(where(is.numeric)) %>% names()

# Histograms
for (col in num_cols) {
  print(
    ggplot(dss23, aes(.data[[col]])) +
      geom_histogram(bins = 40, fill = "steelblue") +
      geom_density(alpha = 0.3, fill = "lightblue") +
      labs(title = paste("Histogram of", col), x = col, y = "Count")
  )
}

# Boxplots
for (col in num_cols) {
  print(
    ggplot(dss23, aes(x = .data[[col]])) +
      geom_boxplot(fill = "lightblue") +
      labs(title = paste("Boxplot of", col), x = col)
  )
}

# 1. Salary Distribution
ggplot(dss23, aes(salary_in_usd)) +
  geom_histogram(bins = 40, fill = "steelblue") +
  labs(title = "Salary Distribution", x = "Salary (USD)", y = "Count")

ggplot(dss23, aes(salary_in_usd)) +
  geom_density(fill = "lightblue") +
  labs(title = "Salary KDE", x = "Salary (USD)", y = "Density")

# 2. Salary by Experience Level
ggplot(dss23, aes(experience_level, salary_in_usd)) +
  geom_boxplot() +
  labs(title = "Salary by Experience Level")

# 3. Salary by Employment Type
ggplot(dss23, aes(employment_type, salary_in_usd)) +
  geom_boxplot() +
  labs(title = "Salary by Employment Type")

# 4. Salary by Remote Ratio
ggplot(dss23, aes(remote_ratio, salary_in_usd)) +
  geom_boxplot() +
  labs(title = "Salary by Remote Ratio")

# 5. Job Title Frequency (Top 20)
top_jobs <- dss23 %>% count(job_title, sort = TRUE) %>% head(20)

ggplot(top_jobs, aes(n, fct_reorder(job_title, n))) +
  geom_col() +
  labs(title = "Top 20 Job Titles", x = "Count", y = "Job Title")

# 6. Salary by Country (Top 20)
top_countries <- dss23 %>% count(employee_residence, sort = TRUE) %>% head(20)

ggplot(filter(dss23, employee_residence %in% top_countries$employee_residence),
       aes(employee_residence, salary_in_usd)) +
  geom_boxplot() +
  coord_flip() +
  labs(title = "Salary by Employee Residence")

# 7. Correlation Heatmap
ggcorrplot(cor_matrix, lab = TRUE, hc.order = TRUE)

# 8. Geographic Salary Plot
country_salary <- dss23 %>%
  group_by(company_location) %>%
  summarize(median_salary = median(salary_in_usd)) %>%
  arrange(desc(median_salary))

ggplot(country_salary, aes(median_salary, fct_reorder(company_location, median_salary))) +
  geom_col() +
  labs(title = "Median Salary by Company Location", x = "Median Salary", y = "Location")

# =========================================================
# ADDITIONAL GRAPHS (R Versions)
# =========================================================

# Salary Over Time
ggplot(dss23, aes(work_year, salary_in_usd)) +
  geom_boxplot() +
  labs(title = "Salary by Work Year")

# Salary by Company Size
ggplot(dss23, aes(company_size, salary_in_usd)) +
  geom_boxplot()

# Salary vs Remote Ratio (Scatter)
ggplot(dss23, aes(remote_ratio, salary_in_usd)) +
  geom_point(alpha = 0.5)

# Pairplot Equivalent
ggpairs(select(dss23, salary_in_usd, remote_ratio, work_year))

# Countplots
ggplot(dss23, aes(experience_level)) + geom_bar()
ggplot(dss23, aes(employment_type)) + geom_bar()
ggplot(dss23, aes(company_size)) + geom_bar()

# Interaction Plot
ggplot(dss23, aes(experience_level, salary_in_usd, fill = company_size)) +
  geom_boxplot() +
  labs(title = "Salary by Experience Level and Company Size")

# =========================================================
# KEY INSIGHTS FROM VISUALIZATION
# =========================================================
# (Same insights as Python version — unchanged)
# =========================================================
