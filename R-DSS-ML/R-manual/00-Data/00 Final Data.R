# =========================================================
# Unified Pipeline: RAW => CLEAN => FINAL
# Created: 2026-09-14
# Author: Aaron Niecestro

# Description:
# After downloading the 7 datasets and merging them together. I have cleaned the
# data and am ready to do the Exploratory Data Analysis (EDA) in a new file. 
# Doing EDA in another file because it is easier to read and check code.

# Master dataset (never modify)
# master = dss2025

# USA-only
# usa_df = master[master["company_location"] == "United States"].copy()

# Global (all countries)
# global_df = master.copy()

# Non-USA (optional)
# non_usa_df = master[master["company_location"] != "United States"].copy()

# =========================================================

library(tidyverse)
library(janitor)
library(data.table)
library(cluster)
library(factoextra)
library(text2vec)
library(yaml)
library(knitr)

# =========================================================
# 1. DOWNLOAD ALL KAGGLE DATASETS (manual in R)
# =========================================================
# In R, KaggleHub equivalent does not exist.
# Assume datasets already downloaded into a folder.

kaggle_sources <- c(
  "saurabhbadole/latest-data-science-job-salaries-2024",
  "arnabchaki/data-science-salaries-2023",
  "yusufdelikkaya/datascience-salaries-2024",
  "josiagiven/data-science-salaries-and-fields",
  "sazidthe1/data-science-salaries",
  "zain280/data-science-salaries",
  "lainguyn123/data-science-salary-landscape"
)

data_dir <- "kaggle_downloads"

csv_files <- list.files(data_dir, pattern = "\\.csv$", full.names = TRUE)

# =========================================================
# 2. LOAD ALL CSV FILES
# =========================================================

dfs <- lapply(csv_files, function(f) {
  df <- fread(f) |> clean_names()
  df <- df[, !duplicated(names(df)), with = FALSE]
  return(df)
})

cat("Loaded", length(dfs), "datasets.\n")

# =========================================================
# 3. REQUIRED SCHEMA
# =========================================================

required_cols <- c(
  "work_year", "experience_level", "employment_type", "job_title",
  "salary", "salary_currency", "salary_in_usd", "employee_residence",
  "remote_ratio", "company_location", "company_size"
)

# =========================================================
# 4. ALIGN SCHEMA
# =========================================================

aligned_dfs <- lapply(dfs, function(df) {
  df <- df[, !duplicated(names(df)), with = FALSE]
  df <- df[, required_cols, drop = FALSE]
  return(df)
})

# =========================================================
# 5. MERGE RAW DATASETS
# =========================================================

raw_master <- rbindlist(aligned_dfs, fill = TRUE)
raw_master <- unique(raw_master)

fwrite(raw_master, "dss_raw_data.csv")

# =========================================================
# 6. INITIAL CLEANING
# =========================================================

clean_df <- raw_master |> filter(!is.na(work_year))
fwrite(clean_df, "dss_clean_data.csv")

# =========================================================
# 7. FULL CLEANING PIPELINE
# =========================================================

df <- clean_df
df$data_age <- 2026 - as.integer(df$work_year)

# ---------------------------------------------------------
# Experience Level Standardization
# ---------------------------------------------------------

experience_map <- c(
  "EN"="Entry-Level","Entry"="Entry-Level","Entry Level"="Entry-Level",
  "Entry-Level"="Entry-Level",
  "MI"="Middle-Level","Mid"="Middle-Level","Middle"="Middle-Level",
  "Mid Level"="Middle-Level","Middle Level"="Middle-Level",
  "Middle-Level"="Middle-Level",
  "SE"="Senior-Level","Senior"="Senior-Level","Senior Level"="Senior-Level",
  "Senior-Level"="Senior-Level",
  "EX"="Executive-Level","Executive"="Executive-Level",
  "Executive Level"="Executive-Level","Executive-Level"="Executive-Level"
)

df$experience_level <- recode(df$experience_level, !!!experience_map)
df$experience_level <- factor(df$experience_level,
                              levels=c("Entry-Level","Middle-Level","Senior-Level","Executive-Level"),
                              ordered=TRUE)

# ---------------------------------------------------------
# Employment Type
# ---------------------------------------------------------

employment_map <- c(
  "FT"="Full-time","Full-time"="Full-time",
  "PT"="Part-time","Part-time"="Part-time",
  "CT"="Contract/Freelance","Contract"="Contract/Freelance",
  "FL"="Contract/Freelance","Freelance"="Contract/Freelance"
)

df$employment_type <- recode(df$employment_type, !!!employment_map)

# ---------------------------------------------------------
# Company Size
# ---------------------------------------------------------

size_map <- c("M"="Medium","L"="Large","S"="Small")
df$company_size <- recode(df$company_size, !!!size_map)

# ---------------------------------------------------------
# Categorical Versions
# ---------------------------------------------------------

df$work_year_cat <- factor(df$work_year)

remote_map <- c("0"="On-Site","50"="Hybrid","100"="Remote")
df$remote_work_cat <- factor(remote_map[as.character(df$remote_ratio)],
                             levels=c("On-Site","Hybrid","Remote"),
                             ordered=TRUE)

# Salary categories
mean_salary <- mean(df$salary_in_usd, na.rm=TRUE)
median_salary <- median(df$salary_in_usd, na.rm=TRUE)
q1 <- quantile(df$salary_in_usd, 0.33, na.rm=TRUE)
q2 <- quantile(df$salary_in_usd, 0.66, na.rm=TRUE)

df$salary_mean_cat <- ifelse(df$salary_in_usd >= mean_salary, "Above-Average", "Below-Average")
df$salary_median_cat <- ifelse(df$salary_in_usd >= median_salary, "Above-Median", "Below-Median")

df$salary_3cat <- cut(df$salary_in_usd,
                      breaks=c(-Inf, q1, q2, Inf),
                      labels=c("Low-Salary","Middle-Salary","High-Salary"),
                      ordered_result=TRUE)

# ---------------------------------------------------------
# ISO Country Mapping
# ---------------------------------------------------------

iso_to_country <- list(
  US="United States", CA="Canada", GB="United Kingdom", DE="Germany", FR="France",
  IN="India", CN="China", JP="Japan", KR="South Korea", BR="Brazil", MX="Mexico",
  AU="Australia", NZ="New Zealand", SG="Singapore", CH="Switzerland", SE="Sweden",
  NO="Norway", FI="Finland", DK="Denmark", IE="Ireland", IT="Italy", ES="Spain",
  PT="Portugal", NL="Netherlands", BE="Belgium", AT="Austria", PL="Poland",
  CZ="Czech Republic", RO="Romania", RU="Russia", AR="Argentina", CL="Chile",
  CO="Colombia", PE="Peru", PH="Philippines", PK="Pakistan", ID="Indonesia",
  MY="Malaysia", TH="Thailand", TR="Turkey", SA="Saudi Arabia",
  AE="United Arab Emirates", ZA="South Africa", NG="Nigeria", KE="Kenya",
  GH="Ghana", HK="Hong Kong", HN="Honduras", MU="Mauritius", BS="Bahamas",
  XK="Kosovo", AM="Armenia", AL="Albania", LT="Lithuania", LV="Latvia",
  LU="Luxembourg", MD="Moldova", MK="North Macedonia", MT="Malta",
  SI="Slovenia", SK="Slovakia", GR="Greece", HR="Croatia", BG="Bulgaria",
  RS="Serbia", UA="Ukraine", VE="Venezuela", VN="Vietnam", IQ="Iraq",
  IR="Iran", IL="Israel", LB="Lebanon", EG="Egypt", DZ="Algeria",
  MA="Morocco", CF="Central African Republic", CD="Democratic Republic of Congo",
  GI="Gibraltar", AS="American Samoa", PR="Puerto Rico", UZ="Uzbekistan",
  BO="Bolivia", HU="Hungary", DO="Dominican Republic", CR="Costa Rica",
  KW="Kuwait", TN="Tunisia", UG="Uganda", GE="Georgia"
)

df$company_location <- recode(df$company_location, !!!iso_to_country)
df$employee_residence <- recode(df$employee_residence, !!!iso_to_country)

# ---------------------------------------------------------
# Filter Locations ≥100
# ---------------------------------------------------------

valid_locations <- names(which(table(df$company_location) >= 100))
df <- df |> filter(company_location %in% valid_locations)

valid_residence <- names(which(table(df$employee_residence) >= 100))
df <- df |> filter(employee_residence %in% valid_residence)

# ---------------------------------------------------------
# Job Title Grouping
# ---------------------------------------------------------

job_title_map <- yaml::read_yaml("job_title_map.yaml")  # optional external file

df$job_title_group <- unlist(job_title_map[df$job_title])
df$job_title_group[is.na(df$job_title_group)] <- "Other"
df$job_title_group <- factor(df$job_title_group)

# =========================================================
# NLP CLUSTERING (TF-IDF + KMeans)
# =========================================================

other_df <- df |> filter(job_title_group == "Other")

titles <- other_df$job_title

it <- itoken(titles, progress_bar = FALSE)
vocab <- create_vocabulary(it, ngram = c(1L, 2L), stopwords = stopwords::stopwords("en"))
vocab <- prune_vocabulary(vocab, term_count_min = 2)

vectorizer <- vocab_vectorizer(vocab)
X <- create_dtm(it, vectorizer)

k <- 12
km <- kmeans(X, centers = k, nstart = 10)

other_df$nlp_cluster <- km$cluster

get_top_terms <- function(cluster_id, n_terms = 8) {
  centroid <- km$centers[cluster_id, ]
  top_idx <- order(centroid, decreasing = TRUE)[1:n_terms]
  vocab$term[top_idx]
}

cluster_names <- sapply(1:k, function(cid) {
  paste(get_top_terms(cid)[1:3], collapse = ", ")
})

other_df$nlp_cluster_name <- cluster_names[other_df$nlp_cluster]

df$job_title_group[df$job_title_group == "Other"] <- other_df$nlp_cluster_name

# ---------------------------------------------------------
# Clean NLP cluster names
# ---------------------------------------------------------

nlp_clean_map <- c(
  "data, consultant, engineer"="Data Consulting & Engineering",
  "business, intelligence, business intelligence"="Business Intelligence / BI",
  "architect, data architect, data"="Data Architecture",
  "analytics engineer, analytics, engineer"="Analytics Engineering",
  "product manager, product, manager"="Product Management",
  "developer, software, software developer"="Software Development",
  "research engineer, research, engineer"="Research Engineering",
  "associate, team lead, team"="Team Lead / Associate Roles",
  "data manager, manager, data"="Data Management",
  "scientist, applied scientist, applied"="Applied Science",
  "solutions architect, solutions, architect"="Solutions Architecture",
  "devops engineer, devops, engineer"="DevOps Engineering"
)

df$job_title_group <- recode(df$job_title_group, !!!nlp_clean_map)

# ---------------------------------------------------------
# Convert Data Types
# ---------------------------------------------------------

numeric_cols <- c("salary_in_usd","remote_ratio","data_age")
df[numeric_cols] <- lapply(df[numeric_cols], as.numeric)

categorical_cols <- c(
  "work_year","experience_level","employment_type","job_title_group",
  "employee_residence","company_location","company_size","work_year_cat",
  "remote_work_cat","salary_mean_cat","salary_median_cat","salary_3cat"
)

df[categorical_cols] <- lapply(df[categorical_cols], factor)

# ---------------------------------------------------------
# Drop salary + salary_currency
# ---------------------------------------------------------

df <- df |> select(-salary, -salary_currency)

# ---------------------------------------------------------
# Remove duplicates & missing target
# ---------------------------------------------------------

df <- df |> distinct() |> filter(!is.na(salary_in_usd))

# =========================================================
# 8. SAVE FINAL CLEANED DATASET
# =========================================================

fwrite(df, "dss2025_final.csv")

# =========================================================
# 9–20. Summaries, profiling, missing values, etc.
# =========================================================

summary(df)
str(df)
skimr::skim(df)

missing_df <- tibble(
  column = names(df),
  missing_count = colSums(is.na(df)),
  missing_percent = colSums(is.na(df)) / nrow(df) * 100
)

print(missing_df)

# Imputation functions
impute_experience_level <- function(df) {
  senior_map <- list(
    "Executive-Level"=c("executive","vp","vice president","chief","cto","ceo","cso","head of"),
    "Senior-Level"=c("senior","sr","lead","principal","staff","director","manager"),
    "Middle-Level"=c("mid","intermediate","associate","specialist","analyst ii","analyst 2"),
    "Entry-Level"=c("junior","jr","entry","intern","assistant","analyst i","analyst 1")
  )
  
  infer_level <- function(title) {
    t <- tolower(title)
    for (lvl in names(senior_map)) {
      if (any(str_detect(t, senior_map[[lvl]]))) return(lvl)
    }
    return("Middle-Level")
  }
  
  df$experience_level <- ifelse(
    is.na(df$experience_level),
    sapply(df$job_title, infer_level),
    df$experience_level
  )
  
  df$experience_level <- factor(df$experience_level,
                                levels=c("Entry-Level","Middle-Level","Senior-Level","Executive-Level"),
                                ordered=TRUE)
  df
}

impute_remote_ratio <- function(df) {
  remote_mask <- is.na(df$remote_ratio) & df$employee_residence != df$company_location
  df$remote_ratio[remote_mask] <- 100
  df$remote_ratio[is.na(df$remote_ratio)] <- 50
  
  remote_map <- c("0"="On-Site","50"="Hybrid","100"="Remote")
  df$remote_work_cat <- factor(remote_map[as.character(df$remote_ratio)],
                               levels=c("On-Site","Hybrid","Remote"),
                               ordered=TRUE)
  df
}

df <- impute_experience_level(df)
df <- impute_remote_ratio(df)

# Final save
fwrite(df, "dss2025_final.csv")
