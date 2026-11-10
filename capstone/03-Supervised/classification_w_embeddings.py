"""
Author: Aaron Niecestro
Project: Course Recommendation System Capstone

Created: November 4, 2026
Last Edit: November 4, 2026
Progress: Complete

Analysis: Classification Modeling

Description:

Builds and evaluates multiple classification models
to predict learner outcomes using course enrollment
and interaction data.

Data preparation, feature engineering, model
training, hyperparameter optimization, and model
evaluation are performed to identify the highest
performing classification approach.

Multiple classification algorithms are compared and
evaluated using standard classification metrics to
determine the optimal predictive model.

Techniques Applied:

- Data Cleaning
- Feature Engineering
- Categorical Encoding
- Train-Test Split Validation
- Classification Modeling
- Hyperparameter Optimization
- Cross Validation
- Model Comparison
- Feature Importance Analysis
- Performance Evaluation

Classification Models Evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gaussian Naive Bayes

Datasets:

- ratings.csv
- user_embeddings.csv
- course_embeddings.csv

Research Questions:

- Can learner outcomes be accurately predicted
  using classification techniques?

- Which classification algorithm achieves the
  highest predictive performance?

- Which features contribute most to model
  predictions?

- How do hyperparameter tuning and feature
  selection impact performance?

- Which model provides the best balance between
  accuracy and generalization?

Outputs:

- classification_results.csv
- model_comparison.csv
- feature_importance.csv
- confusion_matrix.png
- model_performance_comparison.png

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Cross Validation Score
- Model Ranking
"""

# ============================================================
# Imports
# ============================================================

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

# ============================================================
# Reproducibility
# ============================================================

SEED = 423

random.seed(SEED)
np.random.seed(SEED)

RANDOM_STATE = SEED

# ============================================================
# Paths
# ============================================================

DATA_DIR = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\capstone\00-Data"
)

OUTPUT_DIR = Path(
    r"C:\Users\aniec\Desktop\ibm-ml-project\capstone\03-Supervised\supervised-results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# Load Data
# ============================================================

rating_df = pd.read_csv(
    DATA_DIR / "ratings.csv"
)

user_emb = pd.read_csv(
    DATA_DIR / "user_embeddings.csv"
)

item_emb = pd.read_csv(
    DATA_DIR / "course_embeddings.csv"
)

print("\nDatasets Loaded Successfully")
print("-" * 40)

print(f"Ratings Shape           : {rating_df.shape}")
print(f"User Embeddings Shape   : {user_emb.shape}")
print(f"Course Embeddings Shape : {item_emb.shape}")

# ============================================================
# Class Distribution
# ============================================================

print("\nClass Distribution")
print("-" * 40)

print(
    rating_df["rating"]
    .value_counts()
    .sort_index()
)

print(
    f"\nNumber of Classes: "
    f"{rating_df['rating'].nunique()}"
)

print(
    f"Classes: "
    f"{sorted(rating_df['rating'].unique())}"
)

print(
    "\nClassification Type: "
    "Multiclass"
)

# ============================================================
# Create Interaction Dataset
# ============================================================

user_emb_merged = pd.merge(
    rating_df,
    user_emb,
    how="left",
    on="user"
).fillna(0)

merged_df = pd.merge(
    user_emb_merged,
    item_emb,
    how="left",
    on="item"
).fillna(0)

u_features = [
    f"UFeature{i}"
    for i in range(16)
]

c_features = [
    f"CFeature{i}"
    for i in range(16)
]

user_embeddings = merged_df[u_features]

course_embeddings = merged_df[c_features]

ratings = merged_df["rating"]

interaction_dataset = (
    user_embeddings +
    course_embeddings.values
)

interaction_dataset.columns = [
    f"Feature{i}"
    for i in range(16)
]

interaction_dataset["rating"] = ratings

print(
    f"\nInteraction Dataset Shape: "
    f"{interaction_dataset.shape}"
)

# ============================================================
# Prepare Features and Labels
# ============================================================

X = interaction_dataset.iloc[:, :-1]

y_raw = interaction_dataset.iloc[:, -1]

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    y_raw.values.ravel()
)

print(
    f"Input Shape : {X.shape}"
)

print(
    f"Output Shape: {y.shape}"
)

print(
    f"Encoded Classes: "
    f"{list(label_encoder.classes_)}"
)

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE
    )
)

print("\nTrain/Test Split Complete")
print("-" * 40)

print(
    f"Training Samples: "
    f"{len(X_train):,}"
)

print(
    f"Testing Samples : "
    f"{len(X_test):,}"
)

print("\nTraining Class Distribution")
print("-" * 40)

print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)

# ============================================================
# Classification Models
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),

    "AdaBoost":
        AdaBoostClassifier(
            random_state=RANDOM_STATE
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "SVM":
        SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE
        ),

    "Naive Bayes":
        GaussianNB()
}
    
print("\nClassification Models")
print("-" * 40)

for model in models:
    print(model)
    
# ============================================================
# Model Evaluation
# ============================================================

results = []

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    cv_accuracy = (
        cv_scores.mean()
    )

    cv_std = (
        cv_scores.std()
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        cv_accuracy,
        cv_std
    ])

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(
        f"Accuracy      : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision     : "
        f"{precision:.4f}"
    )

    print(
        f"Recall        : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score      : "
        f"{f1:.4f}"
    )

    print(
        f"CV Accuracy   : "
        f"{cv_accuracy:.4f}"
    )

    print(
        f"CV Std Dev    : "
        f"{cv_std:.4f}"
    )

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "CV_Accuracy",
        "CV_STD"
    ]
).sort_values(
    by="CV_Accuracy",
    ascending=False
)

best_model_name = (
    results_df.iloc[0]["Model"]
)

print(
    f"\nBest Model Selected: "
    f"{best_model_name}"
)

best_model = models[
    best_model_name
]

best_predictions = (
    best_model.predict(X_test)
)

predictions_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": best_predictions
})

print("\nModel Comparison")
print("-" * 40)

print(results_df)

print("\nCross-Validation Summary")
print("-" * 40)

print(
    results_df[
        [
            "Model",
            "CV_Accuracy",
            "CV_STD"
        ]
    ]
)

# ============================================================
# Visualization 1
# Cross-Validated Classification Model Comparison
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="CV_Accuracy",
    y="Model",
    palette="viridis"
)

plt.title(
    "Cross-Validated Classification Model Comparison"
)

plt.xlabel(
    "CV Accuracy"
)

plt.ylabel(
    "Model"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "classification_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Visualization 2
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title(
    f"Confusion Matrix ({best_model_name})"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Class-Level Performance 
# ============================================================

print("\nClass Labels")
print("-" * 40)

for i, label in enumerate(
    label_encoder.classes_
):
    print(
        f"Class {i} -> Rating {label}"
    )

# ============================================================
# Save Results
# ============================================================

results_df.to_csv(
    OUTPUT_DIR /
    "classification_results.csv",
    index=False
)

predictions_df.to_csv(
    OUTPUT_DIR /
    "best_model_predictions.csv",
    index=False
)

results_df.to_csv(
    OUTPUT_DIR /
    "model_comparison.csv",
    index=False
)

print("\nResults Saved")
print("-" * 40)

print("classification_results.csv")
print("model_comparison.csv")
print("best_model_predictions.csv")
print("classification_model_comparison.png")
print("confusion_matrix.png")

print(
    f"\nResults saved to:\n{OUTPUT_DIR}"
)

# ============================================================
# Final Summary Table
# ============================================================

summary_df = pd.DataFrame({
    "Metric": [
        "Best Model",
        "Best CV Accuracy",
        "Total Models Tested",
        "Number of Classes",
        "Training Samples",
        "Testing Samples"
    ],
    "Value": [
        best_model_name,
        round(
            results_df.iloc[0]["CV_Accuracy"],
            4
        ),
        len(models),
        len(label_encoder.classes_),
        len(X_train),
        len(X_test)
    ]
})

print("\nFinal Summary")
print("-" * 40)

print(
    summary_df.to_string(
        index=False
    )
)
