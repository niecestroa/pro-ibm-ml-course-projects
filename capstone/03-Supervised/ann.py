"""
Author: Aaron Niecestro
Project: Course Recommendation System Capstone

Created: November 4, 2026
Last Edit: November 4, 2026
Progress: Complete

Analysis: Neural Network Collaborative Filtering

Method:
Embedding-Based Recommendation System
Using Artificial Neural Networks

Description:

Builds a recommendation system using neural
networks and embedding layers to learn latent
user and course representations directly from
interaction data.

User and course identifiers are transformed into
dense embedding vectors which are learned during
model training. Rating predictions are generated
through the interaction of user and course
embeddings and evaluated using Root Mean Squared
Error (RMSE).

Multiple embedding dimensions are evaluated to
determine the optimal neural network architecture
and recommendation performance.

Techniques Applied:

- Collaborative Filtering
- Neural Networks
- Embedding Layers
- Latent Feature Learning
- User Embedding Extraction
- Course Embedding Extraction
- Hyperparameter Optimization
- Deep Learning
- Train/Validation/Test Split
- RMSE Evaluation
- Recommendation System Analysis

Datasets:

- ratings.csv

Research Questions:

- Can neural networks accurately predict
  user-course ratings?

- What embedding size produces the best
  recommendation performance?

- Can embedding layers effectively learn
  latent user and course features?

- How does recommendation accuracy change
  as embedding dimensions increase?

- What neural network configuration
  minimizes RMSE?

Outputs:

- ann_results.csv
- model_comparison.csv
- user_embeddings.csv
- course_embeddings.csv
- ann_rmse_comparison.png
- embedding_size_comparison.png
- learning_curve.png

Evaluation Metrics:

- RMSE
- Validation Loss
- Test Loss
- Embedding Size Comparison
- Model Ranking
"""

# ============================================================
# Imports
# ============================================================

import random
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

# ============================================================
# Reproducibility
# ============================================================

SEED = 423

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

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

print("\nDataset Loaded Successfully")
print("-" * 40)

print(
    f"Dataset Shape : "
    f"{rating_df.shape}"
)

# ============================================================
# Dataset Summary
# ============================================================

print("\nDataset Summary")
print("-" * 40)

print(
    f"Total Ratings : "
    f"{len(rating_df):,}"
)

print(
    f"Unique Users : "
    f"{rating_df['user'].nunique():,}"
)

print(
    f"Unique Courses : "
    f"{rating_df['item'].nunique():,}"
)

print("\nRating Distribution")
print("-" * 40)

print(
    rating_df["rating"]
    .value_counts()
    .sort_index()
)

# ============================================================
# Encode Users and Courses
# ============================================================

def process_dataset(
    raw_data
):

    encoded_data = (
        raw_data.copy()
    )

    user_list = (
        encoded_data["user"]
        .unique()
        .tolist()
    )

    course_list = (
        encoded_data["item"]
        .unique()
        .tolist()
    )

    user_id2idx = {
        x: i
        for i, x in enumerate(user_list)
    }

    course_id2idx = {
        x: i
        for i, x in enumerate(course_list)
    }

    user_idx2id = {
        i: x
        for i, x in enumerate(user_list)
    }

    course_idx2id = {
        i: x
        for i, x in enumerate(course_list)
    }

    encoded_data["user"] = (
        encoded_data["user"]
        .map(user_id2idx)
    )

    encoded_data["item"] = (
        encoded_data["item"]
        .map(course_id2idx)
    )

    encoded_data["rating"] = (
        encoded_data["rating"]
        .astype(int)
    )

    return (
        encoded_data,
        user_idx2id,
        course_idx2id
    )

encoded_data, user_idx2id, course_idx2id = (
    process_dataset(
        rating_df
    )
)

num_users = (
    encoded_data["user"]
    .nunique()
)

num_items = (
    encoded_data["item"]
    .nunique()
)

print("\nEncoded Dataset")
print("-" * 40)

print(
    f"Users   : "
    f"{num_users:,}"
)

print(
    f"Courses : "
    f"{num_items:,}"
)

# ============================================================
# Train / Validation / Test Split
# ============================================================

def generate_train_test_datasets(
    dataset,
    scale=True
):

    min_rating = (
        dataset["rating"]
        .min()
    )

    max_rating = (
        dataset["rating"]
        .max()
    )

    dataset = dataset.sample(
        frac=1,
        random_state=SEED
    )

    x = dataset[
        ["user", "item"]
    ].values

    if scale:

        y = dataset[
            "rating"
        ].apply(
            lambda x:
            (x - min_rating)
            /
            (max_rating - min_rating)
        ).values

    else:

        y = dataset[
            "rating"
        ].values

    train_index = (
        int(
            0.8 *
            len(dataset)
        )
    )

    test_index = (
        int(
            0.9 *
            len(dataset)
        )
    )

    x_train = (
        x[:train_index]
    )

    x_val = (
        x[
            train_index:test_index
        ]
    )

    x_test = (
        x[test_index:]
    )

    y_train = (
        y[:train_index]
    )

    y_val = (
        y[
            train_index:test_index
        ]
    )

    y_test = (
        y[test_index:]
    )

    return (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test
    )

(
    x_train,
    x_val,
    x_test,
    y_train,
    y_val,
    y_test
) = generate_train_test_datasets(
    encoded_data
)

print("\nDataset Splits")
print("-" * 40)

print(
    f"Train Samples      : "
    f"{len(x_train):,}"
)

print(
    f"Validation Samples : "
    f"{len(x_val):,}"
)

print(
    f"Test Samples       : "
    f"{len(x_test):,}"
)

# ============================================================
# RecommenderNet Architecture
# ============================================================

class RecommenderNet(
    keras.Model
):

    def __init__(
        self,
        num_users,
        num_items,
        embedding_size=16,
        **kwargs
    ):

        super(
            RecommenderNet,
            self
        ).__init__(
            **kwargs
        )

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        # User Embeddings

        self.user_embedding_layer = (
            layers.Embedding(
                input_dim=num_users,
                output_dim=embedding_size,
                name="user_embedding_layer",
                embeddings_initializer="he_normal",
                embeddings_regularizer=
                keras.regularizers.l2(
                    1e-6
                )
            )
        )

        self.user_bias = (
            layers.Embedding(
                input_dim=num_users,
                output_dim=1,
                name="user_bias"
            )
        )

        # Item Embeddings

        self.item_embedding_layer = (
            layers.Embedding(
                input_dim=num_items,
                output_dim=embedding_size,
                name="item_embedding_layer",
                embeddings_initializer="he_normal",
                embeddings_regularizer=
                keras.regularizers.l2(
                    1e-6
                )
            )
        )

        self.item_bias = (
            layers.Embedding(
                input_dim=num_items,
                output_dim=1,
                name="item_bias"
            )
        )

    def call(
        self,
        inputs
    ):

        user_vector = (
            self.user_embedding_layer(
                inputs[:, 0]
            )
        )

        user_bias = (
            self.user_bias(
                inputs[:, 0]
            )
        )

        item_vector = (
            self.item_embedding_layer(
                inputs[:, 1]
            )
        )

        item_bias = (
            self.item_bias(
                inputs[:, 1]
            )
        )

        dot_user_item = (
            tf.tensordot(
                user_vector,
                item_vector,
                2
            )
        )

        x = (
            dot_user_item +
            user_bias +
            item_bias
        )

        return tf.nn.relu(x)
    
# ============================================================
# Embedding Size Search
# ============================================================

embedding_sizes = [
    4,
    8,
    16,
    32,
    64
]

results = []

best_history = None
best_model = None
best_rmse = np.inf

print("\nEmbedding Search")
print("-" * 40)

# ============================================================
# Early Stopping
# ============================================================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# ============================================================
# Model Training and Evaluation
# ============================================================

for embedding_size in embedding_sizes:

    print(
        f"\nEmbedding Size = "
        f"{embedding_size}"
    )

    model = RecommenderNet(
        num_users=num_users,
        num_items=num_items,
        embedding_size=embedding_size
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(),
        metrics=[
            tf.keras.metrics.RootMeanSquaredError()
        ]
    )
    
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(
            x_val,
            y_val
        ),
        epochs=20,
        batch_size=64,
        callbacks=[
            early_stop
        ],
        verbose=1
    )

    evaluation = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    test_loss = evaluation[0]
    test_rmse = evaluation[1]

    results.append([
        embedding_size,
        test_loss,
        test_rmse
    ])

    print(
        f"Loss  : "
        f"{test_loss:.4f}"
    )

    print(
        f"RMSE  : "
        f"{test_rmse:.4f}"
    )

    if test_rmse < best_rmse:

        best_rmse = test_rmse
        best_history = history
        best_model = model
        
# ============================================================
# Results DataFrame
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Embedding_Size",
        "Loss",
        "RMSE"
    ]
)

results_df = (
    results_df
    .sort_values(
        by="RMSE",
        ascending=True
    )
)

print("\nANN Results")
print("-" * 40)

print(results_df)

# ============================================================
# Best Configuration
# ============================================================

best_result = (
    results_df.iloc[0]
)

best_embedding = int(
    best_result[
        "Embedding_Size"
    ]
)

best_loss = (
    best_result["Loss"]
)

best_rmse = (
    best_result["RMSE"]
)

print("\nBest Configuration")
print("-" * 40)

print(
    f"Embedding Size : "
    f"{best_embedding}"
)

print(
    f"Loss           : "
    f"{best_loss:.4f}"
)

print(
    f"RMSE           : "
    f"{best_rmse:.4f}"
)

# ============================================================
# Improvement Analysis
# ============================================================

worst_rmse = results_df["RMSE"].max()

improvement_pct = (
    (
        worst_rmse -
        best_rmse
    )
    /
    worst_rmse
) * 100

print("\nImprovement Analysis")
print("-" * 40)

print(
    f"Best RMSE  : "
    f"{best_rmse:.4f}"
)

print(
    f"Worst RMSE : "
    f"{worst_rmse:.4f}"
)

print(
    f"Improvement : "
    f"{improvement_pct:.2f}%"
)

# ============================================================
# Extract Embeddings
# ============================================================

user_latent_features = (
    best_model
    .get_layer(
        "user_embedding_layer"
    )
    .get_weights()[0]
)

item_latent_features = (
    best_model
    .get_layer(
        "item_embedding_layer"
    )
    .get_weights()[0]
)

print("\nEmbedding Shapes")
print("-" * 40)

print(
    f"User Embeddings  : "
    f"{user_latent_features.shape}"
)

print(
    f"Item Embeddings  : "
    f"{item_latent_features.shape}"
)

# ============================================================
# Embedding DataFrames
# ============================================================

user_embeddings_df = pd.DataFrame(
    user_latent_features
)

item_embeddings_df = pd.DataFrame(
    item_latent_features
)

print("\nEmbedding Tables")
print("-" * 40)

print(
    f"User Embeddings Shape : "
    f"{user_embeddings_df.shape}"
)

print(
    f"Item Embeddings Shape : "
    f"{item_embeddings_df.shape}"
)

# ============================================================
# Save Embeddings
# ============================================================

user_embeddings_df.to_csv(
    OUTPUT_DIR /
    "user_embeddings.csv",
    index=False
)

item_embeddings_df.to_csv(
    OUTPUT_DIR /
    "course_embeddings.csv",
    index=False
)

print("\nEmbeddings Saved")
print("-" * 40)

print("user_embeddings.csv")
print("course_embeddings.csv")

# ============================================================
# Visualization 1
# RMSE vs Embedding Size
# ============================================================

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=results_df,
    x="Embedding_Size",
    y="RMSE",
    marker="o",
    color="#1f77b4"
)

plt.title(
    "ANN RMSE by Embedding Size"
)

plt.xlabel(
    "Embedding Size"
)

plt.ylabel(
    "RMSE"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "ann_rmse_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Visualization 2
# Loss vs Embedding Size
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="Embedding_Size",
    y="Loss",
    color="#1f77b4"
)

plt.title(
    "Model Loss by Embedding Size"
)

plt.xlabel(
    "Embedding Size"
)

plt.ylabel(
    "Loss"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "embedding_size_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Visualization 3
# Learning Curve
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    best_history.history["loss"],
    label="Training Loss",
    color="#1f77b4"
)

plt.plot(
    best_history.history["val_loss"],
    label="Validation Loss",
    color="#ff7f0e"
)

plt.title(
    "Training vs Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "learning_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Improvement Analysis
# ============================================================

worst_rmse = (
    results_df["RMSE"]
    .max()
)

rmse_improvement = (
    (
        worst_rmse -
        best_rmse
    )
    /
    worst_rmse
) * 100

print("\nImprovement Analysis")
print("-" * 40)

print(
    f"Best RMSE      : "
    f"{best_rmse:.4f}"
)

print(
    f"Worst RMSE     : "
    f"{worst_rmse:.4f}"
)

print(
    f"Improvement (%) : "
    f"{rmse_improvement:.2f}"
)

# ============================================================
# Save Results
# ============================================================

results_df.to_csv(
    OUTPUT_DIR /
    "ann_results.csv",
    index=False
)

results_df.to_csv(
    OUTPUT_DIR /
    "model_comparison.csv",
    index=False
)

print("\nResults Saved")
print("-" * 40)

print("ann_results.csv")
print("model_comparison.csv")
print("user_embeddings.csv")
print("course_embeddings.csv")
print("ann_rmse_comparison.png")
print("embedding_size_comparison.png")
print("learning_curve.png")

# ============================================================
# ANN Performance Summary
# ============================================================

print("\nANN Results")
print("-" * 40)

print(
    results_df.to_string(
        index=False
    )
)

# ============================================================
# Improvement Analysis
# ============================================================

worst_rmse = (
    results_df["RMSE"]
    .max()
)

rmse_improvement = (
    (
        worst_rmse -
        best_rmse
    )
    /
    worst_rmse
) * 100

print("\nImprovement Analysis")
print("-" * 40)

print(
    f"Best RMSE      : "
    f"{best_rmse:.4f}"
)

print(
    f"Worst RMSE     : "
    f"{worst_rmse:.4f}"
)

print(
    f"Improvement (%) : "
    f"{rmse_improvement:.2f}"
)

# ============================================================
# Final Summary Table
# ============================================================

summary_df = pd.DataFrame({
    "Metric": [
        "Best Embedding Size",
        "Best Loss",
        "Best RMSE",
        "Users",
        "Courses",
        "Ratings"
    ],
    "Value": [
        best_embedding,
        round(best_loss, 4),
        round(best_rmse, 4),
        f"{rating_df['user'].nunique():,}",
        f"{rating_df['item'].nunique():,}",
        f"{len(rating_df):,}"
    ]
})

print("\nFinal Summary")
print("-" * 40)

print(
    summary_df.to_string(
        index=False
    )
)

# ============================================================
# Findings
# ============================================================

print("\nKey Findings")
print("-" * 40)

print("""
• Neural collaborative filtering successfully learned latent user and course
representations.
""")

print("""
• Embedding size 8 achieved the strongest performance with an RMSE of 0.4424.
""")

print("""
• Larger embedding dimensions increased model complexity without improving accuracy.
""")

print("""
• Validation loss increased after the first few epochs, indicating overfitting.
""")

print("""
• Early stopping helped preserve the best-performing model weights.
""")

print("""
• The trained neural network generated compact latent feature representations 
for 33,901 users and 126 courses.:
""")
