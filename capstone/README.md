# **06 — Machine Learning Capstone**

- **Status:** Complete
- **Last Updated:** November 10, 2026

---

## **📌 Module Overview**

This Capstone Project integrates the concepts and techniques developed throughout the IBM Machine Learning Professional Certificate into a complete end-to-end recommendation system.

The project focuses on building and evaluating personalized online course recommenders using both content-based and collaborative-filtering approaches. Multiple machine learning and deep learning models were developed, compared, and evaluated to determine the most effective recommendation strategy.

---

## **🎯 Project Objectives**

- Explore learner enrollment and rating behavior through exploratory data analysis (EDA)
- Develop content-based recommendation systems
- Implement clustering-based recommendation methods
- Build collaborative-filtering recommendation models
- Evaluate matrix factorization and neural network approaches
- Compare model performance using RMSE and other evaluation metrics
- Identify the most effective approach for personalized course recommendation

---

## **📊 Dataset Summary**

- 233,306 course ratings
- 33,901 users
- Course metadata including genres and content features
- User profile and interaction data
- Course similarity and embedding datasets

---

## **🔍 Exploratory Data Analysis**

Completed analyses include:

- Course counts by genre
- Enrollment distribution analysis
- Top 20 most popular courses
- Course title word-cloud analysis
- Learner behavior and engagement patterns

---

## **🤖 Recommendation Systems Developed**

### Content-Based Recommenders
- User Profile-Based Recommender
- Course Similarity-Based Recommender

### Unsupervised Learning
- K-Means Clustering Recommender
- PCA-Based User Segmentation

### Collaborative Filtering
- KNN Collaborative Filtering
- Non-Negative Matrix Factorization (NMF)

### Deep Learning
- Neural Collaborative Filtering
- User and Course Embedding Models

### Supervised Learning
- Embedding-Based Regression Models
- Classification Models

---

## **📈 Key Results**

### Content-Based Recommenders
- Successfully generated personalized recommendations using user interests and course content

### Clustering Recommender
- Optimal Clusters: 2
- PCA improved silhouette score from 0.3973 to 0.4101

### KNN Collaborative Filtering
- Best Configuration:
  - Item-Based KNN
  - MSD Similarity
  - K = 35
- RMSE: 0.8951

### NMF
- Best Latent Factors: 4
- RMSE: 3.2763
- Performance negatively impacted by dataset sparsity

### Neural Collaborative Filtering
- Best Embedding Size: 8
- RMSE: 0.4219
- Best-performing recommendation model

---

## **🏆 Final Conclusion**

Neural Collaborative Filtering achieved the strongest overall performance and produced the most accurate rating predictions among all evaluated approaches.

Content-based filtering and clustering methods effectively generated personalized recommendations, while neural network embeddings provided the most accurate representation of user-course relationships.

---

## **🛠 Technologies Used**

- Python
- Pandas
- NumPy
- Scikit-Learn
- TensorFlow / Keras
- Matplotlib
- Seaborn

---

## **📁 Project Deliverables**

- Exploratory Data Analysis Notebooks
- Content-Based Recommender Systems
- Clustering-Based Recommender System
- KNN Collaborative Filtering
- NMF Recommender
- Neural Collaborative Filtering
- Regression Models
- Classification Models
- PowerPoint Presentation
- Performance Comparison Visualizations

---

## **🔗 Repository**

GitHub:
https://github.com/niecestroa/pro-ibm-ml-course-projects/tree/main/capstone

---

# **(End of README)**
