# Credit Card Fraud Detection using Machine Learning and Sampling Techniques

## 1. Introduction
Credit card fraud detection is a critical real-world problem due to the increasing number of online transactions. One of the major challenges in fraud detection is the **highly imbalanced nature of the dataset**, where fraudulent transactions constitute only a very small fraction of the total data.

This project aims to analyze the impact of **different sampling techniques** on the performance of various **machine learning classifiers** for fraud detection. The study compares model accuracy under multiple data resampling strategies and identifies the best-performing sampling method for each model.

## 2. Objective of the Project
The main objectives of this project are:
- To handle class imbalance using different sampling techniques
- To evaluate multiple machine learning models on the same dataset
- To compare model performance using accuracy
- To determine the most effective sampling strategy for each model

## 3. Dataset Description
- **Dataset Name:** Creditcard_data.csv
- **Target Variable:** `Class`
  - `0` → Legitimate transaction
  - `1` → Fraudulent transaction

## 4. Methodology

### 4.1 Data Preprocessing
1. The dataset is loaded using Pandas.
2. The feature matrix `X` and target vector `y` are separated.
3. **StandardScaler** is applied to normalize all feature values.
   - This ensures that all features contribute equally to model training.
   - Scaling is especially important for distance-based models like KNN and SVM.

### 4.2 Machine Learning Models Used
The following classifiers are used in this study:
Logistic Regression 
Decision Tree Classifier 
Random Forest Classifier 
Support Vector Machine (SVC) 
K-Nearest Neighbors (KNN) 

### 4.3 Sampling Techniques Applied

# 1 Bootstrap Sampling
# 2 Random Under-Sampling
# 3 Random Over-Sampling
# 4 Stratified Sampling
# 5 K-Fold Cross Validation

## 5. Model Training and Evaluation

### 5.1 Train–Test Split
- Dataset split into:
  - 70% training data
  - 30% testing data
- Same random state is used for fair comparison

### 5.2 Evaluation Metric
- **Accuracy (%)** is used as the evaluation metric.
- Accuracy represents the percentage of correctly classified transactions.

<img width="490" height="281" alt="Screenshot 2026-02-03 at 11 37 46 AM" src="https://github.com/user-attachments/assets/37965b78-fd8b-4dd9-bf20-21987bbe9e72" />





