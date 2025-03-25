**Credit Card Fraud Detection 🚀**

**Overview**
This project aims to detect fraudulent credit card transactions using machine learning models. By analyzing transaction patterns, we can identify suspicious activities and help prevent financial fraud.

**Dataset**
The dataset used contains real-world anonymized transactions labeled as fraudulent (1) or non-fraudulent (0).

Features include transaction amounts, time, and engineered attributes derived from transaction behavior.

Due to the highly imbalanced nature of fraud detection (fraud cases are rare), we apply oversampling and undersampling techniques.

**Technologies Used**
Python (for data analysis and model building)

pandas & NumPy (for data preprocessing)

scikit-learn (for machine learning models)

imbalanced-learn (for handling imbalanced datasets)

Matplotlib & Seaborn (for data visualization)

**Project Workflow**
Data Preprocessing

Handle missing values and outliers

Scale numerical features

Encode categorical variables (if any)

Exploratory Data Analysis (EDA)

Visualize fraud vs. non-fraud distributions

Identify trends and correlations

Handling Class Imbalance

Use SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset

Compare different resampling strategies

**Model Selection & Training**

Train models such as Logistic Regression, Random Forest, XGBoost, and Neural Networks

Evaluate models using accuracy, precision, recall, F1-score, and AUC-ROC

Model Optimization

Perform GridSearchCV to tune hyperparameters

Optimize the model for best fraud detection performance
