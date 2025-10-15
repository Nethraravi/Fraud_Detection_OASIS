🕵️‍♂️ Fraud Detection System – Data Analytics Internship Project
📘 Overview

This project focuses on detecting and preventing fraudulent financial transactions using Data Analytics and Machine Learning techniques.
It demonstrates a complete data science workflow — from anomaly detection to model building, real-time simulation, scalability, and interactive visualization using Power BI.

🧠 Project Objectives

Anomaly Detection: Identify unusual transaction patterns.

Machine Learning Models: Build predictive models (Logistic Regression, Decision Trees, Neural Networks).

Feature Engineering: Transform and select relevant features to improve accuracy.

Real-Time Monitoring: Simulate real-time fraud detection.

Scalability: Design data pipelines to handle large transaction volumes efficiently.

🧩 Tools & Technologies
Category	Tools
Programming	Python, Jupyter Notebook
Data Analysis	Pandas, NumPy, Scikit-learn
Visualization	Matplotlib, Seaborn, Plotly, Power BI
Big Data	PySpark (for scalability simulation)
Others	Excel, Git, GitHub
⚙️ Workflow Steps
Step 1 – Anomaly Detection

Techniques: Isolation Forest, Local Outlier Factor (LOF)

Goal: Identify unusual transaction behavior in the dataset.

Output: Confusion matrix + Scatter plots of anomalies

Step 2 – Machine Learning Models

Algorithms: Logistic Regression, Decision Tree, Neural Network

Evaluated using precision, recall, F1-score, and AUC.

Predicts whether a transaction is fraudulent (1) or legitimate (0).

Step 3 – Feature Engineering

Selected significant features from PCA components (V1–V28)

Created new features like Log_Amount and normalized transaction data.

Step 4 – Real-Time Monitoring Simulation

Implemented a real-time fraud probability prediction pipeline.

Generates realtime_fraud_predictions.csv for Power BI visualization.

Step 5 – Scalability

Simulated large-scale processing using PySpark and Pandas chunking.

Efficiently handled large transaction datasets and exported results as Parquet files.

📊 Visualization

The data was visualized using Power BI and Jupyter Notebooks.

Key Charts:

Fraud vs Legit Transactions Distribution

Fraud Probability Heatmap

PCA Clustering of Transactions

Fraud Trends Over Time

🧾 Dataset

Source: (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Rows: 284,807 | Fraudulent: 492 | Legitimate: 284,315

Features: Time, V1–V28, Amount, Class

Note: The dataset is highly imbalanced — handled using undersampling and model tuning techniques.

🧩 Results
✅ Achieved ~99.8% AUC with Logistic Regression and Decision Tree models.
✅ Real-time fraud detection prototype capable of scoring new transactions dynamically.

🧠 Key Learnings
Implementing machine learning for real-world financial data.
Handling imbalanced datasets effectively.
End-to-end data pipeline development with visualization integration.
Experience with scalable data processing and interactive dashboards.

