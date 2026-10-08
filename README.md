# User Behaviour Anomaly Detection using Isolation Forest

An interactive Streamlit application for detecting anomalous user behaviour using unsupervised machine learning and the Isolation Forest algorithm.

## Overview

User Behaviour Analysis can help identify unusual activity patterns that may indicate suspicious behaviour, fraud, account misuse, or potential security issues.

This project implements an unsupervised anomaly detection system that analyzes structured user activity data without requiring pre-labelled normal and anomalous records.

The application provides an interactive Streamlit interface where users can upload CSV datasets, preprocess the data, configure the anomaly detection sensitivity, visualize detected patterns, and export the processed results.

## Key Features

- Upload user activity datasets in CSV format
- Automatic data cleaning and duplicate removal
- Categorical feature encoding using one-hot encoding
- Numerical feature scaling using StandardScaler
- Unsupervised anomaly detection using Isolation Forest
- Adjustable contamination rate
- Detection summary showing normal and anomalous records
- Interactive bar chart visualization
- Interactive pie chart visualization
- Interactive scatter plot for feature comparison
- Downloadable CSV containing anomaly labels and anomaly scores
- Support for different structured user activity datasets

## Machine Learning Approach

### Isolation Forest

Isolation Forest is an unsupervised machine learning algorithm designed for anomaly detection.

Instead of requiring labelled examples of anomalous behaviour, the algorithm isolates observations that are different from the majority of the dataset.

In this project:

- **Learning Type:** Unsupervised Learning
- **Algorithm:** Isolation Forest
- **Number of Trees:** 200
- **Contamination:** User-configurable from 1% to 20%
- **Random State:** 42

The model produces:

- `1` → Normal behaviour
- `-1` → Anomalous behaviour

An anomaly score is also generated using the model's decision function.

## Data Processing Pipeline

The application follows the following workflow:

```text
CSV Dataset
     │
     ▼
Data Upload
     │
     ▼
Missing Value Removal
     │
     ▼
Duplicate Removal
     │
     ▼
Categorical Encoding
     │
     ▼
Feature Scaling
     │
     ▼
Isolation Forest
     │
     ▼
Anomaly Labels & Scores
     │
     ├──► Visualizations
     │
     └──► Downloadable Results
