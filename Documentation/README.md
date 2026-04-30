# Flight Cancellation Prediction System

A machine learning-powered web application for predicting flight cancellations. Built with a robust data pipeline and multiple classification models, this system enables real-time predictions and batch model evaluation through an intuitive web interface.

## Overview

This project implements an end-to-end flight cancellation prediction system that transforms raw flight data through a comprehensive preprocessing pipeline and applies various machine learning classifiers to predict whether a flight will be cancelled.

The system handles the complete data science workflow:
- **Data Preprocessing**: Missing value imputation, categorical encoding, temporal feature engineering
- **Feature Engineering**: Cyclic transformations for temporal variables, ordinal encoding for categorical features
- **Model Training**: Multiple classifier algorithms with configurable hyperparameters
- **Inference Pipeline**: Consistent transformation of new data for real-time predictions
- **Model Evaluation**: Batch evaluation with standard classification metrics

## Features

### Single Flight Prediction
Input individual flight details through a web form to receive instant cancellation predictions. The system automatically handles data transformation and applies the selected model.

### Batch Evaluation
Upload CSV files containing historical flight data to evaluate model performance. The system computes accuracy, precision, recall, and F1-score metrics.

### Multi-Model Support
Choose from six different classification algorithms:
- **Naive Bayes** (GaussianNB)
- **K-Nearest Neighbors** (KNN)
- **Logistic Regression**
- **Decision Tree**
- **Multi-Layer Perceptron** (MLP)
- **Random Forest**

## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, FastAPI |
| ML Framework | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| Frontend | HTML5, CSS3, JavaScript |

## Project Structure

```
├── main.py                   # FastAPI web server and API endpoints
├── pipeline.py               # Data transformation and inference pipeline
├── save_objects.py           # Model training and artifact generation
├── prediction_objects.json   # Pipeline configuration manifest
│
├── static/                   # Frontend assets
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── models/                   # Trained models and preprocessing artifacts
│   ├── naive_bayes.joblib
│   ├── knn.joblib
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── mlp.joblib
│   ├── random_forest.joblib
│   ├── encoder.joblib
│   ├── scaler.joblib
│   └── ...
│
├── datasets/                 # Training and test data
│   ├── flights_best_fs_train.csv
│   ├── flights_best_fs_test.csv
│   └── ...
│
└── codes/                    # Original preprocessing scripts (reference)
```

## Data Pipeline

The preprocessing pipeline applies the following transformations:

1. **Missing Value Imputation**: Mean imputation for numeric features, mode imputation for categorical features
2. **Type Enforcement**: Numeric coercion and data type standardization
3. **Cyclic Encoding**: Sine/cosine transformation for temporal features (Month, DayOfWeek, Quarter, Time blocks)
4. **Ordinal Encoding**: Categorical to numeric conversion for string features
5. **MinMax Scaling**: Feature normalization to [0,1] range
6. **Feature Selection**: Retention of 37 selected features for model input

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Activate environment (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install fastapi uvicorn pandas scikit-learn numpy joblib python-multipart
```

### Running the Application

```bash
# Start the development server
python -m uvicorn main:app --reload
```

Access the application at: `http://127.0.0.1:8000`

### Retraining Models (Optional)

To retrain models with new data:

```bash
python save_objects.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| POST | `/predict` | Single flight prediction |
| POST | `/evaluate` | Batch model evaluation |

## License

This project was developed as part of an academic Data Science course.
