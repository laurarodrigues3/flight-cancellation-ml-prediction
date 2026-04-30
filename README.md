# Flight Cancellation ML Prediction System

<div align="center">
  <p><strong>An end-to-end Machine Learning web application that predicts flight cancellations based on historical aviation data.</strong></p>
  <p><a href="https://teu-link-do-render-aqui.onrender.com" target="_blank"><strong>-> Test the Live API & Web Client Here</strong></a></p>
</div>

## Project Overview

**Flight Cancellation ML Prediction** is a comprehensive Data Science and Machine Learning project focused on predicting flight cancellations. Leveraging the extensive Flight Delay Dataset (2022), this project encompasses the entire Knowledge Discovery in Databases (KDD) process—from rigorous data profiling and feature engineering to model training, evaluation, and deployment via a modern REST API.

The system allows users to evaluate different classification algorithms (Naive Bayes, K-Nearest Neighbors, Logistic Regression, Decision Trees, Multi-layer Perceptron, and Random Forests) on flight data to predict potential disruptions, providing a robust tool for aviation analytics.

## Key Features

- **End-to-End ML Pipeline**: Custom preprocessing pipeline (`PredictionPipeline`) handling missing value imputation, cyclic feature encoding (e.g., temporal data like `CRSDepTime`), scaling, and feature selection.
- **Multiple Classification Models**: Integration and comparison of 6 distinct machine learning algorithms.
- **RESTful API**: Built with **FastAPI** to serve predictions and model evaluations in real-time.
- **Interactive Web Client**: A responsive vanilla HTML/CSS/JS frontend to upload datasets, test single predictions, and visualize model metrics.
- **Comprehensive Data Analysis**: In-depth statistical analysis, data profiling (dimensionality, distribution, sparsity), and preprocessing strategy documented in the accompanying technical report.

## Technology Stack

- **Backend / API**: Python 3.10+, FastAPI, Uvicorn
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Frontend**: HTML5, Vanilla CSS, JavaScript

## Project Structure

```text
├── client/                 # Frontend web interface (HTML, CSS, JS)
├── models/                 # Serialized ML models and pipeline artifacts (.joblib)
├── server/                 # FastAPI application
│   ├── main.py             # API endpoints and server configuration
│   └── pipeline.py         # Custom ML data transformation and prediction pipeline
├── InstallationGuide.txt   # Setup instructions
├── requirements.txt        # Python dependencies
└── Technical_Analysis_Report.pdf # Comprehensive Data Science and EDA Report
```

## Data Analysis & Technical Report

For a deep dive into the data engineering and model evaluation process, please refer to the **[Technical Analysis Report](Technical_Analysis_Report.pdf)** included in this repository. 

The report covers:
- **Data Profiling**: Analysis of feature distributions, dataset sparsity, and granularity.
- **Data Preparation Methodology**: Step-by-step justification for categorical encoding, missing value imputation, and scaling.
- **Model Evaluation**: Critical analysis of parameter tuning, overfitting, and comparative performance metrics across all models.

## Installation and Setup

### Prerequisites
- Python 3.10 or higher

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/flight-cancellation-ml-prediction.git
cd flight-cancellation-ml-prediction
```

### 2. Set up the virtual environment
**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```
**Mac/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt --only-binary :all:
```

### 4. Run the Application
```bash
python -m uvicorn server.main:app --reload
```

Navigate to `http://localhost:8000` in your web browser to access the interactive dashboard.

## API Endpoints

- `GET /api/models`: Returns a list of all available classification models.
- `POST /api/predict-single`: Accepts a JSON payload with flight features and returns a cancellation prediction using the specified model.
- `POST /api/evaluate-models`: Accepts a `.csv` file upload and evaluates the dataset against one or all available models, returning key performance metrics (Accuracy, Precision, Recall, F1-Score).