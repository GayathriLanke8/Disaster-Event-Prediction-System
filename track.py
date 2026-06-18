import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
# pip install optuna-integration[mlflow]
from optuna.integration.mlflow import MLflowCallback

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # or 1

import warnings
warnings.filterwarnings("ignore")

# Name of the experiment
mlflow.set_experiment("KNN_IRIS_Pipeline")

# Load Dataset
df = pd.read_csv(r'synthetic_disaster_events_2025.csv')
data = df.copy()

# Data Cleaning
data = data.drop_duplicates()

# Segregate features and Target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Define Pipeline
pipeline_1 = Pipeline(
    [
        ('Scaler', StandardScaler()),
        ('Model', KNeighborsClassifier())
    ]
)

# Define Objective
def objective(trial):
    # Hyperparameters suggested by Optuna
    scaler_type = trial.suggest_categorical("Scaler__type", ["standard", "minmax"])
    n_neighbors = trial.suggest_int("Model__n_neighbors", 3, 21, 2)
    p = trial.suggest_int("Model__p", 1, 3)
    weights = trial.suggest_categorical("Model__weights", ["uniform", "distance"])

    # Set pipeline params for this trial
    pipeline_1.set_params(
        Scaler=StandardScaler() if scaler_type == "standard" else MinMaxScaler(),
        Model__n_neighbors=n_neighbors,
        Model__p=p,
        Model__weights=weights
    )

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_score = cross_val_score(pipeline_1, X_train, y_train, scoring="accuracy", cv=skf).mean()

    return cv_score

mlflow_callback = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(),
    metric_name="cv_accuracy"
)
    
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, callbacks=[mlflow_callback])

best_params = study.best_trial.params
print("Best hyperparameters:", best_params)
print("Best CV accuracy:", study.best_trial.value)

# Autolog final sklearn model
mlflow.sklearn.autolog()

# Training with Best parameters
scaler_type = best_params['Scaler__type']
Scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
pipeline_1.set_params(Scaler = Scaler)
pipeline_1.set_params(**{'Model__n_neighbors': best_params['Model__n_neighbors'], 'Model__p': best_params['Model__p'], 'Model__weights': best_params['Model__weights']})
pipeline_1.fit(X_train, y_train)
score = pipeline_1.score(X_train, y_train)
print('Training score', score)

# Testing the model
y_pred_test = pipeline_1.predict(X_test)
test_score = accuracy_score(y_test, y_pred_test)
print('Testing score', test_score)

# Log metrics to MLflow
mlflow.log_metric("train_accuracy", score)
mlflow.log_metric("test_accuracy", test_score)


