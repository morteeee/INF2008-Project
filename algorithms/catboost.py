import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, make_scorer, precision_score, recall_score, f1_score
import warnings
import os

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load dataset
file_path = r"C:\EverythingElse\GitHubDesktopProjects\INF2008-Projectdatastuff\creditcard.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

# Prepare dataset
X = data.drop(columns=['Class'])
y = data['Class']

# Split dataset before any processing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define evaluation metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, pos_label=1),
    'recall': make_scorer(recall_score, pos_label=1),
    'f1': make_scorer(f1_score, pos_label=1)
}

# --- Feature Selection and Data Balancing ---
def create_pipeline():
    over = SMOTE(random_state=42)
    under = RandomUnderSampler(random_state=42)
    steps = [('over', over), ('under', under)]
    pipeline = Pipeline(steps=steps)
    return pipeline

X_train_resampled, y_train_resampled = create_pipeline().fit_resample(X_train, y_train)

# --- Model Training and Tuning ---
def train_and_evaluate(X_train, y_train, X_test, y_test):
    start_time = time.time()

    # Define the CatBoost model
    cat_model = CatBoostClassifier(random_state=42, verbose=0, early_stopping_rounds=10) # Reduced early_stopping_rounds

    # Define a grid of hyperparameters to search
    param_grid = {
        'depth': [6, 8],  # Reduced depth range
        'learning_rate': [0.01, 0.05],  # Reduced learning rate range
        'l2_leaf_reg': [1, 3]  # Reduced l2_leaf_reg range
    }

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=cat_model, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1, verbose=1) # Added verbose
    grid_search.fit(X_train, y_train)

    # Get the best model from the grid search
    best_cat_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_cat_model.predict(X_test)
    y_pred_prob = best_cat_model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    end_time = time.time()
    total_time = end_time - start_time

    # Print results
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    return y_test, y_pred, y_pred_prob

# --- Plotting ---
def plot_results(y_test, y_pred_prob):
    # Plot ROC-AUC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'CatBoost (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve - CatBoost")
    plt.legend(loc="lower right")
    plt.show()

# Train and Evaluate
y_test, y_pred, y_pred_prob = train_and_evaluate(X_train_resampled, y_train_resampled, X_test, y_test)

# Plot Results
plot_results(y_test, y_pred_prob)