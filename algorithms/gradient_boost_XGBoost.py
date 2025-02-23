import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
from collections import Counter

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load dataset
file_path = "../data/creditcard.csv"
data = pd.read_csv(file_path)

# Prepare dataset
X = data.drop(columns=['Class'])
y = data['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessing steps
# Use ADASYN for over-sampling
over_xgb = ADASYN(sampling_strategy=0.5, random_state=42)

# Use Tomek Links for under-sampling
under_xgb = TomekLinks()
pipeline = Pipeline(steps=[('over', over_xgb), ('under', under_xgb)])

# Apply data balancing to the training set
X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0.01, 0.1, 1],
    'reg_lambda': [0.01, 0.1, 1],
    'scale_pos_weight': [1, 2, 3]
}


# Initialize XGBoost classifier
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Perform hyperparameter tuning with resampled data
random_search.fit(X_train_resampled, y_train_resampled)

# Get the best parameters and best model
best_params = random_search.best_params_
best_model = random_search.best_estimator_


# Make predictions
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Start timer for hyperparameter tuning
start_time = time.time()


# Get the best parameters and best model
print("Best Parameters:", best_params)

# Make predictions with the best model
y_pred_random = best_model.predict(X_test)
y_pred_prob_random = best_model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy_random = accuracy_score(y_test, y_pred_random)
report_random = classification_report(y_test, y_pred_random)
conf_matrix_random = confusion_matrix(y_test, y_pred_random)

# End timer
end_time = time.time()
total_time = end_time - start_time

# Print results
print("Best Parameters:", best_params)
print("\nClass distribution after resampling (XGBoost):", Counter(y_train_resampled))
print(f"\nTotal Execution Time: {total_time:.2f} seconds")
print(f"\nAccuracy: {accuracy_random:.4f}")
print("\nClassification Report (Random Search):")
print(report_random)
print("\nConfusion Matrix (Random Search):")
print(conf_matrix_random)

# Plot Confusion Matrix for Random Search
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_random, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Search XGBoost")
plt.show()

# Plot ROC-AUC Curve for Random Search
fpr_random, tpr_random, _ = roc_curve(y_test, y_pred_prob_random)
roc_auc_random = auc(fpr_random, tpr_random)
plt.figure(figsize=(6, 4))
plt.plot(fpr_random, tpr_random, color='blue', lw=2, label=f'XGBoost (AUC = {roc_auc_random:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve - Random Search XGBoost")
plt.legend(loc="lower right")
plt.show()