import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load dataset
file_path = "../data/creditcard.csv"
data = pd.read_csv(file_path)

# Prepare dataset
X = data.drop(columns=['Class'])
y = data['Class']

# Split dataset before any processing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection using RFE with Logistic Regression
log_model = LogisticRegression(max_iter=10000, random_state=42)
rfe = RFE(estimator=log_model, n_features_to_select=10)  # Select top 10 features
rfe.fit(X_train, y_train)

# Get the selected features
selected_features = X_train.columns[rfe.support_].tolist()

# Apply feature selection to both train and test sets
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# Perform Data Balancing (SMOTE + Undersampling) only on training data
over_sampler = SMOTE(sampling_strategy=0.3, random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy=0.1, random_state=42)

steps = [('under', under_sampler), ('over', over_sampler)]
pipeline = Pipeline(steps=steps)
X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)

# Define hyperparameter grid for RandomizedSearchCV
# Best Param: {'tol': 0.0001, 'solver': 'saga', 'penalty': 'l2', 'C': np.float64(0.012742749857031334)}
param_dist = {
    'C': np.logspace(-4, 4, 20),  # Regularization strength
    'penalty': ['l2'],  
    'solver': ['saga'],  
    'tol': [1e-4, 1e-3, 1e-2]  # Tolerance for stopping criteria
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=LogisticRegression(max_iter=10000, random_state=42),
    param_distributions=param_dist,
    n_iter=50,  # Number of iterations to sample
    scoring='accuracy',
    cv=3,  
    random_state=42,
    n_jobs=-1  
)

# Train model using RandomizedSearchCV
random_search.fit(X_train_resampled, y_train_resampled)

# Get best model from RandomizedSearchCV
log_model_final = random_search.best_estimator_


# Print the best hyperparameters
print("\nBest Hyperparameters found by RandomizedSearchCV:")
print(random_search.best_params_)


# Start timer
start_time = time.time()

# Predictions on test set
y_pred_prob = log_model_final.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.90).astype(int)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# End timer
end_time = time.time()
total_time = end_time - start_time

# Print results
print("Top 10 Selected Features (RFE - Logistic Regression):")
print(selected_features)
print("\nClass distribution after resampling:", Counter(y_train_resampled))
print(f"\nTotal Execution Time: {total_time:.2f} seconds")
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Plot ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.show()
