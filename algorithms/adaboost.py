import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load dataset
file_path = "/data/creditcard.csv"
data = pd.read_csv(file_path)

# Prepare dataset
X = data.drop(columns=['Class'])
y = data['Class']

# Start timer
start_time = time.time()

# Split dataset before any processing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train AdaBoost model only on training data to get feature importance
ada_model = AdaBoostClassifier()
ada_model.fit(X_train, y_train)

# Extract feature importance from training data only
ada_importances = pd.Series(ada_model.feature_importances_, index=X_train.columns)
ada_top_features = ada_importances.sort_values(ascending=False).index[:10].tolist()

# Apply feature selection to both train and test sets
X_train = X_train[ada_top_features]
X_test = X_test[ada_top_features]

# Perform Data Balancing (SMOTE + Undersampling) only on training data
over_ada = SMOTE(sampling_strategy=0.5, random_state=42)
under_ada = RandomUnderSampler(sampling_strategy=0.1, random_state=42)

steps_ada = [('under', under_ada), ('over', over_ada)]
pipeline_ada = Pipeline(steps=steps_ada)
X_train_resampled, y_train_resampled = pipeline_ada.fit_resample(X_train, y_train)

# Train final AdaBoost model on resampled data
ada_model_final = AdaBoostClassifier()
ada_model_final.fit(X_train_resampled, y_train_resampled)

# Predictions on test set
y_pred = ada_model_final.predict(X_test)
y_pred_prob = ada_model_final.predict_proba(X_test)[:, 1]

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# End timer
end_time = time.time()
total_time = end_time - start_time

# Print results
print("Top 10 Features - AdaBoost:")
print(ada_top_features)
print("\nClass distribution after resampling (Adaboost):", Counter(y_train_resampled))
print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")
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
plt.title("Confusion Matrix - AdaBoost")
plt.show()

# Plot ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AdaBoost (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve - AdaBoost")
plt.legend(loc="lower right")
plt.show()
