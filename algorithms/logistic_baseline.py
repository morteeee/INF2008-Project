import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
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
file_path = "../data/creditcard.csv"
data = pd.read_csv(file_path)

# Prepare dataset
X = data.drop(columns=['Class'])
y = data['Class']

# Start timer
start_time = time.time()

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
over_sampler = SMOTE(sampling_strategy=0.5, random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy=0.1, random_state=42)

steps = [('under', under_sampler), ('over', over_sampler)]
pipeline = Pipeline(steps=steps)
X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)

# Train final Logistic Regression model on resampled data
log_model_final = LogisticRegression(max_iter=10000, random_state=42)
log_model_final.fit(X_train_resampled, y_train_resampled)

# Predictions on test set
y_pred = log_model_final.predict(X_test)
y_pred_prob = log_model_final.predict_proba(X_test)[:, 1]

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
