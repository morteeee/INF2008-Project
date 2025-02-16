import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
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
file_path = "creditcard.csv"
data = pd.read_csv(file_path)

# Prepare dataset
X = data.drop(columns=['Class'])
y = data['Class']

# Start timer
start_time = time.time()

# Split dataset before any processing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model to get feature importance
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# Extract feature importance from training data only
gb_importances = pd.Series(gb_model.feature_importances_, index=X_train.columns)
gb_top_features = gb_importances.sort_values(ascending=False).index[:10].tolist()

# Apply feature selection to both train and test sets
X_train = X_train[gb_top_features]
X_test = X_test[gb_top_features]

# Perform Data Balancing (SMOTE + UnderSampling) only on training data
over_gb = SMOTE(sampling_strategy=0.5, random_state=42)
under_gb = RandomUnderSampler(sampling_strategy=0.1, random_state=42)

steps_gb = [('under', under_gb), ('over', over_gb)]
pipeline_gb = Pipeline(steps=steps_gb)
X_train_resampled, y_train_resampled = pipeline_gb.fit_resample(X_train, y_train)

# Train final Gradient Boosting model on resampled data
gb_model_final = GradientBoostingClassifier()
gb_model_final.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred = gb_model_final.predict(X_test)
y_pred_prob = gb_model_final.predict_proba(X_test)[:, 1]

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# End timer before visualization
end_time = time.time()
total_time = end_time - start_time

# Print results
print("Top 10 Features - Gradient Boosting:")
print(gb_top_features)
print("\nClass distribution after resampling (Gradient Boosting):", Counter(y_train_resampled))
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
plt.title("Confusion Matrix - Gradient Boosting")
plt.show()

# Plot ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Gradient Boosting (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve - Gradient Boosting")
plt.legend(loc="lower right")
plt.show()