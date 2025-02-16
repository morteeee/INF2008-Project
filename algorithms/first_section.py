# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('creditcard.csv')

# Display the first few rows and dataset information
data_info = {
    "head": data.head(),
    "info": data.info(),
    "describe": data.describe()
}

print("=== Data Headings ===")
print(data_info["head"])
print("\n=== Data Information ===")
print(data_info["info"])
print("\n=== Data Description ===")
print(data_info["describe"])

# Check for missing, null, and duplicate values
data_quality = {
    "missing_values": data.isnull().sum(),
    "null_values": data.isna().sum(),
    "duplicate_rows": data.duplicated().sum(),
}

print("=== Missing Values ===")
print(data_quality["missing_values"])
print("\n=== Null Values ===")
print(data_quality["null_values"])
print("\n=== Duplicate Rows ===")
print(data_quality["duplicate_rows"])

#Analysing part

# Split Fraud and Non-Fraud Data
fraud = data[data['Class'] == 1].describe().T
nofraud = data[data['Class'] == 0].describe().T

# Define Colors
fraud_color = "Reds"   # Red for fraud transactions
nofraud_color = "Blues" # Blue for non-fraud transactions

# Determine the Number of Features Dynamically
num_features = len(fraud)  # Get total number of features
split_point = num_features // 2  # Split for multiple plots

# Create Subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

# Fraud Heatmap Part 1
sns.heatmap(fraud[['mean']][:split_point], annot=True, cmap=fraud_color, linewidths=0.5,
            linecolor='black', cbar=False, fmt='.2f', ax=axes[0, 0])
axes[0, 0].set_title('Fraud Transactions - Mean (Part 1)')

# Fraud Heatmap Part 2
sns.heatmap(fraud[['mean']][split_point:], annot=True, cmap=fraud_color, linewidths=0.5,
            linecolor='black', cbar=False, fmt='.2f', ax=axes[0, 1])
axes[0, 1].set_title('Fraud Transactions - Mean (Part 2)')

# Non-Fraud Heatmap Part 1
sns.heatmap(nofraud[['mean']][:split_point], annot=True, cmap=nofraud_color, linewidths=0.5,
            linecolor='black', cbar=False, fmt='.2f', ax=axes[1, 0])
axes[1, 0].set_title('Non-Fraud Transactions - Mean (Part 1)')

# Non-Fraud Heatmap Part 2
sns.heatmap(nofraud[['mean']][split_point:], annot=True, cmap=nofraud_color, linewidths=0.5,
            linecolor='black', cbar=False, fmt='.2f', ax=axes[1, 1])
axes[1, 1].set_title('Non-Fraud Transactions - Mean (Part 2)')

plt.tight_layout()
plt.show()