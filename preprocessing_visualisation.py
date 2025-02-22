import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('data/creditcard.csv')

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
}

print("=== Missing Values ===")
print(data_quality["missing_values"])
print("\n=== Null Values ===")
print(data_quality["null_values"])

# Analyzing fraud vs. non-fraud cases
fraud_count = data['Class'].value_counts()
labels = ['No Fraud', 'Fraud']
colors = ['#1E90FF', '#FF4C4C']

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie Chart for Fraud Distribution
axes[0].pie(fraud_count, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})
axes[0].set_title("Fraud vs Non-Fraud Distribution")

# Bar Chart for Fraud Cases
sns.barplot(x=fraud_count.index, y=fraud_count.values, ax=axes[1], palette=['blue', 'red'])
axes[1].set_title("Number of Fraud Cases")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Count")
for index, value in enumerate(fraud_count.values):
    axes[1].text(index, value + 500, str(value), ha='center', fontsize=12)

plt.tight_layout()
plt.show()

# Analyzing fraud and non-fraud statistics
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
