# Technology Innovation Project
# Project 4 - Data Science Project (Data Processing)
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pylab
import datetime

#Load the dataset
df = pd.read_csv('C:/Users/singh/Desktop/TIP_2/fraudTest.csv', on_bad_lines='skip')
df.info()

#Display the data
df.head()

df.describe()

#Sum the missing value
df.isnull().sum()
# Split the 'trans_date_trans_time' into 'trans_date' and 'trans_time'
df['trans_date'] = pd.to_datetime(df['trans_date_trans_time'], format='%d-%m-%Y %H:%M').dt.date
df['trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%d-%m-%Y %H:%M').dt.time

# Remove the 'trans_date_trans_time' column and 'Unnamed: 0' if it exists
df.drop(columns=['trans_date_trans_time', 'Unnamed: 0'], errors='ignore', inplace=True)

# Reorder the columns to have 'trans_date' first and 'trans_time' second
columns = ['trans_date', 'trans_time'] + [col for col in df.columns if col not in ['trans_date', 'trans_time']]
df = df[columns]

# Convert 'dob' to datetime
df['dob'] = pd.to_datetime(df['dob'], format='%d-%m-%Y')

# Check the data types
print("\nData types of the columns:")
print(df.dtypes)

plt.figure(figsize=(14, 8))
# Filter for transactions where is_fraud is 1
df_fraud = df[df['is_fraud'] == 1]

# Print columns of df_fraud to check column names and existence
print(df_fraud.columns)

# Extract hour from trans_time
df['trans_time'] = pd.to_datetime(df['trans_time'], format='%H:%M:%S')
df['hour'] = df['trans_time'].dt.hour

# Filter for transactions where is_fraud is 1
df_fraud = df[df['is_fraud'] == 1]

# Plot histogram
plt.figure(figsize=(14, 8))
sns.histplot(data=df_fraud, x='hour', bins=24)
plt.title('Fraudulent Transactions Over Time')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.show()

# Load the dataset again for geographical analysis
df = pd.read_csv("C:/Users/singh/Desktop/TIP_2/fraudTest.csv")

# Plot geographical distribution
plt.figure(figsize=(14, 10))
sns.scatterplot(x='long', y='lat', hue='is_fraud', palette=['blue', 'red'], data=df)
plt.title('Geographical Distribution of Transactions')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Fraudulent (1) / Non-Fraudulent (0)')
plt.show()
# Plot State-wise Distribution of Fraudulent Transactions
plt.figure(figsize=(14, 10))
sns.countplot(y='state', hue='is_fraud', palette=['blue', 'red'], data=df, order=df['state'].value_counts().index)
plt.title('State-wise Distribution of Fraudulent Transactions')
plt.xlabel('Count')
plt.ylabel('State')
plt.legend(title='Fraudulent (1) / Non-Fraudulent (0)')
plt.show()


## Transactio Amount vs Age with Fraud indicator
# Load the dataset
df = pd.read_csv('C:/Users/singh/Desktop/TIP_2/fraudTest.csv', on_bad_lines='skip')

# Convert 'trans_date_trans_time' to datetime
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%d-%m-%Y %H:%M')

# Calculate age from 'dob'
df['dob'] = pd.to_datetime(df['dob'], format='%d-%m-%Y', errors='coerce')
df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year

# Plot Transaction Amount vs. Age with Fraud Indicator
plt.figure(figsize=(14, 8))
sns.scatterplot(x='age', y='amt', hue='is_fraud', palette=['blue', 'red'], data=df)
plt.title('Transaction Amount vs. Age with Fraud Indicator')
plt.xlabel('Age')
plt.ylabel('Transaction Amount')
plt.legend(title='Fraudulent (1) / Non-Fraudulent (0)')
plt.show()

# Distribution of transaction amount
plt.figure(figsize=(10, 6))
hist_plot = sns.histplot(df['amt'], bins=50, kde=True)
plt.title('Distribution of Transaction Amount')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')

# Adding data annotations
counts, bins = np.histogram(df['amt'], bins=50)
for count, bin_edge in zip(counts, bins):
    if count > 0:  # Only annotate bins with non-zero counts
        plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, str(count), ha='center', va='bottom', fontsize=8)

plt.show()

# Time-based analysis - transactions over time
plt.figure(figsize=(14, 8))
sns.histplot(data=df, x='hour', hue='is_fraud', multiple='stack', bins=24)
plt.title('Transactions Over Time')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.show()
