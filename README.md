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

# Import the data from fraud_data.csv.
print(os.listdir('/content/'))

#Load the dataset
df = pd.read_csv('/content/fraudTest.csv', on_bad_lines='skip')
df.info()

#Display the data
df.head()

df.describe()

#Sum the missing value
df.isnull().sum()

##Split the column trans_date_trans_time into 2 separate columns, including trans_date and trans_time
df['trans_date'] = pd.to_datetime(df['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S').dt.date
df['trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S').dt.time
df.info()

# Remove the 'trans_date_trans_time' column
# Check if the index column exists and remove it
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
df.drop('trans_date_trans_time', axis=1, inplace=True)

# Reorder the columns to have 'trans_date' first and 'trans_time' second
columns = ['trans_date', 'trans_time'] + [col for col in df.columns if col not in ['trans_date', 'trans_time']]
df = df[columns]

# Convert 'dob' to datetime
df['dob'] = pd.to_datetime(df['dob'], format='%Y-%m-%d')

# Convert 'trans_time' to time
df['trans_time'] = pd.to_datetime(df['trans_time'], format='%H:%M:%S').dt.time

# Convert 'trans_date' to date
df['trans_date'] = pd.to_datetime(df['trans_date']).dt.date

# Check the data types
print("\nData types of the columns:")
print(df.dtypes)

## Exploratory Data Analysis
## the maximum transaction that are fraudelent happen between midnight and early morning hours.

plt.figure(figsize=(14, 8))
# Filter for transactions where is_fraud is 1
df_fraud = df[df['is_fraud'] == 1]

# Print columns of df_fraud to check column names and existence
print(df_fraud.columns)

# Extract hour from trans_date_trans_time as an example (if it contains time information)
df_fraud['hour'] = pd.to_datetime(df_fraud['trans_date_trans_time']).dt.hour

# Plot histogram
sns.histplot(data=df_fraud, x='hour', bins=24)
plt.title('Fraudulent Transactions Over Time')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
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
df['hour'] = df['trans_time'].apply(lambda x: x.hour)
sns.histplot(data=df, x='hour', hue='is_fraud', multiple='stack', bins=24)
plt.title('Transactions Over Time')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.show()
