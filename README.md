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

# Display the data
df.head()

# Sum the missing value
df.isnull().sum()

## Data Processing 
# Check if the index column exists and remove it
import pandas as pd
# Check if the index column exists and remove it
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
df.columns[0]

# Display the data before splitting the column
print("Before splitting the column:")
print(df.head())

# Split 'trans_date_trans_time' into 'trans_date' and 'trans_time'
df['trans_date'] = pd.to_datetime(df['trans_date_trans_time'], format='%d-%m-%Y %H:%M').dt.date
df['trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%d-%m-%Y %H:%M').dt.time

# Drop the original 'trans_date_trans_time' column
df.drop(columns=['trans_date_trans_time'], inplace=True)

# Reorder columns to place 'trans_date' and 'trans_time' at the beginning
columns_order = ['trans_date', 'trans_time'] + [col for col in df.columns if col not in ['trans_date', 'trans_time']]
df = df[columns_order]
# Display the data after splitting the column
print("\nAfter splitting the column:")

# Convert 'amt' to integer
df['amt'] = df['amt'].astype(int)

# Convert 'dob' to datetime
df['dob'] = pd.to_datetime(df['dob'], format='%d-%m-%Y')

# Convert 'trans_time' to time
df['trans_time'] = pd.to_datetime(df['trans_time'], format='%H:%M:%S').dt.time

# Convert 'trans_date' to date
df['trans_date'] = pd.to_datetime(df['trans_date']).dt.date

# Check the data types
print("\nData types of the columns:")
print(df.dtypes)

print(df.head())
