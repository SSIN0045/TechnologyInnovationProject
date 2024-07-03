#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("D:/Swinburne/Sem 3/Innovation Project/Data Sciencne/fraudTest.csv")


# In[3]:


df


# In[4]:


#Display the data
df.head()


# In[5]:


df.info()


# In[6]:


#Sum the missing value
df.isnull().sum()


# In[7]:


#Summary the description of the dataset
df.describe()


# # Data Pre-processing

# In[8]:


# Drop the first column
df.drop(df.columns[0], axis=1, inplace=True)


# In[9]:


# Convert 'trans_date_trans_time'and 'dob' to datetime
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['dob'] = pd.to_datetime(df['dob'])


# In[10]:


import datetime
# Extract the date and time components
df['trans_date'] = df['trans_date_trans_time'].dt.date
df['trans_time'] = df['trans_date_trans_time'].dt.time


# In[11]:


df['trans_date'] = pd.to_datetime(df['trans_date'])


# In[12]:


# Format the datetime object as a string
df['trans_time'] = df['trans_date_trans_time'].dt.strftime('%H:%M:%S')
df.info()


# In[13]:


df.head()


# # EDA

# In[14]:


#Distribute the class
df[('is_fraud')].value_counts()


# In[15]:


# Calculate correlation matrix
corr_matrix = df.corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()


# In[16]:


# Plot the class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='is_fraud', data=df)
plt.title('Class Distribution (Fraud vs Non-Fraud)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[17]:


# Function to plot the distribution of features based on class
def plot_feature_distribution(df, features):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=feature, hue='is_fraud', kde=True, element="step", stat="density", common_norm=False)
        plt.title(f'Distribution of {feature} by Class')
        plt.show()

# List of features to plot
features = ['amt']

# Plot the distributions
plot_feature_distribution(df, features)

# Function to plot boxplots of features based on class
def plot_feature_boxplots(df, features):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='is_fraud', y=feature, data=df)
        plt.title(f'{feature} by Class')
        plt.show()

# Plot the boxplots
plot_feature_boxplots(df, features)


# In[18]:


# Extract month and year from 'trans_date'
df['trans_month_year'] = df['trans_date'].dt.to_period('M')


# In[19]:


# EDA based on trans_month_year
# Example: Count of transactions per month and year
transaction_counts = df['trans_month_year'].value_counts().sort_index()

# Plotting transaction counts over time
plt.figure(figsize=(10, 6))
sns.lineplot(x=transaction_counts.index.astype(str), y=transaction_counts.values, marker='o')
plt.xticks(rotation=45)
plt.title('Transaction Counts per Month-Year')
plt.xlabel('Month-Year')
plt.ylabel('Number of Transactions')
plt.tight_layout()
plt.show()


# In[20]:


# EDA based on trans_month_year for fraudulent transactions
# Example: Count of fraudulent transactions per month and year
fraud_transaction_counts = df[df['is_fraud'] == 1]['trans_month_year'].value_counts().sort_index()

# Plotting fraudulent transaction counts over time
plt.figure(figsize=(10, 6))
sns.lineplot(x=fraud_transaction_counts.index.astype(str), y=fraud_transaction_counts.values, marker='o', color='red')
plt.xticks(rotation=45)
plt.title('Fraudulent Transaction Counts per Month-Year')
plt.xlabel('Month-Year')
plt.ylabel('Number of Fraudulent Transactions')
plt.tight_layout()
plt.show()


# In[21]:


# Plotting the horizontal bar plot for transactions by category
plt.figure(figsize=(10, 8))

# Countplot for both fraudulent and non-fraudulent transactions
sns.countplot(y='category', hue='is_fraud', data=df, palette='viridis', order=df['category'].value_counts().index)

plt.title('Count of Transactions by Category and Fraud Status')
plt.xlabel('Number of Transactions')
plt.ylabel('Category')
plt.legend(title='Fraud', loc='upper right')
plt.show()


# In[22]:


# Assuming 'is_fraud' column exists
fraud_by_hour = df.groupby(df['trans_date_trans_time'].dt.hour)['is_fraud'].mean()

# Plotting fraud analysis by time
plt.figure(figsize=(10, 6))
sns.lineplot(x=fraud_by_hour.index, y=fraud_by_hour.values, marker='o')
plt.title('Fraud Rate Over Time')
plt.xlabel('Hour of the Day')
plt.ylabel('Fraud Rate')
plt.xticks(range(24))
plt.tight_layout()
plt.show()


# In[23]:


# Create new features: 'transaction_hour', 'transaction_day', 'transaction_month', 'age'
df['transaction_hour'] = df['trans_date_trans_time'].dt.hour
df['transaction_day'] = df['trans_date_trans_time'].dt.day
df['transaction_month'] = df['trans_date_trans_time'].dt.month
df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365


# In[24]:


# Calculate the geographical distance between cardholder and merchant
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# In[25]:


df['distance'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])


# In[26]:


# Calculate transaction frequency and average amount within the last 24 hours
df = df.sort_values(by=['cc_num', 'trans_date_trans_time'])
df['trans_24h_count'] = df.groupby('cc_num').apply(lambda x: x.rolling('24h', on='trans_date_trans_time').trans_date_trans_time.count()).reset_index(level=0, drop=True)
df['trans_24h_amount_avg'] = df.groupby('cc_num').apply(lambda x: x.rolling('24h', on='trans_date_trans_time').amt.mean()).reset_index(level=0, drop=True)


# In[27]:


# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Improved Visualization: Transaction Day
plt.figure(figsize=(14, 8))

# Plot density plots
sns.kdeplot(df[df['is_fraud'] == 0]['transaction_day'], color='blue', label='Non-Fraudulent', lw=2)
sns.kdeplot(df[df['is_fraud'] == 1]['transaction_day'], color='red', label='Fraudulent', lw=2)

# Add title and labels
plt.title('Transaction Day Distribution', fontsize=16)
plt.xlabel('Day of the Month', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend()
plt.show()

# Calculate correlation between transaction day and fraud
correlation = df['transaction_day'].corr(df['is_fraud'])

print(f"Correlation between transaction day and fraud: {correlation:.2f}")


# In[28]:


# Calculate the average and total transaction amount per user
df['avg_trans_amount'] = df.groupby('cc_num')['amt'].transform('mean')
df['total_trans_amount'] = df.groupby('cc_num')['amt'].transform('sum')

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Visualization: Average Transaction Amount per User
plt.figure(figsize=(14, 8))

# Boxplot
plt.subplot(2, 1, 1)
sns.boxplot(x='is_fraud', y='avg_trans_amount', data=df, palette={0: 'blue', 1: 'red'})
plt.title('Average Transaction Amount per User')
plt.xlabel('Is Fraud')
plt.ylabel('Average Transaction Amount')

# Calculate correlation between transaction amount ('amt') and fraud ('is_fraud')
correlation = df['amt'].corr(df['is_fraud'])

print(f"Correlation between transaction amount and fraud: {correlation:.2f}")


# In[29]:


# Assuming 'df' is your DataFrame containing transaction data
# Convert 'dob' to datetime and calculate age
df['dob'] = pd.to_datetime(df['dob'])
df['age'] = (pd.Timestamp.now() - df['dob']).astype('<m8[Y]')

# Define age groups (you can adjust these as per your specific age ranges)
bins = [0, 20, 30, 40, 50, 60, 100]  # Define your age bins here
labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61+']  # Labels for age groups

# Assign age groups based on 'age' column
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Group by age group and calculate correlation between age group and fraud
age_group_corr = df.groupby('age_group')['is_fraud'].mean()

print("Correlation between age group and fraud:")
print(age_group_corr)


# In[30]:


# Violin plot
plt.subplot(2, 1, 2)
sns.violinplot(x='is_fraud', y='avg_trans_amount', data=df, palette={0: 'blue', 1: 'red'}, inner='quartile')
plt.title('Average Transaction Amount per User')
plt.xlabel('Is Fraud')
plt.ylabel('Average Transaction Amount')

plt.tight_layout()
plt.show()

# Visualization: Total Transaction Amount per User
plt.figure(figsize=(14, 8))


# # Outliers Handling

# In[33]:


# Calculate quartiles
Q1 = df['amt'].quantile(0.25)
Q3 = df['amt'].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Calculate upper and lower bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['amt'] < lower_bound) | (df['amt'] > upper_bound)]


# In[34]:


plt.figure(figsize=(12, 6))
sns.countplot(x='category', hue='is_fraud', data=outliers)
plt.title('Outliers by Transaction Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[35]:


# Boxplot
plt.figure(figsize=(10, 4))
sns.boxplot(x='amt', data=df)
plt.title('Boxplot of Transaction Amount')
plt.xlabel('Amount')
plt.show()


# In[36]:


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


# In[37]:


plt.figure(figsize=(8, 5))
sns.countplot(x='gender', hue='is_fraud', data=outliers)
plt.title('Outliers by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[38]:


# Assuming 'outliers' DataFrame contains your identified outliers
# Calculate the total number of outliers
total_outliers = outliers.shape[0]

# Calculate the number of outliers that are fraud cases
fraud_outliers = outliers[outliers['is_fraud'] == 1].shape[0]

# Calculate the percentage of outliers that are fraud cases
percentage_fraud_outliers = (fraud_outliers / total_outliers) * 100

print(f"Total number of outliers: {total_outliers}")
print(f"Number of fraud cases among outliers: {fraud_outliers}")
print(f"Percentage of outliers that are fraud cases: {percentage_fraud_outliers:.2f}%")


# In[39]:


# Step 1: Evaluate Outlier Impact
plt.figure(figsize=(14, 8))

# Plot original distribution
plt.subplot(2, 1, 1)
sns.histplot(df['amt'], kde=True, bins=50, color='blue')
plt.title('Distribution of Transaction Amounts (Original)')
plt.xlabel('Transaction Amount')
plt.ylabel('Count')


# In[40]:


# Plot distribution without outliers
plt.subplot(2, 1, 2)
sns.histplot(df[~df.index.isin(outliers.index)]['amt'], kde=True, bins=50, color='green')
plt.title('Distribution of Transaction Amounts (Without Outliers)')
plt.xlabel('Transaction Amount')
plt.ylabel('Count')

plt.tight_layout()
plt.show()


# In[41]:


# Step 2: Consider Analysis Goals
# Example: If fraud detection is the goal, visualize fraudulent transactions with and without outliers
plt.figure(figsize=(14, 8))

# Plot fraudulent transactions with outliers
plt.subplot(2, 1, 1)
sns.histplot(df[df['is_fraud'] == 1]['amt'], kde=True, bins=50, color='red')
plt.title('Distribution of Fraudulent Transaction Amounts (With Outliers)')
plt.xlabel('Transaction Amount')
plt.ylabel('Count')


# In[42]:


# Plot fraudulent transactions without outliers
plt.subplot(2, 1, 2)
sns.histplot(df[(df['is_fraud'] == 1) & (~df.index.isin(outliers.index))]['amt'], kde=True, bins=50, color='purple')
plt.title('Distribution of Fraudulent Transaction Amounts (Without Outliers)')
plt.xlabel('Transaction Amount')
plt.ylabel('Count')

plt.tight_layout()
plt.show()


# # Undersampling

# In[44]:


from sklearn.utils import resample

# Separate the majority and minority classes
df_majority = df[df['is_fraud'] == 0]
df_minority = df[df['is_fraud'] == 1]


# In[63]:


# Downsample the majority class
df_majority_downsampled = resample(df_majority, 
                                   replace=False,    # sample without replacement
                                   n_samples=len(df_minority), # to match minority class
                                   random_state=42) # reproducible results


# In[64]:


# Combine minority class with downsampled majority class
df_undersampled = pd.concat([df_majority_downsampled, df_minority])


# In[65]:


# Display new class counts
print(df_undersampled['is_fraud'].value_counts())


# In[66]:


# Example: Count of transactions per month and year after undersampling
transaction_counts_undersampled = df_undersampled['trans_month_year'].value_counts().sort_index()


# In[67]:


# Plotting transaction counts over time after undersampling
plt.figure(figsize=(10, 6))
sns.lineplot(x=transaction_counts_undersampled.index.astype(str), y=transaction_counts_undersampled.values, marker='o')
plt.xticks(rotation=45)
plt.title('Transaction Counts per Month-Year After Undersampling')
plt.xlabel('Month-Year')
plt.ylabel('Number of Transactions')
plt.tight_layout()
plt.show()


# In[68]:


# Plotting fraudulent transaction counts over time after undersampling
fraud_transaction_counts_undersampled = df_undersampled[df_undersampled['is_fraud'] == 1]['trans_month_year'].value_counts().sort_index()


# In[69]:


plt.figure(figsize=(10, 6))
sns.lineplot(x=fraud_transaction_counts_undersampled.index.astype(str), y=fraud_transaction_counts_undersampled.values, marker='o', color='red')
plt.xticks(rotation=45)
plt.title('Fraudulent Transaction Counts per Month-Year After Undersampling')
plt.xlabel('Month-Year')
plt.ylabel('Number of Fraudulent Transactions')
plt.tight_layout()
plt.show()


# In[ ]:





# In[70]:


# Example: Creating new features based on transaction amounts
df_undersampled['amount_log'] = np.log(df_undersampled['amt'] + 1)


# In[71]:


# Calculate correlation matrix
corr_matrix = df_undersampled.corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix After Undersampling')
plt.show()


# ### Positive Correlations
# - **amt** and **amount_log**: 0.85
# - **merch_lat** and **merch_long**: 0.91
# - **trans_24h_count** and **trans_24h_amount_avg**: 0.75
# - **avg_trans_amount** and **total_trans_amount**: 0.30
# - **amt** and **trans_24h_amount_avg**: 0.73
# - **amt** and **total_trans_amount**: 0.23
# - **total_trans_amount** and **amount_log**: 0.12
# - **trans_month** and **trans_year**: 0.19
# 
# ### Negative Correlations
# - **merch_long** and **merch_lat**: -0.91
# - **cc_num** and **merch_lat**: -0.10
# - **cc_num** and **amount_log**: -0.01
# - **amt** and **unix_time**: -0.04
# - **is_fraud** and **trans_month**: -0.02
# - **transaction_day** and **amount_log**: -0.02
# - **transaction_day** and **unix_time**: -0.04
# 
# ### Additional Notes
# - **is_fraud** has a moderate positive correlation with **amt** (0.26) and **amount_log** (0.57).
# - **trans_24h_amount_avg** has a moderate positive correlation with **amt** (0.73) and a moderate positive correlation with **amount_log** (0.64).

# ### Positive Correlations with target
# - **amt**: 0.26
# - **trans_24h_count**: 0.75
# - **trans_24h_amount_avg**: 0.26
# - **amount_log**: 0.57
# 
# ### Negative Correlations with target
# - **transaction_day**: -0.03
# - **transaction_month**: -0.02
# - **total_trans_amount**: -0.30
# - **zip**: -0.07
# - **merch_lat**: -0.06
# - **city_pop**: -0.02

# In[77]:


# List of features to plot
features = ['amt', 'trans_24h_count', 'trans_24h_amount_avg', 'amount_log', 'total_trans_amount']


# In[78]:


# Create box plots for each feature against the target class 'is_fraud'
plt.figure(figsize=(20, 15))
for i, feature in enumerate(features, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(x='is_fraud', y=feature, data=df_undersampled)
    plt.title(f'Box plot of {feature} by Fraud Status')
    plt.xlabel('Is Fraud')
    plt.ylabel(feature)
plt.tight_layout()
plt.show()


# In[ ]:




