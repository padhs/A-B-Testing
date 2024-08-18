import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# hypothesis testing
from scipy.stats import shapiro
import scipy.stats as stats

# Read Dataset
df = pd.read_csv('./dataset/cookie_cats.csv')
print(df.info())  # No null values

# Check for duplicates:
duplicates = df[df.duplicated(subset='userid', keep=False)]
# Count duplicates:
duplicate_count = duplicates['userid'].value_counts()
print(f"Duplicate count: {duplicate_count}")  # There are no duplicates in the dataset

# descriptive stats
percentiles = [0.01, 0.05, 0.1, 0.15, 0.20, 0.80, 0.90, 0.95, 0.99]
des_stats = df.describe(percentiles)[['sum_gamerounds']]
print(des_stats)

version_stats = df.groupby('version').sum_gamerounds.agg(['count', 'mean', 'std', 'min', 'max'])
print(version_stats)

# Basic EDA to understand Dataset:
# Datapoints in each group:
version_counts = df['version'].value_counts()
plt.figure(figsize=(6, 4))
plt.pie(x=version_counts,
        labels=version_counts.index,
        radius=0.5,
        startangle=90)
plt.title('Distribution of sample in A/B Testing')
plt.axis('equal')  # to ensure the pie chart is perfectly circular
plt.show()

# Distribution of retentions:
retention_1_count = df['retention_1'].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(['False', 'True'], retention_1_count)
plt.xlabel('retention_1')
plt.ylabel('Count')
plt.title('Distribution of retention_1 in A/B Testing')
plt.show()

retention_7_count = df['retention_7'].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(['False', 'True'], retention_7_count)
plt.xlabel('retention_7')
plt.ylabel('Count')
plt.title('Distribution of retention_7 in A/B Testing')
plt.show()

# There are some significant distribution changes in retention_7
'''
Since there are some significant differences in distribution of data
in retention of players after 7 days, it confirms that the results of A/B Testing are consistently repeatable.
Let's analyse further to understand player psychology and retention. What drives player engagement.
'''

# Check for outliers in the dataset: (Boxplot/Violin/Scatter plots) by version vs. sum_gamerounds

plt.figure(figsize=(18, 6))
sns.boxplot(x=df['version'], y=df['sum_gamerounds'])

plt.suptitle('Outliers in the distribution of players at gate_30 & gate_40', fontsize=20)
plt.title('Distribution of control & test groups', fontsize=10)

plt.tight_layout(pad=4)
plt.show()
# There seems to be an outlier for gate_30.

# Removing Outlier:
df = df[df['sum_gamerounds'] < df['sum_gamerounds'].max()]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
df['sum_gamerounds'].hist(ax=axes[0], color="steelblue")
df[(df.version == "gate_30")].hist(df["sum_gamerounds"], ax=axes[1], color="steelblue")
df[(df.version == "gate_40")].hist(df["sum_gamerounds"], ax=axes[2], color="steelblue")
sns.boxplot(x=df['version'], y=df['sum_gamerounds'], ax=axes[3])

plt.suptitle("After Removing The Extreme Value", fontsize=20)
axes[0].set_title("Distribution of Total Game Rounds", fontsize=15)
axes[1].set_title("Distribution of Gate 30 (A)", fontsize=15)
axes[2].set_title("Distribution of Gate 40 (B)", fontsize=15)
axes[1].set_title("Distribution of Two Groups", fontsize=15)
plt.tight_layout(pad=4)
