import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read df
df = pd.read_csv('./dataset/cookie_cats.csv')

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


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
df[(df.version == 'gate_30')].hist('sum_gamerounds', ax=axes[0], color='steelblue')
df[(df.version == 'gate_30')].hist('sum_gamerounds', ax=axes[1], color='steelblue')
sns.boxplot(x=df.version, y=df.sum_gamerounds, ax=axes[2])

plt.suptitle('Before removing outliers', fontsize=20)
axes[0].set_title('Distribution of gate_30 (A)', fontsize=15)
axes[1].set_title('Distribution of gate_40 (B)', fontsize=15)
axes[2].set_title('Distributions of both gates', fontsize=15)

plt.tight_layout(pad=4)
plt.show()

# Feature Range: (There is an outlier for gate_30)
df[df.version == "gate_30"].reset_index().set_index("index").sum_gamerounds.plot(
    legend=True,
    label="gate_30",
    figsize=(20, 5))
df[df.version == "gate_40"].reset_index().set_index("index").sum_gamerounds.plot(
    legend=True,
    label="gate_40")
plt.suptitle("Before Removing The Extreme Value", fontsize=20)

plt.show()

'''
There is outlier in the gate_30 feature.
How to handle outlier ?
1. Delete
2. Set the outlier variable values to the max value of the feature
3. Impute the mean of the feature and set the values of the outliers as the imputed mean
4. Other methods: log, cube-roots
For this case, since there is only 1 outlier, let's just go ahead and delete that value.
'''

# Finding outlier: (z-score) method
'''
~99.7% of datapoints can be found within 3 SDs. Giving them a z-score of 3.
We'll look for values having z-scores greater than 3. They're our outliers skewing the data distribution
'''

mean_value = df.sum_gamerounds.mean()
print(f"Mean value: {mean_value}")
std_dev = df.sum_gamerounds.std()
print(f"Standard Deviation value: {std_dev}")
df['z_score'] = (df['sum_gamerounds'] - mean_value)/std_dev

outliers = df[(df['z_score'] > 3) | (df['z_score'] < -3)]
print(f"Outliers: \n")
print(outliers) # ~0.5% of the dataset is outliers. Let's remove them
# print(f"\nTotal observations: {df.shape[0]}")

# Removing outliers:
df_cleaned = df[(df['z_score'] <= 3) & (df['z_score'] >= -3)] # Keep in df if true or else remove
print(f"Observations of dataFrame before removing outliers: {df.shape[0]}\n")
print(f"Observations of dataFrame after removing outliers: {df_cleaned.shape[0]}\n")

df_cleaned = df_cleaned.drop(columns=['z_score'])
df_cleaned.to_csv('./dataset/cookie_cats_cleaned.csv', index=False)
