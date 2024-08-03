import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Read Dataset
df = pd.read_csv('./dataset/cookie_cats.csv')
print(df.info()) # No null values

# Basic EDA to understand Dataset:
# Datapoints in each group:
version_counts = df['version'].value_counts()
plt.figure(figsize=(6, 4))
plt.pie(x=version_counts,
        labels=version_counts.index,
        radius=0.5,
        startangle=90)
plt.title('Distribution of sample in A/B Testing')
plt.axis('equal') # to ensure the pie chart is perfectly circular
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
