import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Read Dataset
df = pd.read_csv('./dataset/cookie_cats.csv')
print(df.info()) # No null values

'''
About the dataset:
userid, version, sum_gamerounds, retention_1, retention_7
Version: gate_30, gate_40 (control & test groups respectively)
retention_1 => Retention of player for 1 day(s)
retention_7 => Player coming back to play after 7 days ?
'''

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

