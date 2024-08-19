import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro
import scipy.stats as stats


# dataFrame
df = pd.read_csv('./dataset/cookie_cats_cleaned.csv')

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

# Outliers are there in data --> ab_eda.py

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
df.sum_gamerounds.hist(ax=axes[0], color="steelblue")
df[(df.version == "gate_30")].hist("sum_gamerounds", ax=axes[1], color="steelblue")
df[(df.version == "gate_40")].hist("sum_gamerounds", ax=axes[2], color="steelblue")
sns.boxplot(x=df.version, y=df.sum_gamerounds, ax=axes[3])

plt.suptitle("After Removing The Extreme Value", fontsize=20)
axes[0].set_title("Distribution of Total Game Rounds", fontsize=15)
axes[1].set_title("Distribution of gate_30", fontsize=15)
axes[2].set_title("Distribution of gate_40", fontsize=15)
axes[3].set_title("Distribution of Two Groups", fontsize=15)

plt.tight_layout(pad=4)
plt.show()

# After removing all the outliers having z-scores > 3
df[(df.version == "gate_30")].reset_index().set_index("index").sum_gamerounds.plot(legend=True,
                                                                                   label="gate_30",
                                                                                   figsize=(20, 5))
df[df.version == "gate_40"].reset_index().set_index("index").sum_gamerounds.plot(legend=True,
                                                                                 label="gate_40",
                                                                                 alpha=0.8)
plt.suptitle("After Removing The Extreme Value", fontsize=20)

plt.show()

# People who didn't play:
'''
Implied by sum_gamerounds value = 0. They're not at gate_30 or gate_40. They haven't cleared a game round. Why ?
1. Don't play or don't like to play or other prefs.
2. No time to play.
3. Other reasons
'''

'''
As we play & progress in the game, the levels get tougher and the speed of level clearance decreases.
Therefore, we tend to lose engagement. Very few of the players reach the end of the game through the funnel.
The numbers decrease as we progress level by level. This increases the churn rate. KPI for difficulty level measure.
Retention ideas:
1. Make levels easier
2. Introduce Hints/Help for tougher levels. Create Sink Economy for players' monetization
3. Keys to surpass levels
'''

# users vs. sum_gamerounds
fig3, axes = plt.subplots(3, 1, figsize=(25, 20))
df.groupby("sum_gamerounds").userid.count().plot(ax=axes[0])
df.groupby("sum_gamerounds").userid.count()[:200].plot(ax=axes[1])
df.groupby("sum_gamerounds").userid.count()[:85].plot(ax=axes[2])
plt.suptitle("The number of users in the game rounds played", fontsize=25)
axes[0].set_title("How many users are there all game rounds?", fontsize=15)
axes[1].set_title("How many users are there first 200 game rounds?", fontsize=15)
axes[2].set_title("How many users are there in the first 85 game rounds?", fontsize=15)

plt.tight_layout(pad=5)
plt.show()  # ~20 to 25 users are there.

# This confirms that the user count decreases as they progress through the levels.
# L0: 3994 -> players have installed the game maybe played. But not cleared the first level
# L1: 5538 -> max & decreases L1 through L20 onwards
print(df.groupby('sum_gamerounds').userid.count().reset_index().head(40))

# User counts that reached gate_30 & gate_40:
print(df.groupby("sum_gamerounds").userid.count().loc[[30, 40]])
# 30 -> 642,
# 40 -> 505

print(df.groupby("version").sum_gamerounds.agg(["count", "median", "mean", "std", "max"]))
# The number of users at gate_30 & gate_40 more or less seems to be similar.
# We need to check if the results of this is statistically significant

# For retention_1
print(df.groupby(["version", "retention_1"]).sum_gamerounds.agg(["count", "median", "mean", "std", "max"]))

# For retention_2
print(df.groupby(['version', 'retention_7']).sum_gamerounds.agg(["count", "median", "mean", "std", "max"]))
print(f"\n")

# Retention
# There is no statistically significant difference. But False counts in retention_7 is ~23k higher
# which means more players don't come to play the game after 7 days.
# But the ratio of players coming & not-coming for ret_1 is similar.
# Ratio Not similar for ret_7 (non-returns are significantly higher)

retention_df = pd.DataFrame({"RET1_COUNT": df["retention_1"].value_counts(),
                             "RET7_COUNT": df["retention_7"].value_counts(),
                             "RET1_RATIO": df["retention_1"].value_counts() / len(df),
                             "RET7_RATIO": df["retention_7"].value_counts() / len(df)})
print(retention_df)

# Key Insight: 18% of players might continue playing the game when they come back after 7 days.

df["Retention"] = np.where((df.retention_1 == True) & (df.retention_7 == True), 1, 0)
print(df.groupby(["version", "Retention"])["sum_gamerounds"].agg(["count",
                                                                  "median",
                                                                  "mean",
                                                                  "std",
                                                                  "max"]))
# When the retention variables are combined and 2 groups are compared, summary statistics are similar as well.

df["NewRetention"] = list(map(lambda x, y: str(x)+"-"+str(y), df.retention_1, df.retention_7))
print(df.groupby(["version",
                  "NewRetention"]).sum_gamerounds.agg(["count",
                                                       "median",
                                                       "mean",
                                                       "std",
                                                       "max"]).reset_index())

# Hypothesis Testing:

# assigning a/b groups
df['group'] = np.where(df.version == 'gate_30', 'A', 'B')

print(df.head(10))
df.to_csv('./dataset/cookie_cats_hypo.csv', index=False)
