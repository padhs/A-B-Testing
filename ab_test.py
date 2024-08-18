import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

fig, axes = plt.subplots(1, 4, figsize = (18,5))
df.sum_gamerounds.hist(ax = axes[0], color = "steelblue")
df[(df.version == "gate_30")].hist("sum_gamerounds", ax = axes[1], color = "steelblue")
df[(df.version == "gate_40")].hist("sum_gamerounds", ax = axes[2], color = "steelblue")
sns.boxplot(x = df.version, y = df.sum_gamerounds, ax = axes[3])

plt.suptitle("After Removing The Extreme Value", fontsize = 20)
axes[0].set_title("Distribution of Total Game Rounds", fontsize = 15)
axes[1].set_title("Distribution of gate_30", fontsize = 15)
axes[2].set_title("Distribution of gate_40", fontsize = 15)
axes[3].set_title("Distribution of Two Groups", fontsize = 15)

plt.tight_layout(pad = 4)
plt.show()

# After removing all the outliers having z-scores > 3
df[(df.version == "gate_30")].reset_index().set_index("index").sum_gamerounds.plot(legend = True, label = "Gate 30", figsize = (20,5))
df[df.version == "gate_40"].reset_index().set_index("index").sum_gamerounds.plot(legend = True, label = "Gate 40", alpha = 0.8)
plt.suptitle("After Removing The Extreme Value", fontsize = 20)

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
'''


