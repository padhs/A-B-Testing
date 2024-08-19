import pandas as pd
import mpmath
from scipy.stats import shapiro, anderson
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Hypothesis testing
'''
1. split the dataset into control & test groups
2. Apply shapiro test for normal distribution
3. Tests: Levene, T-test, Welch, Mann Whitney U test (non-parametric)
'''

df = pd.read_csv('./dataset/cookie_cats_hypo.csv')

# split into a/b datasets
group_A = df[df['group'] == "A"]['sum_gamerounds']
group_B = df[df['group'] == "B"]['sum_gamerounds']

# check normality by shapiro test (not accurate for large datasets.)
normal_A = shapiro(group_A)[1] < 0.05
normal_B = shapiro(group_B)[1] < 0.05
print(f'Shapiro test: {normal_A} for p-value < 0.05 i.e non-normal distribution')
print(f'Shapiro test: {normal_B} for p-value < 0.05 i.e non-normal distribution')

figA = sns.histplot(group_A, bins=100, kde=False, color='steelblue')
plt.show()

figB = sns.histplot(group_B, bins=100, kde=False, color='steelblue')
plt.show()
# The distributions also show a +ve/ Right-skewed distribution. Not normally distributed.

'''
Since non-normal, non-parametric tests: Mann Whitney U & Chi-squared test.
Chi-squared test is preferred if there is categorical or ordinal data. Since sum_gamerounds is numerical data, 
Mann Whitney U makes more sense as a statistical non-parametric test.
We'll use Mann whitney to compare distributions of the independent groups. The outcome variable is boolean 
H0: Both have the same distribution. i.e group_A == group_B
H1: Both have different distributions. They're not same.
'''

# Mann Whitney U
mw_test = stats.mannwhitneyu(group_A, group_B)[1]
print(f"Mann Whitney p-value: \n{mw_test}")

if mw_test < 0.05:
    # Reject null hypothesis
    print(f"Both have different distributions. They're not same.")
else:
    print(f"Both have same distributions.")

