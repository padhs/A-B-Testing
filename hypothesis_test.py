import pandas as pd
import numpy as np
import scipy.stats as stats
import mpmath
from scipy.stats import shapiro, anderson
import seaborn as sns
import matplotlib.pyplot as plt

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
print(f'Shapiro test: {normal_A}')
print(f'Shapiro test: {normal_B}')

# anderson test
resultA = anderson(group_A)
resultB = anderson(group_B)

# Print the results
print('Anderson-Darling Test Statistic:', resultA.statistic)
print('Critical Values:', resultA.critical_values)
print('Significance Levels:', resultA.significance_level)


print('Anderson-Darling Test Statistic:', resultB.statistic)
print('Critical Values:', resultB.critical_values)
print('Significance Levels:', resultB.significance_level)


def anderson_darling_p_value(ad_statistic):

    # Desired precision
    mpmath.mp.dps = 100

    if ad_statistic >= 0.6:
        p = mpmath.exp(1.2937 - 5.709 * ad_statistic + 0.0186 * ad_statistic ** 2)
    elif 0.34 < ad_statistic < 0.6:
        p = mpmath.exp(0.9177 - 4.279 * ad_statistic - 1.38 * ad_statistic ** 2)
    elif 0.20 < ad_statistic <= 0.34:
        p = 1 - mpmath.exp(-8.318 + 42.796 * ad_statistic - 59.938 * ad_statistic ** 2)
    else:  # ad_statistic <= 0.20
        p = 1 - mpmath.exp(-13.436 + 101.14 * ad_statistic - 223.73 * ad_statistic ** 2)

    return p


p_valueA = anderson_darling_p_value(resultA.statistic)  # AD Statistic result
print(f"P-Value for AD Statistic {resultA.statistic}: {p_valueA}")

p_valueB = anderson_darling_p_value(resultB.statistic)  # AD Statistic result
print(f"P-Value for AD Statistic {resultB.statistic}: {p_valueB}")

print(f"Does group_A follow normal distribution: {p_valueA < 0.05}")
print(f"Does group_B follow normal distribution: {p_valueB < 0.05}")

figA = sns.histplot(group_A, bins=100, kde=False, color='steelblue')
plt.show()

figB = sns.histplot(group_B, bins=100, kde=False, color='steelblue')
plt.show()  # Right skewed normal distribution

# Both have similar kind of distribution plots & both are not normally distributed

