import os
import pandas
from scipy.stats import shapiro, levene, ttest_rel
from analysis_pilot import create_df, plot_group, plot_single
import matplotlib.pyplot as plt
import seaborn as sns

create_df()

path = os.getcwd()

part = "main"

main = pandas.read_csv(path + f"/Results/results_postprocessed.csv")
main = main[main['part'].str.match(f'{part}')]
vc_data = pandas.read_csv(path + f"/Results/results.csv")
vc_data = vc_data[vc_data['part'].str.match(f'{part}')]

#-#-#-# T-tEST

tstats = []
pvalue = []

measures = ['hit_rate', 'd_prime', 'ff1']
for m in measures:

    print(f'Results for {measure}')

    temp = main.pivot(index='subject', columns='condition', values=m)

    ### CHECK THE DISTRIBUTION
    # if Shapiro - Wilk test has p > 0.05 it is good, because
    # it means that the distribution is close to normal
    shapiro(temp['boundary'])
    shapiro(temp['no_boundary'])
    sns.displot(temp, x="boundary", kind="kde")
    sns.displot(temp, x="no_boundary", kind="kde")

    ### CHECK THE VARIANCE - only if you do t-test for independed groups
    # if Levene test has p > 0.05 it is good, because
    # it means that the variance is homogenic
    levene(temp['boundary'], pairwise_d_prime['no_boundary'])

    ### PAIRED T-TEST
    t, p = ttest_rel(temp['boundary'], temp['no_boundary'])

    tstats.append(t)
    pvalue.append(p)


df = pandas.DataFrame(list(zip(measures, tstats, pvalue)),
               columns =['Measure', 'T-stats', 'p-value'])




