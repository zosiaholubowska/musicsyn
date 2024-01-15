import os
import pandas
from scipy.stats import shapiro, levene, ttest_rel
from analysis_pilot import create_df, plot_group, plot_single
import matplotlib.pyplot as plt
import seaborn as sns
from analysis_pilot import create_df
import numpy
import seaborn as sns

create_df()

path = os.getcwd()

part = "main"

main = pandas.read_csv(path + f"/Results/results_postprocessed.csv")
main = main[main['part'].str.match(f'{part}')]
vc_data = pandas.read_csv(path + f"/Results/results.csv")
vc_data = vc_data[vc_data['part'].str.match(f'{part}')]

#-#-#-#-#-#-#-# T-TEST #-#-#-#-#-#-#-#

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




#-#-#-#-#-#-#-# Hit Rate with rhythm #-#-#-#-#-#-#-#


path = os.getcwd()
data = pandas.read_csv(path + f"/Results/results_raw.csv")

data = data[data['stimulus'].str.match('stim')]
data['signal_theory'] = ''

data.loc[(data['loc_change'] == 1) & (data['answer'] == 1), 'signal_theory'] = 'hit'
data.loc[(data['loc_change'] == 1) & (data['answer'] == 0), 'signal_theory'] = 'miss'
data.loc[(data['loc_change'] == 0) & (data['answer'] == 0), 'signal_theory'] = 'corr'
data.loc[(data['loc_change'] == 0) & (data['answer'] == 1), 'signal_theory'] = 'fa'

data = data.reset_index(inplace=False)
data = data.drop(columns='index')

for idx in data.index:
    if idx == 0:
        data.loc[idx, 'prev_duration'] = 0
    else:
        data.loc[idx, 'prev_duration'] = data.loc[idx-1, 'duration']
data_filtered = data[data['visual_cue'] == 1]
counts = data_filtered.groupby(['boundary', 'prev_duration', 'subject', 'signal_theory']).size()
counts = counts.to_frame()
counts.reset_index(inplace=True)
counts = counts.rename(columns={0: 'counts'})
counts_pivot = counts.pivot(index=('subject','boundary', 'prev_duration'), columns='signal_theory', values='counts')
counts_pivot.reset_index(inplace=True)

counts_pivot = counts_pivot.fillna(0)

for i in range(len(counts_pivot)):
    counts_pivot.at[i, "hit_rate"] = counts_pivot.at[i, "hit"] / (counts_pivot.at[i, "hit"] + counts_pivot.at[i, "miss"])
    counts_pivot.at[i, "false_rate"] = counts_pivot.at[i, "fa"] / (counts_pivot.at[i, "fa"] + counts_pivot.at[i, "corr"])

counts_pivot = counts_pivot.fillna(0)

counts_pivot["z_hit_rate"] = ""
counts_pivot["z_false_rate"] = ""

score_columns = ["hit_rate", "false_rate"]

for col in score_columns:
    temp = counts_pivot[col]
    temp_targ = counts_pivot[f"z_{col}"]
    mean = numpy.mean(temp)
    sd = numpy.std(temp)

    for idx in temp.index:
        temp_targ[idx] = (temp[idx] - mean) / sd

    counts_pivot[f"z_{col}"] = temp_targ

counts_pivot['prev_duration'] = counts_pivot['prev_duration'].round(decimals=2)
counts_pivot['boundary'] = counts_pivot['boundary'].map({0: 'within_unit', 1: 'at_boundary'})
g = sns.pointplot(data=counts_pivot, x="prev_duration",  y="hit_rate", hue="boundary", linestyles='none', dodge=True)
g.set(xlabel ="Note duration", ylabel = "Hit Rate", title ='Rhythm control analysis')
plt.savefig(f'{path}/plots/rhythm_and_hit_rate.png', dpi=300)

data_filtered['boundary'] = data_filtered['boundary'].map({0: 'within_unit', 1: 'at_boundary'})
g = sns.displot(data=data_filtered, x="prev_duration", hue="boundary", hue_order=['within_unit', 'at_boundary'], binwidth=0.25)
g.set(xlabel ="Note duration", title ='Rhythm control analysis')

plt.savefig(f'{path}/plots/density_of_rhythm.png', dpi=300)



#-#-#-#-#-#-#-# Hit Rate with frequency #-#-#-#-#-#-#-#


path = os.getcwd()
data = pandas.read_csv(path + f"/Results/results_raw.csv")

data = data[data['stimulus'].str.match('stim')]
data['signal_theory'] = ''

data.loc[(data['loc_change'] == 1) & (data['answer'] == 1), 'signal_theory'] = 'hit'
data.loc[(data['loc_change'] == 1) & (data['answer'] == 0), 'signal_theory'] = 'miss'
data.loc[(data['loc_change'] == 0) & (data['answer'] == 0), 'signal_theory'] = 'corr'
data.loc[(data['loc_change'] == 0) & (data['answer'] == 1), 'signal_theory'] = 'fa'

data = data.reset_index(inplace=False)
data = data.drop(columns='index')

for idx in data.index:
    if idx == 0:
        data.loc[idx, 'prev_freq'] = 0
    else:
        data.loc[idx, 'prev_freq'] = data.loc[idx-1, 'frequencies']
data_filtered = data[data['visual_cue'] == 1]

for index, row in data_filtered.iterrows():
    if row['frequencies'] < row['prev_freq']:
        data_filtered.at[index, 'interval'] = row['prev_freq'] / row['frequencies']
    else:
        data_filtered.at[index, 'interval'] = row['frequencies'] / row['prev_freq']

counts = data_filtered.groupby(['boundary', 'interval', 'subject', 'signal_theory']).size()
counts = counts.to_frame()
counts.reset_index(inplace=True)
counts = counts.rename(columns={0: 'counts'})
counts_pivot = counts.pivot(index=('subject','boundary', 'interval'), columns='signal_theory', values='counts')
counts_pivot.reset_index(inplace=True)

counts_pivot = counts_pivot.fillna(0)

for i in range(len(counts_pivot)):
    counts_pivot.at[i, "hit_rate"] = counts_pivot.at[i, "hit"] / (counts_pivot.at[i, "hit"] + counts_pivot.at[i, "miss"])
    counts_pivot.at[i, "false_rate"] = counts_pivot.at[i, "fa"] / (counts_pivot.at[i, "fa"] + counts_pivot.at[i, "corr"])

counts_pivot = counts_pivot.fillna(0)

counts_pivot["z_hit_rate"] = ""
counts_pivot["z_false_rate"] = ""

score_columns = ["hit_rate", "false_rate"]

for col in score_columns:
    temp = counts_pivot[col]
    temp_targ = counts_pivot[f"z_{col}"]
    mean = numpy.mean(temp)
    sd = numpy.std(temp)

    for idx in temp.index:
        temp_targ[idx] = (temp[idx] - mean) / sd

    counts_pivot[f"z_{col}"] = temp_targ


counts_pivot['interval'] = counts_pivot['interval'].round(decimals=2)
counts_pivot['boundary'] = counts_pivot['boundary'].map({0: 'within_unit', 1: 'at_boundary'})
g = sns.pointplot(data=counts_pivot, x="interval", y="hit_rate", hue="boundary",linestyles="none", dodge=True)
g.set(xlabel ="Interval", ylabel = "Hit Rate", title ='Interval jump control analysis')
plt.savefig(f'{path}/plots/frequency_and_hit_rate.png', dpi=300)

data_filtered['boundary'] = data_filtered['boundary'].map({0: 'within_unit', 1: 'at_boundary'})
g = sns.displot(data=data_filtered, x="interval", hue="boundary", hue_order=['within_unit', 'at_boundary'], binwidth=0.05)

g.set(xticks=numpy.arange(1, 2.25, 0.2))
plt.savefig(f'{path}/plots/density_of_frequency.png', dpi=300)




