import pandas
import numpy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import  ttest_rel
from experiment.analysis_pilot import create_df
import math

# 0. DOWNLOAD THE DATA

create_df()
path = os.getcwd()
df = pandas.read_csv(path + f"/Results/results_postprocessed.csv")
vc = pandas.read_csv(path + f"/Results/results.csv")
raw = pandas.read_csv(path + f"/Results/results_raw.csv")

# 1. RESULTS FOR THE MAIN CONDITION

# specify condition
condition = 'main'

#create a sub_df
main = df[df['condition'].str.match(f'{condition}')]

# create sub_dfs for stats
pairwise_d_prime = main.pivot(index='subject', columns='state', values='d_prime')
pairwise_ff1 = main.pivot(index='subject', columns='state', values='ff1')
pairwise_hit_rate = main.pivot(index='subject', columns='state', values='hit_rate')
pairwise_false_rate = main.pivot(index='subject', columns='state', values='false_rate')

# plot

palette = sns.color_palette(['#a4e0f5'], len(main['subject'].unique()))

def plot_and_test(ax, data, data_plot, x, y, title, a, b, c, d, e):
    sns.lineplot(ax=ax, x=x, y=y, data=data_plot, hue='subject', palette=palette, marker='o', legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['boundary', 'no_boundary'])
    ax.set_xlim(-0.2, 1.2)
    ax.set_title(title)
    ax.set_yticks(numpy.arange(a, b, c))
    ax.set_ylim(d, e)

    t_stat, p_value = ttest_rel(data.iloc[:, 0], data.iloc[:, 1], nan_policy='omit')
    ax.text(0.5, 0.9, f't(7)= {t_stat:.4f} \np = {p_value:.4f}', transform=ax.transAxes, ha='center')

def plot_avg(ax, data_plot, x, y, title, a, b, c, d, e):
    sns.lineplot(ax=ax, x=x, y=y, data=data_plot, color='#008EBF', hue= 'condition',marker='o', legend=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['at boundary', 'within unit'])
    ax.set_xlim(-0.2, 1.2)
    ax.set_title(title)
    ax.set_yticks(numpy.arange(a, b, c))
    ax.set_ylim(d, e)

# Set up subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Comparison between two conditions - {condition}')
main['ff1'] = pandas.to_numeric(main['ff1'], errors='coerce')
main['d_prime'] = pandas.to_numeric(main['d_prime'], errors='coerce')

# Hit rate plot
plot_and_test(axes[0, 0], pairwise_hit_rate, main, 'state', 'hit_rate', 'Hit Rate', 0.0, 1.01, 0.1, 0.0, 1.1)
plot_avg(axes[0, 0], main, 'state', 'hit_rate', 'Hit Rate', 0.0, 1.01, 0.1, 0.0, 1.1)

# False alarm rate plot
plot_and_test(axes[0, 1], pairwise_false_rate, main, 'state', 'false_rate', 'False Alarm Rate', 0.1, 1.01, 0.1, 0.01, 1.1)
plot_avg(axes[0, 1], main, 'state', 'false_rate', 'False Alarm Rate', 0.1, 1.01, 0.1, 0.01, 1.1)

# D-prime plot
plot_and_test(axes[1, 1], pairwise_d_prime, main, 'state', 'd_prime', 'D-prime', -4, 3, 1, -4.1, 2.5)
plot_avg(axes[1, 1], main, 'state', 'd_prime', 'D-prime', -4, 3, 1, -4.1, 2.5)

# F-score plot
plot_and_test(axes[1, 0], pairwise_ff1, main, 'state', 'ff1', 'F-score', 0.5, 1.01, 0.1, 0.45, 1.1)
plot_avg(axes[1, 0], main, 'state', 'ff1', 'F-score', 0.5, 1.01, 0.1, 0.45, 1.1)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'{path}/plots/{condition}_results.png', dpi=300)
plt.show()


# 2. COMPARISON BETWEEN CONDITIONS

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Comparison between control conditions')
cond = df[~df['condition'].str.match('train')]


plot_avg(axes[0, 0], cond, 'state', 'hit_rate', 'Hit Rate', 0.5, 1.01, 0.1, 0.45, 1.1)
plot_avg(axes[0, 1], cond, 'state', 'false_rate', 'False Alarm Rate', 0.1, 1.01, 0.1, 0.01, 1.1)
plot_avg(axes[1, 1], cond, 'state', 'd_prime', 'D-prime', -4, 3, 1, -4.1, 2.5)
plot_avg(axes[1, 0], cond, 'state', 'ff1', 'F-score', 0.5, 1.01, 0.1, 0.45, 1.1)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'{path}/plots/results_control.png', dpi=300)
plt.show()

# 3. AVERAGE D-PRIME FOR EACH MELODY

# prepare a sub df
vc = vc[vc['condition'].str.match("main")]
vc_boundary = vc[vc['boundary'] == 1]
vc_boundary = vc_boundary.round({'time': 0})
group = vc_boundary.groupby(['stimulus','time','signal_theory']).size().reset_index()
group_piv = group.pivot(index=['stimulus','time'],columns='signal_theory', values=0).reset_index()

for idx in group_piv.index:
    a = group_piv['time'][idx]
    b = group_piv['time'][idx+1]

    if (b-a) == 1 :
        group_piv['time'][idx] = group_piv['time'][idx+1]

#there will be an error, because the loop exceeds the length of the df
gg=group_piv.groupby(['stimulus','time']).sum().reset_index()
counts_gg = gg['stimulus'].value_counts()

# calculate d-prime and hit rate for the boundaries

for i in range(len(gg)):
    gg.at[i, "hit_rate"] = gg.at[i, "hit"] / (gg.at[i, "hit"] + gg.at[i, "miss"])
    gg.at[i, "false_rate"] = gg.at[i, "fa"] / (gg.at[i, "fa"] + gg.at[i, "corr"])

gg['false_rate'] = gg['false_rate'].fillna(0)
gg['hit_rate'] = gg['hit_rate'].fillna(0)
gg["z_hit_rate"] = ""
gg["z_false_rate"] = ""
gg["d_prime"] = ""
score_columns = ["hit_rate", "false_rate"]

for col in score_columns:
    temp = gg[col]
    temp_targ = gg[f"z_{col}"]
    mean = numpy.mean(temp)
    sd = numpy.std(temp)

    for idx in temp.index:
        temp_targ[idx] = (temp[idx] - mean) / sd

    gg[f"z_{col}"] = temp_targ

for idx in gg.index:
    gg["d_prime"][idx] = gg["z_hit_rate"][idx] - gg["z_false_rate"][idx]

# plot the average d-prime for boundaries

sns.set_theme(style="ticks")
avg_d_prime = gg.groupby('stimulus')['d_prime'].mean().reset_index()
avg_d_prime = avg_d_prime.sort_values(by='d_prime', ascending=False)  # Sort by d_prime in descending order
grid = sns.FacetGrid(gg, col="stimulus", hue="stimulus", palette="tab20c",
                     col_wrap=4, height=1.5, col_order=avg_d_prime['stimulus'])  # Use col_order to set the order of the mini-plots
grid.refline(y=0, linestyle=":")
grid.map(plt.plot, "time", "d_prime", marker="o")
for ax, (_, row) in zip(grid.axes.flat, avg_d_prime.iterrows()):
    ax.text(0.5, 0.9, f"Avg d_prime: {row['d_prime']:.2f}", transform=ax.transAxes,
            fontsize=8, ha='center')
grid.set(xticks=numpy.arange(0, 65, 8), yticks=[-5, 2], ylim=(-5.5, 2.5))
grid.fig.tight_layout(w_pad=1)
plt.savefig(f'{path}/plots/d-prime_per_boundary.png', dpi=300)
plt.show()

# 4. CONTROL ANALYSIS

### RHYTHM

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
    if idx < (len(data)-1):
        if data.loc[idx, 'time'] < data.loc[idx+1, 'time']:
            data.loc[idx, 'duration'] = data.loc[idx+1, 'time'] - data.loc[idx, 'time']
        else:
            data.loc[idx, 'duration'] = 1

    elif idx == (len(data)-1):
        data.loc[idx, 'duration'] = 1

def round_up_to_0_125(x):
    return math.ceil(x * 8) / 8
data['duration'] = data['duration'].apply(round_up_to_0_125)

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
counts_pivot['d_prime'] = counts_pivot['z_hit_rate'] - counts_pivot['z_false_rate']
g = sns.pointplot(data=counts_pivot, x="prev_duration",  y="d_prime", hue="boundary", linestyles='none', dodge=True)
g.set(xlabel ="Note duration", ylabel = "D-prime", title ='Rhythm control analysis')
plt.savefig(f'{path}/plots/rhythm_and_hit_rate.png', dpi=300)

data_filtered['boundary'] = data_filtered['boundary'].map({0: 'within_unit', 1: 'at_boundary'})
g = sns.displot(data=data_filtered, x="prev_duration", hue="boundary", hue_order=['within_unit', 'at_boundary'], binwidth=0.25)
g.set(xlabel ="Note duration", title ='Rhythm control analysis')

plt.savefig(f'{path}/plots/density_of_rhythm.png', dpi=300)

### FREQUENCIES
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
counts_pivot['d_prime'] = counts_pivot['z_hit_rate'] - counts_pivot['z_false_rate']
g = sns.pointplot(data=counts_pivot, x="interval", y="d_prime", hue="boundary",linestyles="none", dodge=True)
g.set(xlabel ="Interval", ylabel = "D-prime", title ='Interval jump control analysis')
plt.savefig(f'{path}/plots/frequency_and_hit_rate.png', dpi=300)

data_filtered['boundary'] = data_filtered['boundary'].map({0: 'within_unit', 1: 'at_boundary'})
g = sns.displot(data=data_filtered, x="interval", hue="boundary", hue_order=['within_unit', 'at_boundary'], binwidth=0.05)

g.set(xticks=numpy.arange(1, 2.25, 0.2))
plt.savefig(f'{path}/plots/density_of_frequency.png', dpi=300)


# 5. FILTER DATA

path = os.getcwd()
raw = pandas.read_csv(path + f"/Results/results_raw.csv")

condition = 'main'
#create a sub_df
raw = raw[raw['condition'].str.match(f'{condition}')]
worst_melodies = ['stim_min_4', 'stim_maj_4', 'stim_maj_1', 'stim_min_5']
#best_melodies = ['stim_maj_6', 'stim_maj_4', 'stim_min_6', 'stim_maj_5', 'stim_min_3', 'stim_min_4', 'stim_maj_2', 'stim_maj_3'] #dprime
best_melodies = ['stim_maj_2', 'stim_maj_6', 'stim_maj_1', 'stim_min_3', 'stim_min_4', 'stim_maj_3', 'stim_maj_4', 'stim_maj_5'] #dprime at boundary
raw = raw[~raw["stimulus"].isin(worst_melodies)]
#raw = raw[raw["subject"]!='sub06']
#raw = raw[raw["stimulus"]=='stim_maj_1']

vc_data = raw[raw['visual_cue'] == 1].copy()
vc_data['signal_theory'] = ''
vc_data['block'] = ''
vc_data.loc[(vc_data['loc_change'] == 1) & (vc_data['answer'] == 1), 'signal_theory'] = 'hit'
vc_data.loc[(vc_data['loc_change'] == 1) & (vc_data['answer'] == 0), 'signal_theory'] = 'miss'
vc_data.loc[(vc_data['loc_change'] == 0) & (vc_data['answer'] == 0), 'signal_theory'] = 'corr'
vc_data.loc[(vc_data['loc_change'] == 0) & (vc_data['answer'] == 1), 'signal_theory'] = 'fa'
vc_data = vc_data[(vc_data['time_difference'] > 0.2) | (vc_data['answer'] != 1)]
vc_data.loc[vc_data['stimulus'].str.startswith('test'), 'block'] = 'training'
vc_data.loc[vc_data['stimulus'].str.startswith('stim'), 'block'] = 'experiment'
vc_data.loc[vc_data['boundary']==1, 'state'] = 'boundary'
vc_data.loc[vc_data['boundary']==0, 'state'] = 'no_boundary'
vc_data_grouped = vc_data.groupby(['subject', 'block', 'condition', 'state', 'signal_theory']).size().unstack(
    fill_value=0)
main = vc_data_grouped.reset_index(drop=False)

# CALCULATE HIT RATE AND FALSE RATE

for i in range(len(main)):
    main.at[i, "hit_rate"] = main.at[i, "hit"] / (main.at[i, "hit"] + main.at[i, "miss"])
    main.at[i, "false_rate"] = main.at[i, "fa"] / (main.at[i, "fa"] + main.at[i, "corr"])

main['false_rate'] = main['false_rate'].fillna(0)

main["z_hit_rate"] = ""
main["z_false_rate"] = ""
main["d_prime"] = ""
score_columns = ["hit_rate", "false_rate"]

for col in score_columns:
    temp = main[col]
    temp_targ = main[f"z_{col}"]
    mean = numpy.mean(temp)
    sd = numpy.std(temp)

    for idx in temp.index:
        temp_targ[idx] = (temp[idx] - mean) / sd

    main[f"z_{col}"] = temp_targ

for idx in main.index:
    main["d_prime"][idx] = main["z_hit_rate"][idx] - main["z_false_rate"][idx]

for i in range(len(main)):
    main.at[i, "precision"] = main.at[i, "hit"] / (main.at[i, "hit"] + main.at[i, "fa"])
    main.at[i, "recall"] = main.at[i, "hit"] / (main.at[i, "hit"] + main.at[i, "miss"])
    main.at[i, "ff1"] = 2 / ((1 / main.at[i, "recall"]) + (1 / main.at[i, "precision"]))

df = main.copy()

# 6. CHECK THE EFFECT OF MELODY

results_separate = pandas.DataFrame()

stimuli = [file[:-4] for file in os.listdir(path + '/stimuli') if ".mp3" in file]
i = 0
for stimulus in stimuli:
    raw = pandas.read_csv(path + f"/Results/results_raw.csv")
    raw = raw[raw["stimulus"] == stimulus]

    vc_data = raw[raw['visual_cue'] == 1].copy()
    vc_data['signal_theory'] = ''
    vc_data['block'] = ''
    vc_data.loc[(vc_data['loc_change'] == 1) & (vc_data['answer'] == 1), 'signal_theory'] = 'hit'
    vc_data.loc[(vc_data['loc_change'] == 1) & (vc_data['answer'] == 0), 'signal_theory'] = 'miss'
    vc_data.loc[(vc_data['loc_change'] == 0) & (vc_data['answer'] == 0), 'signal_theory'] = 'corr'
    vc_data.loc[(vc_data['loc_change'] == 0) & (vc_data['answer'] == 1), 'signal_theory'] = 'fa'
    vc_data = vc_data[(vc_data['time_difference'] > 0.2) | (vc_data['answer'] != 1)]
    vc_data.loc[vc_data['stimulus'].str.startswith('test'), 'block'] = 'training'
    vc_data.loc[vc_data['stimulus'].str.startswith('stim'), 'block'] = 'experiment'
    vc_data.loc[vc_data['boundary'] == 1, 'state'] = 'boundary'
    vc_data.loc[vc_data['boundary'] == 0, 'state'] = 'no_boundary'
    vc_data_grouped = vc_data.groupby(['subject', 'block', 'condition', 'state', 'signal_theory']).size().unstack(
        fill_value=0)
    main = vc_data_grouped.reset_index(drop=False)

    for ii in range(len(main)):
        main.at[ii, "hit_rate"] = main.at[ii, "hit"] / (main.at[ii, "hit"] + main.at[ii, "miss"])
        main.at[ii, "false_rate"] = main.at[ii, "fa"] / (main.at[ii, "fa"] + main.at[ii, "corr"])

    main['false_rate'] = main['false_rate'].fillna(0)
    main['hit_rate'] = main['hit_rate'].fillna(0)

    main["z_hit_rate"] = ""
    main["z_false_rate"] = ""
    main["d_prime"] = ""
    score_columns = ["hit_rate", "false_rate"]

    for col in score_columns:
        temp = main[col]
        temp_targ = main[f"z_{col}"]
        mean = numpy.mean(temp)
        sd = numpy.std(temp)

        for idx in temp.index:
            temp_targ[idx] = (temp[idx] - mean) / sd

        main[f"z_{col}"] = temp_targ

    for idx in main.index:
        main["d_prime"][idx] = main["z_hit_rate"][idx] - main["z_false_rate"][idx]

    for iii in range(len(main)):
        main.at[iii, "precision"] = main.at[iii, "hit"] / (main.at[iii, "hit"] + main.at[iii, "fa"])
        main.at[iii, "recall"] = main.at[iii, "hit"] / (main.at[iii, "hit"] + main.at[iii, "miss"])
        main.at[iii, "ff1"] = 2 / ((1 / main.at[iii, "recall"]) + (1 / main.at[iii, "precision"]))

    df = main.copy()

    main = df[df['condition'].str.match('main')]

    pairwise_d_prime = main.pivot(index='subject', columns='state', values='d_prime')
    pairwise_ff1 = main.pivot(index='subject', columns='state', values='ff1')
    pairwise_hit_rate = main.pivot(index='subject', columns='state', values='hit_rate')
    pairwise_false_rate = main.pivot(index='subject', columns='state', values='false_rate')

    results_separate.loc[i, 'stimulus'] = stimulus

    t_stat, p_value = ttest_rel(pairwise_d_prime.iloc[:, 0], pairwise_d_prime.iloc[:, 1], nan_policy='omit')
    results_separate.loc[i, 't_d_prime'] = t_stat
    results_separate.loc[i, 'p_d_prime'] = p_value
    t_stat, p_value = ttest_rel(pairwise_ff1.iloc[:, 0], pairwise_ff1.iloc[:, 1], nan_policy='omit')
    results_separate.loc[i, 't_ff1'] = t_stat
    results_separate.loc[i, 'p_ff1'] = p_value
    t_stat, p_value = ttest_rel(pairwise_hit_rate.iloc[:, 0], pairwise_hit_rate.iloc[:, 1], nan_policy='omit')
    results_separate.loc[i, 't_hit_rate'] = t_stat
    results_separate.loc[i, 'p_hit_rate'] = p_value
    t_stat, p_value = ttest_rel(pairwise_false_rate.iloc[:, 0], pairwise_false_rate.iloc[:, 1], nan_policy='omit')
    results_separate.loc[i, 't_false_rate'] = t_stat
    results_separate.loc[i, 'p_false_rate'] = p_value



    i += 1

results_separate = results_separate.sort_values(by=['t_d_prime'])


# 7. EFFECT OF MUSICALITY

gold_msi = pandas.read_excel("/Users/zofiaholubowska/Documents/PhD/1_admin/experiments/participants.xlsx", sheet_name='gold_MSI')
gold_msi = gold_msi.rename(columns={'sub_ID': 'subject'})
main_piv = main.pivot(index='subject', columns='state', values=['hit_rate', 'false_rate', 'd_prime', 'ff1']).reset_index()
main_piv.columns = ['_'.join(col).strip() for col in main_piv.columns.values]
main_piv = main_piv.rename(columns={'subject_': 'subject'})
merged_df = pandas.merge(main_piv, gold_msi[['subject', 'MT [7*7=49]', 'GI [18*7=126]']], on='subject', how='left')
var = [x for x in main_piv.columns if x != 'subject']

for idx in merged_df.index:
    merged_df.at[idx, 'hit_diff'] = merged_df.at[idx, var[0]] - merged_df.at[idx, var[1]]
    merged_df.at[idx, 'false_diff'] = merged_df.at[idx, var[2]] - merged_df.at[idx, var[3]]
    merged_df.at[idx, 'd_prime_diff'] = merged_df.at[idx, var[4]] - merged_df.at[idx, var[5]]
    merged_df.at[idx, 'ff1_diff'] = merged_df.at[idx, var[6]] - merged_df.at[idx, var[7]]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd

# Convert data to numeric type
x = pd.to_numeric(merged_df['MT [7*7=49]'], errors='coerce')
y_vars = ['hit_diff', 'false_diff', 'd_prime_diff', 'ff1_diff']
y_values = [pd.to_numeric(merged_df[var], errors='coerce') for var in y_vars]

# Create a figure and subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Iterate through each subplot and plot the scatter plot with correlation line
for i, ax in enumerate(axs.flat):
    y = y_values[i]
    ax.scatter(x, y)
    ax.set_title(f'Correlation between MT and {y_vars[i]}')
    ax.set_xlabel('MT [7*7=49]')
    ax.set_ylabel(y_vars[i])
    ax.grid(True)

    # Calculate Pearson correlation coefficient and p-value
    mask = ~np.isnan(x) & ~np.isnan(y)
    r, p_value = pearsonr(x[mask], y[mask])

    # Plot correlation line
    z = np.polyfit(x[mask], y[mask], 1)
    p = np.poly1d(z)
    ax.plot(x[mask], p(x[mask]), color='red', linestyle='--', label=f'Corr: {r:.2f}, p-value: {p_value:.4f}')
    ax.legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# 8. ANALYSIS OF A FAKE TRIAL - BLANK

main = vc[vc['condition'].str.match('blank')]
drop = main[['subject', 'boundary', 'visual_cue', 'answer']]
main_grouped = drop.groupby(['subject', 'boundary']).sum().reset_index()
main_grouped['percentage'] = main_grouped['answer'] / main_grouped['visual_cue']