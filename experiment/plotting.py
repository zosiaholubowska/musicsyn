import pandas
import numpy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import  ttest_rel
from experiment.read_data import read_data

path = os.getcwd()
df = pandas.read_csv(path + f"/Results/results_postprocessed.csv")
df = df[df['block'].str.match("experiment")]
raw = pandas.read_csv(path + f"/Results/results_raw.csv")
raw = raw[raw['block'].str.match("experiment")]
vc = pandas.read_csv(path+ f"/Results/results.csv")
vc = vc[vc['block'].str.match("experiment")]
vc = vc[vc['condition'].str.match("main")]
vc_boundary = vc[vc['boundary'] == 1]
vc_boundary = vc_boundary.round({'time': 0})
group = vc_boundary.groupby(['stimulus','time','signal_theory']).size().reset_index()
group_piv = group.pivot(index=['stimulus','time'],columns='signal_theory', values=0).reset_index()
counts = group_piv['stimulus'].value_counts()

for idx in group_piv.index:
    a = group_piv['time'][idx]
    b = group_piv['time'][idx+1]

    if (b-a) == 1 :
        group_piv['time'][idx] = group_piv['time'][idx+1]

gg=group_piv.groupby(['stimulus','time']).sum().reset_index()
counts_gg = gg['stimulus'].value_counts()


# CALCULATE HIT RATE AND FALSE RATE

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

for i in range(len(gg)):
    gg.at[i, "precision"] = gg.at[i, "hit"] / (gg.at[i, "hit"] + gg.at[i, "fa"])
    gg.at[i, "recall"] = gg.at[i, "hit"] / (gg.at[i, "hit"] + gg.at[i, "miss"])
    gg.at[i, "ff1"] = 2 / ((1 / gg.at[i, "recall"]) + (1 / gg.at[i, "precision"]))


sns.set_theme(style="ticks")
# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(gg, col="stimulus", hue="stimulus", palette="tab20c",
                     col_wrap=4, height=1.5)

# Draw a horizontal line to show the starting point
grid.refline(y=0, linestyle=":")

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "time", "d_prime", marker="o")

# Adjust the tick positions and labels
grid.set(xticks=numpy.arange(0, 65, 8), yticks=[-5, 2], ylim=(-5.5, 2.5))

# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=1)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate average d_prime for each stimulus
avg_d_prime = gg.groupby('stimulus')['d_prime'].mean().reset_index()
avg_d_prime = avg_d_prime.sort_values(by='d_prime', ascending=False)  # Sort by d_prime in descending order

# Plotting
sns.set_theme(style="ticks")
grid = sns.FacetGrid(gg, col="stimulus", hue="stimulus", palette="tab20c",
                     col_wrap=4, height=1.5, col_order=avg_d_prime['stimulus'])  # Use col_order to set the order of the mini-plots

# Draw a horizontal line to show the starting point
grid.refline(y=0, linestyle=":")

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "time", "d_prime", marker="o")

# Add average d_prime label to each mini-plot
for ax, (_, row) in zip(grid.axes.flat, avg_d_prime.iterrows()):
    ax.text(0.5, 0.9, f"Avg d_prime: {row['d_prime']:.2f}", transform=ax.transAxes,
            fontsize=8, ha='center')

# Adjust the tick positions and labels
grid.set(xticks=np.arange(0, 65, 8), yticks=[-5, 2], ylim=(-5.5, 2.5))

# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=1)

plt.show()


#### CUMULATIVE HIT RATE - EFFECT OF TRAINING ######

## we use vs data

vc_grouped = vc.groupby(['subject', 'condition', 'condition_n','trial_n', 'signal_theory']).size().unstack(fill_value=0)
vc_grouped.reset_index(inplace=True)
vc_grouped['hit_cum'] = ''
vc_grouped['miss_cum'] = ''
vc_grouped['fa_cum'] = ''
vc_grouped['corr_cum'] = ''

vc_cum = pandas.DataFrame()

subjects = vc_grouped['subject'].unique()

for subject in subjects:
    temp_df = vc_grouped[vc_grouped["subject"] == subject]
    temp_df['hit_cum'] = temp_df['hit'].cumsum()
    temp_df['miss_cum'] = temp_df['miss'].cumsum()
    temp_df['fa_cum'] = temp_df['fa'].cumsum()
    temp_df['corr_cum'] = temp_df['corr'].cumsum()

    vc_cum = pandas.concat([vc_cum, temp_df])

for idx in vc_cum.index:
    vc_cum.at[idx, "hit_rate"] = vc_cum.at[idx, "hit"] / (vc_cum.at[idx, "hit"] + vc_cum.at[idx, "miss"])
    vc_cum.at[idx, "false_alarm_rate"] = vc_cum.at[idx, "fa"] / (vc_cum.at[idx, "fa"] + vc_cum.at[idx, "corr"])
    hr = vc_cum.at[idx, "hit_rate"]
    far = vc_cum.at[idx, "false_alarm_rate"]
    dp = hr - far

vc_cum_melt = pandas.melt(vc_cum, id_vars=["subject", 'condition', 'condition_n', 'trial_n'], value_vars=["hit_rate"])
sns.lineplot(data=vc_cum_melt, x="trial_n", y="value")

gg.to_csv(f'{path}/Results/gg.csv')
avg_d_prime.to_csv(f'{path}/Results/avg_d_prime.csv')
