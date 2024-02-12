import pandas
import matplotlib.pyplot as plt
import numpy
from analysis_pilot import create_df, plot_group, plot_single
import os
import seaborn as sns

# import the data

path = os.getcwd()
main = pandas.read_csv(path + f"/Results/results_postprocessed.csv")
vc_data = pandas.read_csv(path + f"/Results/results.csv")

# add participant's type
main_piv = main.pivot(index='subject', columns='state', values='d_prime')
pivot_df = main.pivot(index=['subject', 'condition', ], columns='state', values='d_prime').reset_index()
pivot_df = pivot_df[pivot_df['condition'].str.match('main')]
pivot_df['diff'] = pivot_df['boundary'] - pivot_df['no_boundary']
pivot_df['type'] = pivot_df['diff'].apply(lambda x: 'increase' if x < 0 else 'decrease')
subject_type_map = dict(zip(pivot_df['subject'], pivot_df['type']))
main['type'] = main['subject'].map(subject_type_map)


## plot the d-prime values for each participant
grid = sns.FacetGrid(main, col="subject", hue="condition", palette="colorblind",
                     col_wrap=4, height=1.5)
grid.map(plt.plot, "state", "d_prime", marker="o")
grid.set(yticks=[-3, 3],ylim=(-3.5, 3.5))

grid.fig.tight_layout(w_pad=1)
grid.add_legend()

## average plot for all participants

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Comparison between three conditions - d-prime')

palette = sns.color_palette(['#a4e0f5'], len(main['subject'].unique()))

conditions = ['main', 'rhythm', 'melody', 'main', 'rhythm', 'melody']
types = ['increase', 'increase', 'increase', 'decrease', 'decrease', 'decrease']

for ax, condition, type in zip(axes.flat, conditions, types):  # Fixed the loop and renamed the variable 'type'
    data = main[(main['condition'] == condition) & (main['type'] == type)]
    sns.lineplot(ax=ax, x="state", y="d_prime", data=data, hue='subject', palette=palette, marker='o', legend=False)
    ax.set_xticks([0, 1])  # Changed from axes[0] to ax
    ax.set_xticklabels(['boundary', 'no_boundary'])
    ax.set_xlim(-0.2, 1.2)  # Changed from axes[0] to ax
    ax.set_title(f"{condition} - {type}")  # Added condition and type_value to the title
    sns.lineplot(ax=ax, x="state", y="d_prime", data=data, color='#008EBF', marker='o', legend=False)



