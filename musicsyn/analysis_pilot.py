import pandas
import numpy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import  ttest_rel

path = os.getcwd()
subjects = [f for f in os.listdir(f"{path}/musicsyn/Results") if f.startswith("p")]
df = pandas.DataFrame()

for subject in subjects:
    folder_path = f"{path}/musicsyn/Results/{subject}"

    file_names = [f for f in os.listdir(folder_path) if "_data_" in f]

    for file in file_names:
        data = pandas.read_csv(f"{folder_path}/{file}")
        for index, row in data.iterrows():
            # Check if Visual_cue is 1
            if row['visual_cue'] == 1:
                # Check for response in the window of 1.1 seconds
                window_condition = (data['time'] >= row['time']) & (data['time'] <= row['time'] + 1.1)
                found_index = data.index[window_condition & (data['answer'] == 1)].tolist()

                if any((data.loc[window_condition, 'answer'] == 1)):
                    # If there is a response in the window, set the Response value for the current row to 1
                    data.at[index, 'answer'] = 1
                    data.at[index, 'prec_time'] = data['prec_time'][found_index[0]]

        df = pandas.concat([df, data])

df2 = df.iloc[:, 1:]

df2["time_difference"] = df2["prec_time"] - df2["time"]

#### PRINT DISTRIBUTION OF SIGNAL THEORY PER CONDITION

df2 = df2[df2['stimulus'].str.match('stim')]

condition_1 = df2[df2['boundary'] == 1]
condition_0 = df2[df2['boundary'] == 0]

# Creating contingency table for condition 1
table_condition_1 = pandas.crosstab(index=condition_1['loc_change'], columns=condition_1['visual_cue'])

# Creating contingency table for condition 0
table_condition_0 = pandas.crosstab(index=condition_0['loc_change'], columns=condition_0['visual_cue'])

print("Condition 'At boundary'")
print(table_condition_1)

print("\nCondition 'No boundary':")
print(table_condition_0)

#### SAVE RAW RESULTS AS .CSV

df2.to_csv(f'{path}/musicsyn/results.csv')

#### FILTER FOR CUED INSTANCES

vc_data = df2[df2['visual_cue'] == 1].copy()


vc_data['signal_theory'] = ''

vc_data.loc[(vc_data['loc_change'] == 1) & (vc_data['answer'] == 1), 'signal_theory'] = 'hit'
vc_data.loc[(vc_data['loc_change'] == 1) & (vc_data['answer'] == 0), 'signal_theory'] = 'miss'
vc_data.loc[(vc_data['loc_change'] == 0) & (vc_data['answer'] == 0), 'signal_theory'] = 'corr'
vc_data.loc[(vc_data['loc_change'] == 0) & (vc_data['answer'] == 1), 'signal_theory'] = 'fa'

vc_data = vc_data[(vc_data['time_difference'] > 0.2) | (vc_data['answer'] != 1)]


#### CALCULATING D-PRIME

at_boundary = vc_data[vc_data['boundary'] == 1]
no_boundary = vc_data[vc_data['boundary'] == 0]
at_boundary_grouped = at_boundary.groupby(['subject', 'signal_theory']).size().unstack(fill_value=0)
at_boundary_grouped.reset_index(inplace=True)
at_boundary_grouped.columns.name = None
### add false alarms = 0
#at_boundary_grouped['fa'] = 0
#no_boundary_grouped = at_boundary_grouped[['subject', 'corr', 'fa', 'hit', 'miss']]
at_boundary_grouped.columns = ['subject', 'corr', 'fa', 'hit', 'miss']

no_boundary_grouped = no_boundary.groupby(['subject', 'signal_theory']).size().unstack(fill_value=0)
no_boundary_grouped.reset_index(inplace=True)
no_boundary_grouped.columns.name = None
### add false alarms = 0
#no_boundary_grouped['fa'] = 0
#no_boundary_grouped = no_boundary_grouped[['subject', 'corr', 'fa', 'hit', 'miss']]
no_boundary_grouped.columns = ['subject', 'corr', 'fa', 'hit', 'miss']

score_columns = ['corr', 'fa', 'hit', 'miss']


at_boundary_grouped["condition"] = "boundary"
no_boundary_grouped["condition"] = "no_boundary"

main = pandas.concat([at_boundary_grouped, no_boundary_grouped])
main = main.reset_index(drop=True)


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

#### SAVE D-PRIME and F-SCORE INTO CSV
main.to_csv("main.csv", index=False)



#### RESHAPE THE DF FOR THE PAIR-WISE COMPARISON
pairwise_d_prime = main.pivot(index='subject', columns='condition', values='d_prime')
pairwise_ff1 = main.pivot(index='subject', columns='condition', values='ff1')
pairwise_hit_rate = main.pivot(index='subject', columns='condition', values='hit_rate')
vc_data_filtered = vc_data[vc_data['answer'] == 1]
vc_data_filtered['condition'] = numpy.where(vc_data_filtered['boundary'] == 0, 'no_boundary', 'boundary')
mean_time_diff = vc_data_filtered.groupby(['subject', 'condition'])['time_difference'].mean().reset_index()
pairwise_time_difference = mean_time_diff.pivot(index='subject', columns='condition', values='time_difference')

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

palette = sns.color_palette(['#a4e0f5'], len(main['subject'].unique()))
def plot_and_test(ax, data, data_plot, x, y, title):
    sns.lineplot(ax=ax, x=x, y=y, data=data_plot, hue='subject', palette=palette, marker='o', legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['boundary', 'no_boundary'])
    ax.set_xlim(-0.2, 1.2)
    ax.set_title(title)

    t_stat, p_value = ttest_rel(data.iloc[:, 0], data.iloc[:, 1], nan_policy='omit')
    ax.text(0.5, 0.9, f'{p_value:.4f}', transform=ax.transAxes, ha='center')

def plot_avg(ax, data_plot, x, y, title):
    sns.lineplot(ax=ax, x=x, y=y, data=data_plot, color='#008EBF', marker='o', legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['boundary', 'no_boundary'])
    ax.set_xlim(-0.2, 1.2)
    ax.set_title(title)


# Set up subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Pilot - preliminary results')
main['ff1'] = pandas.to_numeric(main['ff1'], errors='coerce')
main['d_prime'] = pandas.to_numeric(main['d_prime'], errors='coerce')

# Hit rate plot
plot_and_test(axes[0, 0], pairwise_hit_rate, main,'condition', 'hit_rate', 'Hit Rate')
plot_avg(axes[0, 0], main,'condition', 'hit_rate', 'Hit Rate')

# D-prime plot
plot_and_test(axes[0, 1], pairwise_d_prime, main,'condition', 'd_prime', 'D-prime')
plot_avg(axes[0, 1], main,'condition', 'd_prime', 'D-prime')

# F-score plot
plot_and_test(axes[1, 0], pairwise_ff1, main,'condition', 'ff1', 'F-score')
plot_avg(axes[1, 0], main,'condition', 'ff1', 'F-score')

# Time difference plot

plot_and_test(axes[1, 1], pairwise_time_difference, mean_time_diff,'condition', 'time_difference', 'Time difference')
plot_avg(axes[1, 1], mean_time_diff,'condition', 'time_difference', 'Time difference')


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('second_pilot_results.png', dpi=300)
plt.show()

#### PLOT - SIGNAL THEORY DISTRIBUTION

counts_boundary = vc_data[vc_data['boundary'] == 1]['signal_theory'].value_counts()
print(counts_boundary)

counts_noboundary = vc_data[vc_data['boundary'] == 0]['signal_theory'].value_counts()
print(counts_noboundary)

# Calculate percentages
total_boundary = len(vc_data[vc_data['boundary'] == 1])
total_noboundary = len(vc_data[vc_data['boundary'] == 0])

percentages_boundary = counts_boundary / total_boundary * 100
percentages_noboundary = counts_noboundary / total_noboundary * 100

categories = ['hit', 'miss', 'corr', 'fa']
bar_width = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=False)

bar_positions_boundary = numpy.arange(len(categories))
bar_positions_noboundary = bar_positions_boundary + bar_width

# Updated color names
colors = {'boundary': 'skyblue', 'noboundary': 'cornflowerblue'}

for category_index, category in enumerate(categories):
    counts_boundary_1 = counts_boundary[category] if category in counts_boundary else 0
    counts_noboundary_1 = counts_noboundary[category] if category in counts_noboundary else 0

    # Plot raw values on the upper subplot
    ax1.bar(bar_positions_boundary[category_index], counts_boundary_1, width=bar_width, label=f'{category} - Boundary',
            color=colors['boundary'])
    ax1.bar(bar_positions_noboundary[category_index], counts_noboundary_1, width=bar_width,
            label=f'{category} - No Boundary', color=colors['noboundary'])

    percentages_boundary_1 = percentages_boundary[category] if category in percentages_boundary else 0
    percentages_noboundary_1 = percentages_noboundary[category] if category in percentages_noboundary else 0

    # Plot percentage values on the lower subplot
    ax2.bar(bar_positions_boundary[category_index], percentages_boundary_1, width=bar_width,
            label=f'{category} - Boundary', color=colors['boundary'])
    ax2.bar(bar_positions_noboundary[category_index], percentages_noboundary_1, width=bar_width,
            label=f'{category} - No Boundary', color=colors['noboundary'])

    # Add labels above each bar with the exact values
    ax1.text(bar_positions_boundary[category_index], counts_boundary_1, f'{counts_boundary_1}', ha='center',
             va='bottom')
    ax1.text(bar_positions_noboundary[category_index], counts_noboundary_1, f'{counts_noboundary_1}', ha='center',
             va='bottom')

    ax2.text(bar_positions_boundary[category_index], percentages_boundary_1, f'{percentages_boundary_1:.2f}%',
             ha='center', va='bottom')
    ax2.text(bar_positions_noboundary[category_index], percentages_noboundary_1, f'{percentages_noboundary_1:.2f}%',
             ha='center', va='bottom')

# Rename labels
labels = {'hit': 'Hit Rate', 'fa': 'False Alarm', 'corr': 'Correct Rejection', 'miss': 'Miss'}

ax1.legend(labels=['Boundary', 'No Boundary'])
ax1.set_ylabel('Raw Values')

ax2.legend(labels=['Boundary', 'No Boundary'])
ax2.set_ylabel('Percentage')

ax2.set_xticks(bar_positions_boundary + bar_width / 2)
ax2.set_xticklabels([labels[category] for category in categories])

ax1.set_xticks(bar_positions_boundary + bar_width / 2)
ax1.set_xticklabels([labels[category] for category in categories])
#plt.savefig('distrib_by_condition.png', dpi=300)
plt.show()

#### ALL VALUES #####

categories = ['hit', 'miss', 'corr', 'fa']
bar_width = 0.5

fig, ax = plt.subplots(figsize=(10, 8))

counts = vc_data['signal_theory'].value_counts()
total = len(vc_data)

percentages = counts / total * 100

bar_positions = numpy.arange(len(categories)) + bar_width / 2  # Adjusted to center bars with ticks

colors = {'all_data': '#008EBF'}

for category_index, category in enumerate(categories):
    counts_1 = counts[category] if category in counts else 0
    percentages_1 = percentages[category] if category in percentages else 0

    # Plot raw values
    ax.bar(bar_positions[category_index], counts_1, width=bar_width, label=f'{percentages_1:.2f}%', color=colors['all_data'])

    # Add labels above each bar with the exact values and percentages
    ax.text(bar_positions[category_index], counts_1, f'{percentages_1:.2f}%', ha='center', va='bottom')

# Rename labels
labels = {'hit': 'Hit', 'fa': 'False Alarm', 'corr': 'Correct Rejection', 'miss': 'Miss'}

ax.set_ylabel('Raw Values')
ax.set_xticks(bar_positions)
ax.set_xticklabels([labels[category] for category in categories])
plt.savefig('combined_plot_test.png', dpi=300)
plt.show()


######## SINGLE PARTICIPANT
categories = ['hit', 'miss', 'corr', 'fa']
bar_width = 0.5

fig, ax = plt.subplots(figsize=(10, 8))
sub = vc_data[(vc_data['subject'] == 'p08')]
counts = sub['signal_theory'].value_counts()
total = len(sub)

percentages = counts / total * 100

bar_positions = numpy.arange(len(categories)) + bar_width / 2  # Adjusted to center bars with ticks

colors = {'all_data': 'royalblue'}

for category_index, category in enumerate(categories):
    counts_1 = counts[category] if category in counts else 0
    percentages_1 = percentages[category] if category in percentages else 0

    # Plot raw values
    ax.bar(bar_positions[category_index], counts_1, width=bar_width, label=f'{percentages_1:.2f}%', color=colors['all_data'])

    # Add labels above each bar with the exact values and percentages
    ax.text(bar_positions[category_index], counts_1, f'{percentages_1:.2f}%', ha='center', va='bottom')

# Rename labels
labels = {'hit': 'Hit Rate', 'fa': 'False Alarm', 'corr': 'Correct Rejection', 'miss': 'Miss'}

ax.set_ylabel('Raw Values')
ax.set_xticks(bar_positions)
ax.set_xticklabels([labels[category] for category in categories])
plt.savefig('p08_plot.png', dpi=300)
plt.show()


#### PARAMETERS
sub_vc_data = vc_data[(vc_data['subject'] == 'p07')]
sub_main = main[(main['subject'] == 'p07')]

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

palette = sns.color_palette(['#a4e0f5'], len(main['subject'].unique()))
def plot_sub(ax, data_plot, x, y, title):
    sns.lineplot(ax=ax, x=x, y=y, data=data_plot, hue='subject', palette=palette, marker='o', legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['boundary', 'no_boundary'])
    ax.set_xlim(-0.2, 1.2)
    ax.set_title(title)


# Set up subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Pilot - preliminary results')
main['ff1'] = pandas.to_numeric(main['ff1'], errors='coerce')
main['d_prime'] = pandas.to_numeric(main['d_prime'], errors='coerce')

# Hit rate plot
plot_sub(axes[0, 0], sub_main,'condition', 'hit_rate', 'Hit Rate')


# D-prime plot
plot_sub(axes[0, 1], sub_main,'condition', 'd_prime', 'D-prime')


# F-score plot
plot_sub(axes[1, 0], sub_main,'condition', 'ff1', 'F-score')


# Time difference plot

plot_sub(axes[1, 1], mean_time_diff,'condition', 'time_difference', 'Time difference')




plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('p07_second_pilot_results_out.png', dpi=300)
plt.show()



