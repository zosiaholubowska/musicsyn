import slab
import pandas
import numpy
import os
import matplotlib.pyplot as plt
import seaborn as sns

path = os.getcwd()
subjects = [f for f in os.listdir(f"{path}/musicsyn/Results") if f.startswith("p")]
df = pandas.DataFrame()

for subject in subjects:
    folder_path = f"{path}/musicsyn/Results/{subject}"

    # Get all file names in the folder
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


df2.to_csv(f'{path}/musicsyn/results.csv')

vc_data = df2[df2['visual_cue'] == 1].copy()

# Create a new column 'Signal_theory' based on conditions for vc_data
vc_data['signal_theory'] = ''

# Set 'Signal_theory' based on conditions for vc_data
vc_data.loc[(vc_data['loc_change'] == 1) & (vc_data['answer'] == 1), 'signal_theory'] = 'hit'
vc_data.loc[(vc_data['loc_change'] == 1) & (vc_data['answer'] == 0), 'signal_theory'] = 'miss'
vc_data.loc[(vc_data['loc_change'] == 0) & (vc_data['answer'] == 0), 'signal_theory'] = 'corr'
vc_data.loc[(vc_data['loc_change'] == 0) & (vc_data['answer'] == 1), 'signal_theory'] = 'fa'

# Count occurrences based on 'Signal_theory' with 1 in the 'Boundary' column
counts_boundary = vc_data[vc_data['boundary'] == 1]['signal_theory'].value_counts()
print(counts_boundary)

# Count occurrences based on 'Signal_theory' with 0 in the 'Boundary' column
counts_noboundary = vc_data[vc_data['boundary'] == 0]['signal_theory'].value_counts()
print(counts_noboundary)

# Calculate percentages
total_boundary = len(vc_data[vc_data['boundary'] == 1])
total_noboundary = len(vc_data[vc_data['boundary'] == 0])

percentages_boundary = counts_boundary / total_boundary * 100
percentages_noboundary = counts_noboundary / total_noboundary * 100

#### ------- D PRIME ------- #####

at_boundary = vc_data[vc_data['boundary'] == 1]
no_boundary = vc_data[vc_data['boundary'] == 0]
at_boundary_grouped = at_boundary.groupby(['subject', 'stimulus', 'signal_theory']).size().unstack(fill_value=0)
at_boundary_grouped.reset_index(inplace=True)
at_boundary_grouped.columns.name = None
at_boundary_grouped.columns = ['subject', 'stimulus', 'corr', 'fa', 'hit', 'miss']

no_boundary_grouped = no_boundary.groupby(['subject', 'stimulus', 'signal_theory']).size().unstack(fill_value=0)
no_boundary_grouped.reset_index(inplace=True)
no_boundary_grouped.columns.name = None
no_boundary_grouped.columns = ['subject', 'stimulus', 'corr', 'fa', 'hit', 'miss']

score_columns = ['corr', 'fa', 'hit', 'miss']
dfs = [at_boundary_grouped, no_boundary_grouped]

for frame in dfs:
    frame["z_hit"] = ""
    frame["z_miss"] = ""
    frame["z_fa"] = ""
    frame["z_corr"] = ""
    frame["d_prime"] = ""

    for col in score_columns:
        temp = frame[col]
        temp_targ = frame[f"z_{col}"]
        mean = numpy.mean(temp)
        sd = numpy.std(temp)

        for idx in temp.index:
            temp_targ[idx] = (temp[idx] - mean) / sd

        frame[f"z_{col}"] = temp_targ

    for idx in frame.index:
        frame["d_prime"][idx] = frame["z_hit"][idx] - frame["z_fa"][idx]

at_boundary_grouped["condition"] = "boundary"
no_boundary_grouped["condition"] = "no_boundary"

main = pandas.concat([at_boundary_grouped, no_boundary_grouped])
main = main.reset_index(drop=True)

main['participant'] = main['subject'].str.replace(r'a.*', '', regex=True)


for i in range(len(main)):
    main.at[i, "precision"] = main.at[i, "hit"] / (main.at[i, "hit"] + main.at[i, "fa"])
    main.at[i, "recall"] = main.at[i, "hit"] / (main.at[i, "hit"] + main.at[i, "miss"])
    main.at[i, "ff1"] = 2 / ((1 / main.at[i, "recall"]) + (1 / main.at[i, "precision"]))


main.to_csv("main.csv", index=False)

### BOXPLOT

sns.boxplot(x='d_prime', y='condition', data = main)
sns.stripplot(x='d_prime', y='condition', data=main, color = "grey")
plt.show()

#### ------- PLOT ------- #####

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

plt.show()
