import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
subject = "AP"
file = f"/Users/zofiaholubowska/Documents/PhD/experiment/musicsyn/Results/{subject}/{subject}_data.csv"
data = pd.read_csv(file, sep=",")
print(data)

# Iterate through each row
for index, row in data.iterrows():
    # Check if Visual_cue is 1
    if row['Visual_cue'] == 1:
        # Check for response in the window of 1.5 seconds
        window_condition = (data['Timestamp'] >= row['Timestamp']) & (data['Timestamp'] <= row['Timestamp'] + 1.5)
        if any((data.loc[window_condition, 'Responses'] == 1)):
            # If there is a response in the window, set the Response value for the current row to 1
            data.at[index, 'Responses'] = 1

filtered_data = data[data['Visual_cue'] == 1].copy()

# Create a new column 'Signal_theory' based on conditions for filtered_data
filtered_data['Signal_theory'] = 'Unknown'

# Set 'Signal_theory' based on conditions for filtered_data
filtered_data.loc[(filtered_data['Location_change'] == 1) & (filtered_data['Responses'] == 1), 'Signal_theory'] = 'hit'
filtered_data.loc[(filtered_data['Location_change'] == 1) & (filtered_data['Responses'] == 0), 'Signal_theory'] = 'miss'
filtered_data.loc[(filtered_data['Location_change'] == 0) & (filtered_data['Responses'] == 0), 'Signal_theory'] = 'corr'
filtered_data.loc[(filtered_data['Location_change'] == 0) & (filtered_data['Responses'] == 1), 'Signal_theory'] = 'fa'

# Count occurrences based on 'Signal_theory' with 1 in the 'Boundary' column
counts_boundary = filtered_data[filtered_data['Boundary'] == 1]['Signal_theory'].value_counts()
print(counts_boundary)

# Count occurrences based on 'Signal_theory' with 0 in the 'Boundary' column
counts_noboundary = filtered_data[filtered_data['Boundary'] == 0]['Signal_theory'].value_counts()
print(counts_noboundary)

# Calculate percentages
total_boundary = len(filtered_data[filtered_data['Boundary'] == 1])
total_noboundary = len(filtered_data[filtered_data['Boundary'] == 0])

percentages_boundary = counts_boundary / total_boundary * 100
percentages_noboundary = counts_noboundary / total_noboundary * 100

# Create a bar plot for each category based on 'Boundary' and 'Signal_theory'
categories = ['hit', 'miss', 'corr', 'fa']
bar_width = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=False)

bar_positions_boundary = np.arange(len(categories))
bar_positions_noboundary = bar_positions_boundary + bar_width

# Updated color names
colors = {'boundary': 'mediumseagreen', 'noboundary': 'lightcoral'}

for category_index, category in enumerate(categories):
    counts_boundary_1 = counts_boundary[category] if category in counts_boundary else 0
    counts_noboundary_1 = counts_noboundary[category] if category in counts_noboundary else 0

    # Plot raw values on the upper subplot
    ax1.bar(bar_positions_boundary[category_index], counts_boundary_1, width=bar_width, label=f'{category} - Boundary', color=colors['boundary'])
    ax1.bar(bar_positions_noboundary[category_index], counts_noboundary_1, width=bar_width, label=f'{category} - No Boundary', color=colors['noboundary'])

    percentages_boundary_1 = percentages_boundary[category] if category in percentages_boundary else 0
    percentages_noboundary_1 = percentages_noboundary[category] if category in percentages_noboundary else 0

    # Plot percentage values on the lower subplot
    ax2.bar(bar_positions_boundary[category_index], percentages_boundary_1, width=bar_width, label=f'{category} - Boundary', color=colors['boundary'])
    ax2.bar(bar_positions_noboundary[category_index], percentages_noboundary_1, width=bar_width, label=f'{category} - No Boundary', color=colors['noboundary'])

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

fig.savefig("/Users/zofiaholubowska/Documents/PhD/experiment/musicsyn/analysis.png", bbox_inches='tight', dpi=300)