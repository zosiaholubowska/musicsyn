import pandas as pandas
import matplotlib.pyplot as plt
import numpy as np

# Read the data
subject = "AP"
file = f"/Users/zofiaholubowska/Documents/PhD/experiment/musicsyn/Results/{subject}/{subject}_data.csv"
data = pandas.read_csv(file, sep=",")

boundaries_data = data[data['Boundary'] == 1].copy()
sum_bound_loc = (sum(boundaries_data["Location_change"]))
sum_bound_cue = (sum(boundaries_data["Visual_cue"]))
noboundaries_data = data[data['Boundary'] == 0].copy()
sum_nobound_loc = (sum(noboundaries_data["Location_change"]))
sum_nobound_cue = (sum(noboundaries_data["Visual_cue"]))

print(data)
print(sum_bound_loc)
print(sum_bound_cue)
print(sum_nobound_loc)
print(sum_nobound_cue)
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

counts_all = filtered_data['Signal_theory'].value_counts()

# Count occurrences based on 'Signal_theory' with 1 in the 'Boundary' column
counts_boundary = filtered_data[filtered_data['Boundary'] == 1]['Signal_theory'].value_counts()

# Count occurrences based on 'Signal_theory' with 0 in the 'Boundary' column
counts_noboundary = filtered_data[filtered_data['Boundary'] == 0]['Signal_theory'].value_counts()

print(counts_all)
print(counts_boundary)
print(counts_noboundary)

# Create a bar plot for each category based on 'Boundary' and 'Signal_theory'
categories = ['hit', 'miss', 'corr', 'fa']
bar_width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bar_positions_boundary = np.arange(len(categories))
bar_positions_noboundary = bar_positions_boundary + bar_width

# Updated color names
colors = {'boundary': 'mediumseagreen', 'noboundary': 'lightcoral'}

for category_index, category in enumerate(categories):
    counts_boundary_1 = counts_boundary[category] if category in counts_boundary else 0
    counts_noboundary_1 = counts_noboundary[category] if category in counts_noboundary else 0

    ax.bar(bar_positions_boundary[category_index], counts_boundary_1, width=bar_width, label=f'{category} - Boundary 1', color=colors['boundary'])
    ax.bar(bar_positions_noboundary[category_index], counts_noboundary_1, width=bar_width, label=f'{category} - Boundary 0', color=colors['noboundary'])

# Rename labels
labels = {'hit': 'Hit Rate', 'fa': 'False Alarm', 'corr': 'Correct Rejection', 'miss': 'Miss'}

ax.set_xticks(bar_positions_boundary + bar_width / 2)
ax.set_xticklabels([labels[category] for category in categories])
ax.legend(labels=['Boundary', 'No Boundary'])
plt.show()
