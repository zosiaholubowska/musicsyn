import pandas
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'data'
# Modify the path if needed
subject = "FH"
melody_file = "stim_maj_1"
file = f"/Users/zofiaholubowska/Documents/PhD/3_experiment/musicsyn/Results/{subject}/{subject}_data_{melody_file}.csv"
data = pandas.read_csv(file, sep=",")

print(data.columns)
# Total number of notes
total_notes = 54

# Counting occurrences
total_location_changes = data['Location_change'].sum()
total_visual_cues = data['Visual_cue'].sum()

total_location_with_cue = data[(data['Location_change'] == 1) & (data['Visual_cue'] == 1)].shape[0]
total_location_without_cue = data[(data['Location_change'] == 1) & (data['Visual_cue'] == 0)].shape[0]
total_cue_without_location = data[(data['Location_change'] == 0) & (data['Visual_cue'] == 1)].shape[0]
total_without_cue_and_location = data[(data['Location_change'] == 0) & (data['Visual_cue'] == 0)].shape[0]

# Divide by boundary/no-boundary
boundary_data = data[data['Boundary'] == 1]
noboundary_data = data[data['Boundary'] == 0]

total_location_changes_boundary = boundary_data['Location_change'].sum()
total_visual_cues_boundary = boundary_data['Visual_cue'].sum()

total_location_with_cue_boundary = boundary_data[(boundary_data['Location_change'] == 1) & (boundary_data['Visual_cue'] == 1)].shape[0]
total_location_without_cue_boundary = boundary_data[(boundary_data['Location_change'] == 1) & (boundary_data['Visual_cue'] == 0)].shape[0]
total_cue_without_location_boundary = boundary_data[(boundary_data['Location_change'] == 0) & (boundary_data['Visual_cue'] == 1)].shape[0]
total_without_cue_and_location_boundary = boundary_data[(boundary_data['Location_change'] == 0) & (boundary_data['Visual_cue'] == 0)].shape[0]

total_location_changes_noboundary = noboundary_data['Location_change'].sum()
total_visual_cues_noboundary = noboundary_data['Visual_cue'].sum()

total_location_with_cue_noboundary = noboundary_data[(noboundary_data['Location_change'] == 1) & (noboundary_data['Visual_cue'] == 1)].shape[0]
total_location_without_cue_noboundary = noboundary_data[(noboundary_data['Location_change'] == 1) & (noboundary_data['Visual_cue'] == 0)].shape[0]
total_cue_without_location_noboundary = noboundary_data[(noboundary_data['Location_change'] == 0) & (noboundary_data['Visual_cue'] == 1)].shape[0]
total_without_cue_and_location_noboundary = noboundary_data[(noboundary_data['Location_change'] == 0) & (noboundary_data['Visual_cue'] == 0)].shape[0]

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Total numbers
total_numbers = [total_location_changes, total_visual_cues, total_location_with_cue, total_location_without_cue, total_cue_without_location, total_without_cue_and_location]
labels = ['Location Changes', 'Visual Cues', 'Location with Cue', 'Location without Cue', 'Cue without Location', 'Without Cue and Location']
bar1 = axes[0, 0].bar(labels, total_numbers, color='skyblue')
axes[0, 0].set_title(f"Total number of notes = {total_notes}")

# Add annotations on top of each bar
for rect in bar1:
    height = rect.get_height()
    axes[0, 0].annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom')

# Division for boundary and no-boundary
total_numbers_boundary = [total_location_changes_boundary, total_visual_cues_boundary, total_location_with_cue_boundary,
                           total_location_without_cue_boundary, total_cue_without_location_boundary, total_without_cue_and_location_boundary]
bar2 = axes[1, 1].bar(labels, total_numbers_boundary, color='salmon')
axes[1, 1].set_title(f"Total number of notes = {total_notes} (Boundary)")

# Add annotations on top of each bar
for rect in bar2:
    height = rect.get_height()
    axes[1, 1].annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom')

# Division for no-boundary
total_numbers_noboundary = [total_location_changes_noboundary, total_visual_cues_noboundary, total_location_with_cue_noboundary,
                             total_location_without_cue_noboundary, total_cue_without_location_noboundary, total_without_cue_and_location_noboundary]
bar3 = axes[1, 0].bar(labels, total_numbers_noboundary, color='lightgreen')
axes[1, 0].set_title(f"Total number of notes = {total_notes} (No Boundary)")

# Add annotations on top of each bar
for rect in bar3:
    height = rect.get_height()
    axes[1, 0].annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom')

# Adjusting x-axis labels
for ax in axes.flat:
    ax.tick_params(axis='x', labelrotation=45)

plt.tight_layout()
plt.show()

fig.savefig("/Users/zofiaholubowska/Documents/PhD/experiment/musicsyn/cues_distribution.png", bbox_inches='tight', dpi=300)
