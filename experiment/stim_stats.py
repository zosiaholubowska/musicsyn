import os
import pandas as pd

files = ["stim_maj_1.csv", "stim_maj_2.csv", "stim_maj_3.csv"]
os.chdir('/Users/zofiaholubowska/Documents/PhD/3_experiment/stimuli')

# create an empty df for the stim_stats
stim_stats = pd.DataFrame(columns=['file_name', 'onsets_length', 'sum_boundaries', 'sum_changable_notes', 'boundaries_minus_changable'])


def read_melody(file):
    score_data = pd.read_csv(file, sep=";")  # open the csv file with notes
    onsets = score_data.onset_sec.to_list()  # list of onsets of consecutive notes
    frequencies = score_data.freq.to_list()  # frequencies of consecutive notes
    durations = score_data.duration_sec.to_list()  # note durations
    boundaries = score_data.boundary.to_list()  # 0 or 1 indication if the note is the beginning of a new phrase
    changable_notes = score_data.changable_note.to_list()  # if at note is possible to change direction
    onsets.append(onsets[-1] + durations[-1] + 0.1)  # I add a dummy note here
    durations.append(0.1)  # I add a dummy note here
    return onsets, frequencies, durations, boundaries, changable_notes

idx = 0
# Iterate through each file
for file in files:
    onsets, frequencies, durations, boundaries, changable_notes = read_melody(file)

    onsets_length = len(onsets)
    sum_boundaries = sum(boundaries)
    sum_changable_notes = sum(changable_notes)
    boundaries_minus_changable = sum_changable_notes - sum_boundaries

    stim_stats.at[idx, 'file_name'] = file
    stim_stats.at[idx, 'onsets_length'] = onsets_length
    stim_stats.at[idx, 'sum_boundaries'] = sum_boundaries
    stim_stats.at[idx, 'sum_changable_notes'] = sum_changable_notes
    stim_stats.at[idx, 'boundaries_minus_changable'] = boundaries_minus_changable



    idx += 1



# Print stim_stats
print(stim_stats.to_string())
