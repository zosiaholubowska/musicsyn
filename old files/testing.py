import pandas
import numpy as np
import os

file = "stim_maj_2.csv"
os.chdir('/Users/zofiaholubowska/Documents/PhD/3_experiment/stimuli')

def read_melody(file):
    score_data = pandas.read_csv(file, sep=";")  # open the csv file with notes
    onsets = score_data.onset_sec.to_list()  # list of onsets of consecutive notes
    frequencies = score_data.freq.to_list()  # frequencies of consecutive notes
    durations = score_data.duration_sec.to_list()  # note durations
    boundaries = score_data.boundary.to_list()  # 0 or 1 indication if the note is the beginning of a new phrase
    changable_notes = score_data.changable_note.to_list() #if at note is possible to change direction
    onsets.append(onsets[-1] + durations[-1] + 0.1)  # I add a dummy note here
    durations.append(0.1)  # I add a dummy note here
    return onsets, frequencies, durations, boundaries, changable_notes


onsets, frequencies, durations, boundaries, changable_notes = read_melody(file)

print(len(boundaries))
print(len(changable_notes))

boundaries_df = pandas.DataFrame(np.column_stack([boundaries, changable_notes]),
                                 columns=['boundary', 'changable_notes'])
print(boundaries_df)
