import pandas
import os


def notetofreq(note):
    a = 440
    return (a / 32) * (2 ** ((note - 9) / 12))

os.chdir('/Users/zofiaholubowska/Documents/PhD/3_experiment/stimuli')

files = ["stim_maj_1.csv", "stim_maj_2.csv", "stim_maj_3.csv", "stim_min_1.csv", "stim_min_2.csv", "stim_min_3.csv"]

#for file in files:

score_data = pandas.read_csv("stim_maj_1.csv", sep=";", header=None, index_col=None)
score_data = score_data.drop(0, axis=1)
print(score_data.head(5))
score_data = score_data.rename(columns={1: "onset_beats", 2: "duration_beats", 3: "midi_channel", 4: "midi_pitch", 5: "freq", 6: "velocity", 7: "onset_sec", 8: "duration_sec", 9: "boundary", 10: "changable_note"})
print(list(score_data.columns))
onsets = score_data.onset_sec.to_list()

print(onsets)



"""
freq = []

midi = score_data.midi_pitch.to_list()

for m in midi:
    fr = notetofreq(m)
    freq.append(fr)

score_data.insert(4, "freq", freq)
fl = file[:-4]

score_data.to_csv(f'{fl}.csv')

"""