import pandas
import os
from mido import MidiFile
import json


def notetofreq(note):
    a = 440
    return (a / 32) * (2 ** ((note - 9) / 12))

os.chdir('/Users/zofiaholubowska/Documents/PhD/3_experiment/musicsyn/stimuli')

files = ["stim_maj_4.mid", "stim_min_4.mid"]

for file in files:

    mid = MidiFile(file)


    midi_dict_list = []
    for track in mid.tracks:
        for event in track:
            event_dict = event.dict()
            event_dict['type'] = event.type
            midi_dict_list.append(event_dict)

    # Create a DataFrame from the list of dictionaries
    midi_df = pandas.DataFrame(midi_dict_list)

    midi_df = midi_df[midi_df['type'] == 'note_on']

    columns_to_keep = ['type', 'time', 'note', 'velocity']  # Add other column names you want to keep
    midi_df = midi_df[columns_to_keep]

    midi_df['time'] /= 480
    midi_df['time'] = midi_df['time'].cumsum()

    midi_df['offset'] = 0

    midi_df['offset'] = midi_df['time'].shift(-1)
    midi_df.loc[midi_df['velocity'] == 0, 'offset'] = midi_df['time']

    midi_df = midi_df.loc[midi_df['velocity'] != 0]

    midi_df['duration'] = midi_df['offset'] - midi_df['time']
    midi_df.drop(columns=['type'], inplace=True)
    midi_df['freq'] = midi_df['note'].apply(notetofreq)
    midi_df.rename(columns={'time': 'onset_sec'}, inplace=True)
    fl = file[:-4]
    midi_df.to_csv(f'{fl}.csv')




"""


files = ["stim_maj_1.csv", "stim_maj_2.csv", "stim_maj_3.csv", "stim_min_1.csv", "stim_min_2.csv", "stim_min_3.csv"]

#for file in files:

score_data = pandas.read_csv("stim_maj_1.csv", sep=";", header=None, index_col=None)
score_data = score_data.drop(0, axis=1)
print(score_data.head(5))
score_data = score_data.rename(columns={1: "onset_beats", 2: "duration_beats", 3: "midi_channel", 4: "midi_pitch", 5: "freq", 6: "velocity", 7: "onset_sec", 8: "duration_sec", 9: "boundary", 10: "changable_note"})
print(list(score_data.columns))
onsets = score_data.onset_sec.to_list()

print(onsets)




freq = []

midi = score_data.midi_pitch.to_list()

for m in midi:
    fr = notetofreq(m)
    freq.append(fr)

score_data.insert(4, "freq", freq)
fl = file[:-4]

score_data.to_csv(f'{fl}.csv')

"""