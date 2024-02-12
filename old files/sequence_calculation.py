import time
import subprocess
import itertools
import pickle
import numpy
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pandas
import slab

file = "regular_major_1.csv"


def read_melody(file):
    score_data = pandas.read_csv(file, sep=";")  # open the csv file with notes
    onsets = score_data.onset_sec.to_list()  # list of onsets of consecutive notes
    freq = score_data.freq.to_list()  # frequencies of consecutive notes
    frequencies = [float(f.replace(",", ".")) for f in freq]  # replace , with . in the frequencies
    durations = score_data.duration_sec.to_list()  # note durations
    boundaries = score_data.boundary.to_list()  # 0 or 1 indication if the note is the beginning of a new phrase
    onsets.append(onsets[-1] + durations[-1] + 0.1)  # I add a dummy note here
    durations.append(0.1)  # I add a dummy note here
    return onsets, frequencies, durations, boundaries


onsets, frequencies, durations, boundaries = read_melody(file)

def create_sequence(boundaries, dev_freq, subject):
    boundaries_df = pandas.DataFrame(boundaries, columns=['boundary'])
    boundaries_df['idx'] = range(len(boundaries_df))

    #so this part creates a sequence for location change
    n_notes = boundaries.count(0)
    n_boundaries = boundaries.count(1)

    n_reps_notes = round((100 * n_notes)/(100+100*dev_freq))
    n_reps_boundaries = round((100 * n_boundaries)/(100+100*dev_freq))

    s_notes = slab.Trialsequence(conditions=1, n_reps=n_reps_notes, deviant_freq=dev_freq)
    s_notes_df = pandas.DataFrame(s_notes, columns=['sequence'])
    seq_zeros = s_notes_df['sequence']
    s_zeros = seq_zeros.to_list()

    s_boundaries = slab.Trialsequence(conditions=1, n_reps=10, deviant_freq=dev_freq)
    s_boundaries_df = pandas.DataFrame(s_boundaries, columns=['sequence'])
    seq_ones = s_boundaries_df['sequence']
    s_ones = seq_ones.to_list()

    bound_zeros = boundaries_df.loc[boundaries_df['boundary'] == 0]
    if len(s_zeros) > len(bound_zeros):
        s_zeros = s_ones[ :len(bound_zeros)]
    elif len(s_zeros) < len(bound_zeros):
        s_zeros.append(1)
    bound_zeros.insert(1, 'sequence', s_zeros)

    bound_ones = boundaries_df.loc[boundaries_df['boundary'] == 1]
    if len(s_ones) > len(bound_ones):
        s_ones = s_ones[ :len(bound_ones)]
    bound_ones.insert(1,'sequence', s_ones)

    frames = [bound_zeros, bound_ones]
    sequence = pandas.concat(frames)
    temp_seq = sequence.sort_values(by='idx', ascending=True)

    #this part is to give cues without location change, at 20% rate

    n_locchange = ((temp_seq['sequence']).tolist()).count(0)
    n_locstay = ((temp_seq['sequence']).tolist()).count(1)

    n_reps_locchange = round((100 * n_locchange)/(100+100*dev_freq))
    n_reps_locstay = round((100 * n_locstay)/(100+100*dev_freq))

    s_locchange = slab.Trialsequence(conditions=1, n_reps=n_reps_locchange, deviant_freq=dev_freq)
    s_locchange_df = pandas.DataFrame(s_locchange, columns=['cue'])
    seq_locchange = s_locchange_df['cue']
    s_locchange = seq_locchange.to_list()

    s_locstay = slab.Trialsequence(conditions=1, n_reps=n_locstay, deviant_freq=dev_freq)
    s_locstay_df = pandas.DataFrame(s_locstay, columns=['cue'])
    seq_locstay = s_locstay_df['cue']
    s_locstay = seq_locstay.to_list()

    bound_locchange = temp_seq.loc[temp_seq['sequence'] == 0]
    if len(s_locchange) > len(bound_locchange):
        s_locchange = s_locchange[ :len(bound_locchange)]
    elif len(s_locchange) < len(bound_locchange):
        s_locchange.append(1)
    bound_locchange.insert(1, 'cue', s_locchange)

    bound_locstay = temp_seq.loc[temp_seq['sequence'] == 1]
    if len(s_locstay) > len(bound_locstay):
        s_locstay = s_locstay[ :len(bound_locstay)]
    bound_locstay.insert(1,'cue', s_locstay)

    frames_f = [bound_locchange, bound_locstay]
    seq_f = pandas.concat(frames_f)
    final = seq_f.sort_values(by='idx', ascending=True)
    final.to_csv(
        f"/Users/zofiaholubowska/Documents/PhD/experiment/musicsyn/Results/{subject}/{subject}_seq.csv",
    )
    return final


final = create_sequence(boundaries, 0.25, "ZH")

print(final.columns)
print(len(final.loc[(final['boundary'] == 1) & (final['sequence'] == 0)]))
print(len(final.loc[(final['boundary'] == 0) & (final['sequence'] == 0)]))
print(len(final.loc[(final['boundary'] == 1) & (final['cue'] == 0) & (final['sequence'] == 0)]))
print(len(final.loc[(final['boundary'] == 0) & (final['cue'] == 0) & (final['sequence'] == 1)]))
print(len(final.loc[(final['boundary'] == 1) & (final['cue'] == 0) & (final['sequence'] == 1)]))
print(len(final.loc[(final['boundary'] == 0) & (final['cue'] == 0) & (final['sequence'] == 0)]))

