import time
import subprocess
import itertools
import pickle
import numpy
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pandas
import slab
import random

path = 'C://projects//musicsyn'
randgenerator = default_rng()


def read_melody(file):
    score_data = pandas.read_csv(file, sep=",")  # open the csv file with notes
    onsets = score_data.onset_sec.to_list()  # list of onsets of consecutive notes
    frequencies = score_data.freq.to_list()  # frequencies of consecutive notes
    durations = score_data.duration.to_list()  # note durations
    changable_notes = score_data.changable_note.to_list()  # if at note is possible to change direction
    boundaries = (score_data.boundary.to_list()) # 0 or 1 indication if the note is the beginning of a new phrase

    return onsets, frequencies, durations, boundaries, changable_notes

def expenv(n_samples):
    t = numpy.linspace(start=0, stop=1, num=n_samples)
    return slab.Signal(numpy.exp(-(0.69 * 5) * t))

def note(f0, duration):
    sig = slab.Sound.harmoniccomplex(f0, duration)
    env = expenv(sig.n_samples)
    sig = sig * env
    return slab.Binaural(sig.ramp())


def play(stim):
    stim.write("tmp.wav")
    subprocess.Popen(["afplay", "tmp.wav"])

def shuffle_rhythm(onsets, durations, boundaries):
    shuffling_df = pandas.DataFrame()
    shuffling_df['onsets'] = onsets
    shuffling_df['durations'] = durations
    shuffling_df['boundaries'] = boundaries
    shuffling_df['duration_total'] = ''
    for row in shuffling_df.index:
        try:
            shuffling_df['duration_total'][row] = shuffling_df['onsets'][row + 1] - shuffling_df['onsets'][row]
            if shuffling_df['boundaries'][row] == 1:
                shuffling_df['boundaries'][row - 1] = 2
        except KeyError:
            shuffling_df['duration_total'][row] = shuffling_df['durations'][row]
    shuffling_df['idx'] = range(0, len(shuffling_df))
    sh = shuffling_df[shuffling_df['boundaries'] == 0]
    idx = sh['idx'].tolist()
    sh = sh.sample(frac=1)
    sh['idx'] = idx
    bo = shuffling_df[shuffling_df['boundaries'] != 0]
    frames = [bo, sh]
    mixed = pandas.concat(frames)
    durations_df = mixed.sort_values(by='idx')
    durations_df = durations_df.reset_index(drop=True)
    durations_df['onsets'][0] = 0
    for row in durations_df.index:
        if row != 0:
            durations_df['onsets'][row] = durations_df['onsets'][row - 1] + durations_df['duration_total'][row - 1]
        else:
            durations_df['onsets'][0] = 0
    durations = durations_df['durations'].tolist()
    onsets = durations_df['onsets'].tolist()

    return durations, onsets


def shuffle_melody(frequencies, boundaries):
    shuffling_df = pandas.DataFrame()
    shuffling_df['frequencies'] = frequencies
    shuffling_df['boundaries'] = boundaries
    for row in shuffling_df.index:
        if shuffling_df['boundaries'][row] == 1:
            shuffling_df['boundaries'][row - 1] = 2
    shuffling_df['idx'] = range(0, len(shuffling_df))
    sh = shuffling_df[shuffling_df['boundaries'] == 0]
    idx = sh['idx'].tolist()
    sh = sh.sample(frac=1)
    sh['idx'] = idx
    bo = shuffling_df[shuffling_df['boundaries'] != 0]
    frames = [bo, sh]
    mixed = pandas.concat(frames)
    durations_df = mixed.sort_values(by='idx')
    durations_df = durations_df.reset_index(drop=True)
    frequencies = durations_df['frequencies'].tolist()

    return frequencies


def run(melody_file, subject, condition):
    file = slab.ResultsFile(subject)
    onsets, frequencies, durations, boundaries, changable_notes = read_melody(f"/Users/zofiaholubowska/Documents/PhD/3_experiment/musicsyn/stimuli/{melody_file}")

    if condition == 'rhythm':
        durations, onsets = shuffle_rhythm(onsets, durations, boundaries)
    elif condition == 'melody':
        frequencies = shuffle_melody(frequencies, boundaries)
    start_time = time.time()
    onsets.append(
        onsets[-1] + durations[-1] + 0.1
    )  # add a dummy onset so that the if statement below works during the last note
    i = 0
    try:
        while time.time() - start_time < onsets[-1] + durations[-1]:
            if time.time() - start_time > onsets[i]:  # play the next note
                stim = note(frequencies[i], durations[i])

                if boundaries[i]:
                    print("at boundary!")


                stim = stim.externalize()
                file.write(frequencies[i], tag=f"{time.time()-start_time:.3f}")
                play(stim)
                i += 1
    except IndexError:
        pass



if __name__ == "__main__":
    run("stim_min_1_a.csv", "MS", 'melody')
