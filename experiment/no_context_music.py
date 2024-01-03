import time
import subprocess
import itertools
import pickle
import numpy
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pandas
import slab

path = 'C://projects//musicsyn'
randgenerator = default_rng()


def read_melody(file):
    score_data = pandas.read_csv(file, sep=",")  # open the csv file with notes
    onsets = score_data.onset_sec.to_list()  # list of onsets of consecutive notes
    frequencies = score_data.freq.to_list()  # frequencies of consecutive notes
    durations = score_data.duration.to_list()  # note durations
    changable_notes = score_data.changable_note.to_list()  # if at note is possible to change direction
    boundaries = (score_data.boundary.to_list())  # 0 or 1 indication if the note is the beginning of a new phrase
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


def run(melody_file, subject):
    file = slab.ResultsFile(subject)
    onsets, frequencies, durations, boundaries, changable_notes = read_melody(f"/Users/zofiaholubowska/Documents/PhD/3_experiment/musicsyn/stimuli/{melody_file}")
    start_time = time.time()

    onsets.append(
        onsets[-1] + durations[-1] + 0.1
    )  # add a dummy onset so that the if statement below works during the last note
    i = 0
    while time.time() - start_time < onsets[-1] + durations[-1]:
        if time.time() - start_time > onsets[i]:  # play the next note
            stim = note(frequencies[i], durations[i])

            if boundaries[i]:
                print("at boundary!")


            stim = stim.externalize()
            file.write(frequencies[i], tag=f"{time.time()-start_time:.3f}")
            play(stim)
            i += 1



if __name__ == "__main__":
    run("regular_major_1.csv", "MS")
