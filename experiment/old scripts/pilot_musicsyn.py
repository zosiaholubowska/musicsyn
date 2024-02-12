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
plt.ion()
randgenerator = default_rng()
ils = pickle.load(open(path + f"/stimuli/ils.pickle", "rb"))  # load interaural level spectrum

def read_melody(file):
    score_data = pandas.read_csv(file, sep=",")
    onsets = score_data.onset_sec.to_list()
    freq = score_data.freq.to_list()
    frequencies = [float(f.replace(",", ".")) for f in freq]
    durations = score_data.duration_sec.to_list()
    boundaries = score_data.boundary.to_list()
    return onsets, frequencies, durations, boundaries


def make_deviant_sequence(boundaries):
    seq = slab.Trialsequence(conditions=1, n_reps=174, deviant_freq=0.15)
    sequ = [seq.trials[i] - boundaries[i] for i in range(len(boundaries))]
    return sequ


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
    onsets, frequencies, durations, boundaries = read_melody(path + f"/stimuli/{melody_file}")
    seq = make_deviant_sequence(boundaries)
    directions = itertools.cycle([0, 20])
    direction_jitter = 5
    start_time = time.time()
    # setup the figure for button capture
    fig = plt.figure("stairs")

    def on_key(event):
        print("write key: ", event.key)
        file.write(event.key, tag=f"{time.time()-start_time:.3f}")

    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    onsets.append(
        onsets[-1] + durations[-1] + 0.1
    )  # add a dummy onset so that the if statement below works during the last note
    i = 0
    direction = next(directions)
    while time.time() - start_time < onsets[-1] + durations[-1]:
        if time.time() - start_time > onsets[i]:  # play the next note
            stim = note(frequencies[i], durations[i])
            if seq[i] == 0:
                direction = next(directions)  # toggle direction
                print("direction change")
            if boundaries[i]:
                print("at boundary!")
            direction_addon = randgenerator.uniform(
                low=-direction_jitter / 2, high=direction_jitter / 2
            )
            print(direction + direction_addon)
            stim = stim.at_azimuth(direction + direction_addon, ils)
            stim = stim.externalize()
            file.write(frequencies[i], tag=f"{time.time()-start_time:.3f}")
            play(stim)
            i += 1
        plt.pause(0.01)


if __name__ == "__main__":
    run("regular_major_1.csv", "MS")
