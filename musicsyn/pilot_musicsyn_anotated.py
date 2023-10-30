import time
import subprocess
import itertools
import pickle
import numpy
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pandas
import slab
from musicsyn.balanced_sequence import balanced_sequence
from subject_data import subject_data
import random
import os

os.chdir('/Users/zofiaholubowska/Documents/PhD/3_experiment/musicsyn')
plt.ion()  # enable interactive mode - interactive mode will be on, figures will automatically be shown
randgenerator = default_rng()  # generator of random numbers
ils = pickle.load(open("ils.pickle", "rb"))  # load interaural level spectrum ???


def read_melody(file):
    """
    This function reads a csv file with the description of notes
    - score data (from csv file)
    - note onsets
    - note frequencies
    - note durations
    - if the note is at the phrase boundary
    """
    score_data = pandas.read_csv(file, sep=";")  # open the csv file with notes
    onsets = score_data.onset_sec.to_list()  # list of onsets of consecutive notes
    frequencies = score_data.freq.to_list()  # frequencies of consecutive notes
    durations = score_data.duration_sec.to_list()  # note durations
    changable_notes = score_data.changable_note.to_list()  # if at note is possible to change direction
    boundaries = (
        score_data.boundary.to_list()
    )  # 0 or 1 indication if the note is the beginning of a new phrase
    return onsets, frequencies, durations, boundaries, changable_notes


def expenv(n_samples):
    """
    It is to create more natural sound
    """
    t = numpy.linspace(
        start=0, stop=1, num=n_samples
    )  # returns evenly spaced numbers over a specified interval
    return slab.Signal(numpy.exp(-(0.69 * 5) * t))  # what these numbers specify?


def note(f0, duration):
    """
    This is to create actual sound
    """
    sig = slab.Sound.harmoniccomplex(f0, duration)
    env = expenv(sig.n_samples)
    sig = sig * env
    return slab.Binaural(sig.ramp())


def play(stim):
    """
    Plays a created note
    """
    stim.write("tmp.wav")
    subprocess.Popen(["afplay", "tmp.wav"])

def run(melody_file, subject):
    file = slab.ResultsFile(
        subject
    )  # here we name the results folder with subject name
    file_name = file.name

    onsets, frequencies, durations, boundaries, changable_notes = read_melody(f"/Users/zofiaholubowska/Documents/PhD/3_experiment/stimuli/{melody_file}")  # reading the csv file with the information about the notes
    seq = balanced_sequence(boundaries, changable_notes, subject, melody_file)
    # depending of number of boundaries, we are creating a sequence
    directions = itertools.cycle(
        [0, 20]
    )  # iterator, which will return the numbers from 0 to 20 in a infinite loop
    # but this iterator does only 0 and 20, shouldn't we have -20, as well?
    direction_jitter = 5
    start_time = time.time()  # creates a timestamp in Unix format
    # setup the figure for button capture
    fig = plt.figure("stairs")  # I cannot see the figure - for check later

    def on_key(event):
        print("write key: ", event.key)
        file.write(
            event.key, tag=f"{time.time() - start_time:.3f}"
        )  # logs the key that was pressed on a specified time

    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    onsets.append(
        onsets[-1] + durations[-1] + 0.1
    )  # add a dummy onset so that the if statement below works during the last note
    # durations.append(0.1)  ###
    i = 0
    direction = next(directions)
    try:
        while time.time() - start_time < onsets[-1] + durations[-1]:
            if time.time() - start_time > onsets[i]:  # play the next note
                #print(i)
                stim = note(frequencies[i], durations[i])
                if seq["sequence"][i] == 1:
                    direction = next(directions)  # toggle direction
                    print("direction change")
                if seq["boundary"][i]:  # so if there is 1 in the boundaries list
                    print("at boundary!")
                if seq["cue"][i] == 1:
                    print("########")
                    print("########")
                    print("visual cue!")
                    print("########")
                    print("########")
                direction_addon = randgenerator.uniform(low=-direction_jitter / 2, high=direction_jitter / 2)
                # it creates jitter for the change of the location ranging from -2,5 to 2,5
                #print(direction + direction_addon)
                stim = stim.at_azimuth(
                    direction + direction_addon, ils
                )  # this matches the azimuth with the ils values
                stim = (
                    stim.externalize()
                )  # smooths the sound with HRTF, to simulate external sound source
                file.write(frequencies[i], tag=f"{time.time() - start_time:.3f}")
                play(stim)
                i += 1
            plt.pause(0.01)
    except IndexError:
        subject_data(subject, file, melody_file)


def select_file():
    files = ["stim_maj_1.csv", "stim_maj_2.csv", "stim_maj_3.csv"]
    random.shuffle(files)

    for file in files:
        print(file)
        run(file, 'FH')

        user_input = input("Do you want to continue? (y/n): ")
        if user_input.lower() == 'n':
            break
        elif user_input.lower() == 'y':
            print("Continuing...")

if __name__ == "__main__":
    select_file()
