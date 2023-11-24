import time
import subprocess
import itertools
import numpy
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pandas
import slab
from balanced_sequence import balanced_sequence
from good_luck import good_luck
import random
import freefield

path = 'C://projects//musicsyn'
randgenerator = default_rng()


samplerate = 44828

def read_melody(file):
    """
    This function reads a csv file with the description of notes
    - score data (from csv file)
    - note onsets
    - note frequencies
    - note durations
    - if the note is at the phrase boundary
    """
    score_data = pandas.read_csv(file, sep=",")  # open the csv file with notes
    onsets = score_data.onset_sec.to_list()  # list of onsets of consecutive notes
    frequencies = score_data.freq.to_list()  # frequencies of consecutive notes
    durations = score_data.duration.to_list()  # note durations
    changable_notes = score_data.changable_note.to_list()  # if at note is possible to change direction
    boundaries = (score_data.boundary.to_list() )  # 0 or 1 indication if the note is the beginning of a new phrase
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
    file.write(melody_file, tag=0)
    onsets, frequencies, durations, boundaries, changable_notes = read_melody(path + f"\stimuli\{melody_file}")  # reading the csv file with the information about the notes
    seq = balanced_sequence(boundaries, changable_notes, subject, melody_file)

    directions = [(-17.5, 0), 23, 31]
    [speaker1] = freefield.pick_speakers(directions[0])
    [speaker2] = freefield.pick_speakers(directions[1])
    [speaker3] = freefield.pick_speakers(directions[2])
    speakers = itertools.cycle([speaker1, speaker2, speaker3, speaker2])

    onsets.append(
        onsets[-1] + durations[-1] + 0.1
    )  # add a dummy onset so that the if statement below works during the last note
    # durations.append(0.1)  ###
    i = 0
    curr_speaker = next(speakers)
    file.write(curr_speaker.analog_channel, tag=0)
    start_time = time.time()  # creates a timestamp in Unix format
    led = False
    try:
        while time.time() - start_time < onsets[-1] + durations[-1]:
            if led:
                if time.time() - led_on > 1:
                    freefield.write(tag='bitmask', value=0, processors='RX81')  # turn off LED
            if time.time() - start_time > onsets[i]:  # play the next note

                if seq["sequence"][i] == 1:
                    curr_speaker = next(speakers)  # toggle direction
                    file.write(curr_speaker.analog_channel, tag=f"{time.time() - start_time:.3f}")
                    print(f"direction change")

                if seq["boundary"][i]:  # so if there is 1 in the boundaries list
                    print(f"at boundary!")
                if seq["cue"][i] == 1:
                    led_on = time.time()
                    freefield.write(tag='bitmask', value=1, processors='RX81')  # illuminate LED
                    led = True
                    print("########")
                    print("########")
                    print("visual cue!")
                    print("########")
                    print("########")

                file.write(frequencies[i], tag=f"{time.time() - start_time:.3f}")
                freefield.write('f0', frequencies[i], ['RX81', 'RX82'])


                duration = durations[i] # duration in seconds
                freefield.write('len', int(duration * samplerate * 0.95), ['RX81', 'RX82'])


                freefield.write('chan', curr_speaker.analog_channel, curr_speaker.analog_proc)
                [other_proc] = [item for item in [proc_list[0][0], proc_list[1][0]] if item != curr_speaker.analog_proc]
                freefield.write('chan', 99, other_proc)

                freefield.play()

                while True:
                    response = freefield.read('response', 'RP2', 0)
                    if response != 0:
                        file.write(
                            'p', tag=f"{time.time() - start_time:.3f}"
                        )  # logs the key that was pressed on a specified time
                    if time.time() - start_time > onsets[i]:
                        break

                i += 1
            plt.pause(0.01)
    except IndexError:
        good_luck()
    except KeyError:
        good_luck()



def select_file():
    files = ["stim_maj_1.csv", "stim_maj_2.csv", "stim_maj_3.csv",
             "stim_min_1.csv", "stim_min_2.csv", "stim_min_3.csv"]
    random.shuffle(files)
    i = 0
    for melody_file in files:
        print(melody_file)
        run(melody_file, 'p07a') ########### PARTICIPANT HERE ############
        print(f'That was melody {i+1}.')
        user_input = input("Do you want to continue? (y/n): ")
        if user_input.lower() == 'n':
            break
        elif user_input.lower() == 'y':
            print("Continuing...")

        i += 1

if __name__ == "__main__":

    proc_list = [['RX81', 'RX8',  path + f'/data/rcx/piano.rcx'],
                 ['RX82', 'RX8',  path + f'/data/rcx/piano.rcx'],
                 ['RP2', 'RP2',  path + f'/data/rcx/button.rcx']]
    freefield.initialize('dome', device=proc_list)
    freefield.set_logger('debug')



    select_file()

    freefield.halt()

