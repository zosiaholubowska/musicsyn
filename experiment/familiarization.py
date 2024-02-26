import time
import itertools
from numpy.random import default_rng
import pandas
from experiment.good_luck import good_luck
import random
import freefield
import sys

path = 'C://projects//musicsyn'
randgenerator = default_rng()

samplerate = 44828


def read_sample(file):
    """
    This function reads a csv file with the description of notes
    - score RCX_files (from csv file)
    - note onsets
    - note frequencies
    - note durations
    - if the note is at the phrase boundary
    """
    score_data = pandas.read_csv(file, sep=",")  # open the csv file with notes
    onsets = score_data.onset_sec.to_list()  # list of onsets of consecutive notes
    frequencies = score_data.freq.to_list()  # frequencies of consecutive notes
    durations = score_data.duration.to_list()  # note durations
    return onsets, frequencies, durations


def play_run(melody_file, subject):

    onsets, frequencies, durations, = read_sample(
        path + f"\stimuli\{melody_file}")  # reading the csv file with the information about the notes

    directions = [23, 23, 23]
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

    start_time = time.time()  # creates a timestamp in Unix format

    try:
        while time.time() - start_time < onsets[-1] + durations[-1]:

            if time.time() - start_time > onsets[i]:  # play the next note

                freefield.write('f0', frequencies[i], ['RX81', 'RX82'])

                duration = durations[i]  # duration in seconds
                freefield.write('len', int(duration * samplerate * 0.95), ['RX81', 'RX82'])

                freefield.write('chan', curr_speaker.analog_channel, curr_speaker.analog_proc)
                [other_proc] = [item for item in [proc_list[0][0], proc_list[1][0]] if item != curr_speaker.analog_proc]
                freefield.write('chan', 99, other_proc)

                freefield.play()

                i += 1

    except IndexError:
        good_luck()
    except KeyError:
        good_luck()


def select_file():
    # sound familiarisation
    fam = ['sample_major.csv', 'sample_minor.csv']
    random.shuffle(fam)

    i = 0

    user_input = input("Do you want to start the new task? (y/n): ")
    if user_input.lower() == 'n':
        sys.exit()
    elif user_input.lower() == 'y':
        print("Continuing...")

    for melody_file in fam:
        print(melody_file)
        play_run(melody_file, 'sub06')  ########### PARTICIPANT HERE ############
        print(f'That was melody {i + 1}.')
        user_input = input("Do you want to continue? (y/n): ")
        if user_input.lower() == 'n':
            break
        elif user_input.lower() == 'y':
            print("Continuing...")

            i += 1


if __name__ == "__main__":
    proc_list = [['RX81', 'RX8', path + f'/data/rcx/piano.rcx'],
                 ['RX82', 'RX8', path + f'/data/rcx/piano.rcx'],
                 ['RP2', 'RP2', path + f'/data/rcx/button.rcx']]
    freefield.initialize('dome', device=proc_list)
    # freefield.set_logger('debug')

    select_file()

    freefield.halt()
