import sys
import time
import itertools
from numpy.random import default_rng
import pandas
import slab
from balanced_sequence import balanced_sequence
import random
import freefield
from read_data import read_data
from analysis_pilot import create_df, plot_group, plot_single
from no_context_music import shuffle_melody, shuffle_rhythm


path = 'C://projects//musicsyn'
randgenerator = default_rng()

samplerate = 44828


def read_melody(file):
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
    changable_notes = score_data.changable_notes.to_list()  # if at note is possible to change direction
    boundaries = (score_data.boundary.to_list())  # 0 or 1 indication if the note is the beginning of a new phrase
    return onsets, frequencies, durations, boundaries, changable_notes


def run(melody_file, subject, condition):
    file = slab.ResultsFile(
        subject
    )  # here we name the results folder with subject name
    file_name = file.name
    file.write(melody_file, tag=0)
    file.write(condition, tag=1)
    onsets, frequencies, durations, boundaries, changable_notes = read_melody(
        path + f"\stimuli\{melody_file}")

    seq = balanced_sequence(boundaries, changable_notes, subject, melody_file, 0.2, condition)

    # create control conditions
    if condition == 'melody':
        frequencies = shuffle_melody(frequencies, boundaries)
    elif condition == 'rhythm':
        durations, onsets = shuffle_rhythm(onsets, durations, boundaries)


    directions = [15, 23, 31]
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
    file.write(curr_speaker.azimuth, tag=0)
    start_time = time.time()  # creates a timestamp in Unix format
    prev_response = 0

    led = False
    try:
        while time.time() - start_time < onsets[-1] + durations[-1]:
            if led:
                if time.time() - led_on > 1:
                    freefield.write(tag='bitmask', value=0, processors='RX81')  # turn off LED

            # button

            response = freefield.read('response', 'RP2', 0)

            if response > prev_response:
                print('good')
                file.write('p', tag=f'{time.time() - start_time:.3f}')

            prev_response = response

            if time.time() - start_time > onsets[i]:  # play the next note

                if seq["sequence"][i] == 1:
                    curr_speaker = next(speakers)  # toggle direction
                    file.write(curr_speaker.azimuth, tag=f"{time.time() - start_time:.3f}")

                    print(f"direction change")

                if seq["boundary"][i] and seq["sequence"][i]:  # so if there is 1 in the boundaries list
                    print(f"at boundary!")

                if seq["cue"][i] == 1:
                    led_on = time.time()
                    freefield.write(tag='bitmask', value=speaker2.digital_channel, processors='RX81')  # illuminate LED
                    led = True
                    print("########")
                    print("########")
                    print("visual cue!")
                    print("########")
                    print("########")

                file.write(frequencies[i], tag=f"{time.time() - start_time:.3f}")
                freefield.write('f0', frequencies[i], ['RX81', 'RX82'])

                duration = durations[i]  # duration in seconds
                freefield.write('len', int(duration * samplerate * 0.95), ['RX81', 'RX82'])

                freefield.write('chan', curr_speaker.analog_channel, curr_speaker.analog_proc)
                [other_proc] = [item for item in [proc_list[0][0], proc_list[1][0]] if item != curr_speaker.analog_proc]
                freefield.write('chan', 99, other_proc)

                freefield.play()

                i += 1


    except IndexError:
        read_data(subject, file_name, condition)
    except KeyError:
        read_data(subject, file_name, condition)


def select_file():

    conditions = ['main', 'rhythm', 'melody']

    files = ["stim_maj_1.csv",
            "stim_min_1.csv"]

    random.shuffle(files)

    i = 0

    user_input = input("Do you want to start the new task? (y/n): ")
    if user_input.lower() == 'n':
        sys.exit()
    elif user_input.lower() == 'y':
        print("Continuing...")

    for condition in conditions:
        print(condition)

        for melody_file in files:
            print(melody_file)

            run(melody_file, 'p_Aaron', condition)  ########### PARTICIPANT HERE ############
            print(f'That was melody {i + 1}.')
            user_input = input("Do you want to continue? (y/n): ")
            if user_input.lower() == 'n':
                break
            elif user_input.lower() == 'y':
                print("Continuing...")

                i += 1

    create_df()


if __name__ == "__main__":
    proc_list = [['RX81', 'RX8', path + f'/data/rcx/piano.rcx'],
                ['RX82', 'RX8', path + f'/data/rcx/piano.rcx'],
                 ['RP2', 'RP2', path + f'/data/rcx/button.rcx']]

    freefield.initialize('dome', device=proc_list)
    # freefield.set_logger('debug')

    select_file()

    freefield.halt()