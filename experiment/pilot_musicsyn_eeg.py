import time
import itertools
from numpy.random import default_rng
import pandas
import slab
from balanced_sequence import balanced_sequence
from good_luck import good_luck
import random
import freefield
from read_data import read_data
from analysis_pilot import create_df, plot_group, plot_single


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
    changable_notes = score_data.changable_notes.to_list()  # if at note is possible to change direction
    boundaries = (score_data.boundary.to_list())  # 0 or 1 indication if the note is the beginning of a new phrase
    return onsets, frequencies, durations, boundaries, changable_notes


def run(melody_file, subject, p):
    file = slab.ResultsFile(
        subject
    )  # here we name the results folder with subject name
    file_name = file.name
    file.write(melody_file, tag=0)
    onsets, frequencies, durations, boundaries, changable_notes = read_melody(
        path + f"\stimuli\{melody_file}")
    print(boundaries)
    print(changable_notes)  # reading the csv file with the information about the notes
    seq = balanced_sequence(boundaries, changable_notes, subject, melody_file, p)

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


    try:
        while time.time() - start_time < onsets[-1] + durations[-1]:


            if time.time() - start_time > onsets[i]:  # play the next note
                trig_value = 1 if i == 0 else 0
                if seq["sequence"][i] == 1:
                    curr_speaker = next(speakers)  # toggle direction
                    file.write(curr_speaker.azimuth, tag=f"{time.time() - start_time:.3f}")
                    trig_value = 2
                    print(f"direction change")

                if seq["boundary"][i] and seq["sequence"][i]:  # so if there is 1 in the boundaries list
                    print(f"at boundary!")
                    trig_value = 3
                if seq["cue"][i] == 1:
                    led_on = time.time()
                    freefield.write(tag='bitmask', value=1, processors='RX81')  # illuminate LED
                    led = True
                    # print("########")
                    # print("########")
                    # print("visual cue!")
                    # print("########")
                    # print("########")

                file.write(frequencies[i], tag=f"{time.time() - start_time:.3f}")
                freefield.write('f0', frequencies[i], ['RX81', 'RX82'])

                duration = durations[i]  # duration in seconds
                freefield.write('len', int(duration * samplerate * 0.95), ['RX81', 'RX82'])

                freefield.write('chan', curr_speaker.analog_channel, curr_speaker.analog_proc)
                [other_proc] = [item for item in [proc_list[0][0], proc_list[1][0]] if item != curr_speaker.analog_proc]
                freefield.write('chan', 99, other_proc)

                freefield.write(tag='trigcode', value=trig_value, processors='RX82')

                freefield.play()
                # trigger_time_zBusA = time.time()

                # freefield.play(kind='zBusB', proc="RX82")
                # trigger_time_zBusB = time.time()
                # print("First tone triggered EEG")
                # trigger_time_diff = trigger_time_zBusA - trigger_time_zBusB
                # print("Trigger time diff:", trigger_time_diff)

                i += 1


    except IndexError:
        read_data(subject, file_name)
    except KeyError:
        read_data(subject, file_name)


def select_file():

    music = ["stim_maj_1_a.csv", "stim_maj_2_a.csv", "stim_maj_3_a.csv",
            "stim_min_1_a.csv", "stim_min_2_a.csv", "stim_min_3_a.csv",
            "stim_maj_1_b.csv", "stim_maj_2_b.csv", "stim_maj_3_b.csv",
            "stim_min_1_b.csv", "stim_min_2_b.csv", "stim_min_3_b.csv"
            ]
    random.shuffle(music)
    i = 0
    for m in music:
        melody_file = m


        user_input = input("Do you want to start the new task? (y/n): ")
        if user_input.lower() == 'n':
            break
        elif user_input.lower() == 'y':
            print("Continuing...")


        print(melody_file)
        p = 0.2
        print(p)
        run(melody_file, 'test', p)  ########### PARTICIPANT HERE ############
        print(f'That was melody {i + 1}.')

        i += 1

        create_df()


if __name__ == "__main__":

    proc_list = [['RX81', 'RX8', path + f'/data/rcx/piano_eeg.rcx'],
                 ['RX82', 'RX8', path + f'/data/rcx/piano_eeg.rcx'],
                 ['RP2', 'RP2', path + f'/data/rcx/button.rcx']]
    freefield.initialize('dome', device=proc_list)
    # freefield.set_logger('debug')

    select_file()

    freefield.halt()
