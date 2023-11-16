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
from musicsyn.subject_data import subject_data
import random
import os
import freefield
import win32com.client

path = 'C://projects//musicsyn'
# os.chdir('C:\\projects\\musicsyn')
plt.ion()  # enable interactive mode - interactive mode will be on, figures will automatically be shown
randgenerator = default_rng()  # generator of random numbers
ils = pickle.load(open(path + "/musicsyn/ils.pickle", "rb"))  # load interaural level spectrum ???

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

    onsets, frequencies, durations, boundaries, changable_notes = read_melody(path + f"\stimuli\{melody_file}")  # reading the csv file with the information about the notes
    seq = balanced_sequence(boundaries, changable_notes, subject, melody_file)
    # depending of number of boundaries, we are creating a sequence
    # directions = itertools.cycle(
    #     [0, 20]
    # )
    directions = [(-17.5, 0), 23, 31]
    [speaker1] = freefield.pick_speakers(directions[0])
    [speaker2] = freefield.pick_speakers(directions[1])
    [speaker3] = freefield.pick_speakers(directions[2])
    speakers = itertools.cycle([speaker1, speaker2, speaker3])
    # directions = itertools.cycle(directions) # ([0, 17.5, -17.5])

    # iterator, which will return the numbers from 0 to 20 in a infinite loop
    # but this iterator does only 0 and 20, shouldn't we have -20, as well?
    # direction_jitter = 5
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
    curr_speaker = next(speakers)
    start_time = time.time()  # creates a timestamp in Unix format
    led = False
    try:
        while time.time() - start_time < onsets[-1] + durations[-1]:
            if led:
                if time.time() - led_on > 1:
                    freefield.write(tag='bitmask', value=0, processors='RX81')  # turn off LED
            if time.time() - start_time > onsets[i]:  # play the next note
                #print(i)
                # stim = note(frequencies[i], durations[i])
                if seq["sequence"][i] == 1:
                    curr_speaker = next(speakers)  # toggle direction
                    print("direction change")
                if seq["boundary"][i]:  # so if there is 1 in the boundaries list
                    print("at boundary!")
                if seq["cue"][i] == 1:
                    led_on = time.time()
                    freefield.write(tag='bitmask', value=1, processors='RX81')  # illuminate LED
                    led = True
                    print("########")
                    print("########")
                    print("visual cue!")
                    print("########")
                    print("########")
                # direction_addon = randgenerator.uniform(low=-direction_jitter / 2, high=direction_jitter / 2)
                # it creates jitter for the change of the location ranging from -2,5 to 2,5
                #print(direction + direction_addon)
                # stim = stim.at_azimuth(
                #     direction + direction_addon, ils
                # )  # this matches the azimuth with the ils values
                # stim = (
                #     stim.externalize()
                # )  # smooths the sound with HRTF, to simulate external sound source
                file.write(frequencies[i], tag=f"{time.time() - start_time:.3f}")
                freefield.write('f0', frequencies[i], ['RX81', 'RX82'])

                # rp.SetTagVal('f0', frequencies[i])  # write value to tag
                duration = durations[i] # duration in seconds
                freefield.write('len', int(duration * samplerate * 0.85), ['RX81', 'RX82'])
                # rp.SetTagVal('len', int(duration * samplerate * 0.85))

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
                # play(stim)
                i += 1
            plt.pause(0.01)
    except IndexError:
        subject_data(subject, file, melody_file)


def select_file():
    files = ["stim_maj_1.csv", "stim_maj_2.csv", "stim_maj_3.csv"]
    random.shuffle(files)

    for melody_file in files:
        print(melody_file)
        run(melody_file, 'FH')

        user_input = input("Do you want to continue? (y/n): ")
        if user_input.lower() == 'n':
            break
        elif user_input.lower() == 'y':
            print("Continuing...")

if __name__ == "__main__":
    # freefield.initialize('dome', default='play_rec')
    proc_list = [['RX81', 'RX8',  path + f'/data/rcx/piano.rcx'],
                 ['RX82', 'RX8',  path + f'/data/rcx/piano.rcx'],
                 ['RP2', 'RP2',  path + f'/data/rcx/button.rcx']]
    freefield.initialize('dome', device=proc_list)
    freefield.set_logger('debug')

    # [led_speaker] = freefield.pick_speakers(23)  # get object for center speaker LED
    # freefield.write(tag='bitmask', value=led_speaker.digital_channel, processors=led_speaker.digital_proc)  # illuminate LED

    # rp = win32com.client.Dispatch('RPco.X')
    # zb = win32com.client.Dispatch('ZBUS.x')
    # rp.ConnectRX8('GB', 1)
    # rp.ClearCOF()
    # # rp.LoadCOF(path + f'/data/rcx/proto_.rcx')
    # rp.LoadCOF(path + f'/data/rcx/proto_.rcx')
    # rp.Run()

    select_file()

    freefield.halt()
    # rp.Halt()
