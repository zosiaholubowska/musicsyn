import os
import freefield
import random
import pathlib
import slab

DIR = pathlib.Path(os.getcwd())
rcx_file_path = 'C:/projects/musicsyn/data/rcx/jungle.rcx'

path = 'C://projects//musicsyn'

proc_list = [['RX81', 'RX8', path + f'/data/rcx/jungle_2.rcx'],
                ['RX82', 'RX8', path + f'/data/rcx/jungle_2.rcx'],
                 ['RP2', 'RP2', path + f'/data/rcx/button.rcx']]

freefield.initialize('dome', device=proc_list)

speaker_idxs = random.sample(range(46), 8)
speakers = freefield.pick_speakers(speaker_idxs)
sound_file_names = os.listdir(DIR / "stimuli" / "jungle")

for i in range(len(sound_file_names)):
    speaker = speakers[i]
    sound = slab.Sound(DIR/ "stimuli" / "jungle" / sound_file_names[i])
    sound.data = sound.data[:]
    sound.level = 80 if sound_file_names[i] != 'sirene.mp3' else 90
    freefield.write(tag=f"data{i+1}", value=sound.data.flatten(), processors=speaker.analog_proc)
    freefield.write(tag=f"chan{i+1}", value=speaker.analog_channel, processors=speaker.analog_proc)
freefield.play()

