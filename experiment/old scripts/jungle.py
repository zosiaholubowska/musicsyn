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
sound_file_names = os.listdir('C://projects//musicsyn//stimuli//jungle')

for i in range(len(sound_file_names)):
    speaker = speakers[i]
    sound = slab.Sound(DIR/ "stimuli" / "jungle" / sound_file_names[i])
    sound.data = sound.data[:80000]
    sound.level = 80 if sound_file_names[i] != 'sirene.mp3' else 90
    freefield.write(tag=f"data{i+1}", value=sound.data.flatten(), processors=speaker.analog_proc)
    freefield.write(tag=f"chan{i+1}", value=speaker.analog_channel, processors=speaker.analog_proc)
freefield.play()


### GENERATING SOUNDS

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import sounddevice as sd

# Parameters
duration = 2  # seconds
sample_rate = 44100  # Hz
frequency = 440  # Hz

# Generate the time vector
t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

# Generate the waveform (sine wave)
waveform = np.sin(2 * np.pi * frequency * t)

# Play the sound
print("Playing sound...")
sd.play(waveform, sample_rate)
sd.wait()

# Plot the waveform
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t[:int(0.1 * sample_rate)], waveform[:int(0.1 * sample_rate)])
plt.title('Waveform (440 Hz, First 100 ms)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot the spectrogram
plt.subplot(2, 1, 2)
frequencies, times, Sxx = spectrogram(waveform, fs=sample_rate)
plt.pcolormesh(times, frequencies[frequencies <= 10000], 10 * np.log10(Sxx[frequencies <= 10000]))
plt.title('Spectrogram (up to 2000 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Intensity (dB)')

plt.tight_layout()
plt.show()


#### COMPLEX TONE

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import sounddevice as sd

# Parameters
duration = 2  # seconds
sample_rate = 44100  # Hz
frequency = 440  # Hz
num_harmonics = 10  # Number of harmonics to include in the complex tone

# Generate the time vector
t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

# Generate the waveform (complex tone)
waveform = np.zeros_like(t)
for harmonic in range(1, num_harmonics + 1):
    waveform += np.sin(2 * np.pi * harmonic * frequency * t) / harmonic

# Normalize the waveform to ensure amplitude doesn't exceed 1
waveform /= np.max(np.abs(waveform))

# Play the sound
print("Playing sound...")
sd.play(waveform, sample_rate)
sd.wait()

# Plot the waveform (only the first 100 ms)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t[:int(0.1 * sample_rate)], waveform[:int(0.1 * sample_rate)])
plt.title('Waveform (Complex Tone, First 100 ms)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot the spectrogram (up to 2000 Hz)
plt.subplot(2, 1, 2)
frequencies, times, Sxx = spectrogram(waveform, fs=sample_rate)
plt.pcolormesh(times, frequencies[frequencies <= 10000], 10 * np.log10(Sxx[frequencies <= 10000]))
plt.title('Spectrogram (up to 2000 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Intensity (dB)')

plt.tight_layout()
plt.show()
