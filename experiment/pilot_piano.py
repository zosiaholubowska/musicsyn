import time
import numpy
import pandas
import slab
import os
import matplotlib.pyplot as plt

path = os.getcwd()
duration = 0.5
frequency = 440
def read_melody(file):
    score_data = pandas.read_csv(f"{path}/stimuli/{file}", sep=",")
    onsets = score_data.onset_sec.to_list()
    frequencies = score_data.freq.to_list()
    durations = score_data.duration.to_list()
    boundaries = score_data.boundary.to_list()
    return onsets, frequencies, durations, boundaries

def create_sound(frequency, duration):
    fs = 48828
    t = numpy.arange(0, duration, 1 / fs)  # time points

    # carrier
    fc = frequency
    # amplitude over time
    A = 1  # base amp
    # Alin = 10 ** (AdB / 20)  # convert to linear scale
    AdB = A * 75
    T = 10 * numpy.sqrt(AdB) / numpy.sqrt(fc)  # decay time for linear scaled amp
    At = numpy.exp(-t * 5 / T)  # decay envelope
    # At = numpy.e ** (-t)  # same thing
    #plt.plot(t, At)

    yc = A * numpy.sin(2 * numpy.pi * fc * t)  # carrier (tone)

    # modulating sines
    fm1 = fc
    fm2 = fc * 4
    S = fc / 200  # ymod spectrum is less clean but still sounds fine
    # modulation indices (amplitudes of the components)
    I1 = 17 * (8 - numpy.log(fc)) / (numpy.log(fc) ** 2)
    I2 = 20 * (8 - numpy.log(fc)) / fc
    ym1 = I1 * numpy.sin(2 * numpy.pi * (fm1 + S) * t)
    ym2 = I2 * numpy.sin(2 * numpy.pi * (fm2 + S) * t)
    ymod = A * numpy.sin(2 * numpy.pi * fc * t + ym1 + ym2)  # frequency modulated tone
    ymod = ymod * At  # add decay ramp

    tone = slab.Sound(ymod, samplerate=fs)


    return tone

def run(melody_file, subject):
    file = slab.ResultsFile(subject)
    onsets, frequencies, durations, boundaries = read_melody(melody_file)

    start_time = time.time()
    # setup the figure for button capture

    onsets.append(
        onsets[-1] + durations[-1] + 0.1
    )  # add a dummy onset so that the if statement below works during the last note
    i = 0

    while time.time() - start_time < onsets[-1] + durations[-1]:
        if time.time() - start_time > onsets[i]:  # play the next note
            tone = create_sound(frequencies[i], durations[i])
            tone.play()

            i += 1



if __name__ == "__main__":
    run("stim_maj_1.csv", "MS")
