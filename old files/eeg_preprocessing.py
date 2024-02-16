import sys
sys.path.append('/Users/zofiaholubowska/Documents/PhD/3_experiment')
from EEG_Tools2.core.Pipeline import EEGPipeline
from EEG_Tools2.core.FileHandler import FileHandler
import mne
from EEG_Tools2.core.FileHandler import FileHandler
from matplotlib import pyplot as plt

# fh = FileHandler('/Users/zofiaholubowska/Documents/PhD/3_experiment/eeg_analysis')
root = '/Users/zofiaholubowska/Documents/PhD/3_experiment/eeg_analysis'
pl = EEGPipeline(root)
pl.subjects.pop(0)
pattern = "*raw*"
handle = FileHandler(root)
raws = handle.find(pattern)
for raw in raws:
    file = mne.io.read_raw_fif(raw)
pl.raw = file


pl.concatenate_brainvision('sub02', ref_to_add="FCz")
pl.save(pl.raw, "sub02")

pl.filtering(highpass=0.5, lowpass=35)
tmin = -0.2
tmax = 0.5
event_id = {"note": 1,
            "note/change": 2,
            "boundary/change": 3,
            "boundary": 4,
            "start": 5,
            }
pl.make_epochs(exclude_event_id=None, event_id=event_id)

pl.rereference()

pl.apply_ica(method="fastica")

pl.reject_epochs(mode='threshold', verbose=True)

pl.average_epochs()

#pl.reject_epochs(mode='autoreject', verbose=True)

note = pl.epochs["note/change"].average()
boundary = pl.epochs["boundary/change"].average()
fig, ax = plt.subplots(2)
note.plot(axes=ax[0], show=False)
boundary.plot(axes=ax[1], show=False)
plt.show()

for evk in (note, boundary):
    evk.plot(gfp=True, spatial_colors=True, ylim=dict(eeg=[-12, 12]))


tmin = -0.2
tmax = 0.5
fmin = 1.0
fmax = 90.0
sfreq = pl.epochs.info["sfreq"]

spectrum = pl.epochs.compute_psd(
    "welch",
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0,
    n_per_seg=None,
    tmin=tmin,
    tmax=tmax,
    fmin=fmin,
    fmax=fmax,
    window="boxcar",
    verbose=False,
)
psds, freqs = spectrum.get_data(return_freqs=True)


#-#-# MAX'S TUTORIAL #-#-#

import os
import pathlib
import mne
import matplotlib
from matplotlib import pyplot as plt
from autoreject import AutoReject, Ransac
from mne.preprocessing import ICA

cwd = os.getcwd()
DIR = pathlib.Path(os.getcwd())

data_DIR = DIR / "data" / "sub01"

header_files = [file for file in os.listdir(data_DIR) if ".vhdr" in file]
raw_files = []
for header_file in header_files:
    raw_files.append(mne.io.read_raw_brainvision(os.path.join(data_DIR, header_file), preload=True))  # read BrainVision files.
raw = mne.concatenate_raws(raw_files)  # make raw file
mapping = {"1": "Fp1", "2": "Fp2", "3": "F7", "4": "F3", "5": "Fz", "6": "F4",
           "7": "F8", "8": "FC5", "9": "FC1", "10": "FC2", "11": "FC6",
           "12": "T7", "13": "C3", "14": "Cz", "15": "C4", "16": "T8", "17": "TP9",
           "18": "CP5", "19": "CP1", "20": "CP2", "21": "CP6", "22": "TP10",
           "23": "P7", "24": "P3", "25": "Pz", "26": "P4", "27": "P8", "28": "PO9",
           "29": "O1", "30": "Oz", "31": "O2", "32": "PO10", "33": "AF7", "34": "AF3",
           "35": "AF4", "36": "AF8", "37": "F5", "38": "F1", "39": "F2", "40": "F6",
           "41": "FT9", "42": "FT7", "43": "FC3", "44": "FC4", "45": "FT8", "46": "FT10",
           "47": "C5", "48": "C1", "49": "C2", "50": "C6", "51": "TP7", "52": "CP3",
           "53": "CPz", "54": "CP4", "55": "TP8", "56": "P5", "57": "P1", "58": "P2",
           "59": "P6", "60": "PO7", "61": "PO3", "62": "POz", "63": "PO4", "64": "PO8"}
raw.rename_channels(mapping)  # Look at supplements below for mapping variable.
# Use BrainVision montage file to specify electrode positions.
montage_path = DIR / "settings" / "AS-96_REF.bvef"
montage = mne.channels.read_custom_montage(fname=montage_path)
raw.set_montage(montage)

raw.plot()

raw.plot_psd()

raw.filter(1, 40)

events = mne.events_from_annotations(raw)[0]

tmin = -0.5
tmax = 1.5
event_id = {"deviant": 1,
            "control": 2,
            "20": 3,
            "200": 4}

epochs = mne.Epochs(raw,
                    events=events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=None,
                    preload=True)

epochs.plot_psd()

del raw

epochs.plot_sensors(kind="topomap", ch_type='all')
reference = "average"
ransac = Ransac(n_jobs=-1)
epochs = ransac.fit_transform(epochs)
epochs.set_eeg_reference(ref_channels=reference)

ica = ICA()
ica.fit(epochs)
ica.plot_components(picks=range(10))
ica.plot_sources(epochs)
ica.apply(epochs, exclude=[1, 2])

ar = AutoReject(n_jobs=-1)
epochs_ar = ar.fit_transform(epochs)
# Visually inspect the data.
epochs_ar.plot_drop_log()
fig, ax = plt.subplots(2)
epochs.average().plot(axes=ax[0])
epochs_ar.average().plot(axes=ax[1])

evokeds = [epochs_ar[condition].average() for condition in event_id.keys()]
evokeds[2].plot_joint(times="auto")

#-#-# EXAMPLE 1 #-#-#

import mne
import os
import matplotlib.pyplot as plt
import seaborn
import pathlib
plt.style.use(['seaborn-v0_8-colorblind', 'seaborn-v0_8-darkgrid'])

cwd = os.getcwd()
DIR = pathlib.Path(os.getcwd())

data_path = mne.datasets.sample.data_path(DIR, verbose=True)

raw = mne.io.read_raw_fif(os.path.join(data_path, "MEG/sample/sample_audvis_raw.fif"), preload=True)
raw.info
raw.info["chs"]  # for example a list of all channels
raw.info["dig"]  # or coordinates of points on the surface of the subjects head

picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True)
raw.pick(picks=picks)

raw.plot()
events = mne.read_events(os.path.join(data_path, "MEG/sample/sample_audvis_raw-eve.fif"))

raw.filter(None, 40)


tmin = -0.2  # start of the epoch (relative to the stimulus)
tmax = 0.5  # end of the epoch
event_id = dict(vis_l=3, vis_r=2)  # the stimuli we are interested in
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, preload=True)

epochs.plot()

epochs.average().plot()

baseline = (-0.2, 0)
epochs.apply_baseline(baseline)
epochs.average().plot()

#-#-# EXAMPLE 2 #-#-#

import mne
import os
import matplotlib.pyplot as plt
import seaborn
import pathlib
plt.style.use(['seaborn-v0_8-colorblind', 'seaborn-v0_8-darkgrid'])

cwd = os.getcwd()
DIR = pathlib.Path(os.getcwd())
raw = mne.io.read_raw_fif(os.path.join(DIR, "MNE-sample-data/MEG/sample/sample_audvis_raw.fif"), preload=True)
events = mne.read_events(os.path.join(DIR, "MNE-sample-data/MEG/sample/sample_audvis_raw-eve.fif"))

picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True)
raw.pick(picks=picks)

# filter
raw.filter(None, 40)

# segmentation into epochs
tmin = -0.7  # start of the epoch (relative to the stimulus)
tmax = 0.7  # end of the epoch
event_id = dict(vis_l=3, vis_r=2)  # the stimuli we are interested in
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, preload=True)  # get epochs

# baseline correction
baseline = (-0.2, 0)
epochs.apply_baseline(baseline)

raw.plot()

epochs.average().plot()

epochs.plot()

from autoreject import AutoReject  # import the module
ar = AutoReject(n_interpolate=[3, 6, 12], random_state=42, n_jobs=-1)
epochs_ar, reject_log = ar.fit_transform(epochs, return_log=True)

reject_log.plot_epochs(epochs)

ica = mne.preprocessing.ICA(n_components=0.99, method="fastica")
ica.fit(epochs)

ica.plot_components()

ica_sources = ica.get_sources(epochs)
ica_sources.plot(picks="all")

epochs_ica = ica.apply(epochs, exclude=[1,2])

epochs_ica.plot()

epochs = epochs_ica


#-#-# EXAMPLE 3 #-#-#


import mne
import os
import matplotlib.pyplot as plt
import seaborn
import pathlib
plt.style.use(['seaborn-v0_8-colorblind', 'seaborn-v0_8-darkgrid'])
import numpy as np

cwd = os.getcwd()
DIR = pathlib.Path(os.getcwd())
raw = mne.io.read_raw_fif(os.path.join(DIR, "MNE-sample-data/MEG/sample/sample_audvis_raw.fif"), preload=True)
events = mne.read_events(os.path.join(DIR, "MNE-sample-data/MEG/sample/sample_audvis_raw-eve.fif"))

raw.filter(1, 40)
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False)

thresholds = np.linspace(200, 50, 50)
snr = np.zeros(len(thresholds))
for i, thresh in enumerate(thresholds):
    epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, baseline=None,
                        reject=dict(eeg=thresh*1e-6), proj=False, picks=picks,
                        preload=True)

    signal = epochs.copy().crop(0.0, 0.3).average()
    noise = epochs.copy().crop(None, 0.0).average()
    signal_rms = np.sqrt(np.mean(signal._data**2))
    noise_rms = np.sqrt(np.mean(noise._data**2))
    snr[i] = signal_rms/noise_rms
plt.plot(thresholds, snr)
plt.xlim(200, 50)
plt.xlabel("treshold in microvolts")
plt.ylabel("signal to noise ratio")
plt.show()

thresholds[snr == max(snr)][0]
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, baseline=None,
                    reject=dict(eeg=thresholds[snr == max(snr)][0] * 1e-6),
                    proj=False, picks=picks, preload=True)
epochs.average().plot()

epochs.plot_sensors(show_names=True)

# Let's compare a few different ones:
fig, axes = plt.subplots(2, 2)
epochs.plot_sensors(show_names=["EEG 012", "EEG 017", "EEG 024"],
                    axes=axes[0, 0], show=False)
references = [["EEG 012"], ["EEG 017", "EEG 024"], "average"]
for ref, ax in zip(references, [axes[0, 1], axes[1, 0], axes[1, 1]]):
    epochs.set_eeg_reference(ref)
    epochs.average().plot(axes=ax, show=False)
    ax.set_title("reference electrode: %s" % (ref))

evoked_left = epochs["1"].average()
evoked_right = epochs["2"].average()
fig, ax = plt.subplots(2)
evoked_left.plot(axes=ax[0], show=False)
evoked_right.plot(axes=ax[1], show=False)
plt.show()

lh_channels = ["EEG 001", "EEG 004", "EEG 005", "EEG 008", "EEG 009",
               "EEG 010", "EEG 011", "EEG 017", "EEG 018", "EEG 019",
               "EEG 020", "EEG 025", "EEG 026", "EEG 027", "EEG 028",
               "EEG 029", "EEG 036", "EEG 037", "EEG 038", "EEG 039",
               "EEG 044", "EEG 045", "EEG 046", "EEG 047",
               "EEG 054", "EEG 057"]

rh_channels = ["EEG 003", "EEG 006", "EEG 007", "EEG 013", "EEG 014",
               "EEG 015", "EEG 016", "EEG 021", "EEG 022", "EEG 023",
               "EEG 024", "EEG 031", "EEG 032", "EEG 033", "EEG 034",
               "EEG 035", "EEG 040", "EEG 041", "EEG 042", "EEG 043",
               "EEG 049", "EEG 050", "EEG 051", "EEG 052",
               "EEG 055", "EEG 056"]

evoked_left_lh = evoked_left.copy().pick_channels(lh_channels).crop(0.0, 0.3)
evoked_left_rh = evoked_left.copy().pick_channels(rh_channels).crop(0.0, 0.3)
evoked_right_lh = evoked_right.copy().pick_channels(lh_channels).crop(0.0, 0.3)
evoked_right_rh = evoked_right.copy().pick_channels(rh_channels).crop(0.0, 0.3)


rms = [np.sqrt(np.mean(evoked_left_lh.data**2)),
       np.sqrt(np.mean(evoked_right_lh.data**2)),
       np.sqrt(np.mean(evoked_left_rh.data**2)),
       np.sqrt(np.mean(evoked_right_rh.data**2))]

x = ["left_lh", "right_lh", "left_rh", "right_rh"]
plt.bar(x, rms)
plt.show()

#-#-# ERP ANALYSIS #-#-#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mne

root = mne.datasets.sample.data_path() / "MEG" / "sample"
raw_file = root / "sample_audvis_filt-0-40_raw.fif"
raw = mne.io.read_raw_fif(raw_file, preload=False)

events_file = root / "sample_audvis_filt-0-40_raw-eve.fif"
events = mne.read_events(events_file)

raw.crop(tmax=90)
events = events[events[:, 0] <= raw.last_samp]

raw.pick(["eeg", "eog"]).load_data()
raw.info

channel_renaming_dict = {name: name.replace(" 0", "").lower() for name in raw.ch_names}
_ = raw.rename_channels(channel_renaming_dict)

raw.plot_sensors(show_names=True)
fig = raw.plot_sensors("3d")

for proj in (False, True):
    with mne.viz.use_browser_backend("matplotlib"):
        fig = raw.plot(
            n_channels=5, proj=proj, scalings=dict(eeg=50e-6), show_scrollbars=False
        )
    fig.subplots_adjust(top=0.9)  # make room for title
    ref = "Average" if proj else "No"
    fig.suptitle(f"{ref} reference", size="xx-large", weight="bold")


raw.filter(l_freq=0.1, h_freq=None)

np.unique(events[:, -1])
event_dict = {
    "auditory/left": 1,
    "auditory/right": 2,
    "visual/left": 3,
    "visual/right": 4,
    "face": 5,
    "buttonpress": 32,
}

epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.3, tmax=0.7, preload=True)
fig = epochs.plot(events=events)

reject_criteria = dict(eeg=100e-6, eog=200e-6)  # 100 µV, 200 µV
epochs.drop_bad(reject=reject_criteria)

epochs.plot_drop_log()

l_aud = epochs["auditory/left"].average()
l_vis = epochs["visual/left"].average()

fig1 = l_aud.plot()
fig2 = l_vis.plot(spatial_colors=True)

l_aud.plot_topomap(times=[-0.2, 0.1, 0.4], average=0.05)

l_aud.plot_joint()

for evk in (l_aud, l_vis):
    evk.plot(gfp=True, spatial_colors=True, ylim=dict(eeg=[-12, 12]))

gfp = l_aud.data.std(axis=0, ddof=0)

# Reproducing the MNE-Python plot style seen above
fig, ax = plt.subplots()
ax.plot(l_aud.times, gfp * 1e6, color="lime")
ax.fill_between(l_aud.times, gfp * 1e6, color="lime", alpha=0.2)
ax.set(xlabel="Time (s)", ylabel="GFP (µV)", title="EEG")

left = ["eeg17", "eeg18", "eeg25", "eeg26"]
right = ["eeg23", "eeg24", "eeg34", "eeg35"]

left_ix = mne.pick_channels(l_aud.info["ch_names"], include=left)
right_ix = mne.pick_channels(l_aud.info["ch_names"], include=right)

roi_dict = dict(left_ROI=left_ix, right_ROI=right_ix)
roi_evoked = mne.channels.combine_channels(l_aud, roi_dict, method="mean")
print(roi_evoked.info["ch_names"])
roi_evoked.plot()

evokeds = dict(auditory=l_aud, visual=l_vis)
picks = [f"eeg{n}" for n in range(10, 15)]
mne.viz.plot_compare_evokeds(evokeds, picks=picks, combine="mean")

evokeds = dict(
    auditory=list(epochs["auditory/left"].iter_evoked()),
    visual=list(epochs["visual/left"].iter_evoked()),
)
mne.viz.plot_compare_evokeds(evokeds, combine="mean", picks=picks)

grand_average = mne.grand_average([l_aud, l_vis])
print(grand_average)

# Define a function to print out the channel (ch) containing the
# peak latency (lat; in msec) and amplitude (amp, in µV), with the
# time range (tmin and tmax) that was searched.
# This function will be used throughout the remainder of the tutorial.
def print_peak_measures(ch, tmin, tmax, lat, amp):
    print(f"Channel: {ch}")
    print(f"Time Window: {tmin * 1e3:.3f} - {tmax * 1e3:.3f} ms")
    print(f"Peak Latency: {lat * 1e3:.3f} ms")
    print(f"Peak Amplitude: {amp * 1e6:.3f} µV")


# Get peak amplitude and latency from a good time window that contains the peak
good_tmin, good_tmax = 0.08, 0.12
ch, lat, amp = l_vis.get_peak(
    ch_type="eeg", tmin=good_tmin, tmax=good_tmax, mode="pos", return_amplitude=True
)

# Print output from the good time window that contains the peak
print("** PEAK MEASURES FROM A GOOD TIME WINDOW **")
print_peak_measures(ch, good_tmin, good_tmax, lat, amp)

#-#-# TIME - FREQUENCY ANALYSIS #-#-#

import numpy as np

import mne

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False).crop(tmax=60)

raw.compute_psd(method="multitaper", tmin=10, tmax=20, fmin=5, fmax=30, picks="eeg")
raw.plot_psd(method="multitaper", tmin=10, tmax=20, fmin=5, fmax=30, picks="eeg")

with mne.use_log_level("WARNING"):  # hide some irrelevant info messages
    events = mne.find_events(raw, stim_channel="STI 014")
    event_dict = {
        "auditory/left": 1,
        "auditory/right": 2,
        "visual/left": 3,
        "visual/right": 4,
    }
    epochs = mne.Epochs(
        raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict, preload=True
    )
epo_spectrum = epochs.compute_psd()
psds, freqs = epo_spectrum.get_data(return_freqs=True)
print(f"\nPSDs shape: {psds.shape}, freqs shape: {freqs.shape}")
epo_spectrum

evoked = epochs["auditory"].average()
evk_spectrum = evoked.compute_psd()
# the first 3 frequency bins for the first 4 channels:
print(evk_spectrum[:4, :3])
epochs['auditory'].average().plot_psd()

# get both "visual/left" and "visual/right" epochs:
epo_spectrum["visual"]


import matplotlib.pyplot as plt
import numpy as np

import mne
from mne.datasets import somato
from mne.time_frequency import tfr_morlet

data_path = somato.data_path()
subject = "01"
task = "somato"
raw_fname = data_path / f"sub-{subject}" / "meg" / f"sub-{subject}_task-{task}_meg.fif"

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
# crop and resample just to reduce computation time
raw.crop(120, 360).load_data().resample(200)
events = mne.find_events(raw, stim_channel="STI 014")

# picks MEG gradiometers
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)

# Construct Epochs
event_id, tmin, tmax = 1, -1.0, 3.0
baseline = (None, 0)
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    picks=picks,
    baseline=baseline,
    reject=None,
    preload=True,
)

epochs.compute_psd(fmin=2.0, fmax=40.0).plot(average=True, picks="data", exclude="bads")

epochs.compute_psd().plot_topomap(ch_type="eeg", normalize=True, contours=0)

_, ax = plt.subplots()
spectrum = epochs.compute_psd(fmin=2.0, fmax=40.0, tmax=3.0, n_jobs=None)
# average across epochs first
mean_spectrum = spectrum.average()
psds, freqs = mean_spectrum.get_data(return_freqs=True)
# then convert to dB and take mean & standard deviation across channels
psds = 10 * np.log10(psds)
psds_mean = psds.mean(axis=0)
psds_std = psds.std(axis=0)

ax.plot(freqs, psds_mean, color="k")
ax.fill_between(
    freqs,
    psds_mean - psds_std,
    psds_mean + psds_std,
    color="k",
    alpha=0.5,
    edgecolor="none",
)
ax.set(
    title="Multitaper PSD (gradiometers)",
    xlabel="Frequency (Hz)",
    ylabel="Power Spectral Density (dB)",
)

### FREQUENCY TAGGING ###

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

import mne

# Load raw data
data_path = mne.datasets.ssvep.data_path()
bids_fname = (
    data_path / "sub-02" / "ses-01" / "eeg" / "sub-02_ses-01_task-ssvep_eeg.vhdr"
)

raw = mne.io.read_raw_brainvision(bids_fname, preload=True, verbose=False)
raw.info["line_freq"] = 50.0

# Set montage
montage = mne.channels.make_standard_montage("easycap-M1")
raw.set_montage(montage, verbose=False)

# Set common average reference
raw.set_eeg_reference("average", projection=False, verbose=False)

# Apply bandpass filter
raw.filter(l_freq=0.1, h_freq=None, fir_design="firwin", verbose=False)

# Construct epochs
event_id = {"12hz": 255, "15hz": 155}
events, _ = mne.events_from_annotations(raw, verbose=False)
tmin, tmax = -1.0, 20.0  # in s
baseline = None
epochs = mne.Epochs(
    raw,
    events=events,
    event_id=[event_id["12hz"], event_id["15hz"]],
    tmin=tmin,
    tmax=tmax,
    baseline=baseline,
    verbose=False,
)

tmin = 1.0
tmax = 20.0
fmin = 1.0
fmax = 90.0
sfreq = epochs.info["sfreq"]

spectrum = epochs.compute_psd(
    "welch",
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0,
    n_per_seg=None,
    tmin=tmin,
    tmax=tmax,
    fmin=fmin,
    fmax=fmax,
    window="boxcar",
    verbose=False,
)
psds, freqs = spectrum.get_data(return_freqs=True)

def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise

snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1)

fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))
freq_range = range(
    np.where(np.floor(freqs) == 1.0)[0][0], np.where(np.ceil(freqs) == fmax - 1)[0][0]
)

psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
psds_std = psds_plot.std(axis=(0, 1))[freq_range]
axes[0].plot(freqs[freq_range], psds_mean, color="b")
axes[0].fill_between(
    freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.2
)
axes[0].set(title="PSD spectrum", ylabel="Power Spectral Density [dB]")

# SNR spectrum
snr_mean = snrs.mean(axis=(0, 1))[freq_range]
snr_std = snrs.std(axis=(0, 1))[freq_range]

axes[1].plot(freqs[freq_range], snr_mean, color="r")
axes[1].fill_between(
    freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std, color="r", alpha=0.2
)
axes[1].set(
    title="SNR spectrum",
    xlabel="Frequency [Hz]",
    ylabel="SNR",
    ylim=[-2, 30],
    xlim=[fmin, fmax],
)
fig.show()

# define stimulation frequency
stim_freq = 12.0

# find index of frequency bin closest to stimulation frequency
i_bin_12hz = np.argmin(abs(freqs - stim_freq))
# could be updated to support multiple frequencies

# for later, we will already find the 15 Hz bin and the 1st and 2nd harmonic
# for both.
i_bin_24hz = np.argmin(abs(freqs - 24))
i_bin_36hz = np.argmin(abs(freqs - 36))
i_bin_15hz = np.argmin(abs(freqs - 15))
i_bin_30hz = np.argmin(abs(freqs - 30))
i_bin_45hz = np.argmin(abs(freqs - 45))

i_trial_12hz = np.where(epochs.events[:, 2] == event_id["12hz"])[0]
i_trial_15hz = np.where(epochs.events[:, 2] == event_id["15hz"])[0]

# Define different ROIs
roi_vis = [
    "POz",
    "Oz",
    "O1",
    "O2",
    "PO3",
    "PO4",
    "PO7",
    "PO8",
    "PO9",
    "PO10",
    "O9",
    "O10",
]  # visual roi

# Find corresponding indices using mne.pick_types()
picks_roi_vis = mne.pick_types(
    epochs.info, eeg=True, stim=False, exclude="bads", selection=roi_vis
)

snrs_target = snrs[i_trial_12hz, :, i_bin_12hz][:, picks_roi_vis]
print("sub 2, 12 Hz trials, SNR at 12 Hz")
print(f"average SNR (occipital ROI): {snrs_target.mean()}")

# get average SNR at 12 Hz for ALL channels
snrs_12hz = snrs[i_trial_12hz, :, i_bin_12hz]
snrs_12hz_chaverage = snrs_12hz.mean(axis=0)

# plot SNR topography
fig, ax = plt.subplots(1)
mne.viz.plot_topomap(snrs_12hz_chaverage, epochs.info, vlim=(1, None), axes=ax)

print("sub 2, 12 Hz trials, SNR at 12 Hz")
print("average SNR (all channels): %f" % snrs_12hz_chaverage.mean())
print("average SNR (occipital ROI): %f" % snrs_target.mean())

tstat_roi_vs_scalp = ttest_rel(snrs_target.mean(axis=1), snrs_12hz.mean(axis=1))
print(
    "12 Hz SNR in occipital ROI is significantly larger than 12 Hz SNR over "
    "all channels: t = %.3f, p = %f" % tstat_roi_vs_scalp
)

snrs_roi = snrs[:, picks_roi_vis, :].mean(axis=1)

freq_plot = [12, 15, 24, 30, 36, 45]
color_plot = ["darkblue", "darkgreen", "mediumblue", "green", "blue", "seagreen"]
xpos_plot = [-5.0 / 12, -3.0 / 12, -1.0 / 12, 1.0 / 12, 3.0 / 12, 5.0 / 12]
fig, ax = plt.subplots()
labels = ["12 Hz trials", "15 Hz trials"]
x = np.arange(len(labels))  # the label locations
width = 0.6  # the width of the bars
res = dict()

# loop to plot SNRs at stimulation frequencies and harmonics
for i, f in enumerate(freq_plot):
    # extract snrs
    stim_12hz_tmp = snrs_roi[i_trial_12hz, np.argmin(abs(freqs - f))]
    stim_15hz_tmp = snrs_roi[i_trial_15hz, np.argmin(abs(freqs - f))]
    SNR_tmp = [stim_12hz_tmp.mean(), stim_15hz_tmp.mean()]
    # plot (with std)
    ax.bar(
        x + width * xpos_plot[i],
        SNR_tmp,
        width / len(freq_plot),
        yerr=np.std(SNR_tmp),
        label="%i Hz SNR" % f,
        color=color_plot[i],
    )
    # store results for statistical comparison
    res["stim_12hz_snrs_%ihz" % f] = stim_12hz_tmp
    res["stim_15hz_snrs_%ihz" % f] = stim_15hz_tmp

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("SNR")
ax.set_title("Average SNR at target frequencies")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(["%i Hz" % f for f in freq_plot], title="SNR at:")
ax.set_ylim([0, 70])
ax.axhline(1, ls="--", c="r")
fig.show()