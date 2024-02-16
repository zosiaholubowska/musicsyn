import mne
import pathlib
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from autoreject import AutoReject, Ransac, get_rejection_threshold
from mne.preprocessing import ICA
from collections import Counter

### LOAD THE DATA

# set directory
cwd = os.getcwd()
DIR = pathlib.Path(os.getcwd())

# specify participant's folder
data_DIR = DIR / "data" / "sub02"

# find all .vhdr files in participant's folder
header_files = [file for file in os.listdir(data_DIR) if ".vhdr" in file]
raw_files = []
for header_file in header_files:
    raw_files.append(mne.io.read_raw_brainvision(os.path.join(data_DIR, header_file), preload=True))  # read BrainVision files.

# append all files from a participant
raw = mne.concatenate_raws(raw_files)
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

### here the raw file is ready

# EXCLUDE BAD CHANNELS

raw.plot()
good_raw = mne.pick_types(raw.info, eeg=True)
# apply high-pass and low-pass filter
raw.filter(0.1, 40)

# epoch the data
events = mne.events_from_annotations(raw)[0]  # get events
# sanity check: number of different types of epochs
Counter(events[:, 2])

# making epochs
tmin = -0.2
tmax = 0.5
event_id = {"note": 1,
            "note/change": 2,
            "boundary/change": 3,
            "boundary": 4,
            "start": 5,
            }
epochs = mne.Epochs(raw,
                    events=events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=None,
                    preload=True)
del raw

# re-referencing the data

epochs.plot_sensors(kind="topomap", ch_type='all')
reference = "average"
ransac = Ransac(n_jobs=-1)
epochs = ransac.fit_transform(epochs)
epochs.set_eeg_reference(ref_channels=reference)

# ICA

ica = ICA()
ica.fit(epochs)
ica.plot_components(picks=range(10))
ica.plot_sources(epochs)
ica.apply(epochs, exclude=[1]) # here you specify the number of ICA components you want to reject

# reject criteria
reject_criteria = dict(eeg=100e-6)       # 100 µV
#flat_criteria = dict(eeg=1e-6)           # 1 µV
# Note that these values are very liberal here.

epochs_auto = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5,
                    reject=reject_criteria, #flat=flat_criteria,
                    reject_by_annotation=False, preload=True) # this is the same command for extracting epochs as used above
epochs_auto.plot_drop_log() # summary of rejected epochs per channel


# AutoReject algorithm

ar = AutoReject(n_jobs=-1)
epochs_ar = ar.fit_transform(epochs)
reject = get_rejection_threshold(epochs)
# Visually inspect the data.
epochs_ar.plot_drop_log()
fig, ax = plt.subplots(2)
epochs.average().plot(axes=ax[0])
epochs_ar.average().plot(axes=ax[1])


