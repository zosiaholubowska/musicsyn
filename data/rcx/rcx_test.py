import win32com.client
from pathlib import Path
path = Path.cwd()
samplerate = 48828

# establish connection
rp = win32com.client.Dispatch('RPco.X')
zb = win32com.client.Dispatch('ZBUS.x')
rp.ConnectRX8('GB', 1)

# load circuit
rp.ClearCOF()
rp.LoadCOF(path / 'data' / 'rcx' / 'proto_.rcx')
# rp.LoadCOF(path / 'data' / 'rcx' / 'fail.rcx')
rp.Run()

# set tag values
rp.SetTagVal('f0', 440)  # write value to tag
duration = 0.5  # duration in seconds
n_samples = int(duration*samplerate)
rp.SetTagVal('len', n_samples)

# play
zb.zBusTrigA(0, 0, 20)  # trigger zbus

import numpy
import slab
from matplotlib import pyplot as plt
played = numpy.asarray(rp.ReadTagV('played', 0, n_samples+1000))
played = slab.Sound(played, samplerate=samplerate)
played.waveform()


rp.Halt()

