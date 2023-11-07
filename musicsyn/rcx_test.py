import win32com.client
from pathlib import Path

path = Path.cwd()
samplerate = 44828

rp = win32com.client.Dispatch('RPco.X')
zb = win32com.client.Dispatch('ZBUS.x')
rp.ConnectRX8('GB', 1)

rp.ClearCOF()
rp.LoadCOF(path / 'data' / 'rcx' / 'proto_.rcx')

rp.Run()

rp.SetTagVal('f0', 493.88)  # write value to tag
duration = 2  # duration in seconds
n_samples = int(duration*samplerate)
rp.SetTagVal('len', n_samples)


rp.Halt()

zb.zBusTrigA(0, 0, 20)  # trigger zbus

import numpy
import slab

played = numpy.asarray(rp.ReadTagV('played', 0, n_samples))
played = slab.Sound(played, samplerate=samplerate)
played.waveform()

