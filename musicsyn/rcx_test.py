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
rp.SetTagVal('len', int(duration*samplerate))

rp.Halt()

zb.zBusTrigA(0, 0, 20)  # trigger zbus