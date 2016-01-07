import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import numpy as np
from lasp.timefreq import gaussian_stft
from lasp.sound import plot_spectrogram
import pdb

fname = 'DCCalls16x16/fbird8call1.wav'
wf = read(fname,'r')[1]

# Frames: 30870.00d, Rate: 44100.00

wl = 0.007 # 7ms
ic = 0.001 # 1ms

t,freq,timefreq,rms = gaussian_stft(wf, 44100, wl, ic)

spec = np.abs(timefreq)
spec = spec/spec.max()
nz = spec > 0
spec[nz] = 20*np.log10(spec[nz]) + 50
spec[spec < 0] = 0

plot_spectrogram(t, freq, spec)
plt.show()
