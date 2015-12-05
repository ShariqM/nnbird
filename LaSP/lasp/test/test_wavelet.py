from unittest import TestCase

import numpy as np

import matplotlib.pyplot as plt
from numpy.fft import fftfreq
from scipy.fftpack import fft


class WaveletTest(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_wavelets(self):

        sr = 1e3
        t = np.arange(-10.0, 10.0, 1.0/sr)
        psi = lambda t,f,bw: (np.pi*bw)**(-0.5) * np.exp(2*np.pi*complex(0, 1)*f*t) * np.exp(-t**2 / bw)

        freqs = [1.0, 4.0, 8.0, 15.0, 30.0, 50, 150.0]
        bws = [1e-4, 1e-2, 1e-1, 1]

        tfig = plt.figure()
        ffig = plt.figure()
        nrows = len(freqs)
        ncols = len(bws)

        for k,freq in enumerate(freqs):
            for j,bw in enumerate(bws):

                #compute wavelet
                z = psi(t, freq, bw)

                #compute FFT at wavelet
                max_freq = 190.0
                zfft = fft(z)
                zfreq = fftfreq(len(z), d=1.0/sr)
                fi = (zfreq > 0.0) & (zfreq < max_freq)
                zfft = zfft[fi]
                zfreq = zfreq[fi]
                ps = np.abs(zfft)

                #plot wavelet
                sp = k*ncols + j + 1
                plt.figure(tfig.number)
                ax = plt.subplot(nrows, ncols, sp)
                plt.plot(t, z.real, 'k-')
                plt.plot(t, z.imag, 'r-', alpha=0.5)
                plt.axis('tight')
                plt.title('f=%0.2f, bw=%0.2f' % (freq, bw))

                #plot power spectrum of wavelet
                plt.figure(ffig.number)
                ax = plt.subplot(nrows, ncols, sp)
                plt.plot(zfreq, ps, 'k-', linewidth=2.0)
                plt.axis('tight')
                plt.title('f=%0.2f, bw=%0.2f' % (freq, bw))

        plt.show()