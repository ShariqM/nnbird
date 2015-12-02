import numpy as np
import matplotlib.pyplot as plt
import pdb
import math

simulate_for = 90
x = np.zeros(simulate_for)
v = np.zeros(simulate_for)
x[0] = 1.0
v[0] = 0.0
dt = 0.01
k = math.sqrt(math.pi/2)
k = 50

for i in range(1, simulate_for):
    vdot = -k * x[i-1]
    v[i] = v[i-1] + vdot * dt
    x[i] = x[i-1] + v[i] * dt
    #print "x=%.3f, v=%.3f, vdot=%.3f" % (x[i], v[i], vdot)
    #if i > 10:
        #break

#simulate_for = 256
#t = np.arange(simulate_for)
#x = np.sin(t)
sp = np.fft.fft(x)
#for i in range(simulate_for):
    #if sp.real[i] <  1:
        #sp.real[i] = 0
#plt.plot(sp.real)
#plt.show()
pdb.set_trace()

for k in range(simulate_for):
    total = 0
    for m in range(simulate_for):
        total = total + x[m] * math.cos(2 * math.pi * m * k * (1./simulate_for))
    print 'A_%d = %.2f' % (k,total)

freq = np.fft.fftfreq(simulate_for)
#plt.plot(freq, sp.real, freq, sp.imag)
plt.plot(freq, sp.real)
plt.show()

plt.plot(x)
plt.show()
