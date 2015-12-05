# -*- coding: utf-8 -*-
# Testing ode

from scipy.integrate import odeint
from pylab import *
from lasp.sound import BioSound

def deriv(y,t, alpha, beta, gamma):
# These are the differential equations of the songbird syrinx Sitt et al. PHYSICAL REVIEW E 81, 031927 (2010)
 g2 = gamma*gamma

 dydt = g2*alpha + g2*beta*y[0] + g2*y[0]*y[0] - gamma*y[0]*y[1] - g2*y[0]*y[0]*y[0] - gamma*y[0]*y[1]

 return array([ y[1], dydt] )

def shm_deriv(y,t,k):
    # These are the differential equations of the songbird syrinx Sitt et al. PHYSICAL REVIEW E 81, 031927 (2010)
    dydt = -k * y[0]
    return array([ y[1], dydt] )

# Runs the ode

samprate = 1000000.0 # 1000 kHz sample rate to get solution
soundlen = 2.0 # 200 ms sound
npts = fix(soundlen*samprate)+1
time = linspace(0.0, soundlen, npts)
yinit = array([1,0.2]) # initial values
alpha = -0.8
beta = -0.8
gamma = 2500.0
k = 300.

#y = odeint(shm_deriv,yinit,time, (k,))
y = odeint(deriv,yinit,time, (alpha, beta, gamma) )

figure(3)
plot(time,y[:,0]) # y[:,0] is the first column of y
xlabel('time')
ylabel('displacement')
show()

figure(4)
plot(y[:,0],y[:,1]);
xlabel('displacement')
ylabel('velocity')
show()

print 'sounding off?'
synSound = BioSound(soundWave=y[:,0], fs=samprate)
synSound.spectrum()
synSound.ampenv()
synSound.spectroCalc()
synSound.fundest()
synSound.plot()
synSound.play()
