import numpy as np
import matplotlib.pyplot as plt
import pdb
import math

simulate_for = 5
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
    print "x=%.3f, v=%.3f, vdot=%.3f" % (x[i], v[i], vdot)
    #if i > 10:
        #break

plt.plot(x)
plt.show()
