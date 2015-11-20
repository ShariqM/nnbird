import numpy as np
import matplotlib.pyplot as plt
import pdb
import math

simulate_for = 200
x = np.zeros(simulate_for)
v = np.zeros(simulate_for)
x[0] = 1
v[0] = 0.2
dt = 0.01
k = math.sqrt(math.pi/2)
k = 50

for i in range(1, simulate_for):
    v[i] = v[i-1] - k * x[i-1] * dt
    x[i] = x[i-1] + v[i] * dt

plt.plot(x)
plt.show()
