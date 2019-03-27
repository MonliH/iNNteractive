import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**2*0.01 - x**2*0.0095



t = np.arange(0.0, 1000, 0.01)
s = f(t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.grid()

plt.show()
