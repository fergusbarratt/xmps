from numpy import load
from seaborn import distplot
import matplotlib.pyplot as plt

A = load('data/two_spins_chaos_L2_D1_N10000.npy')
B = load('data/two_spins_int_L2_D1_N10000.npy')
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(A)
ax[1].plot(B)
plt.show()
