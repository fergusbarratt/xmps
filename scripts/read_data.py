from numpy import load
import matplotlib.pyplot as plt

A = load('data/spectra/lyapunovs_L8_D1_N20100.npy')
B = load('data/spectra/lyapunovs_L8_D2_N20100.npy')
C = load('data/spectra/lyapunovs_L8_D3_N20100.npy')
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(A)
ax[1].plot(B)
ax[2].plot(B)
plt.show()
