from numpy import load
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')

A = abs(load('data/spectra/lyapunovs_L8_D1_N20400.npy')[-1])
B = abs(load('data/spectra/lyapunovs_L8_D2_N20400.npy')[-1])
C = abs(load('data/spectra/lyapunovs_L8_D3_N20400.npy')[-1])
D = abs(load('data/spectra/lyapunovs_L8_D4_N20400.npy')[-1])
E = abs(load('data/spectra/lyapunovs_L8_D5_N20400.npy')[-1])
F = abs(load('data/spectra/lyapunovs_L8_D6_N20400.npy')[-1])
G = abs(load('data/spectra/lyapunovs_L8_D7_N20400.npy')[-1])
data = [A, B, C, D, E, F, G]
Ds = [1, 2, 3, 4, 5, 6, 7]
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].scatter(Ds, list(map(sum, data)), marker='x')
ax[0].set_title('KS', loc='right')
ax[1].scatter(Ds, list(map(max, data)), marker='x')
ax[1].set_title('max', loc='right')
plt.savefig('images/spectra/max_ks.pdf', bbox_inches='tight')
plt.show()
