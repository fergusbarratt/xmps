from numpy import load
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')

A = load('data/spectra/lyapunovs_L8_D1_N20400.npy')[-1]
B = load('data/spectra/lyapunovs_L8_D2_N20400.npy')[-1]
C = load('data/spectra/lyapunovs_L8_D3_N20400.npy')[-1]
D = load('data/spectra/lyapunovs_L8_D4_N20400.npy')[-1]
E = load('data/spectra/lyapunovs_L8_D5_N20400.npy')[-1]
F = load('data/spectra/lyapunovs_L8_D7_N20400.npy')[-1]
data = [A, B, C, D, E, F]
Ds = [1, 2, 3, 4, 5, 7]
fig, ax = plt.subplots(6, 1, sharex=True, figsize=(4, 10))
for m, x in enumerate(data):
    ax[m].hist(x, bins=8)
    ax[m].set_title('D='+str(Ds[m]), loc='right')
fig.tight_layout()
plt.savefig('images/spectra/hists.pdf', bbox_inches='tight')
plt.show()
