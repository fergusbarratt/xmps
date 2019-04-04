from numpy import load
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')

A = load('data/spectra/lyapunovs_L8_D2_N1200.npy')[-1]
#B = load('data/spectra/lyapunovs_L8_D2_N20400.npy')[-1]
#C = load('data/spectra/lyapunovs_L8_D3_N20400.npy')[-1]
#D = load('data/spectra/lyapunovs_L8_D4_N20400.npy')[-1]
#E = load('data/spectra/lyapunovs_L8_D5_N20400.npy')[-1]
#F = load('data/spectra/lyapunovs_L8_D6_N20400.npy')[-1]
#G = load('data/spectra/lyapunovs_L8_D7_N20400.npy')[-1]
#H = load('data/spectra/lyapunovs_L8_D8_N21000.npy')[-1]
#I = load('data/spectra/lyapunovs_L8_D9_N21000.npy')[-1]
#J = load('data/spectra/lyapunovs_L8_D10_N21000.npy')[-1]
#K = load('data/spectra/lyapunovs_L8_D12_N21600.npy')[-1]
data = [A]#, B, C, D, E, F, G, H, I, J, K]
Ds = [1]#, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
fig, ax = plt.subplots(len(Ds), 1, sharex=True, figsize=(4, 10))
ax = [ax] if len(Ds)==1 else ax
for m, x in enumerate(data):
    ax[m].hist(x, bins=15)
    ax[m].set_title('D='+str(Ds[m]), loc='right')
fig.tight_layout()
#plt.savefig('images/spectra/hists.pdf', bbox_inches='tight')
plt.show()
