from numpy import load, cumsum as cs, expand_dims as ed, arange as ar, array
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')
k = 10000
def av(lys, dt):
    return (1/(dt))*cs(array(lys), axis=0)/ed(ar(1, len(lys)+1), 1)

A = load('data/spectra/lyapunovs_L8_D2_N120.npy')[-k:]
avA = av(A, 5e-3)
#B = load('data/spectra/lyapunovs_L8_D2_N20400.npy')[-1000:]
#C = load('data/spectra/lyapunovs_L8_D3_N20400.npy')[-1000:]
#D = load('data/spectra/lyapunovs_L8_D4_N20400.npy')[-1000:]
#E = load('data/spectra/lyapunovs_L8_D5_N20400.npy')[-1000:]
#F = load('data/spectra/lyapunovs_L8_D6_N20400.npy')[-1000:]
#G = load('data/spectra/lyapunovs_L8_D7_N20400.npy')[-1000:]
#H = load('data/spectra/lyapunovs_L8_D8_N21000.npy')[-1000:]
#I = load('data/spectra/lyapunovs_L8_D9_N21000.npy')[-1000:]
#J = load('data/spectra/lyapunovs_L8_D10_N21000.npy')[-1000:]
#K = load('data/spectra/lyapunovs_L8_D12_N21600.npy')[-1000:]

data = [(A, avA)]#, B, C, D, E, F, G, H, I, J, K]
Ds = [1]#, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
fig, ax = plt.subplots(2, len(Ds), sharex=True, figsize=(3, 6))
ax = ed(ax, -1) if len(Ds)==1 else ax
for m, (lys, exps) in enumerate(data):
    ax[0][m].plot(lys)
    ax[0][m].set_title('D='+str(Ds[m]), loc='right')
    ax[1][m].plot(exps)
fig.tight_layout()
#plt.savefig('images/spectra/conv.pdf', bbox_inches='tight')
plt.show()

