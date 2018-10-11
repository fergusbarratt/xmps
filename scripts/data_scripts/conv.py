from numpy import load, cumsum as cs, expand_dims as ed, arange as ar, array
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.style.use('ggplot')
k = 30000
step = 3
def av(lys, dt):
    return (1/(dt))*cs(array(lys), axis=0)/ed(ar(1, len(lys)+1), 1)

A = load('data/spectra/run_3/lyapunovs_L8_D1_N40000.npy')[-k:][::step]
avA = av(A, 5e-3)
B = load('data/spectra/run_3/lyapunovs_L8_D2_N40000.npy')[-k:][::step]
avB = av(B, 5e-3)
C = load('data/spectra/run_3/lyapunovs_L8_D3_N40000.npy')[-k:][::step]
avC = av(C, 5e-3)
D = load('data/spectra/run_3/lyapunovs_L8_D4_N40000.npy')[-k:][::step]
avD = av(D, 5e-3)
E = load('data/spectra/run_3/lyapunovs_L8_D5_N40000.npy')[-k:][::step]
avE = av(E, 5e-3)
F = load('data/spectra/run_3/lyapunovs_L8_D6_N40000.npy')[-k:][::step]
avF = av(F, 5e-3)
G = load('data/spectra/run_3/lyapunovs_L8_D7_N40000.npy')[-k:][::step]
avG = av(G, 5e-3)
H = load('data/spectra/run_3/lyapunovs_L8_D8_N40000.npy')[-k:][::step]
avH = av(H, 5e-3)

data = [(A, avA), (B, avB), (C, avC), (D, avD), (E, avE), (F, avF), (G, avG), (H, avH)]
Ds = [1, 2, 3, 4, 5, 6, 7, 8]#, 10]
fig, ax = plt.subplots(2, len(Ds), sharex=True, sharey='row')
ax = ed(ax, -1) if len(Ds)==1 else ax
for m, (lys, exps) in enumerate(data):
    ax[0][m].plot(lys)
    ax[0][m].set_title('D='+str(Ds[m]), loc='right')
    ax[1][m].plot(exps)
fig.tight_layout()
#plt.savefig('images/spectra/conv.pdf', bbox_inches='tight')
plt.show()

