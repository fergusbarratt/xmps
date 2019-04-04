import matplotlib as mpl
from numpy import load, cumsum as cs, expand_dims as ed, arange as ar
from numpy import array, concatenate as ct
import matplotlib.pyplot as plt
#mpl.style.use('ggplot')
k = 5
step = 50
def av(lys):
    return cs(array(lys), axis=0)/ed(ar(1, len(lys)+1), 1)

A = load('data/spectra/lyapunovs_L6_D1_N80000.npy')[::step][k:]
avA = av(A)
B = load('data/spectra/lyapunovs_L6_D3_N80000.npy')[::step][k:]
avB = av(B)
C = load('data/spectra/lyapunovs_L6_D5_N80000.npy')[::step][k:]
avC = av(C)
D = load('data/spectra/lyapunovs_L6_D7_N80000.npy')[::step][k:]
avD = av(D)

data = [(A, avA), (B, avB), (C, avC), (D, avD)]#(E, avE), (F, avF), (G, avG)]
Ds = [1, 3, 5, 7]#, 5, 6, 7, 8]
fig, ax = plt.subplots(2, len(Ds), sharex=True, sharey='row')
ax = ed(ax, -1) if len(Ds)==1 else ax
for m, (lys, exps) in enumerate(data):
    ax[0][m].plot(lys)
    ax[0][m].set_title('D='+str(Ds[m]), loc='right')
    ax[1][m].plot(exps)
fig.tight_layout()
#plt.savefig('images/spectra/conv.pdf', bbox_inches='tight')
plt.show()

