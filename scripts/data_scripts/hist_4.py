from numpy import load, cumsum as cs, expand_dims as ed, arange as ar, 
from numpy import array, concatenate as ct
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.style.use('ggplot')
k = 5 
step = 1  
def av(lys):
    return cs(array(lys), axis=0)/ed(ar(1, len(lys)+1), 1)

A = load('data/spectra/lyapunovs_L6_D1_N5000.npy')[::step][k:]
avA = av(A)                                                          
A_ = load('data/spectra/2_lyapunovs_L6_D1_N5000.npy')[::step][k:]
B = load('data/spectra/lyapunovs_L6_D2_N5000.npy')[::step][k:]
avB = av(B)                                                          
C = load('data/spectra/lyapunovs_L6_D3_N5000.npy')[::step][k:]
avC = av(C)                                                          
D = load('data/spectra/lyapunovs_L6_D4_N5000.npy')[::step][k:]
avD = av(D)                                                          
E = load('data/spectra/lyapunovs_L6_D5_N5000.npy')[::step][k:]
avE = av(E)                                                          
F = load('data/spectra/lyapunovs_L6_D6_N5000.npy')[::step][k:]
avF = av(F)                                                          
G = load('data/spectra/lyapunovs_L6_D7_N5000.npy')[::step][k:]
avG = av(G)                                                          
H = load('data/spectra/lyapunovs_L6_D8_N5000.npy')[::step][k:]
avH = av(H)                                                          

data = [(A, avA), (B, avB), (C, avC), (D, avD), (E, avE), (F, avF), (G, avG)]
Ds = [1, 2, 3, 4, 5, 6, 7, 8]
fig, ax = plt.subplots(len(Ds), 1, sharex=True, sharey='row')
ax = ed(ax, -1) if len(Ds)==1 else ax
for m, x in enumerate(data):
    ax[m].hist(x[1][-1], bins=10)
    ax[m].set_title('D='+str(Ds[m]), loc='right')
fig.tight_layout()
#plt.savefig('images/spectra/hists.pdf', bbox_inches='tight')
plt.show()
