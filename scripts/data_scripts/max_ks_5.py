from numpy import load, cumsum as cs, expand_dims as ed, arange as ar
from numpy import array, concatenate as ct, abs
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.style.use('ggplot')
k = 5 
step = 5  
def av(lys):
    return cs(array(lys), axis=0)/ed(ar(1, len(lys)+1), 1)

A = load('data/spectra/run_5/lyapunovs_L6_D1_N20000.npy')[::step][k:]
avA = av(A)                                                         
B = load('data/spectra/run_5/lyapunovs_L6_D2_N20000.npy')[::step][k:]
avB = av(B)                                                         
C = load('data/spectra/run_5/lyapunovs_L6_D3_N20000.npy')[::step][k:]
avC = av(C)                                                         
D = load('data/spectra/run_5/lyapunovs_L6_D4_N20000.npy')[::step][k:]
avD = av(D)                                                         
E = load('data/spectra/run_5/lyapunovs_L6_D5_N20000.npy')[::step][k:]
avE = av(E)                                                         
F = load('data/spectra/run_5/lyapunovs_L6_D6_N20000.npy')[::step][k:]
avF = av(F)                                                         
G = load('data/spectra/run_5/lyapunovs_L6_D7_N20000.npy')[::step][k:]
avG = av(G)                                                         
H = load('data/spectra/run_5/lyapunovs_L6_D8_N20000.npy')[::step][k:]
avH = av(H)                                                          

data = [(A, avA), (B, avB), (C, avC), (D, avD), (E, avE), (F, avF), (G, avG), (H, avH)]
Ds = [1, 2, 3, 4, 5, 6, 7, 8]
data = [abs(dat[1][-1]) for dat in data]

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].scatter(Ds, list(map(sum, data)), marker='x')
ax[0].set_title('KS', loc='right')
ax[1].scatter(Ds, list(map(max, data)), marker='x')
ax[1].set_title('max', loc='right')
ax[0].set_ylim([0, 0.005])
ax[1].set_ylim([0, 0.0002])
#plt.savefig('images/spectra/max_ks_6.pdf', bbox_inches='tight')
plt.show()