from numpy import load, cumsum as cs, expand_dims as ed, arange as ar
from numpy import array, concatenate as ct
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.style.use('ggplot')
k = 5 
step = 5  
burn = 500
def av(lys):
    return cs(array(lys), axis=0)/ed(ar(1, len(lys)+1), 1)

A = load('data/spectra/run_6/lyapunovs_L6_D1_N80000.npy')[burn:][::step][k:]
avA = av(A)                                              
B = load('data/spectra/run_6/lyapunovs_L6_D2_N80000.npy')[burn:][::step][k:]
avB = av(B)                                             
C = load('data/spectra/run_6/lyapunovs_L6_D3_N80000.npy')[burn:][::step][k:]
avC = av(C)                                            
D = load('data/spectra/run_6/lyapunovs_L6_D4_N80000.npy')[burn:][::step][k:]
avD = av(D)                                           
E = load('data/spectra/run_6/lyapunovs_L6_D5_N80000.npy')[burn:][::step][k:]
avE = av(E)                                          
F = load('data/spectra/run_6/lyapunovs_L6_D6_N80000.npy')[burn:][::step][k:]
avF = av(F)                                         
G = load('data/spectra/run_6/lyapunovs_L6_D7_N80000.npy')[burn:][::step][k:]
avG = av(G)                                        
H = load('data/spectra/run_6/lyapunovs_L6_D8_N80000.npy')[burn:][::step][k:]
avH = av(H)                                                          

data = [(A, avA), (B, avB), (C, avC), (D, avD), (E, avE), (F, avF), (G, avG)]
Ds = [1, 2, 3, 4, 5, 6, 7, 8]
fig, ax = plt.subplots(2, len(Ds), sharex=True, sharey='row')
ax = ed(ax, -1) if len(Ds)==1 else ax
for m, (lys, exps) in enumerate(data):
    ax[0][m].plot(lys)
    ax[0][m].set_title('D='+str(Ds[m]), loc='right')
    ax[1][m].plot(exps)
fig.tight_layout()
#plt.savefig('images/spectra/conv.pdf', bbox_inches='tight')
plt.show()

