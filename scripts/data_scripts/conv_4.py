from numpy import load, cumsum as cs, expand_dims as ed, arange as ar, array
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.style.use('ggplot')
k = 5 
step = 1  
def av(lys):
    return cs(array(lys), axis=0)/ed(ar(1, len(lys)+1), 1)

A = load('data/spectra/lyapunovs_L6_D1_N5000.npy')[::step][k:]
avA = av(A)                                                          
#B = load('data/spectra/lyapunovs_L6_D2_N1000.npy')[::step][k:]
#avB = av(B)                                                          

data = [(A, avA)]#, (B, avB)]
Ds = [1]#, 2]
fig, ax = plt.subplots(2, len(Ds), sharex=True, sharey='row')
ax = ed(ax, -1) if len(Ds)==1 else ax
for m, (lys, exps) in enumerate(data):
    ax[0][m].plot(lys)
    ax[0][m].set_title('D='+str(Ds[m]), loc='right')
    ax[1][m].plot(exps)
fig.tight_layout()
#plt.savefig('images/spectra/conv.pdf', bbox_inches='tight')
plt.show()

