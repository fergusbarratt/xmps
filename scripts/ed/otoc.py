import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

from fMPS import fMPS
from tdvp.tdvp_fast import MPO_TFI
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum, log, array, cumsum as cs
from numpy import arange as ar, mean
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensor import H as cT, C as c
mpl.style.use('ggplot')
L = 6
S_list = [N_body_spins(0.5, n, L) for n in range(1, L+1)]
i = 2
j = 2
Sxi, Syi, Szi = S_list[i]
Sxj, Syj, Szj = S_list[j]

Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

ent = Sz1@Sz2
loc = (Sx1+Sz1), (Sx2+Sz2)

listH = [ent+loc[0]+loc[1]] + [ent+loc[1] for _ in range(L-2)]
fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)

mpss = Trajectory(fMPS().load('fixtures/product{}.npy'.format(L)),
                  H=listH,
                  W=L*[MPO_TFI(0, 0.25, 0.5, 0.5)]).invfreeint(
                          linspace(0, 1000, 2000), 'high').mps_list()
otocss = []
for mps in mpss:
    T = linspace(0, 20, 100)
    ops = Szi, Szj 
    #print(c(mps.recombine().reshape(-1))@fullH@mps.recombine().reshape(-1))
    otocs = array(Trajectory(mps, fullH, fullH=True).ed_OTOC(T, ops))
    otocss.append(otocs)
otocss = array(otocss)
plt.plot(otocss[::10].T)
plt.show()

ma = mean(otocss, axis=0)
plt.plot(log(ma)-log(ma)[1])
plt.show()
#save('data/otocs', log(otocs)-log(otocs)[1])

#fig, ax = plt.subplots(1, 1, sharex=True)
#ax.set_title('$Sz{}, Sz{}$'.format(i+1, j+1))
##ax.plot(T, log(otocs)-log(otocs)[1])
#fig.tight_layout()
##fig.savefig('images/spectra/otoc.pdf')
#plt.show()
