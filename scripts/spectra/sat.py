import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,os.path.dirname(parentdir))

from fTDVP import Trajectory
from fMPS import fMPS
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum, array, concatenate as ct, stack as st, log, exp
from numpy import ceil, real as re, cumsum as cs, arange as ar, array, sqrt
from numpy import log2
import matplotlib.pyplot as plt
import matplotlib as mpl
from tdvp.tdvp_fast import MPO_TFI
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
mpl.style.use('ggplot')
L = 6  
S_list = [N_body_spins(0.5, n, L) for n in range(1, L+1)]
i = 2
j = 2
Sxi, Syi, Szi = S_list[i]
Sxj, Syj, Szj = S_list[j]

Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

g, h = (sqrt(5)+2)/8, (sqrt(5)+1)/4
#ent = 4*Sz1@Sz2
#loc = 2*g*Sx1+2*h*Sz1, 2*g*Sx2+2*h*Sz2
ent = Sz1@Sz2
loc = Sx1+Sz1, Sx2+Sz2

listH = [ent+loc[0]+loc[1]] + [ent+loc[1] for _ in range(L-2)]
#listH[0] += 2*(h-1)*(Sz1)
#listH[-1]-= 2*(h-1)*(Sz2)
fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)

dt = 0.1
t_fin = 300
T = linspace(0, t_fin, int(t_fin//dt)+1)
T = linspace(0, 1000, 10000)
#psi_0 = load('fixtures/mat{}x{}.npy'.format(L,L))

#mps = fMPS().left_from_state(psi_0).left_canonicalise(1).expand(D)
fMPS().random(L, 2, 1).right_canonicalise().store('fixtures/product{}.npy'.format(L))
D = 8
for D in range(2, 8):
    mps = fMPS().load('fixtures/product{}.npy'.format(L)).right_canonicalise().expand(D)
    W = L*[MPO_TFI(0, 0.5, 0.25, 0.25)]

    F = Trajectory(mps, H=listH, W=W, fullH=False)
    F.run_name = 'spectra/entanglement'
    F.invfreeint(T)
    sch = array(F.schmidts())
    λ = array([max([-re(s@log(s)) for s in S]) for S in sch])
    save('data/S_{}'.format(D), λ)

fig, ax = plt.subplots(1, 1, sharex=True)
ax.plot(T[1:], λ)
ax.set_title('$S_E$', loc='right')
#fig.savefig('images/spectra/entanglement.pdf')

fig.tight_layout()
plt.show()
