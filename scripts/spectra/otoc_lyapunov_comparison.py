import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

from fMPS import fMPS
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum, log, array
import matplotlib.pyplot as plt
L = 6
S_list = [N_body_spins(0.5, n, L) for n in range(1, L+1)]
n = 2
m = 2
Sx1, Sy1, Sz1 = S_list[n]
SxL, SyL, SzL = S_list[m]

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)
listH = [Sz12@Sz22+Sx12] + [Sz12@Sz22+Sx12+Sz12+Sx22 for _ in range(L-2)]
fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)

def numdiff(F, T):
    return array([F[n+1]-F[n]/(T[n+1]-T[n]) for n in range(len(F)-1)])

mps = fMPS().load('fixtures/product{}.npy'.format(L))

T = linspace(0, 50, 200)

ops = Sz1, SzL
otocs = Trajectory(mps, fullH, fullH=True).ed_OTOC(T, ops)


fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].set_title('$Sz^{}, Sz^{}, L={}, \lambda={}$'.format(n+1, m+1, L, (log(otocs)[2]-log(otocs)[1])/(T[2]-T[1])), loc='right')
ax[0].plot(T, otocs)
ax[1].plot(T, otocs)
ax[1].set_yscale('log')
fig.tight_layout()
fig.savefig('images/Sz^{}, Sz^{}, L={}, l={}, int.pdf'.format(n+1, m+1, L, (log(otocs)[2]-log(otocs)[1])/(T[2]-T[1])), bbox_inches='tight')
plt.show()
