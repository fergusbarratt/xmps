import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fMPS import fMPS
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum, log, array
import matplotlib.pyplot as plt
L = 8
S_list = [N_body_spins(0.5, n, L) for n in range(1, L+1)]
n = 2
m = 4
Sx1, Sy1, Sz1 = S_list[n]
SxL, SyL, SzL = S_list[m]

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)
listH = [Sz12@Sz22+Sx12+Sz12] + [Sz12@Sz22+Sx12+Sz12+Sx22+Sz22 for _ in range(L-3)] + [Sz12@Sz22+Sx22+Sz22]
fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)

def numdiff(F, T):
    return array([F[n+1]-F[n]/(T[n+1]-T[n]) for n in range(len(F)-1)])

mat = load('fixtures/mat{}x{}.npy'.format(L, L))
mps = fMPS().left_from_state(mat)

T = linspace(0, 50, 200)

ops = Sz1, SzL
otocs = Trajectory(mps, fullH, fullH=True).ed_OTOC(T, ops)

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].set_title('$Sz^{}, Sz^{}, L={}$'.format(n+1, m+1, L), loc='right')
ax[0].plot(T, otocs)
ax[1].plot(T, otocs)
ax[2].plot(T[:-1], numdiff(log(otocs), T))
ax[1].set_yscale('log')
fig.tight_layout()
#fig.savefig('images/Sz1Sz7.pdf', bbox_inches='tight')
plt.show()
