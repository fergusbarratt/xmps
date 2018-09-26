import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fMPS import fMPS
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum, log, array
import matplotlib.pyplot as plt
L = 2
S_list = [N_body_spins(0.5, n, L) for n in range(1, L+1)]
i = 1
j = 1
Sxi, Syi, Szi = S_list[i]
Sxj, Syj, Szj = S_list[j]

Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 2)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 2)

ent = Sx1@Sx2+Sy1@Sy2+Sz1@Sz2
loc = Sx1, -Sz2

listH = [ent+loc[0]+loc[1]] + [ent+loc[1] for _ in range(L-2)]
fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)

def numdiff(F, T):
    return array([F[n+1]-F[n]/(T[n+1]-T[n]) for n in range(len(F)-1)])

mat = load('fixtures/mat{}x{}.npy'.format(L, L))
mps = fMPS().left_from_state(mat)

T = linspace(0, 30, 200)

ops = Szi, Szj
otocs = Trajectory(mps, fullH, fullH=True).ed_OTOC(T, ops)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].set_title('$Sz^{}, Sz^{}, L={}, \lambda={}$'.format(n+1, m+1, L, (log(otocs)[2]-log(otocs)[1])/(T[2]-T[1])), loc='right')
ax[0].plot(T, otocs)
ax[1].plot(T, otocs)
ax[1].set_yscale('log')
fig.tight_layout()
fig.savefig('images/Sz^{}, Sz^{}, L={}, l={}, int.pdf'.format(n+1, m+1, L, (log(otocs)[2]-log(otocs)[1])/(T[2]-T[1])), bbox_inches='tight')
plt.show()
