import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fMPS import fMPS
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum, log, array
import matplotlib.pyplot as plt
Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 7)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 7)
Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 7)
Sx4, Sy4, Sz4 = N_body_spins(0.5, 4, 7)
Sx5, Sy5, Sz5 = N_body_spins(0.5, 5, 7)
Sx6, Sy6, Sz6 = N_body_spins(0.5, 6, 7)
Sx7, Sy7, Sz7 = N_body_spins(0.5, 7, 7)

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)
listH = [Sz12@Sz22+Sx12+Sz12] + [Sz12@Sz22+Sx12+Sz12+Sx22+Sz22 for _ in range(4)] + [Sz12@Sz22+Sx22+Sz22]
fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)

def numdiff(F, T):
    return array([F[n+1]-F[n]/(T[n+1]-T[n]) for n in range(len(F)-1)])

mat = load('fixtures/mat7x7.npy')
mps = fMPS().left_from_state(mat)

T = linspace(0, 30, 200)

ops = Sz3, Sz5
otocs = Trajectory(mps, fullH, fullH=True).OTOC(T, ops)

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(T, otocs)
ax[1].plot(T, log(otocs))
ax[2].plot(T[:-1], numdiff(log(array(otocs)), T))
ax[3].plot(T[:-2], numdiff(numdiff(log(array(otocs)), T), T[:-1]))
ax[3].set_ylim([-200, 100])
plt.show()
