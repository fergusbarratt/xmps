import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fMPS import fMPS
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum
import matplotlib.pyplot as plt
Sx1, Sy1, Sz1 = N_body_spins(0.5, 1, 5)
Sx2, Sy2, Sz2 = N_body_spins(0.5, 2, 5)
Sx3, Sy3, Sz3 = N_body_spins(0.5, 3, 5)
Sx4, Sy4, Sz4 = N_body_spins(0.5, 4, 5)
Sx5, Sy5, Sz5 = N_body_spins(0.5, 5, 5)

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

Sx, Sy, Sz = spins(0.5)

listH = [Sz12@Sz22+Sx12+Sz12] + [Sz12@Sz22+Sx12+Sz12+Sx22+Sz22 for _ in range(2)] + [Sz12@Sz22+Sx22+Sz22]
fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)

mps = fMPS().left_from_state(load('../fixtures/mat5x5.npy'))

T = linspace(0, 10, 200)

#exps, _, _ = Trajectory(mps, listH).lyapunov(T, D=2, bar=True)
#save('data/lyapunovs_L5_D2', exps)
exps = load('../data/lyapunovs_L5_D2.npy')

op = Sx2
otocs = Trajectory(mps, fullH, fullH=True).OTOC(T, op)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(exps)
ax[1].plot(otocs)
plt.show()
