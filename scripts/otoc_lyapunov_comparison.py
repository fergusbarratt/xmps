import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fMPS import fMPS
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum
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

Sx, Sy, Sz = spins(0.5)

listH = [Sz12@Sz22+Sx12+Sz12] + [Sz12@Sz22+Sx12+Sz12+Sx22+Sz22 for _ in range(4)] + [Sz12@Sz22+Sx22+Sz22]
fullH = sum([n_body(a, i, len(listH), d=2) for i, a in enumerate(listH)], axis=0)

mps = fMPS().random(7, 2, 1).left_canonicalise()

T = linspace(0, 20, 400)

#exps, _, _ = Trajectory(mps, listH).lyapunov(T, D=2, bar=True)
#save('data/lyapunovs_L5_D2', exps)
#exps = load('../data/lyapunovs_L5_D2.npy')

op = Sz4
otocs = Trajectory(mps, fullH, fullH=True).OTOC(T, op)
mpss = Trajectory(mps, listH).trajectory(T)

fig, ax = plt.subplots(1, 1, sharex=True)
#ax[0].plot(exps)
ax.plot(otocs)
plt.show()
