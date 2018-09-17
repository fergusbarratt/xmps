import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fMPS import fMPS
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum
import matplotlib.pyplot as plt

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

Sx, Sy, Sz = spins(0.5)

L, D = 5, 2
bulkH =Sz12@Sz22+Sx12+Sx22
H = [Sz12@Sz22+Sx12] + [bulkH for _ in range(L-3)] + [Sz12@Sz22+Sx22]

T = linspace(0, 10, 1000)
mps = fMPS().random(L, 2, D).left_canonicalise()
exps, lys, _ = Trajectory(mps, H).lyapunov(T)
plt.plot(exps)
plt.show()
