import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fMPS import fMPS
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum
import pandas as pd
import matplotlib.pyplot as plt

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

Sx, Sy, Sz = spins(0.5)

L, D = 6, 1
bulkH =Sz12@Sz22+Sx12+Sx22
H_i = [Sz12@Sz22+Sx12] + [bulkH for _ in range(L-3)] + [Sz12@Sz22+Sx22]
H_c = [H_i[0]+Sz12]+[H_i[i]+Sz12+Sz22 for i in range(1, L-2)]+[H_i[-1]+Sz22]

T = linspace(0, 10, 1000)
mps_1 = fMPS().random(L, 2, D).left_canonicalise()
mps_2 = mps_1.copy()

exps_i, _, _ = Trajectory(mps_1, H_i).lyapunov(T)
exps_c, _, _ = Trajectory(mps_2, H_c).lyapunov(T)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax[0].plot(exps_i)
ax[1].plot(exps_c)

plt.show()
