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
from seaborn import distplot

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

Sx, Sy, Sz = spins(0.5)

L, D = 6, 5
bulkH =Sz12@Sz22+Sx12+Sx22
#H_i = [Sz12@Sz22+Sx12] + [bulkH for _ in range(L-3)] + [Sz12@Sz22+Sx22]
H = [H_i[0]+Sz12]+[H_i[i]+Sz12+Sz22 for i in range(1, L-2)]+[H_i[-1]+Sz22]

T = linspace(0, 20, 20)
mps_1 = fMPS().random(L, 2, D).left_canonicalise()
mps_2 = mps_1.copy()

#exps_i, _, _ = Trajectory(mps_1, H_i).lyapunov(T)
exps_c, _, _ = Trajectory(mps_2, H_c).lyapunov(T)

#save('../data/i_lyapunovs_L6_D5', exps_i)
#save('../data/c_lyapunovs_L6_D5', exps_c)

#exps_i = load('../data/i_lyapunovs_L6_D1.npy')
#exps_c = load('../data/c_lyapunovs_L6_D1.npy')

#fig, ax = plt.subplots(2, 2, sharex='col', sharey='col')
#ax[0][0].plot(exps_i)
#distplot(exps_i[-1], ax=ax[0][1], bins=6)
#ax[1][0].plot(exps_c)
#distplot(exps_c[-1], ax=ax[1][1], bins=6)

#plt.savefig('../images/lyapunovs_int_reg.pdf', bbox_inches='tight')
plt.show()
