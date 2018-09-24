import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fMPS import fMPS
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum
import matplotlib.pyplot as plt
from seaborn import distplot

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

Sx, Sy, Sz = spins(0.5)

L = 9 
bulkH =Sz12@Sz22+Sx12+Sx22
H_i = [Sz12@Sz22+Sx12] + [bulkH for _ in range(L-3)] + [Sz12@Sz22+Sx22]
H = [H_i[0]+Sz12]+[H_i[i]+Sz12+Sz22 for i in range(1, L-2)]+[H_i[-1]+Sz22]

dt = 1e-2
N = 2000
T = linspace(0, N*dt, N)

psi_0 = load('fixtures/mat{}x{}.npy'.format(L,L))
mps = fMPS().left_from_state(psi_0).right_canonicalise(1)
Ds = [3]
for D in Ds:
    exps, _ = Trajectory(mps, H).lyapunov(T, D, False)
    plt.plot(exps)
    plt.show()
#    save('../data/lyapunovs_L{}_D{}_N{}'.format(L, D, N), exps)

#Ds = [1, 2, 3, 4]
#exps_D = []
#fig, ax = plt.subplots(4, 1, sharex=True, sharey=True)
#for m, D in enumerate(Ds):
#    exps = load('../data/lyapunovs_L{}_D{}_N{}.npy'.format(L, D, N))
#    ax[m].plot(exps)
#    #exps_D.append(sum(abs(exps[-1])))

#plt.plot(exps_D)
#fig.tight_layout()
#plt.show()


