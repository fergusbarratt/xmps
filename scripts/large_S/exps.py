'''store a list of instantaneous lyapunov exponents'''
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

from fMPS import fMPS
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum, cumsum as cs, arange as ar
from numpy import array, expand_dims as ed
import matplotlib.pyplot as plt
from tdvp.tdvp_fast import MPO_TFI, MPO_XXZ

def av(lys, dt=1):
    return (1/dt)*cs(array(lys), axis=0)/ed(ar(1, len(lys)+1), -1)

D = 2
Ss = [0.5, 1, 1.5, 2, 2.5, 3]
fig, ax = plt.subplots(len(Ss), 1, sharey=True, sharex=True)
for m, S in enumerate(Ss):

    Sx12, Sy12, Sz12 = N_body_spins(S, 1, 2)
    Sx22, Sy22, Sz22 = N_body_spins(S, 2, 2)

    Sx, Sy, Sz = spins(S)

    L = 2 
    H = [Sx12@Sx22+Sy12@Sy22+Sz12@Sz22+Sx12-Sz22]# + [bulkH for _ in range(L-2)]
    W = L*[MPO_XXZ(1, 0., 0.5, 0., 1, Sx, Sy, Sz)]

    dt = 2e-2
    t_fin = 5
    T = linspace(0, t_fin, int(t_fin//dt)+1)
    t_burn = 5

    mps = fMPS().random(L, int(2*S+1), 2).left_canonicalise(1)

    F = Trajectory(mps, H=H, W=W)
    F.run_name = 'large_S/lyapunovs'
    exps, lys = F.lyapunov(T, D, t_burn=t_burn)
    F.stop()

    ax[m].plot(av(lys[5:], dt))
plt.show()
