'''store a list of instantaneous lyapunov exponents'''
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

from fMPS import fMPS
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum
import matplotlib.pyplot as plt
from tdvp.tdvp_fast import MPO_TFI, MPO_XXZ

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

Sx, Sy, Sz = spins(0.5)

L = 6
bulkH =Sx12@Sx22+Sz22@Sz22+Sx12
H = [Sx12@Sx22+Sz22@Sz22+Sx12+Sx22] + [bulkH for _ in range(L-2)]
W = L*[MPO_TFI(0.25, 0.25, 0.5, 0)]
dt = 1e-1
t_fin = 100  
T = linspace(0, t_fin, int(t_fin//dt)+1)
t_burn = 5
#(100, 10)

mps = fMPS().load('fixtures/product{}.npy'.format(L)).right_canonicalise()

Ds = [2]
for D in Ds:

    F = Trajectory(mps, H=H, W=W)
    F.run_name = 'spectra/lyapunovs'

    exps_ = F.lyapunov2(T, D, t_burn=t_burn)
    F.clear()

    exps, lys, _ = F.lyapunov(T, D, t_burn=t_burn)

    fig, ax = plt.subplots(2, 1)
    print(exps[-1])
    print(exps_)
    ax[0].hist(exps[-1])
    ax[1].hist(exps_)
    plt.show()
