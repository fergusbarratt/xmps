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
from tdvp.tdvp_fast import MPO_TFI

Sx12, Sy12, Sz12 = N_body_spins(1, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(1, 2, 2)

Sx, Sy, Sz = spins(1)

L = 4 
bulkH =Sz12@Sz22+Sx22+Sz22
H = [Sz12@Sz22+Sx12+Sz12+Sx22+Sz22] + [bulkH for _ in range(L-2)]
W = L*[MPO_TFI(0, 0.25, 0.5, 0.5, Sx, Sz)]

dt = 2e-2
t_fin = 5
T = linspace(0, t_fin, int(t_fin//dt)+1)
t_burn = 5
load_basis = False
Q = None
#(100, 10)

mps = fMPS().random(L, 3, 20).left_canonicalise(1)

Ds = [1]
for D in Ds:
    #if load_basis and 8>=D:
    #    Q = load('data/bases/spectra/40000s/lyapunovs_L8_D{}_N40000_basis.npy'.format(D))

    F = Trajectory(mps, H=H, W=W)
    F.run_name = 'large_S/lyapunovs'
    exps, lys = F.lyapunov(T, D, t_burn=t_burn)
    F.stop()

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(lys)
    ax[1].plot(exps)
    plt.show()
