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

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

Sx, Sy, Sz = spins(0.5)

L = 4
bulkH =Sz12@Sz22+Sx22+Sz22
H = [Sz12@Sz22+Sx12+Sz12+Sx22+Sz22] + [bulkH for _ in range(L-2)]
W = L*[MPO_TFI(0, 0.25, 0.5, 0.5)]

dt = 1e-2
t_fin = 100
T = linspace(0, t_fin, int(t_fin//dt)+1)
t_burn = 5
#(100, 10)

mps = fMPS().load('fixtures/product{}.npy'.format(L)).right_canonicalise()

Ds = [1]
for D in Ds:

    F = Trajectory(mps, H=H, W=W)
    F.run_name = 'spectra/lyapunovs'
    exps, lys, lys_ = F.lyapunov(T, D, t_burn=t_burn)
    exps, lys = F.lyapunov2(T, D, t_burn=T_burn)
    F.stop()
