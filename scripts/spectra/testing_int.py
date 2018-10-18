import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from fMPS import fMPS
from numpy import array
from fTDVP import Trajectory
from spin import N_body_spins, spins, n_body
from numpy import load, linspace, save, sum
import matplotlib.pyplot as plt
from tdvp.tdvp_fast import MPO_TFI

Sx12, Sy12, Sz12 = N_body_spins(0.5, 1, 2)
Sx22, Sy22, Sz22 = N_body_spins(0.5, 2, 2)

Sx, Sy, Sz = spins(0.5)

L = 4 
bulkH =Sz12@Sz22+Sx22
H = [Sz12@Sz22+Sx12+Sx22] + [bulkH for _ in range(L-2)] 
W = L*[MPO_TFI(0, 0.25, 0.5, 0.)]

dt = 5e-3
t_fin = 2 
T = linspace(0, t_fin, int(t_fin//dt)+1)

if L<10:
    psi_0 = load('fixtures/mat{}x{}.npy'.format(L,L))
    mps = fMPS().left_from_state(psi_0).right_canonicalise(1)
else:
    mps = fMPS().load('fixtures/product{}.npy'.format(L))

Ds = [2]
for D in Ds:
    dt = 5e-3
    t_fin = 100 
    T = linspace(0, t_fin, int(t_fin//dt)+1)
    F = Trajectory(mps, H=H, W=W)
    exps = F.invfreeint(T).mps_evs([Sx, Sy, Sz], 0)
    F.clear()
    dt = 5e-2
    t_fin = 100 
    T_ = linspace(0, t_fin, int(t_fin//dt)+1)
    exps_ = F.invfreeint(T_).mps_evs([Sx, Sy, Sz], 0)
    plt.plot(T, exps)
    plt.plot(T_, exps_)
    plt.show()
